"""Read-only PostgreSQL inspection and catalog-validation adapter.

The adapter queries PostgreSQL's system catalogs; it never inspects business
rows and never writes to the source.  Credential lookup is injected as a
connection factory so connection references remain opaque to YoDb.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import re
from typing import Any, Protocol

from ..catalog import Catalog, SourceKind
from ..connections.contracts import SourceConnectionAdapter
from ..errors import ErrorCode, ErrorDetail, SourceInspectionError, YoDbError
from .relational import error as _error, validate_datasets
from .contracts import (
    FindingSeverity,
    InspectionCapability,
    InspectionRequest,
    PhysicalCheckConstraint,
    PhysicalField,
    PhysicalForeignKey,
    PhysicalIndex,
    PhysicalKey,
    PhysicalResource,
    PhysicalUniqueConstraint,
    ResourceKind,
    SourceInspection,
    SourceValidationReport,
    ValidationFinding,
)


class PostgresCursor(Protocol):
    description: Sequence[Sequence[Any]] | None

    def execute(self, query: str) -> Any: ...

    def fetchall(self) -> Sequence[Sequence[Any] | Mapping[str, Any]]: ...

    def close(self) -> Any: ...


class PostgresConnection(Protocol):
    def cursor(self) -> PostgresCursor: ...

    def rollback(self) -> Any: ...

    def close(self) -> Any: ...


_VECTOR_DIMENSIONS = re.compile(r"^vector\((?P<dimensions>\d+)\)$", re.IGNORECASE)
_ACTION_CODES = {"a": "no_action", "r": "restrict", "c": "cascade", "n": "set_null", "d": "set_default"}

_SERVER_VERSION_SQL = "/* yodb:server_version */ SHOW server_version"
_EXTENSIONS_SQL = """/* yodb:extensions */
SELECT extname, extversion
FROM pg_extension
ORDER BY extname
"""
_RESOURCES_SQL = """/* yodb:resources */
SELECT n.nspname || '.' || c.relname AS resource_name,
       CASE WHEN c.relkind IN ('v', 'm') THEN 'view' ELSE 'table' END AS resource_kind
FROM pg_class AS c
JOIN pg_namespace AS n ON n.oid = c.relnamespace
WHERE c.relkind IN ('r', 'p', 'v', 'm', 'f')
  AND n.nspname NOT IN ('pg_catalog', 'information_schema')
  AND n.nspname NOT LIKE 'pg_toast%'
ORDER BY resource_name
"""
_COLUMNS_SQL = """/* yodb:columns */
SELECT n.nspname || '.' || c.relname AS resource_name,
       a.attname AS field_name,
       format_type(a.atttypid, a.atttypmod) AS native_type,
       NOT a.attnotnull AS nullable,
       pg_get_expr(ad.adbin, ad.adrelid) AS default_expression,
       (a.attgenerated <> '' OR a.attidentity <> '') AS generated
FROM pg_attribute AS a
JOIN pg_class AS c ON c.oid = a.attrelid
JOIN pg_namespace AS n ON n.oid = c.relnamespace
LEFT JOIN pg_attrdef AS ad ON ad.adrelid = a.attrelid AND ad.adnum = a.attnum
WHERE c.relkind IN ('r', 'p', 'v', 'm', 'f')
  AND n.nspname NOT IN ('pg_catalog', 'information_schema')
  AND n.nspname NOT LIKE 'pg_toast%'
  AND a.attnum > 0
  AND NOT a.attisdropped
ORDER BY resource_name, a.attnum
"""
_KEYS_AND_CHECKS_SQL = """/* yodb:keys_and_checks */
SELECT n.nspname || '.' || c.relname AS resource_name,
       con.conname AS constraint_name,
       con.contype AS constraint_type,
       array_agg(att.attname ORDER BY key_part.ordinality)
         FILTER (WHERE att.attname IS NOT NULL) AS fields,
       con.condeferrable AS deferrable,
       con.condeferred AS initially_deferred,
       pg_get_constraintdef(con.oid, true) AS definition
FROM pg_constraint AS con
JOIN pg_class AS c ON c.oid = con.conrelid
JOIN pg_namespace AS n ON n.oid = c.relnamespace
LEFT JOIN LATERAL unnest(con.conkey) WITH ORDINALITY AS key_part(attnum, ordinality) ON true
LEFT JOIN pg_attribute AS att ON att.attrelid = c.oid AND att.attnum = key_part.attnum
WHERE con.contype IN ('p', 'u', 'c')
  AND n.nspname NOT IN ('pg_catalog', 'information_schema')
  AND n.nspname NOT LIKE 'pg_toast%'
GROUP BY n.nspname, c.relname, con.oid
ORDER BY resource_name, constraint_name
"""
_FOREIGN_KEYS_SQL = """/* yodb:foreign_keys */
SELECT n.nspname || '.' || c.relname AS resource_name,
       con.conname AS constraint_name,
       array_agg(local_field.attname ORDER BY key_part.ordinality) AS fields,
       target_namespace.nspname || '.' || target_table.relname AS target_resource,
       array_agg(target_field.attname ORDER BY key_part.ordinality) AS target_fields,
       con.confupdtype AS update_action,
       con.confdeltype AS delete_action,
       con.condeferrable AS deferrable,
       con.condeferred AS initially_deferred
FROM pg_constraint AS con
JOIN pg_class AS c ON c.oid = con.conrelid
JOIN pg_namespace AS n ON n.oid = c.relnamespace
JOIN pg_class AS target_table ON target_table.oid = con.confrelid
JOIN pg_namespace AS target_namespace ON target_namespace.oid = target_table.relnamespace
JOIN LATERAL unnest(con.conkey, con.confkey) WITH ORDINALITY
  AS key_part(local_attnum, target_attnum, ordinality) ON true
JOIN pg_attribute AS local_field ON local_field.attrelid = c.oid AND local_field.attnum = key_part.local_attnum
JOIN pg_attribute AS target_field ON target_field.attrelid = target_table.oid AND target_field.attnum = key_part.target_attnum
WHERE con.contype = 'f'
  AND n.nspname NOT IN ('pg_catalog', 'information_schema')
  AND n.nspname NOT LIKE 'pg_toast%'
GROUP BY n.nspname, c.relname, con.oid, target_namespace.nspname, target_table.relname
ORDER BY resource_name, constraint_name
"""
# Apache AGE's own catalog: every graph and its vertex ('v') and edge ('e') labels.  Read only when the
# extension is installed; it needs no LOAD, so a read-only role can inspect a graph before using it.
_GRAPHS_SQL = """/* yodb:graphs */
SELECT g.name::text AS graph_name,
       l.name::text AS label_name,
       l.kind::text AS label_kind
FROM ag_catalog.ag_label AS l
JOIN ag_catalog.ag_graph AS g ON g.graphid = l.graph
ORDER BY graph_name, label_name
"""
_INDEXES_SQL = """/* yodb:indexes */
SELECT n.nspname || '.' || c.relname AS resource_name,
       index_class.relname AS index_name,
       access_method.amname AS method,
       index_data.indisunique AS is_unique,
       index_data.indisvalid AS is_valid,
       array_agg(attribute.attname ORDER BY key_part.ordinality)
         FILTER (WHERE key_part.ordinality <= index_data.indnkeyatts AND attribute.attname IS NOT NULL) AS fields,
       array_agg(attribute.attname ORDER BY key_part.ordinality)
         FILTER (WHERE key_part.ordinality > index_data.indnkeyatts AND attribute.attname IS NOT NULL) AS include_fields,
       pg_get_expr(index_data.indpred, index_data.indrelid) AS predicate,
       pg_get_indexdef(index_data.indexrelid) AS definition
FROM pg_index AS index_data
JOIN pg_class AS c ON c.oid = index_data.indrelid
JOIN pg_namespace AS n ON n.oid = c.relnamespace
JOIN pg_class AS index_class ON index_class.oid = index_data.indexrelid
JOIN pg_am AS access_method ON access_method.oid = index_class.relam
LEFT JOIN LATERAL unnest(index_data.indkey) WITH ORDINALITY AS key_part(attnum, ordinality) ON true
LEFT JOIN pg_attribute AS attribute ON attribute.attrelid = c.oid AND attribute.attnum = key_part.attnum
WHERE c.relkind IN ('r', 'p', 'v', 'm', 'f')
  AND n.nspname NOT IN ('pg_catalog', 'information_schema')
  AND n.nspname NOT LIKE 'pg_toast%'
GROUP BY n.nspname, c.relname, index_class.relname, access_method.amname, index_data.indexrelid,
         index_data.indisunique, index_data.indisvalid, index_data.indnkeyatts, index_data.indpred, index_data.indrelid
ORDER BY resource_name, index_name
"""


class PostgresSourceInspector:
    """Inspect PostgreSQL catalogs through an injected read-only connection factory."""

    source_kind = SourceKind.POSTGRES

    def __init__(self, connections: SourceConnectionAdapter[PostgresConnection]) -> None:
        self._connections = connections

    def inspect(self, request: InspectionRequest) -> SourceInspection:
        if request.source.kind is not SourceKind.POSTGRES:
            raise SourceInspectionError(
                ErrorDetail(
                    code=ErrorCode.SOURCE_KIND_UNSUPPORTED,
                    message="The PostgreSQL inspector only supports postgres sources.",
                    retryable=False,
                    source_name=request.source_name,
                )
            )

        try:
            with self._connections.acquire(request.source.connection_ref) as connection:
                cursor = connection.cursor()
                try:
                    cursor.execute("SET TRANSACTION READ ONLY")
                    version_rows = _fetch_rows(cursor, _SERVER_VERSION_SQL)
                    extension_rows = _fetch_rows(cursor, _EXTENSIONS_SQL)
                    resource_rows = _fetch_rows(cursor, _RESOURCES_SQL)
                    column_rows = _fetch_rows(cursor, _COLUMNS_SQL)
                    constraint_rows = _fetch_rows(cursor, _KEYS_AND_CHECKS_SQL)
                    foreign_key_rows = _fetch_rows(cursor, _FOREIGN_KEYS_SQL)
                    index_rows = _fetch_rows(cursor, _INDEXES_SQL)
                    graph_rows = (
                        _fetch_rows(cursor, _GRAPHS_SQL)
                        if any(str(row.get("extname")) == "age" for row in extension_rows)
                        else []
                    )
                finally:
                    cursor.close()

            return _build_inspection(
                request,
                version_rows,
                extension_rows,
                resource_rows,
                column_rows,
                constraint_rows,
                foreign_key_rows,
                index_rows,
                graph_rows,
            )
        except YoDbError:
            raise      # already safe and specific (a connection that was refused, a missing reference, ...)
        except Exception as error:
            raise _source_error_from_exception(request.source_name, error) from error


class PostgresCatalogValidator:
    """Validate PostgreSQL bindings and identity claims against one snapshot."""

    def validate(self, catalog: Catalog, inspection: SourceInspection) -> SourceValidationReport:
        findings: list[ValidationFinding] = []
        source = catalog.sources.get(inspection.source_name)
        if source is None:
            return SourceValidationReport(
                source_name=inspection.source_name,
                inspected_at=inspection.inspected_at,
                findings=(
                    _error(
                        "catalog_source_not_found",
                        "The inspection source is not declared by this catalog.",
                        f"sources.{inspection.source_name}",
                    ),
                ),
            )
        if source.kind is not SourceKind.POSTGRES or inspection.source_kind is not SourceKind.POSTGRES:
            findings.append(
                _error(
                    "source_kind_mismatch",
                    "PostgreSQL validation requires a postgres catalog source and inspection snapshot.",
                    f"sources.{inspection.source_name}.kind",
                )
            )
            return SourceValidationReport(
                source_name=inspection.source_name,
                inspected_at=inspection.inspected_at,
                findings=tuple(findings),
            )

        if source.graph is not None:
            findings.extend(_validate_graph(catalog, source.graph, inspection))

        findings.extend(validate_datasets(catalog, source, inspection, "PostgreSQL"))

        return SourceValidationReport(
            source_name=inspection.source_name,
            inspected_at=inspection.inspected_at,
            findings=tuple(findings),
        )


def _validate_graph(catalog: Catalog, graph, inspection: SourceInspection) -> list[ValidationFinding]:
    """The declared AGE graph, its vertex labels and the edge types the relationships use must exist."""

    location = f"sources.{inspection.source_name}.graph"
    if "age" not in inspection.extensions:
        return [
            _error(
                "extension_not_installed",
                "The catalog declares a graph, but the Apache AGE extension is not installed in this PostgreSQL source.",
                location,
            )
        ]
    found = (inspection.provider_metadata.get("graphs") or {}).get(graph.name)
    if found is None:
        return [_error("graph_not_found", f"Graph '{graph.name}' was not found in Apache AGE.", f"{location}.name")]
    findings = []
    for dataset_name, vertex in graph.vertices.items():
        if vertex.label not in found.get("vertex_labels", ()):
            findings.append(
                _error(
                    "vertex_label_not_found",
                    f"Vertex label '{vertex.label}' was not found in graph '{graph.name}'.",
                    f"{location}.vertices.{dataset_name}.label",
                )
            )
    for relationship_name, relationship in catalog.relationships.items():
        for implementation in relationship.implementations:
            if implementation.edge_type is None or implementation.from_endpoint.source != inspection.source_name:
                continue
            if implementation.edge_type not in found.get("edge_types", ()):
                findings.append(
                    _error(
                        "edge_type_not_found",
                        f"Edge type '{implementation.edge_type}' was not found in graph '{graph.name}'.",
                        f"relationships.{relationship_name}.implementations.edge_type",
                    )
                )
    return findings


def _fetch_rows(cursor: PostgresCursor, query: str) -> list[dict[str, Any]]:
    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [column[0] for column in cursor.description or ()]
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, Mapping):
            normalized.append(dict(row))
        else:
            normalized.append(dict(zip(columns, row, strict=True)))
    return normalized


def _build_inspection(
    request: InspectionRequest,
    version_rows: list[dict[str, Any]],
    extension_rows: list[dict[str, Any]],
    resource_rows: list[dict[str, Any]],
    column_rows: list[dict[str, Any]],
    constraint_rows: list[dict[str, Any]],
    foreign_key_rows: list[dict[str, Any]],
    index_rows: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]] = (),
) -> SourceInspection:
    fields_by_resource: dict[str, dict[str, PhysicalField]] = defaultdict(dict)
    for row in column_rows:
        native_type = str(row["native_type"])
        fields_by_resource[str(row["resource_name"])][str(row["field_name"])] = PhysicalField(
            name=str(row["field_name"]),
            native_type=native_type,
            type_family=_postgres_type_family(native_type),
            nullable=_as_optional_bool(row.get("nullable")),
            default=_as_optional_string(row.get("default_expression")),
            generated=_as_optional_bool(row.get("generated")),
            dimensions=_vector_dimensions(native_type),
        )

    primary_keys: dict[str, PhysicalKey] = {}
    unique_constraints: dict[str, list[PhysicalUniqueConstraint]] = defaultdict(list)
    checks: dict[str, list[PhysicalCheckConstraint]] = defaultdict(list)
    for row in constraint_rows:
        resource_name = str(row["resource_name"])
        constraint_type = str(row["constraint_type"])
        name = str(row["constraint_name"])
        fields = _as_strings(row.get("fields"))
        if constraint_type == "p":
            primary_keys[resource_name] = PhysicalKey(name=name, fields=fields)
        elif constraint_type == "u":
            unique_constraints[resource_name].append(PhysicalUniqueConstraint(name=name, fields=fields))
        elif constraint_type == "c":
            definition = _as_optional_string(row.get("definition"))
            if definition:
                checks[resource_name].append(PhysicalCheckConstraint(name=name, expression=definition))

    foreign_keys: dict[str, list[PhysicalForeignKey]] = defaultdict(list)
    for row in foreign_key_rows:
        resource_name = str(row["resource_name"])
        foreign_keys[resource_name].append(
            PhysicalForeignKey(
                name=_as_optional_string(row.get("constraint_name")),
                fields=_as_strings(row.get("fields")),
                target_resource=str(row["target_resource"]),
                target_fields=_as_strings(row.get("target_fields")),
                on_update=_ACTION_CODES.get(str(row.get("update_action", "")).lower()),
                on_delete=_ACTION_CODES.get(str(row.get("delete_action", "")).lower()),
                deferrable=_as_optional_bool(row.get("deferrable")),
                initially_deferred=_as_optional_bool(row.get("initially_deferred")),
            )
        )

    indexes: dict[str, list[PhysicalIndex]] = defaultdict(list)
    for row in index_rows:
        resource_name = str(row["resource_name"])
        indexes[resource_name].append(
            PhysicalIndex(
                name=str(row["index_name"]),
                method=_as_optional_string(row.get("method")),
                fields=_as_strings(row.get("fields")),
                include=_as_strings(row.get("include_fields")),
                unique=bool(row.get("is_unique")),
                predicate=_as_optional_string(row.get("predicate")),
                definition=_as_optional_string(row.get("definition")),
                valid=_as_optional_bool(row.get("is_valid")),
            )
        )

    resources: dict[str, PhysicalResource] = {}
    for row in resource_rows:
        resource_name = str(row["resource_name"])
        resources[resource_name] = PhysicalResource(
            name=resource_name,
            kind=ResourceKind(str(row["resource_kind"])),
            fields=fields_by_resource[resource_name],
            primary_key=primary_keys.get(resource_name),
            unique_constraints=tuple(unique_constraints[resource_name]),
            foreign_keys=tuple(foreign_keys[resource_name]),
            check_constraints=tuple(checks[resource_name]),
            indexes=tuple(indexes[resource_name]),
        )

    extensions = {str(row["extname"]): str(row["extversion"]) for row in extension_rows}
    capabilities = {
        InspectionCapability.RELATIONAL_SCHEMA,
        InspectionCapability.PRIMARY_KEYS,
        InspectionCapability.FOREIGN_KEYS,
        InspectionCapability.UNIQUE_CONSTRAINTS,
        InspectionCapability.INDEXES,
    }
    vector_fields = [field for resource in resources.values() for field in resource.fields.values() if field.type_family == "vector"]
    vector_indexes = [index for resource in resources.values() for index in resource.indexes if index.method in {"hnsw", "ivfflat"}]
    if vector_fields:
        capabilities.add(InspectionCapability.VECTOR_COLUMNS)
    if vector_indexes:
        capabilities.add(InspectionCapability.VECTOR_INDEXES)

    graphs: dict[str, dict[str, list[str]]] = {}
    for row in graph_rows:
        graph = graphs.setdefault(str(row["graph_name"]), {"vertex_labels": [], "edge_types": []})
        label, kind = str(row["label_name"]), str(row["label_kind"])
        if label.startswith("_ag_label_"):
            continue                      # AGE's own default labels
        graph["vertex_labels" if kind == "v" else "edge_types"].append(label)
    if graphs:
        capabilities.add(InspectionCapability.GRAPH_SCHEMA)

    version = _as_optional_string(version_rows[0].get("server_version")) if version_rows else None
    return SourceInspection(
        source_name=request.source_name,
        source_kind=SourceKind.POSTGRES,
        inspected_at=datetime.now(UTC),
        engine_version=version,
        capabilities=frozenset(capabilities),
        resources=resources,
        extensions=extensions,
        provider_metadata={"graphs": graphs} if graphs else {},
    )


def _postgres_type_family(native_type: str) -> str:
    normalized = native_type.lower()
    if normalized.startswith("vector"):
        return "vector"
    if normalized in {"smallint", "integer", "bigint", "smallserial", "serial", "bigserial"}:
        return "int"
    if normalized.startswith(("numeric", "decimal", "real", "double precision")):
        return "float"
    if normalized in {"boolean", "bool"}:
        return "bool"
    if normalized == "uuid":
        return "uuid"
    if normalized in {"json", "jsonb"}:
        return "json"
    if normalized == "bytea":
        return "bytes"
    if normalized.startswith(("timestamp", "time", "date")):
        return "timestamp"
    if normalized.startswith(("character", "varchar", "text", "citext", "name")):
        return "string"
    return "other"


def _vector_dimensions(native_type: str) -> int | None:
    match = _VECTOR_DIMENSIONS.fullmatch(native_type)
    return int(match.group("dimensions")) if match else None


def _as_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(str(item) for item in value)


def _as_optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _as_optional_bool(value: Any) -> bool | None:
    return bool(value) if value is not None else None


def _source_error_from_exception(source_name: str, error: Exception) -> SourceInspectionError:
    sqlstate = getattr(error, "sqlstate", None) or getattr(error, "pgcode", None)
    if sqlstate == "28P01":
        code, retryable, message = (
            ErrorCode.SOURCE_AUTHENTICATION_FAILED,
            False,
            "PostgreSQL authentication failed while inspecting the source.",
        )
    elif sqlstate == "42501":
        code, retryable, message = (
            ErrorCode.SOURCE_PERMISSION_DENIED,
            False,
            "The configured role lacks permission to inspect PostgreSQL metadata.",
        )
    elif isinstance(sqlstate, str) and sqlstate.startswith("08"):
        code, retryable, message = (
            ErrorCode.SOURCE_UNAVAILABLE,
            True,
            "PostgreSQL is unavailable while inspecting the source.",
        )
    else:
        code, retryable, message = (
            ErrorCode.SOURCE_INSPECTION_FAILED,
            False,
            "PostgreSQL metadata inspection failed.",
        )
    return SourceInspectionError(
        ErrorDetail(
            code=code,
            message=message,
            retryable=retryable,
            source_name=source_name,
            details={"exception_type": type(error).__name__},
        )
    )
