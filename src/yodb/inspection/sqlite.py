"""Read-only SQLite inspection and catalog-validation adapter.

Reads ``sqlite_schema`` and the ``PRAGMA`` table-valued functions (``table_xinfo``, ``index_list``,
``index_info``, ``foreign_key_list``); it never reads business rows and never writes.  Resources are
named by table or view name (the ``main`` schema).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
import sqlite3
from typing import Any

from ..catalog import Catalog, SourceKind
from ..connections.contracts import SourceConnectionAdapter
from ..errors import ErrorCode, ErrorDetail, SourceInspectionError, YoDbError
from .contracts import (
    InspectionCapability,
    InspectionRequest,
    PhysicalField,
    PhysicalForeignKey,
    PhysicalIndex,
    PhysicalKey,
    PhysicalResource,
    PhysicalUniqueConstraint,
    ResourceKind,
    SourceInspection,
    SourceValidationReport,
)
from .relational import error, validate_datasets

_RESOURCES_SQL = (
    "SELECT name, type FROM sqlite_schema "
    "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' ORDER BY name"
)
_COLUMNS_SQL = 'SELECT name, type, "notnull", dflt_value, pk, hidden FROM pragma_table_xinfo(?) ORDER BY cid'
_INDEXES_SQL = 'SELECT name, "unique", origin, partial FROM pragma_index_list(?) ORDER BY seq'
_INDEX_COLUMNS_SQL = "SELECT name FROM pragma_index_info(?) ORDER BY seqno"
_FOREIGN_KEYS_SQL = (
    'SELECT id, "table", "from", "to", on_update, on_delete FROM pragma_foreign_key_list(?) ORDER BY id, seq'
)


class SqliteSourceInspector:
    """Inspect a SQLite file through an injected read-only connection adapter."""

    source_kind = SourceKind.SQLITE

    def __init__(self, connections: SourceConnectionAdapter[sqlite3.Connection]) -> None:
        self._connections = connections

    def inspect(self, request: InspectionRequest) -> SourceInspection:
        if request.source.kind is not SourceKind.SQLITE:
            raise SourceInspectionError(
                ErrorDetail(
                    code=ErrorCode.SOURCE_KIND_UNSUPPORTED,
                    message="The SQLite inspector only supports sqlite sources.",
                    retryable=False,
                    source_name=request.source_name,
                )
            )
        try:
            with self._connections.acquire(request.source.connection_ref) as connection:
                version = connection.execute("SELECT sqlite_version()").fetchone()[0]
                resources = _read_resources(connection)
        except YoDbError:
            raise
        except Exception as error_:
            raise _source_error(request.source_name, error_) from error_
        return SourceInspection(
            source_name=request.source_name,
            source_kind=SourceKind.SQLITE,
            inspected_at=datetime.now(UTC),
            engine_version=str(version),
            capabilities=frozenset(
                {
                    InspectionCapability.RELATIONAL_SCHEMA,
                    InspectionCapability.PRIMARY_KEYS,
                    InspectionCapability.FOREIGN_KEYS,
                    InspectionCapability.UNIQUE_CONSTRAINTS,
                    InspectionCapability.INDEXES,
                }
            ),
            resources=resources,
        )


class SqliteCatalogValidator:
    """Validate SQLite bindings and identity claims against one snapshot.

    The rules are the relational ones every SQL source shares (see :mod:`.relational`); a unique index,
    like a unique constraint, backs an identity.
    """

    def validate(self, catalog: Catalog, inspection: SourceInspection) -> SourceValidationReport:
        source = catalog.sources.get(inspection.source_name)
        if source is None:
            findings = [
                error(
                    "catalog_source_not_found",
                    "The inspection source is not declared by this catalog.",
                    f"sources.{inspection.source_name}",
                )
            ]
        elif source.kind is not SourceKind.SQLITE or inspection.source_kind is not SourceKind.SQLITE:
            findings = [
                error(
                    "source_kind_mismatch",
                    "SQLite validation requires a sqlite catalog source and inspection snapshot.",
                    f"sources.{inspection.source_name}.kind",
                )
            ]
        else:
            findings = validate_datasets(catalog, source, inspection, "SQLite")
        return SourceValidationReport(
            source_name=inspection.source_name, inspected_at=inspection.inspected_at, findings=tuple(findings)
        )


def _read_resources(connection: sqlite3.Connection) -> dict[str, PhysicalResource]:
    resources: dict[str, PhysicalResource] = {}
    for name, kind in connection.execute(_RESOURCES_SQL).fetchall():
        resources[name] = _read_resource(connection, name, ResourceKind.VIEW if kind == "view" else ResourceKind.TABLE)
    return resources


def _read_resource(connection: sqlite3.Connection, name: str, kind: ResourceKind) -> PhysicalResource:
    fields: dict[str, PhysicalField] = {}
    primary: dict[int, str] = {}
    for column, declared, not_null, default, pk, hidden in connection.execute(_COLUMNS_SQL, (name,)).fetchall():
        if hidden == 1:
            continue                                  # a virtual table's hidden argument column
        declared = declared or ""
        fields[column] = PhysicalField(
            name=column,
            native_type=declared or "ANY",
            type_family=sqlite_type_family(declared),
            nullable=not (not_null or pk),           # a primary key column is never NULL in practice
            default=None if default is None else str(default),
            generated=hidden in (2, 3),
        )
        if pk:
            primary[pk] = column

    unique_constraints: list[PhysicalUniqueConstraint] = []
    indexes: list[PhysicalIndex] = []
    primary_key_name = None
    for index_name, unique, origin, partial in connection.execute(_INDEXES_SQL, (name,)).fetchall():
        columns = [row[0] for row in connection.execute(_INDEX_COLUMNS_SQL, (index_name,)).fetchall()]
        on_columns = tuple(column for column in columns if column is not None)     # an expression has no name
        indexes.append(
            PhysicalIndex(
                name=index_name,
                fields=on_columns,
                method="btree",
                unique=bool(unique),
                predicate="partial" if partial else None,
                provider="sqlite",
            )
        )
        if origin == "pk":
            primary_key_name = index_name
        if unique and not partial and on_columns and len(on_columns) == len(columns):
            if origin != "pk":                       # origin 'u' is UNIQUE, 'c' is CREATE UNIQUE INDEX
                unique_constraints.append(PhysicalUniqueConstraint(name=index_name, fields=on_columns))

    primary_key = None
    if primary:
        primary_key = PhysicalKey(
            name=primary_key_name or f"{name}_pkey", fields=tuple(primary[position] for position in sorted(primary))
        )
    return PhysicalResource(
        name=name,
        kind=kind,
        fields=fields,
        primary_key=primary_key,
        unique_constraints=tuple(unique_constraints),
        foreign_keys=_read_foreign_keys(connection, name),
        indexes=tuple(indexes),
    )


def _read_foreign_keys(connection: sqlite3.Connection, name: str) -> tuple[PhysicalForeignKey, ...]:
    grouped: dict[int, list[Any]] = defaultdict(list)
    for row in connection.execute(_FOREIGN_KEYS_SQL, (name,)).fetchall():
        grouped[row[0]].append(row)
    keys = []
    for rows in grouped.values():
        target = rows[0][1]
        fields = tuple(row[2] for row in rows)
        if any(row[3] is None for row in rows):      # REFERENCES target without columns: the target's primary key
            target_fields = _primary_key_of(connection, target)
            if len(target_fields) != len(fields):
                continue
        else:
            target_fields = tuple(row[3] for row in rows)
        keys.append(
            PhysicalForeignKey(
                fields=fields,
                target_resource=target,
                target_fields=target_fields,
                on_update=_action(rows[0][4]),
                on_delete=_action(rows[0][5]),
            )
        )
    return tuple(keys)


def _primary_key_of(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
    rows = connection.execute(_COLUMNS_SQL, (table,)).fetchall()
    return tuple(row[0] for row in sorted((row for row in rows if row[4]), key=lambda row: row[4]))


def _action(text: str | None) -> str | None:
    return None if text is None else text.strip().lower().replace(" ", "_")


def sqlite_type_family(declared_type: str) -> str:
    """The family of a declared column type.

    SQLite's own affinity rules decide integer, text, blob and real; the names a column is commonly given
    for a boolean, a timestamp, a UUID or JSON are recognised first, since SQLite would call them numeric
    or text.  A column with no declared type holds anything, and is "other".
    """

    name = declared_type.upper()
    if not name.strip():
        return "other"
    if "BOOL" in name:
        return "bool"
    if "DATE" in name or "TIME" in name:
        return "timestamp"
    if "UUID" in name or "GUID" in name:
        return "uuid"
    if "JSON" in name:
        return "json"
    if "INT" in name:
        return "int"
    if "CHAR" in name or "CLOB" in name or "TEXT" in name:
        return "string"
    if "BLOB" in name:
        return "bytes"
    return "float"                                    # REAL, FLOA, DOUB, NUMERIC, DECIMAL: numeric affinity


def _source_error(source_name: str, error_: Exception) -> SourceInspectionError:
    message = str(error_).lower()
    if "not authorized" in message or "permission" in message:
        code, retryable, text = (
            ErrorCode.SOURCE_PERMISSION_DENIED,
            False,
            "The SQLite file cannot be read by this process.",
        )
    elif "locked" in message or "busy" in message:
        code, retryable, text = ErrorCode.SOURCE_UNAVAILABLE, True, "The SQLite file is locked while inspecting the source."
    else:
        code, retryable, text = ErrorCode.SOURCE_INSPECTION_FAILED, False, "SQLite metadata inspection failed."
    return SourceInspectionError(
        ErrorDetail(
            code=code,
            message=text,
            retryable=retryable,
            source_name=source_name,
            details={"exception_type": type(error_).__name__},
        )
    )
