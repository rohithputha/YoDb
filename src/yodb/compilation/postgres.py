"""Compile one resolved single-source YoDb query into parameterized PostgreSQL SQL."""

from __future__ import annotations

import json
import re

from ..catalog import SourceKind, VectorMetric
from ..errors import ErrorCode
from ..planning import GraphEdgeScan, RemoteScan
from ..query.resolution import QuerySourceShape, SourceResolvedQuery
from .contracts import CompiledOutputColumn, CompiledPostgresQuery
from .sql import (
    CompileContext,
    SqlDialect,
    column_name as _column,
    compile_expression,
    effective_limit,
    fail as _fail,
    field_map,
    order_term,
    quote_identifier as _quote_identifier,
    require_resolved_field,
    resource_name as _resource,
)

POSTGRES_DIALECT = SqlDialect(
    name="PostgreSQL",
    source_kind=SourceKind.POSTGRES,
    placeholder="%s",
    text_collation='"C"',          # byte order, i.e. code-point order for UTF-8
)

# pgvector distance operators: smaller distance = nearer for all three.
_VECTOR_OPERATORS = {
    VectorMetric.COSINE: "<=>",
    VectorMetric.L2: "<->",
    VectorMetric.INNER_PRODUCT: "<#>",
}

# The column a ranked scan adds when the distance of each row is wanted (a helper field: it is
# reported beside the rows and never projected into a result).
DISTANCE_COLUMN = "_distance"

# What the catalog admits for a graph name, label, edge type or property (checked again here,
# because these names are written into the Cypher text).
_GRAPH_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# The dollar-quote tag around the Cypher text; the text can only hold identifiers and ``$ids``.
_CYPHER_TAG = "$yodb_cypher$"


class PostgresQueryCompiler:
    """Compile the first source-local relational slice of YoDb queries.

    This compiler intentionally has no database connection. The subsequent
    executor will lease ``connection_ref`` from the PostgreSQL connection
    adapter and submit the compiled statement with its parameter tuple.
    """

    source_kind = SourceKind.POSTGRES

    def compile(self, query: SourceResolvedQuery) -> CompiledPostgresQuery:
        """Compile a single-source PostgreSQL query or raise a structured error."""

        if query.shape is not QuerySourceShape.SINGLE_SOURCE:
            _fail(
                ErrorCode.QUERY_SOURCE_SHAPE_UNSUPPORTED,
                "The PostgreSQL compiler currently accepts single-source queries only.",
            )
        source = query.identity_source
        if source.source_kind is not SourceKind.POSTGRES:
            _fail(
                ErrorCode.QUERY_COMPILATION_UNSUPPORTED,
                "The PostgreSQL compiler requires a PostgreSQL source binding.",
            )

        context = CompileContext(query=query, fields_by_name=field_map(query), dialect=POSTGRES_DIALECT)
        selected = tuple(
            require_resolved_field(context, field.name, "select") for field in query.query.select
        )
        projections = ", ".join(
            f"{_column(field)} AS {_quote_identifier(field.field.name)}" for field in selected
        )
        parameters: list[object] = []
        where = compile_expression(query.query.where, context, parameters)
        order_by = ", ".join(
            order_term(require_resolved_field(context, term.field.name, "order_by"), term.direction.value, POSTGRES_DIALECT)
            for term in query.query.order_by
        )
        limit = effective_limit(query)
        parameters.append(limit)

        statements = [f"SELECT {projections}", f"FROM {_resource(source.resource)}"]
        if where is not None:
            statements.append(f"WHERE {where}")
        statements.append(f"ORDER BY {order_by}")
        statements.append("LIMIT %s")
        return CompiledPostgresQuery(
            source_name=source.source_name,
            connection_ref=source.connection_ref,
            sql="\n".join(statements),
            parameters=tuple(parameters),
            output_columns=tuple(
                CompiledOutputColumn(sql_alias=field.field.name, logical_field=field.field.name)
                for field in selected
            ),
        )

    def compile_scan(self, scan: RemoteScan) -> CompiledPostgresQuery:
        """Compile one planner-produced PostgreSQL fragment.

        The planner has already established field ownership and pushdown
        legality. This compiler repeats the source-kind and field checks while
        translating only the accepted fragment into parameterized SQL.
        """

        if scan.source.source_kind is not SourceKind.POSTGRES:
            _fail(
                ErrorCode.QUERY_COMPILATION_UNSUPPORTED,
                "The PostgreSQL compiler requires a PostgreSQL remote scan.",
            )
        fields_by_name = {field.field.name: field for field in scan.projection}
        if scan.source.logical_id.field.name not in fields_by_name:
            _fail(
                ErrorCode.QUERY_COMPILATION_UNSUPPORTED,
                "Every PostgreSQL remote scan must project its logical ID.",
            )
        context = CompileContext(query=None, fields_by_name=fields_by_name, dialect=POSTGRES_DIALECT)
        projections = ", ".join(
            f"{_column(field)} AS {_quote_identifier(field.field.name)}" for field in scan.projection
        )
        parameters: list[object] = []
        with_distance = scan.vector_search is not None and scan.vector_search.return_distance
        if with_distance:
            # The distance is selected as well as ordered by, so the same vector is bound twice.
            vector = scan.vector_search
            if vector.query_vector is None:
                _fail(
                    ErrorCode.QUERY_COMPILATION_UNSUPPORTED,
                    "A vector-search scan requires the embedded query vector before compilation.",
                )
            parameters.append(_vector_literal(vector.query_vector))
            projections += (
                f", {_quote_identifier(vector.column)} {_VECTOR_OPERATORS[vector.metric]} %s::vector "
                f"AS {_quote_identifier(DISTANCE_COLUMN)}"
            )
        where = compile_expression(scan.pushed_filter, context, parameters)
        if scan.key_filter is not None:
            if not scan.key_filter:
                _fail(
                    ErrorCode.QUERY_COMPILATION_UNSUPPORTED,
                    "A key-restricted PostgreSQL scan requires at least one logical ID.",
                )
            parameters.extend(scan.key_filter)
            placeholders = ", ".join("%s" for _ in scan.key_filter)
            key_where = f"{_column(scan.source.logical_id)} IN ({placeholders})"
            where = key_where if where is None else f"({where}) AND {key_where}"
        vector = scan.vector_search
        if vector is not None:
            if vector.query_vector is None:
                _fail(
                    ErrorCode.QUERY_COMPILATION_UNSUPPORTED,
                    "A vector-search scan requires the embedded query vector before compilation.",
                )
            # Rows without an embedding cannot be ranked; keep them out of the shortlist.
            not_null = f"{_quote_identifier(vector.column)} IS NOT NULL"
            where = not_null if where is None else f"({where}) AND {not_null}"
        statements = [f"SELECT {projections}", f"FROM {_resource(scan.source.resource)}"]
        if where is not None:
            statements.append(f"WHERE {where}")
        if vector is not None:
            parameters.append(_vector_literal(vector.query_vector))
            statements.append(
                f"ORDER BY {_quote_identifier(vector.column)} {_VECTOR_OPERATORS[vector.metric]} %s::vector, "
                f"{order_term(scan.source.logical_id, 'asc', POSTGRES_DIALECT)}"
            )
        elif scan.order_by:
            statements.append(
                "ORDER BY "
                + ", ".join(
                    order_term(require_resolved_field(context, term.field.name, "order_by"), term.direction.value, POSTGRES_DIALECT)
                    for term in scan.order_by
                )
            )
        limit = scan.limit if scan.limit is not None else (
            scan.maximum_rows + 1 if scan.maximum_rows is not None else None
        )
        if limit is not None:
            parameters.append(limit)
            statements.append("LIMIT %s")
        return CompiledPostgresQuery(
            source_name=scan.source.source_name,
            connection_ref=scan.source.connection_ref,
            sql="\n".join(statements),
            parameters=tuple(parameters),
            output_columns=tuple(
                CompiledOutputColumn(sql_alias=field.field.name, logical_field=field.field.name)
                for field in scan.projection
            )
            + ((CompiledOutputColumn(sql_alias=DISTANCE_COLUMN, logical_field=DISTANCE_COLUMN),) if with_distance else ()),
        )

    def compile_edges(self, scan: GraphEdgeScan) -> CompiledPostgresQuery:
        """Compile one read of Apache AGE edges: a Cypher ``MATCH`` inside ``ag_catalog.cypher()``.

        The graph name, labels, edge type and property names come from the catalog, which
        admits only letters, digits and underscores for them; the IDs travel as the one
        Cypher parameter (``$ids``, an agtype list), never as text in the statement.
        """

        if scan.source_kind is not SourceKind.POSTGRES:
            _fail(ErrorCode.QUERY_COMPILATION_UNSUPPORTED, "The PostgreSQL compiler requires a PostgreSQL graph edge scan.")
        if not scan.ids:
            _fail(ErrorCode.QUERY_COMPILATION_UNSUPPORTED, "A graph edge scan requires at least one starting ID.")
        for name in (scan.graph, scan.edge_type, scan.from_label, scan.from_id_property, scan.to_label, scan.to_id_property):
            if not _GRAPH_IDENTIFIER.fullmatch(name):
                _fail(ErrorCode.QUERY_COMPILATION_UNSUPPORTED, "A graph name, label or property is not a plain identifier.")
        if scan.anchored == "from":
            pattern = (
                f"(a:{scan.from_label})-[:{scan.edge_type}]->(b:{scan.to_label})"
            )
            start, end = f"a.{scan.from_id_property}", f"b.{scan.to_id_property}"
        else:
            pattern = (
                f"(b:{scan.from_label})-[:{scan.edge_type}]->(a:{scan.to_label})"
            )
            start, end = f"a.{scan.to_id_property}", f"b.{scan.from_id_property}"
        cypher = f"UNWIND $ids AS yodb_id MATCH {pattern} WHERE {start} = yodb_id RETURN {start}, {end}"
        parameters: list[object] = [json.dumps({"ids": list(scan.ids)})]
        statements = [
            "SELECT start_id, end_id",
            f"FROM ag_catalog.cypher('{scan.graph}', {_CYPHER_TAG} {cypher} {_CYPHER_TAG}, %s)",
            "AS (start_id ag_catalog.agtype, end_id ag_catalog.agtype)",
        ]
        if scan.maximum_rows is not None:
            parameters.append(scan.maximum_rows + 1)
            statements.append("LIMIT %s")
        return CompiledPostgresQuery(
            source_name=scan.source_name,
            connection_ref=scan.connection_ref,
            sql="\n".join(statements),
            parameters=tuple(parameters),
            output_columns=(
                CompiledOutputColumn(sql_alias="start_id", logical_field="start_id"),
                CompiledOutputColumn(sql_alias="end_id", logical_field="end_id"),
            ),
            requires_extension="age",
        )


def _vector_literal(values: tuple[float, ...]) -> str:
    return "[" + ",".join(repr(float(x)) for x in values) + "]"
