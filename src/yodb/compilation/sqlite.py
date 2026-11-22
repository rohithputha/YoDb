"""Compile one resolved single-source YoDb query into parameterized SQLite SQL.

The differences from the PostgreSQL compiler are all in :data:`SQLITE_DIALECT` (``?`` placeholders,
``COLLATE BINARY``, explicit NULL order, the bound-variable limit) and in what is absent: SQLite has
no vector search and no graph engine, so a scan carrying either is refused here, not mis-compiled.
"""

from __future__ import annotations

import sqlite3

from ..catalog import SourceKind
from ..errors import ErrorCode
from ..planning import GraphEdgeScan, RemoteScan
from ..query.resolution import QuerySourceShape, ResolvedField, SourceResolvedQuery
from .contracts import CompiledOutputColumn, CompiledSqliteQuery
from .sql import (
    CompileContext,
    SqlDialect,
    check_variable_limit,
    column_name,
    compile_expression,
    effective_limit,
    fail,
    field_map,
    order_term,
    quote_identifier,
    require_resolved_field,
    resource_name,
)

# SQLITE_MAX_VARIABLE_NUMBER is 32766 from SQLite 3.32 and 999 before; stay under whichever this build has.
SQLITE_MAX_VARIABLES = 32_766 if sqlite3.sqlite_version_info >= (3, 32, 0) else 999

SQLITE_DIALECT = SqlDialect(
    name="SQLite",
    source_kind=SourceKind.SQLITE,
    placeholder="?",
    text_collation="BINARY",
    collate_comparisons=True,         # a column may be declared COLLATE NOCASE; YoDb compares text in byte order
    explicit_null_order=True,         # SQLite sorts NULLs first for ASC; the coordinator sorts them last
    maximum_variables=SQLITE_MAX_VARIABLES,
)


class SqliteQueryCompiler:
    """Compile source-local queries and planner scan fragments for a SQLite source.

    Like the PostgreSQL compiler it holds no connection: the executor leases one and submits
    the statement with its parameters.
    """

    source_kind = SourceKind.SQLITE

    def compile(self, query: SourceResolvedQuery) -> CompiledSqliteQuery:
        if query.shape is not QuerySourceShape.SINGLE_SOURCE:
            fail(
                ErrorCode.QUERY_SOURCE_SHAPE_UNSUPPORTED,
                "The SQLite compiler currently accepts single-source queries only.",
            )
        source = query.identity_source
        if source.source_kind is not SourceKind.SQLITE:
            fail(ErrorCode.QUERY_COMPILATION_UNSUPPORTED, "The SQLite compiler requires a SQLite source binding.")

        context = CompileContext(query=query, fields_by_name=field_map(query), dialect=SQLITE_DIALECT)
        selected = tuple(require_resolved_field(context, field.name, "select") for field in query.query.select)
        parameters: list[object] = []
        where = compile_expression(query.query.where, context, parameters)
        order_by = ", ".join(
            order_term(require_resolved_field(context, term.field.name, "order_by"), term.direction.value, SQLITE_DIALECT)
            for term in query.query.order_by
        )
        parameters.append(effective_limit(query))
        check_variable_limit(parameters, SQLITE_DIALECT)

        statements = [f"SELECT {_projections(selected)}", f"FROM {resource_name(source.resource)}"]
        if where is not None:
            statements.append(f"WHERE {where}")
        statements.append(f"ORDER BY {order_by}")
        statements.append("LIMIT ?")
        return CompiledSqliteQuery(
            source_name=source.source_name,
            connection_ref=source.connection_ref,
            sql="\n".join(statements),
            parameters=tuple(parameters),
            output_columns=_output_columns(selected),
        )

    def compile_scan(self, scan: RemoteScan) -> CompiledSqliteQuery:
        if scan.source.source_kind is not SourceKind.SQLITE:
            fail(ErrorCode.QUERY_COMPILATION_UNSUPPORTED, "The SQLite compiler requires a SQLite remote scan.")
        if scan.vector_search is not None:
            fail(ErrorCode.QUERY_COMPILATION_UNSUPPORTED, "SQLite has no vector search.")
        fields_by_name = {field.field.name: field for field in scan.projection}
        if scan.source.logical_id.field.name not in fields_by_name:
            fail(ErrorCode.QUERY_COMPILATION_UNSUPPORTED, "Every SQLite remote scan must project its logical ID.")
        context = CompileContext(query=None, fields_by_name=fields_by_name, dialect=SQLITE_DIALECT)
        parameters: list[object] = []
        where = compile_expression(scan.pushed_filter, context, parameters)
        if scan.key_filter is not None:
            if not scan.key_filter:
                fail(
                    ErrorCode.QUERY_COMPILATION_UNSUPPORTED,
                    "A key-restricted SQLite scan requires at least one logical ID.",
                )
            parameters.extend(scan.key_filter)
            placeholders = ", ".join("?" for _ in scan.key_filter)
            key_where = f"{column_name(scan.source.logical_id)} IN ({placeholders})"
            where = key_where if where is None else f"({where}) AND {key_where}"
        statements = [f"SELECT {_projections(scan.projection)}", f"FROM {resource_name(scan.source.resource)}"]
        if where is not None:
            statements.append(f"WHERE {where}")
        if scan.order_by:
            statements.append(
                "ORDER BY "
                + ", ".join(
                    order_term(
                        require_resolved_field(context, term.field.name, "order_by"),
                        term.direction.value,
                        SQLITE_DIALECT,
                    )
                    for term in scan.order_by
                )
            )
        limit = scan.limit if scan.limit is not None else (
            scan.maximum_rows + 1 if scan.maximum_rows is not None else None
        )
        if limit is not None:
            parameters.append(limit)
            statements.append("LIMIT ?")
        check_variable_limit(parameters, SQLITE_DIALECT)
        return CompiledSqliteQuery(
            source_name=scan.source.source_name,
            connection_ref=scan.source.connection_ref,
            sql="\n".join(statements),
            parameters=tuple(parameters),
            output_columns=_output_columns(scan.projection),
        )

    def compile_edges(self, scan: GraphEdgeScan):
        fail(ErrorCode.QUERY_COMPILATION_UNSUPPORTED, "SQLite has no graph engine: its edges cannot be read.")


def _projections(fields: tuple[ResolvedField, ...]) -> str:
    return ", ".join(f"{column_name(field)} AS {quote_identifier(field.field.name)}" for field in fields)


def _output_columns(fields: tuple[ResolvedField, ...]) -> tuple[CompiledOutputColumn, ...]:
    return tuple(
        CompiledOutputColumn(
            sql_alias=field.field.name, logical_field=field.field.name, logical_type=field.field.spec.type
        )
        for field in fields
    )
