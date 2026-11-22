"""The parts of compiling a source-local query into parameterized SQL that every SQL backend shares.

A backend describes itself with a :class:`SqlDialect` (its placeholder, how it makes text compare
in byte order, how it orders NULLs, how many bound values one statement may hold); everything else
(the boolean tree, the predicates, field lookup, quoting, the page limit) is written once here.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..catalog import LogicalType, SourceKind
from ..errors import ErrorCode, ErrorDetail, QueryError
from ..query.models import (
    BoundAllExpression,
    BoundAnyExpression,
    BoundFilterExpression,
    BoundNotExpression,
    BoundPredicate,
    ComparisonOperator,
)
from ..query.resolution import ResolvedField, SourceResolvedQuery

COMPILED_OPERATORS = frozenset(
    {
        ComparisonOperator.EQ,
        ComparisonOperator.NE,
        ComparisonOperator.IN,
        ComparisonOperator.NOT_IN,
        ComparisonOperator.IS_NULL,
        ComparisonOperator.IS_NOT_NULL,
        ComparisonOperator.GT,
        ComparisonOperator.GTE,
        ComparisonOperator.LT,
        ComparisonOperator.LTE,
    }
)

_COMPARISON_TOKENS = {
    ComparisonOperator.EQ: "=",
    ComparisonOperator.NE: "<>",
    ComparisonOperator.GT: ">",
    ComparisonOperator.GTE: ">=",
    ComparisonOperator.LT: "<",
    ComparisonOperator.LTE: "<=",
}


@dataclass(frozen=True)
class SqlDialect:
    """What differs between SQL backends, as far as this compiler family is concerned."""

    name: str                                # "PostgreSQL", "SQLite": for messages
    source_kind: SourceKind
    placeholder: str                         # "%s" (psycopg) or "?" (sqlite3)
    text_collation: str                      # byte-order collation: '"C"' or 'BINARY'
    collate_comparisons: bool = False        # also pin the collation on text =, <>, <, IN ... (a column may declare its own)
    explicit_null_order: bool = False        # spell NULLS LAST / NULLS FIRST (the coordinator sorts NULLs last for ASC)
    maximum_variables: int | None = None     # most bound values one statement may carry


@dataclass(frozen=True)
class CompileContext:
    query: SourceResolvedQuery | None
    fields_by_name: dict[str, ResolvedField]
    dialect: SqlDialect


def compile_expression(
    expression: BoundFilterExpression | None,
    context: CompileContext,
    parameters: list[object],
) -> str | None:
    if expression is None:
        return None
    if isinstance(expression, BoundPredicate):
        return _compile_predicate(expression, context, parameters)
    if isinstance(expression, BoundAllExpression):
        return "(" + " AND ".join(
            compile_expression(child, context, parameters) for child in expression.expressions
        ) + ")"
    if isinstance(expression, BoundAnyExpression):
        return "(" + " OR ".join(
            compile_expression(child, context, parameters) for child in expression.expressions
        ) + ")"
    if isinstance(expression, BoundNotExpression):
        child = compile_expression(expression.expression, context, parameters)
        assert child is not None
        return f"(NOT {child})"
    raise AssertionError(f"Unknown bound expression: {expression!r}")


def _compile_predicate(
    predicate: BoundPredicate,
    context: CompileContext,
    parameters: list[object],
) -> str:
    dialect = context.dialect
    field = require_resolved_field(context, predicate.field.name, "where")
    operator = predicate.operator
    if operator not in COMPILED_OPERATORS:
        fail(
            ErrorCode.QUERY_COMPILATION_UNSUPPORTED,
            f"Operator '{operator.value}' is not compiled for {dialect.name} yet.",
            "where.op",
        )
    column = comparison_column(field, dialect)
    if operator is ComparisonOperator.IS_NULL:
        return f"{column} IS NULL"
    if operator is ComparisonOperator.IS_NOT_NULL:
        return f"{column} IS NOT NULL"
    if operator in {ComparisonOperator.IN, ComparisonOperator.NOT_IN}:
        assert isinstance(predicate.value, tuple)
        placeholders = ", ".join(dialect.placeholder for _ in predicate.value)
        parameters.extend(predicate.value)
        keyword = "IN" if operator is ComparisonOperator.IN else "NOT IN"
        return f"{column} {keyword} ({placeholders})"
    token = _COMPARISON_TOKENS.get(operator)
    if token is None:
        raise AssertionError(f"Unhandled compiled operator: {operator!r}")
    parameters.append(predicate.value)
    return f"{column} {token} {dialect.placeholder}"


def check_variable_limit(parameters: list[object] | tuple[object, ...], dialect: SqlDialect) -> None:
    """A statement with more bound values than the backend allows is refused here, not by the driver."""

    if dialect.maximum_variables is not None and len(parameters) > dialect.maximum_variables:
        fail(
            ErrorCode.QUERY_COMPILATION_UNSUPPORTED,
            f"The {dialect.name} statement would bind {len(parameters)} values; the limit is {dialect.maximum_variables}.",
        )


def field_map(query: SourceResolvedQuery) -> dict[str, ResolvedField]:
    source = query.identity_source
    fields = {field.field.name: field for field in source.fields}
    fields.setdefault(source.logical_id.field.name, source.logical_id)
    return fields


def require_resolved_field(context: CompileContext, field_name: str, location: str) -> ResolvedField:
    try:
        return context.fields_by_name[field_name]
    except KeyError:
        fail(
            ErrorCode.SOURCE_BINDING_UNAVAILABLE,
            f"Field '{field_name}' is not available in the selected {context.dialect.name} source binding.",
            location,
        )


def effective_limit(query: SourceResolvedQuery) -> int:
    page_limit = query.query.page.first
    assert page_limit is not None
    requested_maximum = query.query.constraints.maximum_results
    return min(page_limit, requested_maximum) if requested_maximum is not None else page_limit


def resource_name(resource: str) -> str:
    parts = resource.split(".")
    if not all(part for part in parts):
        fail(
            ErrorCode.SOURCE_BINDING_UNAVAILABLE,
            "The selected source resource has an invalid qualified name.",
        )
    return ".".join(quote_identifier(part) for part in parts)


def column_name(field: ResolvedField) -> str:
    return quote_identifier(field.physical_name)


def _collatable(field: ResolvedField) -> bool:
    """A column the inspection saw as non-text (an integer or uuid ``id``) cannot take a collation."""

    return field.physical_family == "string" or (
        field.physical_family is None and field.field.spec.type in (LogicalType.STRING, LogicalType.TEXT)
    )


def comparison_column(field: ResolvedField, dialect: SqlDialect) -> str:
    """The column in a predicate; where a column may carry its own collation (SQLite's ``COLLATE NOCASE``),
    text is compared in byte order, as the coordinator compares it."""

    if dialect.collate_comparisons and _collatable(field):
        return f"{column_name(field)} COLLATE {dialect.text_collation}"
    return column_name(field)


def order_column(field: ResolvedField, dialect: SqlDialect) -> str:
    """A text column is ordered in byte order (code-point order for UTF-8), so a source orders text exactly
    as the coordinator does and a result never depends on the database's locale."""

    if _collatable(field):
        return f"{column_name(field)} COLLATE {dialect.text_collation}"
    return column_name(field)


def order_term(field: ResolvedField, direction: str, dialect: SqlDialect) -> str:
    """One ``ORDER BY`` term.  The coordinator sorts NULLs last for ASC and first for DESC; a backend whose
    default differs says so explicitly."""

    direction = direction.upper()
    term = f"{order_column(field, dialect)} {direction}"
    if dialect.explicit_null_order:
        term += " NULLS LAST" if direction == "ASC" else " NULLS FIRST"
    return term


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def fail(code: ErrorCode, message: str, location: str | None = None) -> None:
    raise QueryError(ErrorDetail(code=code, message=message, retryable=False, location=location))
