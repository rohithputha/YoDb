"""Bind parsed logical queries to one activated, immutable YoDb catalog."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
import math
from typing import TYPE_CHECKING, Any
from uuid import UUID

from ..catalog import FieldSpec, LogicalType, Visibility
from ..errors import ErrorCode, ErrorDetail, QueryError
from ..runtime.contracts import CatalogEvaluation
from ..runtime.service import InMemoryCatalogRuntime
from .fingerprint import catalog_fingerprint, query_fingerprint
from .models import (
    AggregateFunction,
    AggregateTerm,
    AllExpression,
    AnyExpression,
    BoundAggregateTerm,
    BoundAggregation,
    BoundAllExpression,
    BoundAnyExpression,
    BoundDataset,
    BoundField,
    BoundFilterExpression,
    BoundNotExpression,
    BoundOrderTerm,
    BoundPredicate,
    BoundExtensionTerm,
    BoundQuery,
    ComparisonOperator,
    ExtensionTerm,
    FilterExpression,
    NotExpression,
    OrderTerm,
    PageRequest,
    Predicate,
    QueryConstraints,
    QueryRequest,
    RuleReference,
    SortDirection,
)
from .extensions import TermRegistry, extension_terms, validate_placement
from .fields import bind_public_field
from .registry import DEFAULT_TERMS

if TYPE_CHECKING:
    from ..rules import RuleExpansion


_EQUALITY_OPERATORS = frozenset(
    {
        ComparisonOperator.EQ,
        ComparisonOperator.NE,
        ComparisonOperator.IN,
        ComparisonOperator.NOT_IN,
        ComparisonOperator.IS_NULL,
        ComparisonOperator.IS_NOT_NULL,
    }
)
_RANGE_OPERATORS = frozenset(
    {ComparisonOperator.GT, ComparisonOperator.GTE, ComparisonOperator.LT, ComparisonOperator.LTE}
)
_TEXT_OPERATORS = frozenset({ComparisonOperator.CONTAINS, ComparisonOperator.STARTS_WITH})
_EQUALITY_TYPES = frozenset(
    {
        LogicalType.ID,
        LogicalType.STRING,
        LogicalType.TEXT,
        LogicalType.INT,
        LogicalType.FLOAT,
        LogicalType.BOOL,
        LogicalType.TIMESTAMP,
        LogicalType.UUID,
    }
)


@dataclass(frozen=True)
class QueryValidationPolicy:
    """Deployment-local bounds for the backend-neutral query validator."""

    default_page_size: int = 100
    maximum_page_size: int = 500
    maximum_in_values: int = 1_000
    # Per-extension limits by name (e.g. {"semantic": 2}); an extension supplies its own default.
    limits: Mapping[str, int] = field(default_factory=dict)

    def limit(self, name: str, default: int) -> int:
        """The configured limit for an extension, or its own default."""

        return self.limits.get(name, default)

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.limits.values()):
            raise ValueError("extension limits must not be negative")
        if self.default_page_size <= 0:
            raise ValueError("default_page_size must be positive")
        if self.maximum_page_size < self.default_page_size:
            raise ValueError("maximum_page_size must be at least default_page_size")
        if self.maximum_in_values <= 0:
            raise ValueError("maximum_in_values must be positive")


def validate_query(
    request: QueryRequest,
    runtime: InMemoryCatalogRuntime,
    *,
    policy: QueryValidationPolicy = QueryValidationPolicy(),
    terms: TermRegistry = DEFAULT_TERMS,
) -> BoundQuery:
    """Bind a query to the runtime's active catalog or raise a structured error."""

    return bind_query(request, runtime.require_active(), policy=policy, terms=terms)


def bind_query(
    request: QueryRequest,
    active: CatalogEvaluation,
    *,
    policy: QueryValidationPolicy = QueryValidationPolicy(),
    terms: TermRegistry = DEFAULT_TERMS,
) -> BoundQuery:
    """Validate a parsed request against exactly one active catalog evaluation."""

    catalog = active.catalog
    dataset = catalog.datasets.get(request.root.dataset)
    if dataset is None:
        _fail(ErrorCode.DATASET_NOT_FOUND, f"Unknown dataset '{request.root.dataset}'.", "from.dataset")
    root = BoundDataset(name=request.root.dataset, scope=request.root.scope, spec=dataset)

    aggregation: BoundAggregation | None = None
    if request.aggregate or request.group_by:
        if request.select is not None:
            _fail(
                ErrorCode.QUERY_AGGREGATE_INVALID,
                "'select' cannot be combined with 'aggregate' or 'group_by': the result's columns are the group keys "
                "and the aggregate aliases.",
                "select",
            )
        aggregation = _bind_aggregation(request, root)
        selected_fields: tuple[BoundField, ...] = ()
    else:
        selected_names = (
            request.select
            if request.select is not None
            else tuple(
                field_name
                for field_name, field in dataset.fields.items()
                if field.visibility is Visibility.PUBLIC
            )
        )
        selected_fields = tuple(
            bind_public_field(root, field_name, f"select[{index}]")
            for index, field_name in enumerate(selected_names)
        )
        if "id" not in selected_names:
            selected_fields = (bind_public_field(root, "id", "select"), *selected_fields)

    from ..rules import RuleExpansion       # rules sit above the query package
    from ..rules.expand import check_expanded_size

    expansion = RuleExpansion(catalog)
    bound_filter = (
        _bind_expression(request.where, root, "where", policy, terms, expansion) if request.where is not None else None
    )
    if expansion.uses:
        check_expanded_size(bound_filter)
    validate_placement(bound_filter, terms, policy)
    if aggregation is not None:
        if extension_terms(bound_filter):
            _fail(
                ErrorCode.QUERY_AGGREGATE_UNSUPPORTED,
                "Aggregates cannot be combined with a semantic or similarity condition yet.",
                "where",
            )
        bound_order = _bind_aggregate_order(request.order_by, aggregation)
        effective_order = _with_group_tiebreaker(bound_order, aggregation)
    else:
        bound_order = _bind_order(request.order_by, root)
        effective_order = _with_id_tiebreaker(bound_order, root)
    page = _validate_page(request.page, policy)
    constraints = request.constraints or QueryConstraints()
    _validate_constraints(constraints, policy)
    if constraints.minimum_quality is not None:
        quality_nouns = sorted({e.noun for e in terms if e.uses_minimum_quality})
        if not any(terms.for_bound(term).uses_minimum_quality for term in extension_terms(bound_filter)):
            _fail(
                ErrorCode.QUERY_LIMIT_INVALID,
                f"'constraints.minimum_quality' applies to {' / '.join(quality_nouns) or 'extension'}s; the query has none.",
                "constraints.minimum_quality",
            )

    payload = {
        "query_language_version": "yodb/v0.1-query-core",
        "from": {"dataset": root.name, "scope": root.scope},
        "select": sorted(field.name for field in selected_fields),
        **_aggregation_payload(aggregation),
        "where": _expression_payload(bound_filter, terms),
        "order_by": [
            {"field": term.field.name, "direction": term.direction.value} for term in effective_order
        ],
        "constraints": _constraints_payload(constraints),
    }
    return BoundQuery(
        root=root,
        select=selected_fields,
        where=bound_filter,
        order_by=effective_order,
        page=page,
        constraints=constraints,
        query_fingerprint=query_fingerprint(payload),
        catalog_fingerprint=catalog_fingerprint(catalog),
        rules=expansion.uses,
        explicit_order=bool(request.order_by),
        aggregation=aggregation,
    )


def bind_filter(
    expression: FilterExpression,
    root: BoundDataset,
    location: str,
    policy: QueryValidationPolicy,
    terms: TermRegistry,
    expansion: "RuleExpansion",
) -> BoundFilterExpression:
    """Bind one filter to a dataset, expanding the rules in it (a rule's condition is bound this way)."""

    return _bind_expression(expression, root, location, policy, terms, expansion)


def _bind_expression(
    expression: FilterExpression,
    root: BoundDataset,
    location: str,
    policy: QueryValidationPolicy,
    terms: TermRegistry,
    expansion: "RuleExpansion",
) -> BoundFilterExpression:
    if isinstance(expression, RuleReference):
        return expansion.expand(
            expression, root, location, lambda inner, at: _bind_expression(inner, root, at, policy, terms, expansion)
        )
    if isinstance(expression, Predicate):
        field = bind_public_field(root, expression.field, f"{location}.field")
        _validate_operator(field, expression, location)
        normalized_value = _normalize_predicate_value(field, expression, location, policy)
        return BoundPredicate(field, expression.operator, normalized_value, expression.value_supplied)
    if isinstance(expression, ExtensionTerm):
        return terms.for_parsed(expression).bind(expression, root, location, policy)
    if isinstance(expression, AllExpression):
        return BoundAllExpression(
            tuple(
                _bind_expression(item, root, f"{location}.all[{index}]", policy, terms, expansion)
                for index, item in enumerate(expression.expressions)
            )
        )
    if isinstance(expression, AnyExpression):
        return BoundAnyExpression(
            tuple(
                _bind_expression(item, root, f"{location}.any[{index}]", policy, terms, expansion)
                for index, item in enumerate(expression.expressions)
            )
        )
    if isinstance(expression, NotExpression):
        return BoundNotExpression(_bind_expression(expression.expression, root, f"{location}.not", policy, terms, expansion))
    raise AssertionError(f"Unknown expression: {expression!r}")


def _bind_order(terms: tuple[OrderTerm, ...], root: BoundDataset) -> tuple[BoundOrderTerm, ...]:
    bound: list[BoundOrderTerm] = []
    for index, term in enumerate(terms):
        field = bind_public_field(root, term.field, f"order_by[{index}].field")
        if field.spec.type in {LogicalType.JSON, LogicalType.BYTES}:
            _fail(
                ErrorCode.QUERY_OPERATOR_NOT_SUPPORTED,
                f"Field '{field.name}' cannot be ordered with its logical type.",
                f"order_by[{index}].field",
            )
        bound.append(BoundOrderTerm(field, term.direction))
    return tuple(bound)


_SUMMABLE_TYPES = frozenset({LogicalType.INT, LogicalType.FLOAT})
_EXTREME_TYPES = frozenset(
    {LogicalType.ID, LogicalType.STRING, LogicalType.TEXT, LogicalType.INT, LogicalType.FLOAT, LogicalType.TIMESTAMP}
)
_UNGROUPABLE_TYPES = frozenset({LogicalType.JSON, LogicalType.BYTES})


def _bind_aggregation(request: QueryRequest, root: BoundDataset) -> BoundAggregation:
    """Check ``group_by`` and ``aggregate`` against the dataset: public fields, types the function can take,
    and aliases that cannot be mistaken for a field.  ``group_by`` alone is the distinct combinations of its fields."""

    keys: list[BoundField] = []
    for index, name in enumerate(request.group_by):
        key = bind_public_field(root, name, f"group_by[{index}]")
        if key.spec.type in _UNGROUPABLE_TYPES:
            _fail(
                ErrorCode.QUERY_AGGREGATE_INVALID,
                f"Field '{name}' cannot be grouped by with its logical type '{key.spec.type.value}'.",
                f"group_by[{index}]",
            )
        keys.append(key)
    terms: list[BoundAggregateTerm] = []
    seen: set[str] = {key.name for key in keys}
    for index, term in enumerate(request.aggregate):
        location = f"aggregate[{index}]"
        if term.alias in root.spec.fields or term.alias in seen:
            _fail(
                ErrorCode.QUERY_AGGREGATE_INVALID,
                f"The alias '{term.alias}' is already a field of '{root.name}' or another alias: choose a name of its own.",
                f"{location}.as",
            )
        seen.add(term.alias)
        terms.append(_bind_aggregate_term(term, root, location))
    return BoundAggregation(group_by=tuple(keys), terms=tuple(terms))


def _bind_aggregate_term(term: AggregateTerm, root: BoundDataset, location: str) -> BoundAggregateTerm:
    function = term.function
    field: BoundField | None = None
    if term.field is None:
        if function is not AggregateFunction.COUNT:
            _fail(ErrorCode.QUERY_AGGREGATE_INVALID, f"'{function.value}' needs a 'field'.", f"{location}.field")
    else:
        field = bind_public_field(root, term.field, f"{location}.field")
        kind = field.spec.type
        if function in {AggregateFunction.SUM, AggregateFunction.AVG} and kind not in _SUMMABLE_TYPES:
            _fail(
                ErrorCode.QUERY_AGGREGATE_INVALID,
                f"'{function.value}' needs a numeric field; '{field.name}' is '{kind.value}'.",
                f"{location}.field",
            )
        if function in {AggregateFunction.MIN, AggregateFunction.MAX} and kind not in _EXTREME_TYPES:
            _fail(
                ErrorCode.QUERY_AGGREGATE_INVALID,
                f"'{function.value}' cannot take field '{field.name}' of type '{kind.value}'.",
                f"{location}.field",
            )
        if function is AggregateFunction.COUNT_DISTINCT and kind in _UNGROUPABLE_TYPES:
            _fail(
                ErrorCode.QUERY_AGGREGATE_INVALID,
                f"'count_distinct' cannot take field '{field.name}' of type '{kind.value}'.",
                f"{location}.field",
            )
    result = BoundField(
        dataset_name=root.name,
        scope=root.scope,
        name=term.alias,
        spec=FieldSpec(type=_result_type(function, field), description=f"{function.value} aggregate"),
    )
    return BoundAggregateTerm(function=function, field=field, alias=term.alias, result=result)


def _result_type(function: AggregateFunction, field: BoundField | None) -> LogicalType:
    if function in {AggregateFunction.COUNT, AggregateFunction.COUNT_DISTINCT}:
        return LogicalType.INT
    if function is AggregateFunction.AVG:
        return LogicalType.FLOAT
    assert field is not None
    return field.spec.type              # sum of an int is an int, sum of a float a float; min/max keep the type


def _bind_aggregate_order(terms: tuple[OrderTerm, ...], aggregation: BoundAggregation) -> tuple[BoundOrderTerm, ...]:
    by_name = {key.name: key for key in aggregation.group_by}
    by_name.update({term.alias: term.result for term in aggregation.terms})
    bound: list[BoundOrderTerm] = []
    for index, term in enumerate(terms):
        field = by_name.get(term.field)
        if field is None:
            _fail(
                ErrorCode.QUERY_AGGREGATE_INVALID,
                f"'order_by' of an aggregate query may name only its group keys or aggregate aliases, not '{term.field}'.",
                f"order_by[{index}].field",
            )
        bound.append(BoundOrderTerm(field, term.direction))
    return tuple(bound)


def _with_group_tiebreaker(
    terms: tuple[BoundOrderTerm, ...], aggregation: BoundAggregation
) -> tuple[BoundOrderTerm, ...]:
    """Groups have no ``id``: a group is identified by its keys, so ties break on them in ``group_by`` order."""

    ordered = {term.field.name for term in terms}
    return (*terms, *(BoundOrderTerm(key, SortDirection.ASC) for key in aggregation.group_by if key.name not in ordered))


def _aggregation_payload(aggregation: BoundAggregation | None) -> dict[str, object]:
    if aggregation is None:
        return {}
    return {
        "group_by": [key.name for key in aggregation.group_by],
        "aggregate": [
            {"fn": term.function.value, "field": None if term.field is None else term.field.name, "as": term.alias}
            for term in aggregation.terms
        ],
    }


def _with_id_tiebreaker(
    terms: tuple[BoundOrderTerm, ...], root: BoundDataset) -> tuple[BoundOrderTerm, ...]:
    if terms and terms[-1].field.name == "id":
        return terms
    return (*terms, BoundOrderTerm(bind_public_field(root, "id", "order_by"), SortDirection.ASC))


def _validate_operator(field: BoundField, predicate: Predicate, location: str) -> None:
    operator = predicate.operator
    field_type = field.spec.type
    if operator in _EQUALITY_OPERATORS and field_type not in _EQUALITY_TYPES:
        _unsupported_operator(field, operator, location)
    if operator in _RANGE_OPERATORS and field_type not in {
        LogicalType.INT,
        LogicalType.FLOAT,
        LogicalType.TIMESTAMP,
    }:
        _unsupported_operator(field, operator, location)
    if operator in _TEXT_OPERATORS and field_type not in {LogicalType.STRING, LogicalType.TEXT}:
        _unsupported_operator(field, operator, location)
    if operator in {ComparisonOperator.IS_NULL, ComparisonOperator.IS_NOT_NULL}:
        if predicate.value_supplied:
            _fail(
                ErrorCode.QUERY_VALUE_TYPE_INVALID,
                f"Operator '{operator.value}' does not take a value.",
                f"{location}.value",
            )
    elif not predicate.value_supplied or predicate.value is None:
        _fail(
            ErrorCode.QUERY_VALUE_TYPE_INVALID,
            f"Operator '{operator.value}' requires a non-null value.",
            f"{location}.value",
        )


def _unsupported_operator(field: BoundField, operator: ComparisonOperator, location: str) -> None:
    _fail(
        ErrorCode.QUERY_OPERATOR_NOT_SUPPORTED,
        f"Operator '{operator.value}' is not supported for field '{field.name}' of type '{field.spec.type.value}'.",
        f"{location}.op",
    )


def _normalize_predicate_value(
    field: BoundField,
    predicate: Predicate,
    location: str,
    policy: QueryValidationPolicy,
) -> object | None:
    if predicate.operator in {ComparisonOperator.IS_NULL, ComparisonOperator.IS_NOT_NULL}:
        return None
    if predicate.operator in {ComparisonOperator.IN, ComparisonOperator.NOT_IN}:
        if not isinstance(predicate.value, list) or not predicate.value:
            _fail(
                ErrorCode.QUERY_VALUE_TYPE_INVALID,
                f"Operator '{predicate.operator.value}' requires a non-empty array.",
                f"{location}.value",
            )
        if len(predicate.value) > policy.maximum_in_values:
            _fail(ErrorCode.QUERY_LIMIT_INVALID, "Filter list exceeds the configured maximum.", f"{location}.value")
        return tuple(_normalize_scalar(field, item, f"{location}.value[{index}]") for index, item in enumerate(predicate.value))
    return _normalize_scalar(field, predicate.value, f"{location}.value")


def _normalize_scalar(field: BoundField, value: Any, location: str) -> object:
    logical_type = field.spec.type
    if logical_type in {LogicalType.ID, LogicalType.STRING, LogicalType.TEXT}:
        if not isinstance(value, str) or (logical_type is LogicalType.ID and not value.strip()):
            _type_error(field, location)
        return value
    if logical_type is LogicalType.UUID:
        if not isinstance(value, str):
            _type_error(field, location)
        try:
            return str(UUID(value))
        except ValueError:
            _type_error(field, location)
    if logical_type is LogicalType.INT:
        if type(value) is not int:
            _type_error(field, location)
        return value
    if logical_type is LogicalType.FLOAT:
        if type(value) not in {int, float} or isinstance(value, bool) or not math.isfinite(value):
            _type_error(field, location)
        return float(value)
    if logical_type is LogicalType.BOOL:
        if type(value) is not bool:
            _type_error(field, location)
        return value
    if logical_type is LogicalType.TIMESTAMP:
        if not isinstance(value, str):
            _type_error(field, location)
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            _type_error(field, location)
        if parsed.tzinfo is None:
            _type_error(field, location)
        return parsed.astimezone(UTC)
    _type_error(field, location)


def _type_error(field: BoundField, location: str) -> None:
    _fail(
        ErrorCode.QUERY_VALUE_TYPE_INVALID,
        f"Expected a value compatible with logical type '{field.spec.type.value}'.",
        location,
    )


def _validate_page(page: PageRequest | None, policy: QueryValidationPolicy) -> PageRequest:
    resolved = page or PageRequest()
    first = resolved.first if resolved.first is not None else policy.default_page_size
    if first > policy.maximum_page_size:
        _fail(
            ErrorCode.QUERY_LIMIT_INVALID,
            f"'page.first' exceeds the maximum page size of {policy.maximum_page_size}.",
            "page.first",
        )
    return PageRequest(first=first)


def _validate_constraints(constraints: QueryConstraints, policy: QueryValidationPolicy) -> None:
    if constraints.maximum_results is not None and constraints.maximum_results > policy.maximum_page_size:
        _fail(
            ErrorCode.QUERY_LIMIT_INVALID,
            f"'constraints.maximum_results' exceeds the maximum of {policy.maximum_page_size}.",
            "constraints.maximum_results",
        )
    if constraints.minimum_quality is not None and constraints.minimum_quality > 1:
        _fail(
            ErrorCode.QUERY_LIMIT_INVALID,
            "'constraints.minimum_quality' must not exceed 1.",
            "constraints.minimum_quality",
        )


def _expression_payload(expression: BoundFilterExpression | None, terms: TermRegistry) -> Any:
    if expression is None:
        return None
    if isinstance(expression, BoundPredicate):
        return {
            "field": expression.field.name,
            "op": expression.operator.value,
            "value": expression.value,
            "value_supplied": expression.value_supplied,
        }
    if isinstance(expression, BoundExtensionTerm):
        return terms.for_bound(expression).describe(expression)
    if isinstance(expression, BoundAllExpression):
        return {"all": sorted((_expression_payload(item, terms) for item in expression.expressions), key=_canonical_sort_key)}
    if isinstance(expression, BoundAnyExpression):
        return {"any": sorted((_expression_payload(item, terms) for item in expression.expressions), key=_canonical_sort_key)}
    if isinstance(expression, BoundNotExpression):
        return {"not": _expression_payload(expression.expression, terms)}
    raise AssertionError(f"Unknown bound expression: {expression!r}")


def _constraints_payload(constraints: QueryConstraints) -> dict[str, object]:
    """Include only constraints that can alter a query's logical result meaning."""

    return {
        "maximum_results": constraints.maximum_results,
        "minimum_quality": constraints.minimum_quality,
    }


def _canonical_sort_key(value: Any) -> str:
    import json

    return json.dumps(value, default=str, sort_keys=True, separators=(",", ":"))


def _fail(code: ErrorCode, message: str, location: str | None = None) -> None:
    raise QueryError(ErrorDetail(code=code, message=message, retryable=False, location=location))
