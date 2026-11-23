"""Handler for the AGGREGATE operator when YoDb (not a source) groups the rows.

The semantics are SQL's, so a source that computes the same aggregate returns the same answer: ``COUNT(*)`` counts
rows and every other function skips NULLs, ``SUM``/``MIN``/``MAX``/``AVG`` of no values is NULL, ``AVG`` is a float,
NULL keys form one group, text compares in code-point order, and with no group keys an empty input still yields
one row.  The rows it reads were bounded by the scan's row guard.
"""

from __future__ import annotations

from ...errors import ErrorCode
from ...planning import AggregateMeasure, AggregateSpec, CoordinatorAggregate
from ...query.models import AggregateFunction
from ..contracts import LogicalRow
from .base import ExecutionContext, Run, fail


def execute(ctx: ExecutionContext, node: CoordinatorAggregate, run: Run) -> tuple[LogicalRow, ...]:
    rows = ctx.execute(node.input, run)
    if len(rows) > ctx.policy.maximum_coordinator_rows:
        fail(
            ErrorCode.QUERY_COORDINATOR_LIMIT_EXCEEDED,
            f"Coordinator processing exceeded {ctx.policy.maximum_coordinator_rows} rows.",
        )
    return aggregate_rows(node.spec, rows)


def aggregate_rows(spec: AggregateSpec, rows: tuple[LogicalRow, ...]) -> tuple[LogicalRow, ...]:
    key_names = tuple(key.field.name for key in spec.group_by)
    groups: dict[tuple[object, ...], list[LogicalRow]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(name) for name in key_names), []).append(row)
    if not key_names and not groups:
        groups[()] = []                      # SQL: an aggregate over no rows is still one row (count 0, the rest NULL)
    return tuple(
        {
            **dict(zip(key_names, key, strict=True)),
            **{measure.alias: _compute(measure, members) for measure in spec.measures},
        }
        for key, members in groups.items()
    )


def _compute(measure: AggregateMeasure, rows: list[LogicalRow]) -> object:
    function = measure.function
    if measure.field is None:
        return len(rows)                                         # COUNT(*)
    name = measure.field.field.name
    values = [value for value in (row.get(name) for row in rows) if value is not None]
    if function is AggregateFunction.COUNT:
        return len(values)
    if function is AggregateFunction.COUNT_DISTINCT:
        return len(set(values))
    if not values:
        return None
    if function is AggregateFunction.SUM:
        return sum(values)
    if function is AggregateFunction.AVG:
        return sum(values) / len(values)
    if function is AggregateFunction.MIN:
        return min(values)
    if function is AggregateFunction.MAX:
        return max(values)
    raise AssertionError(f"Unknown aggregate function: {function!r}")
