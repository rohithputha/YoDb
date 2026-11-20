"""Handler for the JOIN operator: run the driving side, probe the other in key batches, match in memory."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cmp_to_key

from ...errors import ErrorCode, QueryExecutionError
from ...planning import EdgeJoin, HashJoin
from ...query.models import SortDirection
from ..contracts import LogicalRow
from .base import ExecutionContext, Run, fail, run_concurrently
from .rows import compare_rows


@dataclass(frozen=True)
class _Field:
    name: str


@dataclass(frozen=True)
class _Term:
    """What ``compare_rows`` reads of an order term: a field name and a direction."""

    field: _Field
    direction: SortDirection


def execute(ctx: ExecutionContext, node: HashJoin, run: Run) -> tuple[LogicalRow, ...]:
    """The driving side runs once and gives its distinct join keys; the probing side runs per batch
    of keys, planned for that batch, under the same deadline; the rows are matched in memory, then
    ordered, paged and projected.

    Each side is an ordinary plan, run by the same executor, so its own guards, key transfer and
    extensions apply.  The join adds only its own bounds (driving rows, probing rows, joined rows).
    """

    policy = node.policy
    if node.independent is not None:
        driver_rows, probe_rows = _both_sides(ctx, node, run)
    else:
        driver_rows = _driver(ctx, node, run)
        keys = list(dict.fromkeys(row[node.driver_key] for row in driver_rows if row.get(node.driver_key) is not None))
        probe_rows = []
        for start in range(0, len(keys), policy.batch_size):
            _probe(ctx, node, keys[start:start + policy.batch_size], probe_rows, run)
    by_key: dict[object, list[LogicalRow]] = {}
    for row in probe_rows:
        by_key.setdefault(row[node.probe_key], []).append(row)

    def matches_of(driver_row: LogicalRow) -> tuple[LogicalRow, ...]:
        key = driver_row.get(node.driver_key)
        return tuple(by_key.get(key, ())) if key is not None else ()

    return _finish(node, driver_rows, matches_of, run)


def execute_edges(ctx: ExecutionContext, node: EdgeJoin, run: Run) -> tuple[LogicalRow, ...]:
    """Like :func:`execute`, with the graph between the two sides: the driving side's IDs go to the graph in
    batches, the edges it returns say which pairs of rows match, and the other side is read only for the IDs
    the edges reach (in key batches, under the same guards as a hash join)."""

    policy = node.policy
    other_rows: list[LogicalRow] | None = None
    if node.independent is not None:
        driver_rows, other_rows = _both_sides(ctx, node, run)
    else:
        driver_rows = _driver(ctx, node, run)
    ids = list(dict.fromkeys(str(row["id"]) for row in driver_rows if row.get("id") is not None))
    pairs: dict[tuple[str, str], None] = {}             # (driver ID, other ID), each once, in edge order
    for start in range(0, len(ids), policy.batch_size):
        run.remaining()
        edges = ctx.execute(replace(node.edges, ids=tuple(ids[start:start + policy.batch_size]), maximum_rows=policy.maximum_probe_rows), run)
        for edge in edges:
            pairs[(edge["start_id"], edge["end_id"])] = None
        if len(pairs) > policy.maximum_probe_rows:
            fail(
                ErrorCode.QUERY_COORDINATOR_LIMIT_EXCEEDED,
                f"The relationship reaches more than {policy.maximum_probe_rows} rows; add a filter to one side of the join.",
            )
    if other_rows is None:
        reached = list(dict.fromkeys(other for _, other in pairs))
        other_rows = []
        for start in range(0, len(reached), policy.batch_size):
            _probe(ctx, node, reached[start:start + policy.batch_size], other_rows, run)
    by_id = {str(row["id"]): row for row in other_rows if row.get("id") is not None}
    others: dict[str, list[str]] = {}
    for mine, other in pairs:
        others.setdefault(mine, []).append(other)

    def matches_of(driver_row: LogicalRow) -> tuple[LogicalRow, ...]:
        if driver_row.get("id") is None:
            return ()
        found = (by_id.get(other) for other in others.get(str(driver_row["id"]), ()))
        return tuple(row for row in found if row is not None)       # an edge to a row the step's filter dropped: no match

    return _finish(node, driver_rows, matches_of, run)


def _both_sides(ctx: ExecutionContext, node, run: Run) -> tuple[tuple[LogicalRow, ...], list[LogicalRow]]:
    """An independent join: the driver and the other side run at the same time, each as its own plan
    with its own trace, then the traces are merged and the join's bounds applied to each side."""

    driver_rows, other_rows = run_concurrently(ctx, (node.driver, node.independent), run)
    if len(driver_rows) > node.policy.maximum_driver_rows:
        fail(
            ErrorCode.QUERY_ROW_LIMIT_EXCEEDED,
            f"The {node.driver_side} side of the join produced more than {node.policy.maximum_driver_rows} rows; "
            "add a filter to one side of the join.",
        )
    if len(other_rows) > node.policy.maximum_probe_rows:
        fail(
            ErrorCode.QUERY_COORDINATOR_LIMIT_EXCEEDED,
            f"The other side of the join produced more than {node.policy.maximum_probe_rows} rows; add a filter to one side of the join.",
        )
    return driver_rows, list(other_rows)


def _driver(ctx: ExecutionContext, node, run: Run) -> tuple[LogicalRow, ...]:
    driver_rows = _side(ctx, node.driver, run)
    if len(driver_rows) > node.policy.maximum_driver_rows:
        fail(
            ErrorCode.QUERY_ROW_LIMIT_EXCEEDED,
            f"The {node.driver_side} side of the join produced more than {node.policy.maximum_driver_rows} rows; "
            "add a filter to one side of the join.",
        )
    return driver_rows


def _finish(node, driver_rows: tuple[LogicalRow, ...], matches_of, run: Run) -> tuple[LogicalRow, ...]:
    """Match, then order, page and project the joined rows."""

    policy = node.policy
    joined: list[dict[str, object]] = []
    for driver_row in driver_rows:
        matches = matches_of(driver_row)
        if not matches and node.optional:
            joined.append(_merge(node, driver_row, None))   # a left join keeps the root row; its right fields are NULL
        for match in matches:
            joined.append(_merge(node, driver_row, match) if node.driver_side == "left" else _merge(node, match, driver_row))
        if len(joined) > policy.maximum_joined_rows:
            fail(
                ErrorCode.QUERY_COORDINATOR_LIMIT_EXCEEDED,
                f"The join produced more than {policy.maximum_joined_rows} rows; add a filter to one side.",
            )
    terms = [_Term(_Field(o.column), SortDirection.DESC if o.descending else SortDirection.ASC) for o in node.order]
    ranking = _relevance_ranking(node, run)
    if ranking is not None and not node.order:
        # The root side was ranked by similarity and the caller gave no order: keep that ranking.
        joined.sort(key=lambda row: (ranking.get(row.get("id"), len(ranking)), str(row.get(f"{node.alias}.id"))))
    else:
        terms += [_Term(_Field("id"), SortDirection.ASC), _Term(_Field(f"{node.alias}.id"), SortDirection.ASC)]
        joined.sort(key=cmp_to_key(lambda a, b: compare_rows(a, b, tuple(terms))))
    columns = node.columns
    return tuple({name: row.get(name) for name in columns} for row in joined[: node.first])


def _relevance_ranking(node, run: Run) -> dict[object, int] | None:
    """The rank of each root row when the driving (root) side ran a similarity search, else None."""

    if run.trace is None or node.driver_side != "left":
        return None
    report = run.trace.reports.get("similar")
    if report is None or not hasattr(report, "records"):
        return None
    return {logical_id: record.rank for logical_id, record in report.records.items()}


def _probe(ctx: ExecutionContext, node, keys: list[object], collected: list[LogicalRow], run: Run) -> None:
    """Read the probing side for ``keys``.  A batch whose matches overflow a read guard (a few very popular keys)
    is split in two and each half read on its own: smaller key sets match fewer rows."""

    policy = node.policy
    run.remaining()                                        # the whole query's budget, between batches too
    budget = policy.maximum_probe_rows - len(collected)
    try:
        rows = _side(ctx, node.probe_for(keys, budget + 1), run)
    except QueryExecutionError as error:
        if error.code is ErrorCode.QUERY_ROW_LIMIT_EXCEEDED and len(keys) > 1:
            middle = len(keys) // 2
            _probe(ctx, node, keys[:middle], collected, run)
            _probe(ctx, node, keys[middle:], collected, run)
            return
        raise
    collected.extend(rows)
    if len(collected) > policy.maximum_probe_rows:
        fail(
            ErrorCode.QUERY_COORDINATOR_LIMIT_EXCEEDED,
            f"The other side of the join matched more than {policy.maximum_probe_rows} rows; add a filter to one side.",
        )


def _side(ctx: ExecutionContext, plan, run: Run) -> tuple[LogicalRow, ...]:
    """Run one side as its own query, sharing the deadline and the trace but not the per-query flags."""

    return ctx.execute(plan, Run(run.timeout_seconds, run.trace, started=run.started))


def _merge(node, left: LogicalRow, right: LogicalRow | None) -> dict[str, object]:
    row: dict[str, object] = dict(left)
    # Every column of the traversed side is present (NULL when unmatched), including those only used to order.
    names = set(node.right_columns) | {o.column[len(node.alias) + 1:] for o in node.order if o.column.startswith(node.alias + ".")}
    for name in names | {"id"}:
        row[f"{node.alias}.{name}"] = None if right is None else right.get(name)
    return row
