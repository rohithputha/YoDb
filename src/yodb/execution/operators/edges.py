"""Handler for a graph edge read: compile the Cypher-in-SQL statement, run it, decode the pairs.

Apache AGE returns ``agtype`` values, which arrive as their JSON text (``"c01"``
for a string, ``7`` for a number, ``null`` when the property is missing).  YoDb's
logical IDs are strings, so every decoded value is kept as text: a graph that
stores the number 7 as a vertex ID matches the dataset row whose ``id`` is ``"7"``.
"""

from __future__ import annotations

import json

from ...errors import ErrorCode
from ...planning import GraphEdgeScan
from .base import ExecutionContext, Run, fail

Pair = tuple[str, str]


def execute(ctx: ExecutionContext, node: GraphEdgeScan, run: Run) -> tuple[dict[str, object], ...]:
    """The (start ID, end ID) pairs of the edges leaving (or entering) ``node.ids``, as rows."""

    compiled = ctx.compilers.adapter_for(node.source_kind).compile_edges(node)
    rows = ctx.executors.adapter_for(compiled.source_kind).execute(compiled, timeout_seconds=run.remaining())
    if run.trace is not None:
        run.trace.record_read(len(rows))
    if node.maximum_rows is not None and len(rows) > node.maximum_rows:
        fail(
            ErrorCode.QUERY_COORDINATOR_LIMIT_EXCEEDED,
            f"The graph in source '{node.source_name}' returned more than {node.maximum_rows} edges for one batch of "
            "IDs; add a filter to the side of the join that drives it.",
            source_name=node.source_name,
        )
    decoded = []
    for row in rows:
        start, end = decode_agtype(row.get("start_id")), decode_agtype(row.get("end_id"))
        if start is None or end is None:
            continue                     # an edge to a vertex without the ID property can match no row
        decoded.append({"start_id": start, "end_id": end})
    return tuple(decoded)


def decode_agtype(value: object) -> str | None:
    """One scalar agtype value as text, or None for a missing one."""

    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("::vertex") or text.endswith("::edge"):
            return None                  # a whole vertex or edge, not an ID: the compiler never asks for one
        try:
            value = json.loads(text)
        except ValueError:
            return text                  # a driver that already decoded the text
    if value is None or isinstance(value, (dict, list)):
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
