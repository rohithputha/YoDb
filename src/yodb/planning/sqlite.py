"""SQLite's declared capabilities for the planner.

Everything here is a promise the SQLite compiler keeps exactly.  What SQLite cannot promise is left
out, so the planner keeps it in YoDb: timestamps and UUIDs are not pushed (SQLite stores them as text,
numbers or blobs, so a comparison in SQL could silently disagree with the coordinator's), there is
no vector search and no graph engine, and a key lookup is smaller than PostgreSQL's so one ``IN`` list
stays well under SQLite's bound-variable limit.
"""

from __future__ import annotations

from ..catalog import LogicalType, SourceKind
from ..query.models import AggregateFunction, ComparisonOperator
from .adapter import CapabilityPlanningAdapter
from .capabilities import AggregateCapability, BooleanOperator, KeyLookupCapability, SourceCapabilities, TextOrdering

_PUSHED_TYPES = frozenset(
    {
        LogicalType.ID,
        LogicalType.STRING,
        LogicalType.TEXT,
        LogicalType.INT,
        LogicalType.FLOAT,
        LogicalType.BOOL,
    }
)

SQLITE_CAPABILITIES = SourceCapabilities(
    source_kind=SourceKind.SQLITE,
    filter_operators=frozenset(
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
    ),
    filterable_types=_PUSHED_TYPES,
    boolean_operators=frozenset(BooleanOperator),
    supports_order=True,
    supports_limit=True,
    orderable_types=_PUSHED_TYPES,
    text_ordering=TextOrdering.CODE_POINT,      # COLLATE BINARY, with NULLs placed as the coordinator places them
    maximum_rows=None,
    key_lookup=KeyLookupCapability(maximum_keys=500),
    vector_search=None,
    graph=None,
    # Grouping and min/max follow the same types as filtering: a timestamp or UUID grouped in SQL would split
    # what YoDb treats as one value (text '...Z' and an epoch number), so those are grouped in YoDb.
    aggregate=AggregateCapability(
        functions=frozenset(AggregateFunction),
        group_types=_PUSHED_TYPES,
        measure_types=_PUSHED_TYPES,
    ),
)


class SqlitePlanningAdapter(CapabilityPlanningAdapter):
    def __init__(self) -> None:
        super().__init__(SQLITE_CAPABILITIES)
