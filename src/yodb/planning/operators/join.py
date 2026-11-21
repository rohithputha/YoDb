"""JOIN: match the rows of two datasets over a declared relationship.

A join is not a new kind of read.  Each side is an ordinary planned query over one dataset
(so every pushdown, key transfer, guard and extension applies to it).  The *driving* side runs
once; its distinct join keys are sent to the *probing* side in batches (an ``in`` filter on the
probing dataset's join field, planned for each batch); the rows are then matched in memory.
The node keeps a template of the probing side so that explaining and fingerprinting a plan show
both sides; the plan for a real batch comes from ``probe_for``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import ClassVar, Literal

from ...catalog import SourceKind
from ...operators import OperatorKind
from ..capabilities import OperatorSupport, SourceCapabilities, SupportLevel, support_view
from ..contracts import PhysicalNode, PlanExplanationNode


@dataclass(frozen=True)
class JoinPolicy:
    """Bounds on the in-memory join (a query over them fails with a clear error)."""

    batch_size: int = 1_000                  # join keys sent to the probing side at a time
    maximum_driver_rows: int = 10_000        # rows the driving side may produce
    maximum_probe_rows: int = 50_000         # rows all probing batches together may produce
    maximum_joined_rows: int = 50_000        # rows the join may hold before ordering and paging
    # When both sides are filtered on their own, they can be read at the same time and matched afterwards
    # (an *independent* join) instead of probing the second with the first's keys.  "auto" does so when
    # statistics expect both sides to fit their guards; "always" whenever both sides have a filter; "never"
    # keeps every join sequential (the forced-sequential ablation).
    parallel: Literal["auto", "always", "never"] = "auto"
    # ``auto`` reads the other side whole (instead of probing it with the driver's keys) only when that is expected to
    # move about as many rows as probing would: the other side's estimate must be at most ``parallel_ratio`` times the
    # driver's and at most ``parallel_maximum_other_rows``.  Otherwise the time saved by reading both sides at once is
    # lost many times over in rows transferred and matched (measured: a 20-row driver against 10,000 filtered rows ran
    # 10-100x slower read whole than probed in batches of keys).
    parallel_ratio: float = 4.0
    parallel_maximum_other_rows: int = 5_000

    def __post_init__(self) -> None:
        if min(self.batch_size, self.maximum_driver_rows, self.maximum_probe_rows, self.maximum_joined_rows) < 1:
            raise ValueError("join limits must be positive")
        if self.parallel not in ("auto", "always", "never"):
            raise ValueError("parallel must be 'auto', 'always' or 'never'")


@dataclass(frozen=True)
class OrderSpec:
    column: str            # a key of the joined row: ``name`` or ``alias.name``
    descending: bool


@dataclass(frozen=True)
class HashJoin(PhysicalNode):
    """Run ``driver``, probe the other side in key batches, match in memory, order, page, project."""

    operator: ClassVar[OperatorKind] = OperatorKind.JOIN

    driver: PhysicalNode                       # the side that runs first, once
    probe: PhysicalNode                        # the other side, planned for a sample batch (explain, fingerprint)
    driver_side: Literal["left", "right"]      # left is the root dataset, right the traversed one
    driver_key: str                            # join field in the driving rows
    probe_key: str                             # join field in the probing rows
    relationship: str
    alias: str
    optional: bool                             # keep root rows that have no match (left join)
    left_columns: tuple[str, ...]              # output columns from the root dataset
    right_columns: tuple[str, ...]             # output columns from the traversed dataset (before the alias prefix)
    order: tuple[OrderSpec, ...]
    first: int
    policy: JoinPolicy
    key_description: str
    # The probing side for a real batch of keys and a row budget.  Not part of equality or the fingerprint.
    probe_for: Callable[[Sequence[object], int], PhysicalNode] = field(compare=False, repr=False, default=None)  # type: ignore[assignment]
    # The other side read on its own, concurrently with the driver (an independent join); None: probe by keys.
    independent: PhysicalNode | None = None

    @property
    def columns(self) -> tuple[str, ...]:
        return (*self.left_columns, *(f"{self.alias}.{name}" for name in self.right_columns))

    @property
    def strategy(self) -> str:
        return "parallel" if self.independent is not None else "probe"

    def inputs(self) -> tuple[PhysicalNode, ...]:
        return (self.driver, self.probe if self.independent is None else self.independent)

    def with_inputs(self, inputs: tuple[PhysicalNode, ...]) -> PhysicalNode:
        driver, other = inputs
        return replace(self, driver=driver, **({"probe": other} if self.independent is None else {"independent": other}))

    def shape(self) -> dict[str, object]:
        return {
            "kind": "hash_join",
            "relationship": self.relationship,
            "driver": self.driver_side,
            "optional": self.optional,
            "order": [(o.column, o.descending) for o in self.order],
            "first": self.first,
            "batch_size": self.policy.batch_size,
            "strategy": self.strategy,
            "driver_plan": self.driver.shape(),
            "probe_plan": (self.probe if self.independent is None else self.independent).shape(),
        }

    def describe(self) -> PlanExplanationNode:
        probe_side = "right" if self.driver_side == "left" else "left"
        how = (
            f"both sides are read at the same time (the {probe_side} side on its own, not by keys), then matched"
            if self.independent is not None
            else f"driving side ({self.driver_side}) is planned first; the probing side ({probe_side}) is shown for one "
            f"batch and runs once per batch of up to {self.policy.batch_size} keys"
        )
        return PlanExplanationNode(
            kind="hash_join",
            location="coordinator",
            fields=self.columns,
            ordering=tuple(f"{o.column} {'desc' if o.descending else 'asc'}" for o in self.order),
            limit=self.first,
            key_transfer_max_keys=None if self.independent is not None else self.policy.batch_size,
            detail=(
                f"relationship={self.relationship} as={self.alias} kind={'left' if self.optional else 'inner'} strategy={self.strategy}",
                f"on {self.key_description}",
                how,
            ),
        )


@dataclass(frozen=True)
class GraphEdgeScan(PhysicalNode):
    """Read the edges of one type between two vertex labels of a graph, starting from a batch of IDs.

    The graph is asked for the pairs (start ID, end ID) of the ``anchored`` end's
    vertices whose ID is in ``ids``.  It is a leaf, like a :class:`RemoteScan`:
    the compiler of the source turns it into a native statement (Cypher inside
    SQL for Apache AGE) and the executor runs it.  ``ids`` is filled per batch by
    the join handler, never by the planner.
    """

    operator: ClassVar[OperatorKind] = OperatorKind.TRAVERSE

    source_name: str
    source_kind: SourceKind
    connection_ref: str
    graph: str
    edge_type: str
    from_label: str
    from_id_property: str
    to_label: str
    to_id_property: str
    anchored: Literal["from", "to"]            # the end whose IDs are given
    ids: tuple[str, ...] = ()
    maximum_rows: int | None = None            # the executor refuses more edges than this (a guard)

    def shape(self) -> dict[str, object]:
        return {
            "kind": "graph_edge_scan",
            "source": self.source_name,
            "graph": self.graph,
            "edge_type": self.edge_type,
            "from": [self.from_label, self.from_id_property],
            "to": [self.to_label, self.to_id_property],
            "anchored": self.anchored,
        }

    def describe(self) -> PlanExplanationNode:
        start, end = (self.from_label, self.to_label) if self.anchored == "from" else (self.to_label, self.from_label)
        arrow = "-[edge]->" if self.anchored == "from" else "<-[edge]-"
        return PlanExplanationNode(
            kind="graph_edge_scan",
            location=self.source_name,
            fields=("start_id", "end_id"),
            limit=self.maximum_rows,
            detail=(
                f"graph={self.graph} edge_type={self.edge_type}",
                f"({start}) {arrow} ({end}), for one batch of {start} IDs",
            ),
        )


@dataclass(frozen=True)
class EdgeJoin(PhysicalNode):
    """Run ``driver``, follow the graph edges from its IDs, read the other side for the IDs reached, match in memory.

    The shape of a :class:`HashJoin` with the edges between: the driving side's IDs are sent to the graph
    in batches, the (start, end) pairs it returns say which rows match, and the other side is read only
    for the IDs the edges reach.  Both sides join on their ``id``.
    """

    operator: ClassVar[OperatorKind] = OperatorKind.TRAVERSE

    driver: PhysicalNode
    probe: PhysicalNode                        # the other side, planned for a sample batch (explain, fingerprint)
    edges: GraphEdgeScan                       # anchored at the driving side, with no IDs yet
    driver_side: Literal["left", "right"]
    relationship: str
    alias: str
    optional: bool
    left_columns: tuple[str, ...]
    right_columns: tuple[str, ...]
    order: tuple[OrderSpec, ...]
    first: int
    policy: JoinPolicy
    probe_for: Callable[[Sequence[object], int], PhysicalNode] = field(compare=False, repr=False, default=None)  # type: ignore[assignment]
    # The other side read on its own, concurrently with the driver; the edges then say which rows match.
    independent: PhysicalNode | None = None

    @property
    def columns(self) -> tuple[str, ...]:
        return (*self.left_columns, *(f"{self.alias}.{name}" for name in self.right_columns))

    @property
    def strategy(self) -> str:
        return "parallel" if self.independent is not None else "probe"

    def inputs(self) -> tuple[PhysicalNode, ...]:
        return (self.driver, self.edges, self.probe if self.independent is None else self.independent)

    def with_inputs(self, inputs: tuple[PhysicalNode, ...]) -> PhysicalNode:
        driver, edges, other = inputs
        return replace(self, driver=driver, edges=edges, **({"probe": other} if self.independent is None else {"independent": other}))

    def shape(self) -> dict[str, object]:
        return {
            "kind": "edge_join",
            "relationship": self.relationship,
            "driver": self.driver_side,
            "optional": self.optional,
            "order": [(o.column, o.descending) for o in self.order],
            "first": self.first,
            "batch_size": self.policy.batch_size,
            "strategy": self.strategy,
            "driver_plan": self.driver.shape(),
            "edges": self.edges.shape(),
            "probe_plan": (self.probe if self.independent is None else self.independent).shape(),
        }

    def describe(self) -> PlanExplanationNode:
        probe_side = "right" if self.driver_side == "left" else "left"
        how = (
            f"both sides are read at the same time (the {probe_side} side on its own); the driver's IDs then go to the "
            f"graph in batches of up to {self.policy.batch_size} and the edges say which rows match"
            if self.independent is not None
            else f"driving side ({self.driver_side}) runs first; its IDs go to the graph in batches of up to "
            f"{self.policy.batch_size}; the other side ({probe_side}) is read for the IDs the edges reach"
        )
        return PlanExplanationNode(
            kind="edge_join",
            location="coordinator",
            fields=self.columns,
            ordering=tuple(f"{o.column} {'desc' if o.descending else 'asc'}" for o in self.order),
            limit=self.first,
            key_transfer_max_keys=self.policy.batch_size,
            detail=(
                f"relationship={self.relationship} as={self.alias} kind={'left' if self.optional else 'inner'} strategy={self.strategy}",
                f"over graph edges {self.edges.edge_type} in graph {self.edges.graph} (source {self.edges.source_name})",
                how,
            ),
        )


@support_view(OperatorKind.JOIN)
def _join_support(caps: SourceCapabilities) -> OperatorSupport:
    strategies = ("driver_left", "driver_right")
    return OperatorSupport(
        OperatorKind.JOIN, SupportLevel.COORDINATOR, strategies,
        "YoDb joins in memory; each side is read by its own source, the probing side restricted to a batch of keys",
    )
