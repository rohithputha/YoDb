"""Reads that do not depend on each other run at the same time: the sides of an independent join, the
first reads of a record assembly.  Results never change; only the order and the overlap of the reads do."""

from __future__ import annotations

from dataclasses import replace
import threading
import time
import unittest

from yodb.catalog import SourceKind
from yodb.compilation import PostgresQueryCompiler, QueryCompilerRegistry
from yodb.errors import ErrorCode, QueryExecutionError
from yodb.execution import QueryExecutionAdapterRegistry, QueryExecutionEngine
from yodb.execution.federated import ExecutionTrace, FederatedPlanExecutor
from yodb.execution.operators import FederatedExecutionPolicy
from yodb.planning import (
    AssemblyStep,
    FederatedPhysicalPlanner,
    HashJoin,
    JoinPolicy,
    PostgresPlanningAdapter,
    RecordAssembly,
    SourcePlanningRegistry,
    SourceStatistics,
    StatisticsService,
    transform_plan,
)
from yodb.query import bind_query, parse_query, resolve_query_sources

from support.catalogs import StaticRuntime
from support.graph import GraphWorld, graph_active
from support.shop import shop_active, shop_world
from support.statistics import MapProvider
from test_joins import p, query, rows_of, traverse

ACTIVE = shop_active()


class SlowWorld:
    """Wraps a world: every read takes ``delay`` seconds and records which reads overlapped."""

    source_kind = SourceKind.POSTGRES

    def __init__(self, inner, delay: float = 0.05) -> None:
        self.inner = inner
        self.delay = delay
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()
        self.order: list[str] = []

    def execute(self, query, *, timeout_seconds=None):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.order.append(query.source_name)
        try:
            time.sleep(self.delay)
            return self.inner.execute(query, timeout_seconds=timeout_seconds)
        finally:
            with self.lock:
                self.active -= 1

    @property
    def queries(self):
        return self.inner.queries

    def reads_of(self, source):
        return self.inner.reads_of(source)


def counts(customers: int, tickets: int) -> StatisticsService:
    provider = MapProvider({
        "crm.customers": SourceStatistics(row_count=customers), "billing.plans": SourceStatistics(row_count=customers),
        "desk.tickets": SourceStatistics(row_count=tickets), "triage.assignments": SourceStatistics(row_count=tickets),
    })
    return StatisticsService({SourceKind.POSTGRES: provider})


def engine(world, *, statistics=None, join_policy=JoinPolicy(), execution_policy=FederatedExecutionPolicy(), active=ACTIVE):
    planner = FederatedPhysicalPlanner(SourcePlanningRegistry([PostgresPlanningAdapter()]), statistics=statistics)
    return QueryExecutionEngine(
        StaticRuntime(active), QueryCompilerRegistry([PostgresQueryCompiler()]), QueryExecutionAdapterRegistry([world]),
        planner=planner, statistics=statistics, join_policy=join_policy, execution_policy=execution_policy,
    )


BOTH_FILTERED = query(select=("name",), where=p("country", "eq", "US"), steps=traverse(select=["subject"], where=p("status", "eq", "open")))


class IndependentJoinTests(unittest.TestCase):
    def test_with_statistics_saying_both_sides_fit_the_sides_run_at_the_same_time(self) -> None:
        world = SlowWorld(shop_world())
        db = engine(world, statistics=counts(1_000, 1_000))
        explanation = db.explain(BOTH_FILTERED)
        self.assertIn("strategy_join=parallel", explanation.optimizer)
        self.assertTrue(any("strategy=parallel" in d for n in explanation.nodes if n.kind == "hash_join" for d in n.detail))
        self.assertEqual(explanation.nodes[-1].key_transfer_max_keys, None)
        got = rows_of(db.execute(BOTH_FILTERED))
        self.assertEqual(world.peak, 2)                                            # the two sides overlapped
        self.assertTrue(all("IN (" not in q.sql for q in world.queries))          # neither side was probed by keys
        expected = rows_of(engine(shop_world(), join_policy=JoinPolicy(parallel="never")).execute(BOTH_FILTERED))
        self.assertEqual(got, expected)
        self.assertTrue(got)

    def test_a_lopsided_join_probes_by_keys_even_when_both_sides_would_fit(self) -> None:
        """Reading 10,000 filtered rows whole to match 20 driver rows moves far more than probing with 20 keys."""

        for customers, tickets in ((20, 10_000), (1_000, 6_000)):
            with self.subTest(customers=customers, tickets=tickets):
                explanation = engine(shop_world(), statistics=counts(customers, tickets)).explain(BOTH_FILTERED)
                self.assertIn("strategy_join=sequential", explanation.optimizer)
                self.assertTrue(any("far more than probing" in note for note in explanation.optimizer), explanation.optimizer)
        balanced = engine(shop_world(), statistics=counts(1_000, 3_000)).explain(BOTH_FILTERED)
        self.assertIn("strategy_join=parallel", balanced.optimizer)
        loose = engine(shop_world(), statistics=counts(20, 10_000), join_policy=JoinPolicy(parallel_ratio=1000, parallel_maximum_other_rows=50_000)).explain(BOTH_FILTERED)
        self.assertIn("strategy_join=parallel", loose.optimizer)

    def test_parallel_always_needs_no_statistics_and_never_turns_it_off(self) -> None:
        world = SlowWorld(shop_world())
        db = engine(world, join_policy=JoinPolicy(parallel="always"))
        self.assertIn("strategy_join=parallel", db.explain(BOTH_FILTERED).optimizer)
        db.execute(BOTH_FILTERED)
        self.assertEqual(world.peak, 2)
        off = engine(SlowWorld(shop_world()), statistics=counts(10, 10), join_policy=JoinPolicy(parallel="never"))
        self.assertIn("strategy_join=sequential", off.explain(BOTH_FILTERED).optimizer)

    def test_a_side_without_a_filter_or_a_left_join_stays_sequential(self) -> None:
        db = engine(shop_world(), join_policy=JoinPolicy(parallel="always"))
        self.assertIn("reason=a side has no filter of its own", db.explain(query(steps=traverse(where=p("status", "eq", "open")))).optimizer)
        self.assertIn("reason=a left join probes by keys", db.explain(query(where=p("country", "eq", "US"), steps=traverse(where=p("status", "eq", "open"), optional=True))).optimizer)

    def test_a_side_expected_too_large_is_probed_by_keys_instead(self) -> None:
        db = engine(shop_world(), statistics=counts(100_000_000, 10))        # 5,000,000 US customers expected
        notes = db.explain(BOTH_FILTERED).optimizer
        self.assertIn("strategy_join=sequential", notes)
        self.assertTrue(any("too large" in n for n in notes))

    def test_concurrency_one_runs_the_same_plan_one_side_after_the_other(self) -> None:
        world = SlowWorld(shop_world())
        db = engine(world, join_policy=JoinPolicy(parallel="always"), execution_policy=FederatedExecutionPolicy(concurrency=1))
        got = rows_of(db.execute(BOTH_FILTERED))
        self.assertEqual(world.peak, 1)
        self.assertEqual(got, rows_of(engine(shop_world(), join_policy=JoinPolicy(parallel="always")).execute(BOTH_FILTERED)))

    def test_the_other_sides_guard_applies_to_a_parallel_read(self) -> None:
        db = engine(shop_world(), join_policy=JoinPolicy(parallel="always", maximum_probe_rows=2))
        with self.assertRaises(QueryExecutionError) as caught:
            db.execute(BOTH_FILTERED)
        self.assertEqual(caught.exception.code, ErrorCode.QUERY_COORDINATOR_LIMIT_EXCEEDED)

    def test_an_edge_join_reads_both_sides_at_once_and_the_edges_decide_the_matches(self) -> None:
        active = graph_active()
        world = SlowWorld(GraphWorld())
        db = engine(world, join_policy=JoinPolicy(parallel="always"), active=active)
        raw = query(where=p("country", "eq", "US"), steps=traverse(select=["subject"], where=p("status", "eq", "open")))
        explanation = db.explain(raw)
        self.assertEqual(explanation.plan_kind, "edge_join")
        self.assertIn("strategy_join=parallel", explanation.optimizer)
        got = rows_of(db.execute(raw))
        self.assertGreaterEqual(world.peak, 2)
        expected = rows_of(engine(GraphWorld(), join_policy=JoinPolicy(parallel="never"), active=active).execute(raw))
        self.assertEqual(got, expected)
        self.assertTrue(got)

    def test_a_failure_on_one_side_is_the_querys_error(self) -> None:
        db = engine(shop_world(), join_policy=JoinPolicy(parallel="always", maximum_driver_rows=1))
        with self.assertRaises(QueryExecutionError) as caught:
            db.execute(BOTH_FILTERED)
        self.assertEqual(caught.exception.code, ErrorCode.QUERY_ROW_LIMIT_EXCEEDED)

    def test_the_plan_fingerprint_sees_the_strategy(self) -> None:
        parallel = engine(shop_world(), join_policy=JoinPolicy(parallel="always")).explain(BOTH_FILTERED).plan_fingerprint
        sequential = engine(shop_world(), join_policy=JoinPolicy(parallel="never")).explain(BOTH_FILTERED).plan_fingerprint
        self.assertNotEqual(parallel, sequential)


class AssemblyWaveTests(unittest.TestCase):
    """A record assembly whose plan reads two sources unrestricted reads them at the same time."""

    def plan_and_run(self, world, schedule_fix, concurrency=4):
        active = ACTIVE
        planner = FederatedPhysicalPlanner(SourcePlanningRegistry([PostgresPlanningAdapter()]))
        raw = {"from": {"dataset": "customer"}, "select": ["name", "plan"], "where": {"all": [p("country", "eq", "US"), p("plan", "eq", "pro")]}, "page": {"first": 50}}
        planned = planner.plan(resolve_query_sources(bind_query(parse_query(raw), active), active))

        def fix(node):
            if isinstance(node, RecordAssembly):
                return replace(node, schedule=tuple(schedule_fix(step) for step in node.schedule))
            return node

        plan = transform_plan(planned.plan, fix)
        trace = ExecutionTrace()
        executor = FederatedPlanExecutor(QueryCompilerRegistry([PostgresQueryCompiler()]), QueryExecutionAdapterRegistry([world]), policy=FederatedExecutionPolicy(concurrency=concurrency))
        rows = executor.execute(plan, trace=trace)
        return [dict(r) for r in rows], trace

    def test_unrestricted_reads_run_together_and_the_result_is_unchanged(self) -> None:
        world = SlowWorld(shop_world())
        rows, trace = self.plan_and_run(world, lambda step: replace(step, restrict=False))
        self.assertEqual(world.peak, 2)
        self.assertEqual([(n, label) for n, label in trace.concurrent_batches], [(2, "assembly")])
        plain_rows, plain_trace = self.plan_and_run(SlowWorld(shop_world()), lambda step: step)
        self.assertEqual(rows, plain_rows)
        self.assertEqual(plain_trace.concurrent_batches, [])                        # the second read needs the first's keys
        self.assertTrue(rows)

    def test_with_concurrency_one_the_same_plan_reads_one_source_at_a_time(self) -> None:
        world = SlowWorld(shop_world())
        rows, trace = self.plan_and_run(world, lambda step: replace(step, restrict=False), concurrency=1)
        self.assertEqual(world.peak, 1)
        self.assertEqual(trace.concurrent_batches, [])
        self.assertTrue(rows)


class PolicyTests(unittest.TestCase):
    def test_invalid_settings_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            JoinPolicy(parallel="sometimes")
        with self.assertRaises(ValueError):
            FederatedExecutionPolicy(concurrency=0)

    def test_a_hash_join_node_swaps_the_independent_side_as_its_second_input(self) -> None:
        db = engine(shop_world(), join_policy=JoinPolicy(parallel="always"))
        planned = db._joins.plan(BOTH_FILTERED, ACTIVE)
        node: HashJoin = planned.plan
        self.assertIsNotNone(node.independent)
        self.assertIs(node.inputs()[1], node.independent)
        self.assertIs(node.with_inputs(node.inputs()).independent, node.independent)
        self.assertEqual(node.strategy, "parallel")


if __name__ == "__main__":
    unittest.main()
