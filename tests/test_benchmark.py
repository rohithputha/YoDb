"""The benchmark: templates, instantiation, the gold-plan harness and the scorer (no database needed)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.build import FIXED_POOLS, derive, fill, instances_of, instantiate, split_of, with_gold  # noqa: E402
from benchmark.harness import GoldPlanSystem, Prediction, run_plan, substitute_ids  # noqa: E402
from benchmark.score import bootstrap, f1, percentile, question_scores, score  # noqa: E402
from benchmark.templates import TEMPLATES, by_class  # noqa: E402

from yodb.compilation import PostgresQueryCompiler, QueryCompilerRegistry  # noqa: E402
from yodb.errors import YoDbError  # noqa: E402
from yodb.execution import QueryExecutionAdapterRegistry, QueryExecutionEngine  # noqa: E402
from yodb.joins import JoinPlanner  # noqa: E402
from yodb.planning import FederatedPhysicalPlanner, PostgresPlanningAdapter, SourcePlanningRegistry  # noqa: E402
from yodb.query import bind_query, parse_query, resolve_query_sources  # noqa: E402
from yodb.semantic import EmbeddingResult, ProviderInfo  # noqa: E402
from yodb.similar import SimilarExtension  # noqa: E402

from support.catalogs import StaticRuntime  # noqa: E402
from support.graph import GraphWorld, graph_active  # noqa: E402
from support.sql_world import evaluation  # noqa: E402

YELP = Path(__file__).resolve().parent.parent / "examples" / "yelp" / "catalog"
POOLS = {
    **FIXED_POOLS,
    "city": ["Philadelphia", "Tucson"],
    "category": [{"category": "coffee_tea", "category_name": "Coffee & Tea"}],
    "user_id": ["u1", "u2"],
    "business": [{"business_name": "Pat's", "business_city": "Philadelphia"}],
}


class FakeEmbedder:
    info = ProviderInfo("fake", "BAAI/bge-small-en-v1.5", "1")
    dimensions = 384

    def embed(self, request):
        return EmbeddingResult(tuple((0.0,) * 384 for _ in request.texts), self.info, 384)


def yelp_planners():
    active = evaluation(*(YELP.joinpath(f).read_text() for f in ("datasets.yaml", "sources.yaml", "relations.yaml", "rules.yaml")))
    planner = FederatedPhysicalPlanner(SourcePlanningRegistry([PostgresPlanningAdapter()]), extensions=(SimilarExtension(FakeEmbedder()),))
    return active, planner, JoinPlanner(planner)


class TemplateTests(unittest.TestCase):
    def test_the_workload_has_the_proposals_shape(self) -> None:
        self.assertEqual(by_class(), {"single": 60, "independent": 60, "dependent": 80, "stress": 40})
        self.assertEqual(len({t.id for t in TEMPLATES}), len(TEMPLATES))

    def test_every_gold_plan_validates_against_the_yelp_catalog_with_the_expected_outcome(self) -> None:
        active, planner, joins = yelp_planners()

        def plan(raw):
            if "traverse" in raw:
                return joins.plan(raw, active).explain.plan_kind
            return planner.plan(resolve_query_sources(bind_query(parse_query(raw), active), active)).explain.plan_kind

        for template in TEMPLATES:
            with self.subTest(template=template.id):
                record = instantiate(template, 1, instances_of(template, POOLS, "s")[0], "s")
                outcomes = []
                for step in record["plan"]:
                    concrete = substitute_ids(step, [[{"business.id": "x", "category.id": "x", "friend.id": "x"}]])
                    try:
                        outcomes.append(plan(concrete))
                    except YoDbError as error:
                        outcomes.append(error.code.value)
                if template.expect_error is None:
                    self.assertTrue(all(o in ("single_source", "hash_join", "edge_join") for o in outcomes), outcomes)
                    self.assertTrue(template.oracle and template.answer_dataset)
                elif template.expect_error == "refusal":
                    self.assertEqual(record["plan"], [])
                else:
                    self.assertIn(template.expect_error, outcomes)

    def test_every_non_stress_template_has_an_oracle_answer_column(self) -> None:
        for template in TEMPLATES:
            if template.oracle:
                self.assertIn("answer_id", template.oracle, template.id)
                for slot in template.slots:
                    # every slot the plan uses reaches the oracle too (directly, derived, or as the embedded phrase)
                    expected = {
                        "business": ("business_name", "business_city"), "category": ("category",), "year": ("year", "year_start", "month_start"),
                        "month": ("month_start",), "phrase": ("vector",),
                    }.get(slot, (slot,))
                    self.assertTrue(any(f"%({name})s" in template.oracle for name in expected), (template.id, slot))


class BuildTests(unittest.TestCase):
    def test_fill_keeps_the_type_of_a_bare_slot_and_formats_text(self) -> None:
        params = {"stars_hi": 4.5, "city": "Tucson", "phrase": "good coffee"}
        self.assertEqual(fill({"field": "stars", "op": "gte", "value": "{stars_hi}"}, params)["value"], 4.5)
        self.assertEqual(fill("the {top} reviews about {phrase}", {**params, "top": 10}), "the 10 reviews about good coffee")
        self.assertEqual(fill(["{city}", 3], params), ["Tucson", 3])

    def test_derived_slots(self) -> None:
        self.assertEqual(derive({"year": 2019, "month": 12})["month_end"], "2020-01-01T00:00:00Z")
        self.assertEqual(derive({"year": 2019, "month": 4})["month_name"], "April")
        self.assertEqual(derive({"year": 2018})["year_end"], "2019-01-01T00:00:00Z")

    def test_instances_are_deterministic_distinct_and_counted(self) -> None:
        template = next(t for t in TEMPLATES if t.id == "I01")
        first, second = instances_of(template, POOLS, "seed"), instances_of(template, POOLS, "seed")
        self.assertEqual(first, second)
        self.assertEqual(len(first), template.instances)
        distinct = {json.dumps(p, sort_keys=True) for p in first}
        self.assertEqual(len(distinct), min(template.instances, len(POOLS["city"]) * len(POOLS["stars_hi"])))
        self.assertNotEqual(instances_of(template, POOLS, "other"), first)

    def test_the_split_is_roughly_forty_in_two_hundred_and_forty(self) -> None:
        ids = [f"{t.id}-{i:03d}" for t in TEMPLATES for i in range(1, t.instances + 1)]
        tune = sum(1 for i in ids if split_of(i, "yodb-bench-v1") == "tune")
        self.assertTrue(25 <= tune <= 60, tune)

    def test_the_oracle_rows_become_answer_and_evidence_sets(self) -> None:
        class Cursor:
            description = [type("c", (), {"name": "answer_id"})(), type("c", (), {"name": "evidence_business"})()]

            def __init__(self, rows):
                self.rows, self.sql, self.params = rows, None, None

            def execute(self, sql, params):
                self.sql, self.params = sql, params

            def fetchall(self):
                return self.rows

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class Connection:
            def __init__(self, rows):
                self.cursor_ = Cursor(rows)

            def cursor(self):
                return self.cursor_

        template = next(t for t in TEMPLATES if t.id == "I01")
        record = instantiate(template, 1, {"city": "Tucson", "stars_hi": 4.0}, "s")
        connection = Connection([("r1", "b1"), ("r2", "b1"), ("r2", "b2")])
        gold = with_gold(record, template, connection)
        self.assertEqual(gold["gold_answer"], ["r1", "r2"])
        self.assertEqual(gold["gold_evidence"], {"business": ["b1", "b2"], "review": ["r1", "r2"]})
        self.assertEqual(connection.cursor_.params["city"], "Tucson")
        self.assertIn("Tucson", gold["question"])
        self.assertEqual(gold["split"], split_of("I01-001", "s"))


class HarnessTests(unittest.TestCase):
    """The gold-plan runner over the shop graph world: a one-step edge join and a two-step chain."""

    def setUp(self) -> None:
        active = graph_active()
        planner = FederatedPhysicalPlanner(SourcePlanningRegistry([PostgresPlanningAdapter()]))
        engine = QueryExecutionEngine(StaticRuntime(active), QueryCompilerRegistry([PostgresQueryCompiler()]), QueryExecutionAdapterRegistry([GraphWorld()]), planner=planner)

        class DB:
            catalog = active.catalog

            def query(self, raw, *, timeout_seconds=None):
                return engine.execute(raw, timeout_seconds=timeout_seconds)

        self.db = DB()

    def test_substitute_ids_takes_the_distinct_values_of_a_previous_step(self) -> None:
        rows = [[{"ticket.id": "t1"}, {"ticket.id": "t2"}, {"ticket.id": "t1"}, {"ticket.id": None}]]
        step = {"where": {"field": "id", "op": "in", "value": {"$ids": {"step": 0, "field": "ticket.id"}}}}
        self.assertEqual(substitute_ids(step, rows)["where"]["value"], ["t1", "t2"])

    def test_a_two_step_chain_runs_and_reports_answers_evidence_sources_and_transfer(self) -> None:
        plan = [
            {"from": {"dataset": "customer"}, "select": ["name"], "where": {"field": "name", "op": "eq", "value": "Ann"},
             "traverse": [{"relationship": "customer_has_ticket", "as": "ticket", "select": ["subject"]}]},
            {"from": {"dataset": "ticket"}, "select": ["subject", "status"], "where": {"all": [
                {"field": "id", "op": "in", "value": {"$ids": {"step": 0, "field": "ticket.id"}}},
                {"field": "status", "op": "eq", "value": "open"}]}},
        ]
        prediction = run_plan(self.db, plan, "id", "ticket")
        self.assertIsNone(prediction.error)
        self.assertTrue(prediction.plan_valid)
        self.assertEqual(prediction.answer, ["t1", "t3"])                 # Ann's open tickets, over the graph edges
        self.assertEqual(set(prediction.evidence["customer"]), {"c1"})
        self.assertEqual(set(prediction.evidence["ticket"]), {"t1", "t2", "t3", "t13"})
        self.assertEqual(prediction.sources, ["graph", "relational"])
        self.assertEqual(prediction.records_transferred, 4)
        self.assertIsNotNone(prediction.latency_ms)

    def test_an_empty_first_step_ends_the_chain_without_a_second_read(self) -> None:
        plan = [
            {"from": {"dataset": "customer"}, "where": {"field": "name", "op": "eq", "value": "Nobody"},
             "traverse": [{"relationship": "customer_has_ticket", "as": "ticket"}]},
            {"from": {"dataset": "ticket"}, "where": {"field": "id", "op": "in", "value": {"$ids": {"step": 0, "field": "ticket.id"}}}},
        ]
        prediction = run_plan(self.db, plan, "id", "ticket")
        self.assertEqual(prediction.answer, [])
        self.assertIsNone(prediction.error)

    def test_an_invalid_plan_is_recorded_as_its_error(self) -> None:
        prediction = run_plan(self.db, [{"from": {"dataset": "customer"}, "where": {"field": "nope", "op": "eq", "value": 1}}], "id", "customer")
        self.assertEqual(prediction.error, "field_not_found")
        self.assertFalse(prediction.plan_valid)

    def test_the_gold_system_refuses_what_has_no_plan(self) -> None:
        prediction = GoldPlanSystem(self.db).answer({"id": "X06-001", "plan": [], "answer_field": "id", "answer_dataset": ""})
        self.assertTrue(prediction.refused)
        self.assertEqual(prediction.error, "refusal")


class ScoreTests(unittest.TestCase):
    def test_set_f1(self) -> None:
        self.assertEqual(f1({"a", "b"}, {"b", "c"}), (0.5, 0.5, 0.5))
        self.assertEqual(f1(set(), set()), (1.0, 1.0, 1.0))
        self.assertEqual(f1({"a"}, set()), (0.0, 0.0, 0.0))

    def test_question_scores_for_an_answer_and_for_a_stress_case(self) -> None:
        question = {"id": "q", "gold_answer": ["a", "b"], "sources": ["graph", "relational"], "gold_evidence": {"user": ["u1"], "business": ["a", "b"]}, "expect_error": None}
        good = {"answer": ["a", "b"], "sources": ["graph", "relational"], "evidence": {"user": ["u1"], "business": ["a", "b"]}, "plan_valid": True}
        self.assertEqual(question_scores(question, good)["answer_em"], 1.0)
        self.assertEqual(question_scores(question, good)["provenance_f1"], 1.0)
        partial = {"answer": ["a", "c"], "sources": ["relational"], "evidence": {"business": ["a"]}, "plan_valid": True}
        scores = question_scores(question, partial)
        self.assertEqual(scores["answer_em"], 0.0)
        self.assertAlmostEqual(scores["answer_f1"], 0.5)
        self.assertAlmostEqual(scores["source_f1"], 2 / 3)
        self.assertAlmostEqual(scores["provenance_recall"], 1 / 3)
        stress = {"id": "x", "expect_error": "field_not_found", "gold_answer": []}
        self.assertEqual(question_scores(stress, {"error": "field_not_found"})["answer_f1"], 1.0)
        self.assertEqual(question_scores(stress, {"answer": ["a"], "plan_valid": True})["answer_f1"], 0.0)
        self.assertEqual(question_scores({"id": "y", "expect_error": "refusal"}, {"refused": True})["plan_valid"], 1.0)

    def test_percentiles_and_bootstrap(self) -> None:
        self.assertEqual(percentile([10, 20, 30, 40], 0.5), 25)
        self.assertEqual(percentile([5], 0.95), 5)
        low, high = bootstrap([1.0, 0.0, 1.0, 1.0], resamples=200)
        self.assertTrue(0.0 <= low <= 0.75 <= high <= 1.0)

    def test_the_report_groups_by_class_and_uses_medians_over_runs(self) -> None:
        questions = [
            {"id": "S01-001", "class": "single", "gold_answer": ["a"], "sources": ["relational"], "gold_evidence": {"business": ["a"]}, "expect_error": None},
            {"id": "X01-001", "class": "stress", "gold_answer": [], "sources": [], "gold_evidence": {}, "expect_error": "field_not_found"},
        ]
        rows = [
            {"id": "S01-001", "answer": ["a"], "sources": ["relational"], "evidence": {"business": ["a"]}, "plan_valid": True, "latency_ms": 10, "run": 0, "tokens_in": 100, "repairs": 1},
            {"id": "S01-001", "answer": ["a"], "sources": ["relational"], "evidence": {"business": ["a"]}, "plan_valid": True, "latency_ms": 30, "run": 1},
            {"id": "X01-001", "error": "field_not_found", "latency_ms": 5, "run": 0},
        ]
        report = score(questions, rows, resamples=50)
        self.assertEqual(report["classes"]["all"]["answer_f1"], 1.0)
        self.assertEqual(report["classes"]["single"]["latency_ms_p50"], 20)
        self.assertEqual(report["classes"]["stress"]["plan_valid"], 1.0)
        self.assertEqual(report["classes"]["single"]["tokens_in_mean"], 100)
        self.assertEqual(report["classes"]["all"]["errors"], ["field_not_found"])


if __name__ == "__main__":
    unittest.main()
