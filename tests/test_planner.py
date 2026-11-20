"""The natural-language planner with a scripted model: prompts, validation, repair, execution, refusals, the SSSI."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.harness import run  # noqa: E402
from planner import NaturalLanguagePlanner, PlannerConfig  # noqa: E402
from planner.grammar import OUTPUT, output_schema_json  # noqa: E402
from planner.llm import ScriptedGenerator  # noqa: E402
from planner.planner import PlanRejected, answer_dataset_of, parse_output  # noqa: E402
from planner.prompt import messages, system_prompt  # noqa: E402
from planner.schema import estimate_tokens, render, render_descriptions, render_full  # noqa: E402
from planner.sssi import SchemaIndex, entries_of, tokens  # noqa: E402

from yodb.compilation import PostgresQueryCompiler, QueryCompilerRegistry  # noqa: E402
from yodb.execution import QueryExecutionAdapterRegistry, QueryExecutionEngine  # noqa: E402
from yodb.planning import FederatedPhysicalPlanner, PostgresPlanningAdapter, SourcePlanningRegistry  # noqa: E402
from yodb.semantic import EmbeddingResult, ProviderInfo  # noqa: E402

from support.catalogs import StaticRuntime  # noqa: E402
from support.graph import GraphWorld, graph_active  # noqa: E402
from support.shop import RULES  # noqa: E402
from support.sql_world import evaluation  # noqa: E402
from support.graph import DATASETS, RELATIONS, SOURCES  # noqa: E402

ACTIVE = evaluation(DATASETS, SOURCES, RELATIONS, RULES)       # the shop, its graph edges and its rules


class DB:
    """What the planner needs of a YoDb client, over the in-memory shop."""

    def __init__(self) -> None:
        planner = FederatedPhysicalPlanner(SourcePlanningRegistry([PostgresPlanningAdapter()]))
        self.engine = QueryExecutionEngine(StaticRuntime(ACTIVE), QueryCompilerRegistry([PostgresQueryCompiler()]), QueryExecutionAdapterRegistry([GraphWorld()]), planner=planner)
        self.catalog = ACTIVE.catalog
        self.limits = self.engine.limits

    def similar_fields(self):
        return {}

    def explain(self, raw):
        return self.engine.explain(raw)

    def query(self, raw, *, timeout_seconds=None):
        return self.engine.execute(raw, timeout_seconds=timeout_seconds)


US_TICKETS = {
    "plan": [{"from": {"dataset": "customer"}, "select": ["name"], "where": {"rule": "in_the_us"},
              "traverse": [{"relationship": "customer_has_ticket", "as": "ticket", "select": ["subject"], "where": {"rule": "urgent"}}],
              "order_by": [{"field": "name", "direction": "asc"}], "page": {"first": 500}}],
    "answer_field": "ticket.id", "sources": ["relational", "graph"],
}
QUESTION = {"id": "D01-001", "question": "Which urgent tickets belong to US customers?", "class": "dependent"}


class PromptTests(unittest.TestCase):
    def test_the_full_schema_is_typed_and_names_relationships_rules_and_edges(self) -> None:
        text = render_full(ACTIVE.catalog)
        self.assertIn("customer (aliases:", text) if ACTIVE.catalog.datasets["customer"].aliases else self.assertIn("customer:", text)
        self.assertIn("priority: int", text)
        self.assertIn("rules (use as", text)
        self.assertIn("over graph edge HAS_TICKET", text)
        self.assertIn("reversible", text)
        self.assertNotIn("legacy_key", text)                                   # internal fields stay hidden

    def test_the_descriptions_representation_has_no_types_keys_or_conditions(self) -> None:
        text = render_descriptions(ACTIVE.catalog)
        self.assertIn("customer_has_ticket:", text)
        self.assertNotIn(": int", text)
        self.assertNotIn("over graph edge", text)
        self.assertNotIn("->", text)
        self.assertNotIn("reversible", text)
        self.assertLess(estimate_tokens(text), estimate_tokens(render_full(ACTIVE.catalog)))

    def test_the_system_prompt_carries_the_query_language_the_schema_and_the_contract(self) -> None:
        prompt = system_prompt(render_full(ACTIVE.catalog), DB().limits)
        for needle in ('"traverse"', "answer_field", '"$ids"', "refuse", "priority: int", "at most 500"):
            self.assertIn(needle, prompt)
        conversation = messages(render_full(ACTIVE.catalog), "Who?", DB().limits)
        self.assertEqual([m["role"] for m in conversation][:3], ["system", "user", "assistant"])
        self.assertEqual(conversation[-1], {"role": "user", "content": "Question: Who?"})

    def test_the_output_schema_is_valid_json_and_covers_both_answers(self) -> None:
        schema = json.loads(output_schema_json())
        self.assertEqual(len(schema["anyOf"]), 2)
        self.assertIn("condition", OUTPUT["$defs"])


class ParseTests(unittest.TestCase):
    def test_json_is_found_inside_fences_and_prose(self) -> None:
        self.assertEqual(parse_output('```json\n{"refuse": {"code": "ambiguous", "reason": "x"}}\n```')["refuse"]["code"], "ambiguous")
        self.assertIn("plan", parse_output('Sure! {"plan": [{"from": {"dataset": "customer"}}], "answer_field": "id", "sources": []}'))
        for bad in ("no json here", "[1, 2]", '{"something": 1}', '{"refuse": {}}'):
            with self.subTest(bad=bad), self.assertRaises(PlanRejected):
                parse_output(bad)

    def test_the_answer_dataset_follows_the_alias(self) -> None:
        db = DB()
        self.assertEqual(answer_dataset_of(db, US_TICKETS["plan"], "ticket.id"), "ticket")
        self.assertEqual(answer_dataset_of(db, US_TICKETS["plan"], "id"), "customer")


class LoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = DB()

    def planner(self, generator, **config):
        return NaturalLanguagePlanner(self.db, generator, PlannerConfig(**config))

    def test_a_valid_plan_runs_and_the_prediction_carries_answers_evidence_and_tokens(self) -> None:
        generator = ScriptedGenerator([US_TICKETS])
        outcome = self.planner(generator).plan(QUESTION["question"], question_id="q1")
        prediction = outcome.prediction
        self.assertIsNone(prediction.error)
        self.assertTrue(prediction.plan_valid)
        self.assertEqual(prediction.repairs, 0)
        self.assertEqual(prediction.sources, ["graph", "relational"])
        self.assertTrue(prediction.answer)                                       # urgent tickets of US customers over the edges
        self.assertTrue(set(prediction.evidence) >= {"customer", "ticket"})
        self.assertGreater(prediction.tokens_in, 0)
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(generator.calls[0][-1]["content"], f"Question: {QUESTION['question']}")
        self.assertGreater(outcome.schema_tokens, 0)

    def test_a_refused_plan_is_repaired_with_yodbs_error_and_counted(self) -> None:
        wrong = json.loads(json.dumps(US_TICKETS).replace('"rule": "in_the_us"', '"field": "county", "op": "eq", "value": "US"'))
        generator = ScriptedGenerator([wrong, US_TICKETS])
        prediction = self.planner(generator).plan(QUESTION["question"]).prediction
        self.assertEqual(prediction.repairs, 1)
        self.assertIsNone(prediction.error)
        repair = generator.calls[1][-1]["content"]
        self.assertIn("[field_not_found]", repair)
        self.assertIn("county", repair)
        self.assertEqual(generator.calls[1][-2]["content"], json.dumps(wrong))   # the model sees its own answer

    def test_after_the_repair_budget_the_last_error_is_recorded_and_nothing_runs(self) -> None:
        wrong = json.loads(json.dumps(US_TICKETS).replace('"rule": "in_the_us"', '"field": "county", "op": "eq", "value": "US"'))
        generator = ScriptedGenerator([wrong, wrong, wrong])
        prediction = self.planner(generator, repairs=2).plan(QUESTION["question"]).prediction
        self.assertEqual(prediction.error, "field_not_found")
        self.assertFalse(prediction.plan_valid)
        self.assertEqual(prediction.repairs, 2)
        self.assertEqual(prediction.answer, [])
        self.assertEqual(len(generator.calls), 3)
        self.assertEqual(prediction.plan, wrong["plan"])

    def test_malformed_output_and_a_bad_answer_field_are_repaired_too(self) -> None:
        bad_field = {**US_TICKETS, "answer_field": "nope.id"}
        generator = ScriptedGenerator(["not json at all", bad_field, US_TICKETS])
        prediction = self.planner(generator).plan(QUESTION["question"]).prediction
        self.assertEqual(prediction.repairs, 2)
        self.assertIn("[output_invalid]", generator.calls[1][-1]["content"])
        self.assertIn("answer_field", generator.calls[2][-1]["content"])
        self.assertIsNone(prediction.error)

    def test_a_two_step_plan_is_validated_with_sample_ids_and_runs(self) -> None:
        plan = {
            "plan": [
                {"from": {"dataset": "customer"}, "where": {"field": "name", "op": "eq", "value": "Ann"},
                 "traverse": [{"relationship": "customer_has_ticket", "as": "ticket"}], "page": {"first": 500}},
                {"from": {"dataset": "ticket"}, "select": ["subject"], "where": {"all": [
                    {"field": "id", "op": "in", "value": {"$ids": {"step": 0, "field": "ticket.id"}}}, {"field": "status", "op": "eq", "value": "open"}]},
                 "page": {"first": 500}},
            ],
            "answer_field": "id", "sources": ["relational", "graph"],
        }
        prediction = self.planner(ScriptedGenerator([plan])).plan("Ann's open tickets").prediction
        self.assertEqual(prediction.answer, ["t1", "t3"])
        self.assertEqual(prediction.records_transferred, 4)

    def test_a_refusal_is_a_refused_prediction_with_its_code(self) -> None:
        refusal = {"refuse": {"code": "unsupported", "reason": "No counting."}}
        prediction = self.planner(ScriptedGenerator([refusal])).plan("How many tickets are there?").prediction
        self.assertTrue(prediction.refused)
        self.assertEqual(prediction.refusal_code, "unsupported")
        self.assertIsNone(prediction.error)
        self.assertEqual(prediction.answer, [])

    def test_the_planner_is_a_benchmark_system(self) -> None:
        import tempfile

        planner = self.planner(ScriptedGenerator([US_TICKETS]))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "p.jsonl"
            predictions = run(planner, [QUESTION], out, quiet=True)
            self.assertEqual(predictions[0].id, "D01-001")
            self.assertTrue(json.loads(out.read_text().splitlines()[0])["answer"])
        self.assertTrue(planner.name.startswith("planner-full-"))

    def test_the_grammar_is_handed_to_the_generator_unless_disabled(self) -> None:
        seen = []

        class Recording(ScriptedGenerator):
            def generate(self, messages, *, json_schema=None):
                seen.append(json_schema)
                return super().generate(messages, json_schema=json_schema)

        self.planner(Recording([US_TICKETS])).plan("q")
        self.planner(Recording([US_TICKETS]), grammar=False).plan("q")
        self.assertIs(seen[0], OUTPUT)
        self.assertIsNone(seen[1])


class HashEmbedder:
    """A deterministic 'embedding': word-hash buckets, so related texts share dimensions."""

    info = ProviderInfo("fake", "hash", "1")
    dimensions = 64

    def embed(self, request):
        vectors = []
        for text in request.texts:
            vector = [0.0] * 64
            for word in tokens(text):
                vector[hash(word) % 64] += 1.0
            vectors.append(tuple(vector))
        return EmbeddingResult(tuple(vectors), self.info, 64)


class SSSITests(unittest.TestCase):
    def test_entries_cover_datasets_fields_rules_and_relationships(self) -> None:
        kinds = {}
        for entry in entries_of(ACTIVE.catalog):
            kinds[entry.kind] = kinds.get(entry.kind, 0) + 1
        self.assertEqual(set(kinds), {"dataset", "field", "rule", "relationship"})
        self.assertNotIn("ticket.legacy_key", [e.key for e in entries_of(ACTIVE.catalog)])

    def test_retrieval_keeps_the_relevant_fields_and_closes_over_relationships(self) -> None:
        index = SchemaIndex(ACTIVE.catalog, HashEmbedder())
        fragment = index.retrieve("Which urgent tickets belong to customers in the US?", top_k=6)
        self.assertIn("ticket", fragment.datasets)
        self.assertIn("customer", fragment.datasets)
        self.assertTrue({"customer_has_ticket", "customer_ticket_both_ways"} & set(fragment.relationships))
        text = render("retrieved", ACTIVE.catalog, question="urgent tickets of US customers", index=index, top_k=6)
        self.assertIn("ticket", text)
        self.assertLessEqual(estimate_tokens(text), estimate_tokens(render_full(ACTIVE.catalog)))

    def test_lexical_only_retrieval_works_without_an_embedder(self) -> None:
        index = SchemaIndex(ACTIVE.catalog)
        fragment = index.retrieve("customers with many seats", top_k=3)
        self.assertEqual(fragment.datasets[0], "customer")
        self.assertIn("seats", fragment.fields.get("customer", []))

    def test_tokens_split_snake_case_and_stem(self) -> None:
        self.assertEqual(tokens("Customers' review_count and categories"), ["customer", "review", "count", "category"])


if __name__ == "__main__":
    unittest.main()


class EndpointTests(unittest.TestCase):
    """The OpenAI-compatible generator finds a request mode the server accepts and keeps it."""

    def test_the_schema_mode_falls_back_until_the_server_accepts_one(self) -> None:
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        from planner.llm import OpenAICompatibleGenerator

        seen = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append("guided_json" if "guided_json" in body else (body.get("response_format") or {}).get("type", "none"))
                assert body["messages"][-1]["content"].endswith("/no_think")
                if (body.get("response_format") or {}).get("type") == "json_schema":
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"{}")
                    return
                payload = json.dumps({"choices": [{"message": {"content": '{"ok": true}'}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            generator = OpenAICompatibleGenerator("m", base_url=f"http://127.0.0.1:{server.server_port}/v1")
            first = generator.generate([{"role": "user", "content": "q"}], json_schema={"type": "object"})
            generator.generate([{"role": "user", "content": "q"}], json_schema={"type": "object"})
            self.assertEqual((first.text, first.tokens_in, first.tokens_out), ('{"ok": true}', 7, 3))
            self.assertEqual(seen, ["json_schema", "guided_json", "guided_json"])
        finally:
            server.shutdown()

    def test_thinking_blocks_are_removed_and_the_soft_switch_is_added(self) -> None:
        from planner.llm import strip_thinking

        self.assertEqual(strip_thinking('<think>\nlet me see {a: 1}\n</think>\n{"plan": []}'), '{"plan": []}')
        self.assertEqual(strip_thinking('{"ok": 1}'), '{"ok": 1}')
        self.assertEqual(strip_thinking('{"ok": 1}<think>unfinished'), '{"ok": 1}')
