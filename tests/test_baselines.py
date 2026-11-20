"""The baselines with a scripted model and in-memory stores: the tool agent (B1) and the router (B2)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines.tool_agent import TOOL_CALL_SCHEMA, ToolAgent, parse_tool_call  # noqa: E402
from baselines.tools import ToolResult, render_rows  # noqa: E402
from planner.llm import ScriptedGenerator  # noqa: E402
from planner.planner import PlanRejected  # noqa: E402

BUSINESSES = [{"business_id": "b1", "name": "Pat's", "city": "Philadelphia", "stars": 4.5}, {"business_id": "b2", "name": "Geno's", "city": "Philadelphia", "stars": 4.0}]
REVIEWS = [{"review_id": "r1", "business_id": "b1", "stars": 5}, {"review_id": "r2", "business_id": "b2", "stars": 2}]
EDGES = {"REVIEWED": [("u1", "b1"), ("u2", "b1")]}


class FakeStores:
    """Answers a few SQL shapes, a vector search and a Cypher pattern; anything else is a store error."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def schema_text(self) -> str:
        return "POSTGRESQL TABLES: yelp.business(business_id text, name text, city text, stars real)\nGRAPH yelp_graph"

    def sql(self, query):
        self.calls.append(("sql", query))
        if "stars >= 4.5" in query:
            rows = [b for b in BUSINESSES if b["stars"] >= 4.5]
            return ToolResult(True, render_rows(rows), rows, latency_ms=3.0)
        if query.lower().startswith("select") and "business" in query:
            return ToolResult(True, render_rows(BUSINESSES), list(BUSINESSES), latency_ms=3.0)
        return ToolResult(False, 'error: relation "yelp.nope" does not exist', [], native_error=True, latency_ms=1.0)

    def vector_search(self, text, top=10, where=None):
        self.calls.append(("vector_search", text))
        rows = [{"id": "r1", "distance": 0.1}, {"id": "r2", "distance": 0.3}][:top]
        return ToolResult(True, render_rows(rows), rows, latency_ms=5.0)

    def cypher(self, query, columns=1):
        self.calls.append(("cypher", query))
        if "REVIEWED" in query:
            rows = [{"c0": a, "c1": b} for a, b in EDGES["REVIEWED"]]
            return ToolResult(True, render_rows(rows), rows, latency_ms=4.0)
        return ToolResult(False, "error: syntax error at or near 'MACTH'", [], native_error=True, latency_ms=1.0)


def call(tool, **arguments):
    return {"tool": tool, "arguments": arguments}


class ParseTests(unittest.TestCase):
    def test_tool_calls_are_parsed_from_text_and_rejected_when_malformed(self) -> None:
        self.assertEqual(parse_tool_call('```json\n{"tool": "sql", "arguments": {"query": "SELECT 1"}}\n```')["tool"], "sql")
        self.assertEqual(parse_tool_call('I will run {"tool": "final", "arguments": {"answer_ids": []}} now')["tool"], "final")
        for bad in ("hello", '{"tool": "ssh", "arguments": {}}', '{"tool": "sql", "arguments": 3}'):
            with self.subTest(bad=bad), self.assertRaises(PlanRejected):
                parse_tool_call(bad)
        self.assertEqual(TOOL_CALL_SCHEMA["properties"]["tool"]["enum"][-1], "refuse")


class ToolAgentTests(unittest.TestCase):
    def agent(self, answers, **options):
        self.stores = FakeStores()
        self.generator = ScriptedGenerator(answers)
        return ToolAgent(self.stores, self.generator, **options)

    def test_a_single_sql_call_then_a_final_answer(self) -> None:
        agent = self.agent([
            call("sql", query="SELECT business_id FROM yelp.business WHERE city = 'Philadelphia' AND stars >= 4.5 ORDER BY 1 LIMIT 500"),
            call("final", answer_dataset="business", answer_ids=["b1"], evidence={}),
        ])
        prediction = agent.answer({"id": "S01-001", "question": "Which businesses in Philadelphia are rated at least 4.5 stars?"})
        self.assertEqual(prediction.answer, ["b1"])
        self.assertEqual(prediction.evidence, {"business": ["b1"]})
        self.assertEqual(prediction.sources, ["relational"])
        self.assertEqual((prediction.native_queries, prediction.native_errors, prediction.tool_calls), (1, 0, 2))
        self.assertTrue(prediction.plan_valid)
        self.assertIsNone(prediction.error)
        self.assertGreater(prediction.tokens_in, 0)
        self.assertEqual(prediction.retrieval_ms, 3.0)
        self.assertEqual(len(prediction.plan), 1)
        self.assertIn("TOOL RESULT #1 (sql):", self.generator.calls[1][-1]["content"])
        self.assertIn('"business_id": "b1"', self.generator.calls[1][-1]["content"])

    def test_a_rejected_native_query_is_counted_and_shown_to_the_model_which_may_fix_it(self) -> None:
        agent = self.agent([
            call("sql", query="SELECT * FROM yelp.nope"),
            call("sql", query="SELECT business_id FROM yelp.business WHERE stars >= 4.5"),
            call("final", answer_dataset="business", answer_ids=["b1"]),
        ])
        prediction = agent.answer({"id": "q", "question": "q"})
        self.assertEqual((prediction.native_queries, prediction.native_errors), (2, 1))
        self.assertIn("error: relation", self.generator.calls[1][-1]["content"])
        self.assertEqual(prediction.answer, ["b1"])

    def test_the_three_stores_feed_the_source_kinds_and_the_evidence(self) -> None:
        agent = self.agent([
            call("vector_search", text="great cheesesteak", top=2),
            call("cypher", query="MATCH (u:User)-[:REVIEWED]->(b:Business) WHERE b.id = 'b1' RETURN u.id, b.id", columns=2),
            call("final", answer_dataset="user", answer_ids=["u1", "u2"], evidence={"review": ["r1"], "business": ["b1"]}),
        ])
        prediction = agent.answer({"id": "q", "question": "q"})
        self.assertEqual(prediction.sources, ["graph", "relational", "vector"])
        self.assertEqual(prediction.evidence, {"review": ["r1"], "business": ["b1"], "user": ["u1", "u2"]})
        self.assertEqual(prediction.retrieval_ms, 9.0)

    def test_a_refusal(self) -> None:
        prediction = self.agent([call("refuse", code="not_a_read", reason="I cannot delete.")]).answer({"id": "x", "question": "Delete everything"})
        self.assertTrue(prediction.refused)
        self.assertEqual(prediction.refusal_code, "not_a_read")
        self.assertEqual(prediction.native_queries, 0)

    def test_the_agent_gives_up_after_its_turn_budget(self) -> None:
        agent = self.agent([call("cypher", query="MACTH (n) RETURN n")] * 4, max_turns=3)
        prediction = agent.answer({"id": "x", "question": "q"})
        self.assertEqual(prediction.error, "agent_gave_up")
        self.assertEqual(prediction.native_errors, 3)
        self.assertFalse(prediction.plan_valid)
        self.assertEqual(prediction.answer, [])

    def test_malformed_replies_cost_a_turn_and_are_explained(self) -> None:
        agent = self.agent(["not a tool call", call("final", answer_dataset="business", answer_ids=["b2"])])
        prediction = agent.answer({"id": "x", "question": "q"})
        self.assertEqual(prediction.repairs, 1)
        self.assertEqual(prediction.answer, ["b2"])
        self.assertIn("not a JSON tool call", self.generator.calls[1][-1]["content"])

    def test_the_final_answer_may_refer_to_a_result_column_instead_of_copying_ids(self) -> None:
        agent = self.agent([
            call("sql", query="SELECT business_id, name FROM yelp.business WHERE stars >= 4.5 ORDER BY name LIMIT 500"),
            call("cypher", query="MATCH (u:User)-[:REVIEWED]->(b:Business) RETURN u.id, b.id", columns=2),
            call("final", answer_dataset="business", answer_from={"result": 1, "column": "business_id"}, evidence={"user": {"result": 2, "column": "c0"}}),
        ])
        prediction = agent.answer({"id": "q", "question": "q"})
        self.assertEqual(prediction.answer, [b["business_id"] for b in BUSINESSES if b["stars"] >= 4.5])
        self.assertEqual(prediction.evidence["user"], list(dict.fromkeys(a for a, _ in EDGES["REVIEWED"])))
        self.assertEqual(prediction.evidence["business"], prediction.answer)
        self.assertIn("TOOL RESULT #2 (cypher):", self.generator.calls[2][-1]["content"])

    def test_a_reference_that_does_not_resolve_is_explained_and_corrected(self) -> None:
        agent = self.agent([
            call("sql", query="SELECT business_id FROM yelp.business WHERE stars >= 4.5"),
            call("final", answer_dataset="business", answer_from={"result": 1, "column": "nope"}),
            call("final", answer_dataset="business", answer_from={"result": 3, "column": "business_id"}),
            call("final", answer_dataset="business", answer_from={"result": 1, "column": "business_id"}),
        ])
        prediction = agent.answer({"id": "q", "question": "q"})
        self.assertEqual(prediction.repairs, 2)
        self.assertIn("column 'nope' of result #1, whose columns are", self.generator.calls[2][-1]["content"])
        self.assertIn("numbered 1 to 1", self.generator.calls[3][-1]["content"])
        self.assertEqual(prediction.answer, [b["business_id"] for b in BUSINESSES if b["stars"] >= 4.5])
        self.assertTrue(prediction.plan_valid)

    def test_a_reference_to_a_name_column_is_refused_as_the_answer_is_ids(self) -> None:
        agent = self.agent([
            call("sql", query="SELECT business_id FROM yelp.business WHERE stars >= 4.5"),
            call("final", answer_dataset="business", answer_from={"result": 1, "column": "name"}),
            call("final", answer_dataset="business", answer_from={"result": 1, "column": "business_id"}),
        ])
        prediction = agent.answer({"id": "q", "question": "Give their names"})
        self.assertEqual(prediction.repairs, 1)
        self.assertIn("not an id column", self.generator.calls[2][-1]["content"])
        self.assertEqual(prediction.answer, ["b1"])

    def test_the_system_prompt_carries_the_raw_schemas_and_the_tools(self) -> None:
        agent = self.agent([call("refuse", code="other", reason="x")])
        agent.answer({"id": "x", "question": "q"})
        system = self.generator.calls[0][0]["content"]
        for needle in ("POSTGRESQL TABLES", "vector_search", "cypher", "8 tool calls before you must answer", "GRAPH yelp_graph"):
            self.assertIn(needle, system)


class RenderTests(unittest.TestCase):
    def test_rows_render_as_json_lines_with_long_values_cut(self) -> None:
        text = render_rows([{"id": "a", "text": "x" * 400, "d": 0.123456789}])
        self.assertIn('"id": "a"', text)
        self.assertIn("...", text)
        self.assertIn("0.1235", text)
        self.assertEqual(render_rows([]), "(no rows)")
        self.assertNotIn("more rows", render_rows([{"i": i} for i in range(500)]))     # a baseline sees every row it is allowed
        self.assertIn("more rows", render_rows([{"i": i} for i in range(70)], limit=60))


if __name__ == "__main__":
    unittest.main()


class RouterTests(unittest.TestCase):
    """B2 in plain Python: decompose and route, one native query per engine, aggregate."""

    def router(self, answers, **options):
        from baselines.router import RouterSystem

        self.stores = FakeStores()
        self.generator = ScriptedGenerator(answers)
        return RouterSystem(self.stores, self.generator, **options)

    def test_one_sub_question_routed_to_sql_then_aggregated(self) -> None:
        router = self.router([
            {"sub_questions": [{"question": "businesses in Philadelphia rated >= 4.5", "tool": "sql"}]},
            {"query": "SELECT business_id FROM yelp.business WHERE city = 'Philadelphia' AND stars >= 4.5 ORDER BY 1 LIMIT 500"},
            {"answer_dataset": "business", "answer_ids": ["b1"], "evidence": {}},
        ])
        prediction = router.answer({"id": "S01-001", "question": "Which businesses in Philadelphia are rated at least 4.5?"})
        self.assertEqual(prediction.answer, ["b1"])
        self.assertEqual(prediction.sources, ["relational"])
        self.assertEqual((prediction.native_queries, prediction.native_errors, prediction.tool_calls), (1, 0, 1))
        self.assertTrue(prediction.plan_valid)
        self.assertEqual(prediction.plan[0]["tool"], "sql")
        self.assertIn("Native query", self.generator.calls[2][-1]["content"])       # the aggregator sees the sub-answers
        self.assertEqual(prediction.evidence, {"business": ["b1"]})

    def test_two_sub_questions_with_the_earlier_answer_as_context_for_the_later(self) -> None:
        router = self.router([
            {"sub_questions": [{"question": "reviews about cheesesteak", "tool": "vector"}, {"question": "who reviewed b1", "tool": "graph"}]},
            {"text": "cheesesteak", "top": 2, "where": None},
            {"query": "MATCH (u:User)-[:REVIEWED]->(b:Business) WHERE b.id = 'b1' RETURN u.id, b.id", "columns": 2},
            {"answer_dataset": "user", "answer_ids": ["u1", "u2"], "evidence": {"review": ["r1"]}},
        ])
        prediction = router.answer({"id": "q", "question": "q"})
        self.assertEqual(prediction.answer, ["u1", "u2"])
        self.assertEqual(prediction.sources, ["graph", "relational", "vector"])
        self.assertIn("Earlier sub-answer (vector", self.generator.calls[2][-1]["content"])
        self.assertEqual(prediction.retrieval_ms, 9.0)

    def test_the_aggregator_may_answer_by_reference_to_a_numbered_sub_answer(self) -> None:
        router = self.router([
            {"sub_questions": [{"question": "businesses rated >= 4.5", "tool": "sql"}]},
            {"query": "SELECT business_id, name FROM yelp.business WHERE stars >= 4.5 ORDER BY 1 LIMIT 500"},
            {"answer_dataset": "business", "answer_from": {"sub_answer": 2, "column": "business_id"}},
            {"answer_dataset": "business", "answer_from": {"sub_answer": 1, "column": "business_id"}},
        ])
        prediction = router.answer({"id": "q", "question": "q"})
        self.assertEqual(prediction.answer, [b["business_id"] for b in BUSINESSES if b["stars"] >= 4.5])
        self.assertIn("Sub-answer #1 (sql)", self.generator.calls[2][-1]["content"])
        self.assertIn("numbered 1 to 1", self.generator.calls[3][-1]["content"])
        self.assertEqual(prediction.evidence, {"business": prediction.answer})

    def test_a_rejected_native_query_is_counted_and_the_plan_is_not_valid(self) -> None:
        router = self.router([
            {"sub_questions": [{"question": "x", "tool": "graph"}]},
            {"query": "MACTH (n) RETURN n", "columns": 1},
            {"answer_dataset": "user", "answer_ids": []},
        ])
        prediction = router.answer({"id": "q", "question": "q"})
        self.assertEqual((prediction.native_queries, prediction.native_errors), (1, 1))
        self.assertFalse(prediction.plan_valid)
        self.assertEqual(prediction.answer, [])

    def test_a_refusal_and_malformed_routing(self) -> None:
        self.assertTrue(self.router([{"refuse": {"code": "not_a_read", "reason": "no"}}]).answer({"id": "x", "question": "Delete"}).refused)
        bad = self.router(["???"]).answer({"id": "x", "question": "q"})
        self.assertEqual(bad.error, "router_output_invalid")
        unrouted = self.router([{"sub_questions": [{"question": "q", "tool": "teleport"}]}]).answer({"id": "x", "question": "q"})
        self.assertEqual(unrouted.error, "router_output_invalid")

    def test_the_sqlalchemy_url_is_derived_from_a_conninfo(self) -> None:
        from baselines.router import _sqlalchemy_url

        self.assertEqual(_sqlalchemy_url("host=h port=5 dbname=d user=u password=p"), "postgresql+psycopg://u:p@h:5/d")
        self.assertEqual(_sqlalchemy_url("postgresql://u:p@h/d"), "postgresql+psycopg://u:p@h/d")
