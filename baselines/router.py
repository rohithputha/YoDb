"""B2: framework routing — one engine per source, a router, sub-question decomposition and aggregation.

This is the shape of LlamaIndex's ``RouterQueryEngine`` over three query engines (an NL-to-SQL
engine on the tables, a vector-store engine on the review embeddings, an NL-to-Cypher engine on
the graph) with a ``SubQuestionQueryEngine`` in front: the LLM decomposes the question into
sub-questions, each is routed to one engine, each engine turns its sub-question into one native
query, and the LLM aggregates the sub-answers.  No typed plan, no validation before execution, no
dependency between sub-questions other than the text of earlier answers.

Two implementations share the same prompts and scoring:

* :class:`RouterSystem` — the design in plain Python over :class:`baselines.tools.RawStores`, so the
  baseline runs with the same generator as every other system and needs no extra package;
* :class:`LlamaIndexRouterSystem` — the same pipeline built from LlamaIndex's own classes when the
  ``llama-index`` packages are installed (``pip install llama-index llama-index-llms-openai-like``),
  for the comparison the proposal names.  Its answer IDs are read from the engines' source rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import re
import time
from typing import Any

from benchmark.harness import Prediction
from planner.llm import Generator
from planner.planner import PlanRejected, parse_output

ENGINES = ("sql", "vector", "graph")
ENGINE_KINDS = {"sql": "relational", "vector": "vector", "graph": "graph"}

DECOMPOSE = """\
You split a question about data into the sub-questions that different tools must answer, and route each to a tool.
Tools:
  sql     - the PostgreSQL tables (structured facts: businesses, reviews, users, categories, friendships, business_category)
  vector  - a similarity search over review text (finds the reviews nearest to a phrase)
  graph   - the Apache AGE graph (friendships, who reviewed what, business categories, as edges between ids)
Reply with exactly one JSON object:
  {"sub_questions": [{"question": "...", "tool": "sql" | "vector" | "graph"}, ...]}
or, if the question cannot be answered with these tools (a count, a write, missing data, too ambiguous):
  {"refuse": {"code": "unsupported" | "ambiguous" | "not_a_read" | "no_such_data" | "other", "reason": "..."}}
Use as few sub-questions as possible (usually one); a later sub-question may refer to the answer of an earlier one.

SCHEMAS
{schema}
"""

ENGINE_PROMPT = {
    "sql": """\
Write one PostgreSQL SELECT statement that answers the sub-question. Return ids (the *_id columns) and the columns
asked for; ORDER BY a column and LIMIT 500. Reply with exactly one JSON object: {"query": "<SQL>"}.
Earlier sub-answers, if any, are given as context; use their ids literally in your query.

SCHEMAS
{schema}
""",
    "vector": """\
Turn the sub-question into a similarity search over review text. Reply with exactly one JSON object:
{"text": "<the phrase to search for>", "top": <how many reviews>, "where": "<optional SQL condition on yelp.review, or null>"}
Earlier sub-answers, if any, are given as context; a condition may use their ids literally.

SCHEMAS
{schema}
""",
    "graph": """\
Write one read-only Cypher query (MATCH ... RETURN ...) for the Apache AGE graph that answers the sub-question. Vertices
hold only the property id (text); return ids. Reply with exactly one JSON object: {"query": "<Cypher>", "columns": <number of RETURN items>}.
Earlier sub-answers, if any, are given as context; use their ids literally in your query.

SCHEMAS
{schema}
""",
}

AGGREGATE = """\
You are given a question and the sub-answers tools produced for it (rows with ids). Decide the final answer: the set of ids
of one dataset that answers the question, using only ids that appear in the sub-answers. Reply with exactly one JSON object:
{"answer_dataset": "<business|review|user|category>", "answer_ids": ["..."], "evidence": {"<dataset>": ["..."]}}
"""

_SCHEMA_DECOMPOSE: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sub_questions": {"type": "array", "items": {"type": "object", "properties": {"question": {"type": "string"}, "tool": {"type": "string", "enum": list(ENGINES)}}, "required": ["question", "tool"], "additionalProperties": False}, "minItems": 1, "maxItems": 4},
        "refuse": {"type": "object", "properties": {"code": {"type": "string"}, "reason": {"type": "string"}}, "required": ["code", "reason"]},
    },
}
_SCHEMA_FINAL: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer_dataset": {"type": "string"},
        "answer_ids": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "object"},
    },
    "required": ["answer_dataset", "answer_ids"],
}
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class SubAnswer:
    question: str
    tool: str
    native: dict[str, Any]
    text: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    native_error: bool = False
    latency_ms: float = 0.0


class RouterSystem:
    """The router/sub-question design over the raw stores, with any generator."""

    def __init__(self, stores, generator: Generator, *, grammar: bool = True, schema_text: str | None = None, max_sub_questions: int = 4) -> None:
        self.stores = stores
        self.generator = generator
        self.grammar = grammar
        self.schema_text = schema_text if schema_text is not None else stores.schema_text()
        self.max_sub_questions = max_sub_questions
        self.name = f"router-{getattr(generator, 'name', 'llm')}"

    def answer(self, question: Mapping[str, Any]) -> Prediction:
        return self.run(question["question"], question_id=question.get("id", ""))

    def run(self, question: str, *, question_id: str = "") -> Prediction:
        started = time.perf_counter()
        prediction = Prediction(id=question_id)
        # 1. decompose and route
        try:
            plan = self._json(
                [{"role": "system", "content": DECOMPOSE.replace("{schema}", self.schema_text)}, {"role": "user", "content": f"Question: {question}"}],
                prediction, _SCHEMA_DECOMPOSE,
            )
        except PlanRejected as rejected:
            prediction.error = "router_output_invalid"
            prediction.plan_valid = False
            prediction.latency_ms = (time.perf_counter() - started) * 1000
            return prediction
        if "refuse" in plan:
            prediction.refused = True
            prediction.refusal_code = str((plan.get("refuse") or {}).get("code") or "other")
            prediction.latency_ms = (time.perf_counter() - started) * 1000
            return prediction
        sub_questions = [s for s in plan.get("sub_questions", []) if isinstance(s, dict) and s.get("tool") in ENGINES][: self.max_sub_questions]
        if not sub_questions:
            prediction.error = "router_output_invalid"
            prediction.plan_valid = False
            prediction.latency_ms = (time.perf_counter() - started) * 1000
            return prediction
        # 2. each sub-question through its engine
        answers: list[SubAnswer] = []
        for sub in sub_questions:
            answers.append(self._engine(sub["tool"], str(sub["question"]), answers, prediction))
        prediction.plan = [{"tool": a.tool, "question": a.question, **a.native} for a in answers]
        prediction.sources = sorted({ENGINE_KINDS[a.tool] for a in answers if not a.native_error} | ({"relational"} if any(not a.native_error for a in answers) else set()))
        prediction.retrieval_ms = sum(a.latency_ms for a in answers)
        # 3. aggregate
        context = "\n\n".join(f"Sub-question ({a.tool}): {a.question}\nNative query: {json.dumps(a.native)}\nResult:\n{a.text}" for a in answers)
        try:
            final = self._json(
                [{"role": "system", "content": AGGREGATE}, {"role": "user", "content": f"Question: {question}\n\n{context}"}],
                prediction, _SCHEMA_FINAL,
            )
        except PlanRejected:
            prediction.error = "router_output_invalid"
            prediction.plan_valid = False
            prediction.latency_ms = (time.perf_counter() - started) * 1000
            return prediction
        ids = final.get("answer_ids") or []
        prediction.answer = list(dict.fromkeys(str(i) for i in ids if i is not None)) if isinstance(ids, list) else []
        evidence = final.get("evidence") or {}
        prediction.evidence = {str(k): [str(i) for i in v] for k, v in evidence.items() if isinstance(v, list)} if isinstance(evidence, dict) else {}
        dataset = str(final.get("answer_dataset") or "")
        if dataset:
            prediction.evidence[dataset] = list(dict.fromkeys([*prediction.evidence.get(dataset, []), *prediction.answer]))
        prediction.plan_valid = all(not a.native_error for a in answers)
        prediction.latency_ms = (time.perf_counter() - started) * 1000
        return prediction

    # --- the engines ----------------------------------------------------------------------------------

    def _engine(self, tool: str, sub_question: str, earlier: Sequence[SubAnswer], prediction: Prediction) -> SubAnswer:
        context = "\n\n".join(f"Earlier sub-answer ({a.tool}: {a.question}):\n{a.text}" for a in earlier)
        messages = [
            {"role": "system", "content": ENGINE_PROMPT[tool].replace("{schema}", self.schema_text)},
            {"role": "user", "content": f"Sub-question: {sub_question}" + (f"\n\n{context}" if context else "")},
        ]
        try:
            native = self._json(messages, prediction, None)
        except PlanRejected as rejected:
            prediction.native_queries += 1
            prediction.native_errors += 1
            return SubAnswer(sub_question, tool, {}, f"error: {rejected.message}", native_error=True)
        prediction.native_queries += 1
        prediction.tool_calls += 1
        if tool == "sql":
            result = self.stores.sql(str(native.get("query", "")))
        elif tool == "vector":
            result = self.stores.vector_search(str(native.get("text", "")), int(native.get("top") or 10), native.get("where") or None)
        else:
            result = self.stores.cypher(str(native.get("query", "")), int(native.get("columns") or 1))
        prediction.native_errors += int(result.native_error)
        return SubAnswer(sub_question, tool, native, result.text, result.rows, result.native_error, result.latency_ms)

    def _json(self, messages: list[dict[str, str]], prediction: Prediction, schema: dict[str, Any] | None) -> dict[str, Any]:
        generation = self.generator.generate(messages, json_schema=schema if self.grammar else None)
        prediction.tokens_in += generation.tokens_in
        prediction.tokens_out += generation.tokens_out
        return parse_json_object(generation.text)


def parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = candidate[4:] if candidate.startswith("json") else candidate
    try:
        value = json.loads(candidate)
    except ValueError:
        match = _JSON_OBJECT.search(text)
        if match is None:
            raise PlanRejected("the reply was not a JSON object") from None
        try:
            value = json.loads(match.group(0))
        except ValueError:
            raise PlanRejected("the reply was not valid JSON") from None
    if not isinstance(value, dict):
        raise PlanRejected("the reply must be a JSON object")
    return value


# --- the LlamaIndex build of the same pipeline -------------------------------------------------------------


class LlamaIndexRouterSystem:
    """``RouterQueryEngine`` + ``SubQuestionQueryEngine`` over three LlamaIndex engines, when the packages are installed.

    The engines: ``NLSQLTableQueryEngine`` over the ``yelp`` tables, a ``VectorStoreIndex`` over a ``PGVectorStore``
    bound to ``yelp.review.text_embedding``, and a ``CustomQueryEngine`` that asks the LLM for Cypher and runs it through
    the raw stores.  Answer IDs are taken from the engines' source rows (SQL rows, vector node ids, Cypher rows); the
    final text is kept for inspection.  Import errors are reported once, with the install hint.
    """

    def __init__(self, stores, *, dsn: str, model: str, endpoint: str, embedder, schema: str = "yelp") -> None:
        try:
            from llama_index.core import SQLDatabase, Settings, VectorStoreIndex
            from llama_index.core.query_engine import CustomQueryEngine, RouterQueryEngine, SubQuestionQueryEngine
            from llama_index.core.indices.struct_store import NLSQLTableQueryEngine
            from llama_index.core.selectors import LLMSingleSelector
            from llama_index.core.tools import QueryEngineTool
            from llama_index.llms.openai_like import OpenAILike
        except ImportError as error:
            raise RuntimeError('B2 with LlamaIndex needs: pip install llama-index llama-index-llms-openai-like llama-index-vector-stores-postgres') from error
        from sqlalchemy import create_engine

        self.stores = stores
        self.name = f"llamaindex-router-{model}"
        Settings.llm = OpenAILike(model=model, api_base=endpoint, api_key="none", is_chat_model=True, temperature=0)
        Settings.embed_model = _LlamaEmbedding(embedder)
        sql_database = SQLDatabase(create_engine(_sqlalchemy_url(dsn)), schema=schema, include_tables=["business", "review", "users", "category", "business_category", "friendship"])
        sql_engine = NLSQLTableQueryEngine(sql_database=sql_database, synthesize_response=False)
        vector_engine = self._vector_engine(dsn, schema, embedder, VectorStoreIndex)
        graph_engine = _CypherEngine(stores, CustomQueryEngine)
        tools = [
            QueryEngineTool.from_defaults(sql_engine, name="sql", description="structured facts about businesses, reviews, users, categories, friendships (PostgreSQL)"),
            QueryEngineTool.from_defaults(vector_engine, name="vector", description="reviews whose text is similar to a phrase (vector search)"),
            QueryEngineTool.from_defaults(graph_engine, name="graph", description="friendships, who reviewed what, business categories, as graph edges (Cypher)"),
        ]
        self.router = RouterQueryEngine(selector=LLMSingleSelector.from_defaults(), query_engine_tools=tools)
        self.engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=tools, use_async=False)

    @staticmethod
    def _vector_engine(dsn, schema, embedder, VectorStoreIndex):
        try:
            from llama_index.vector_stores.postgres import PGVectorStore
        except ImportError as error:
            raise RuntimeError("pip install llama-index-vector-stores-postgres") from error
        store = PGVectorStore.from_params(connection_string=_sqlalchemy_url(dsn), schema_name=schema, table_name="review", embed_dim=embedder.dimensions, hybrid_search=False)
        return VectorStoreIndex.from_vector_store(store).as_query_engine(similarity_top_k=10)

    def answer(self, question: Mapping[str, Any]) -> Prediction:
        started = time.perf_counter()
        prediction = Prediction(id=question.get("id", ""))
        try:
            response = self.engine.query(question["question"])
        except Exception as error:
            prediction.error = f"framework_error:{type(error).__name__}"
            prediction.plan_valid = False
            prediction.latency_ms = (time.perf_counter() - started) * 1000
            return prediction
        ids: dict[str, None] = {}
        for node in getattr(response, "source_nodes", []) or []:
            metadata = getattr(node, "metadata", {}) or {}
            for key in ("review_id", "business_id", "user_id", "category_id", "id"):
                if key in metadata:
                    ids[str(metadata[key])] = None
            text = getattr(node, "text", "") or ""
            for match in re.finditer(r"'([A-Za-z0-9_-]{6,})'", text):
                ids[match.group(1)] = None
        prediction.answer = list(ids)
        prediction.plan = [{"response": str(response)[:2000]}]
        prediction.plan_valid = True
        prediction.latency_ms = (time.perf_counter() - started) * 1000
        return prediction


class _LlamaEmbedding:
    """YoDb's embedding provider as a LlamaIndex embedding model (duck-typed, enough for querying)."""

    def __init__(self, embedder) -> None:
        self.embedder = embedder

    def get_query_embedding(self, query: str):
        return list(next(iter(self.embedder.embed_texts([query]))))

    def get_text_embedding(self, text: str):
        return self.get_query_embedding(text)


def _CypherEngine(stores, CustomQueryEngine):
    class CypherEngine(CustomQueryEngine):
        def custom_query(self, query_str: str):
            from llama_index.core import Settings

            prompt = ENGINE_PROMPT["graph"].replace("{schema}", stores.schema_text()) + f"\nSub-question: {query_str}"
            text = str(Settings.llm.complete(prompt))
            try:
                native = parse_json_object(text)
            except PlanRejected:
                return "error: the model did not produce a Cypher query"
            result = stores.cypher(str(native.get("query", "")), int(native.get("columns") or 1))
            return result.text

    return CypherEngine()


def _sqlalchemy_url(dsn: str) -> str:
    """A libpq conninfo (``host=... dbname=...``) or URI as a SQLAlchemy URL."""

    if dsn.startswith("postgres"):
        return dsn.replace("postgres://", "postgresql+psycopg://").replace("postgresql://", "postgresql+psycopg://")
    parts = dict(item.split("=", 1) for item in dsn.split() if "=" in item)
    user, password, host, port, dbname = parts.get("user", ""), parts.get("password", ""), parts.get("host", "localhost"), parts.get("port", "5432"), parts.get("dbname", "")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}"
