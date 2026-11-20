"""B1: the monolithic tool agent.

The same model as the planner, but no typed layer: it sees the raw schemas of all three stores
and answers by calling the three tools itself (SQL, vector search, Cypher), one call per turn,
passing intermediate IDs along by hand, and finally reports the answer IDs.  Everything a store
refuses comes back to it as the store's own error text; each such refusal is one *invalid native
query* (the quantity H1 is about).  There is no validation before a query runs and no grammar
over the native query languages, only over the tool-call envelope.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import re
import time
from typing import Any

from benchmark.harness import Prediction
from planner.llm import Generator
from planner.planner import PlanRejected

MAX_TURNS = 8
LONG_ANSWER_TOKENS = 12_000          # a final answer may list up to 500 ids
TOOL_KINDS = {"sql": "relational", "vector_search": "vector", "cypher": "graph"}

SYSTEM = """\
You answer questions about data stored in three places: PostgreSQL tables, a pgvector column of review-text
embeddings, and an Apache AGE graph. You have three read-only tools and must do everything yourself: choose the
store, write the native query, pass intermediate ids from one call to the next, and reconcile the results.

Reply with exactly one JSON object per turn, nothing else:
  {"tool": "sql", "arguments": {"query": "<one SELECT statement>"}}
  {"tool": "vector_search", "arguments": {"text": "<text to embed>", "top": <k>, "where": "<optional SQL condition on yelp.review>"}}
      returns the review ids nearest to the text with their distance (smaller is nearer)
  {"tool": "cypher", "arguments": {"query": "MATCH ... RETURN ...", "columns": <number of RETURN items>}}
      the query runs inside ag_catalog.cypher(); vertices hold only the property id (text); return ids, not vertices
  {"tool": "final", "arguments": {"answer_dataset": "<business|review|user|category>",
                                  "answer_from": {"result": <n>, "column": "<column>"}  or  "answer_ids": ["..."],
                                  "evidence": {"<dataset>": {"result": <n>, "column": "<column>"} or ["..."]}}}
      the answer is the set of ids of one dataset: primary-key values (business_id, review_id, user_id, category_id),
      never names. Prefer answer_from: it means every value of that column of tool result #n, in its order, and
      costs nothing to write; list answer_ids only to hand-pick ids. evidence names the ids of the other datasets the
      answer rests on, in the same two forms.
  {"tool": "refuse", "arguments": {"code": "unsupported" | "ambiguous" | "not_a_read" | "no_such_data" | "other", "reason": "..."}}

Each tool result comes back as a message starting with TOOL RESULT #n. A result starting with "error:" means the
store rejected your query: fix it and try again. Reads are capped at 500 rows; use ORDER BY and LIMIT 500, and always
select the id column of the entity the question asks about (and of any entity you will cite as evidence). You have at
most {turns} tool calls before you must answer; answer as soon as you can. Never guess ids: every id in your answer
must come from a tool result. If the question cannot be answered from these stores (it asks for a write, or for data
that is not there, or is too ambiguous), refuse.

SCHEMAS
{schema}
"""

TOOL_CALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": ["sql", "vector_search", "cypher", "final", "refuse"]},
        "arguments": {"type": "object"},
    },
    "required": ["tool", "arguments"],
    "additionalProperties": False,
}

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Turn:
    call: dict[str, Any] | None
    result_text: str
    tokens_in: int = 0
    tokens_out: int = 0
    native_error: bool = False
    tool_ms: float = 0.0


@dataclass
class AgentOutcome:
    prediction: Prediction
    turns: list[Turn] = field(default_factory=list)


class ToolAgent:
    """The benchmark system: ``answer(question)`` runs the tool loop and returns a Prediction."""

    def __init__(self, stores, generator: Generator, *, max_turns: int = MAX_TURNS, grammar: bool = True, schema_text: str | None = None) -> None:
        self.stores = stores
        self.generator = generator
        self.max_turns = max_turns
        self.grammar = grammar
        self.schema_text = schema_text if schema_text is not None else stores.schema_text()
        self.name = f"tool-agent-{getattr(generator, 'name', 'llm')}"

    def answer(self, question: Mapping[str, Any]) -> Prediction:
        return self.run(question["question"], question_id=question.get("id", "")).prediction

    def run(self, question: str, *, question_id: str = "") -> AgentOutcome:
        started = time.perf_counter()
        prediction = Prediction(id=question_id)
        outcome = AgentOutcome(prediction)
        conversation = [
            {"role": "system", "content": SYSTEM.replace("{turns}", str(self.max_turns)).replace("{schema}", self.schema_text)},
            {"role": "user", "content": f"Question: {question}"},
        ]
        sources: set[str] = set()
        results: list[list[dict[str, Any]]] = []            # the rows of every tool result, by number (1-based)
        retrieval_ms = 0.0
        finished = False
        for _ in range(self.max_turns + 1):
            generation = self.generator.generate(conversation, json_schema=TOOL_CALL_SCHEMA if self.grammar else None, max_tokens=LONG_ANSWER_TOKENS)
            prediction.tokens_in += generation.tokens_in
            prediction.tokens_out += generation.tokens_out
            conversation.append({"role": "assistant", "content": generation.text})
            try:
                call = parse_tool_call(generation.text)
            except PlanRejected as rejected:
                outcome.turns.append(Turn(None, rejected.message, generation.tokens_in, generation.tokens_out))
                conversation.append({"role": "user", "content": f"TOOL RESULT: {rejected.message} Reply with one JSON tool call."})
                prediction.repairs += 1
                continue
            tool, arguments = call["tool"], call.get("arguments") or {}
            prediction.tool_calls += 1
            if tool == "final":
                problem = self._finish(prediction, arguments, results)
                if problem:
                    outcome.turns.append(Turn(call, problem, generation.tokens_in, generation.tokens_out))
                    conversation.append({"role": "user", "content": f"TOOL RESULT (final): {problem} Reply with one JSON tool call."})
                    prediction.repairs += 1
                    continue
                outcome.turns.append(Turn(call, "final", generation.tokens_in, generation.tokens_out))
                finished = True
                break
            if tool == "refuse":
                prediction.refused = True
                prediction.refusal_code = str(arguments.get("code") or "other")
                outcome.turns.append(Turn(call, "refuse", generation.tokens_in, generation.tokens_out))
                finished = True
                break
            if prediction.native_queries >= self.max_turns:
                outcome.turns.append(Turn(call, "turn budget spent", generation.tokens_in, generation.tokens_out))
                break
            result = self._dispatch(tool, arguments)
            prediction.native_queries += 1
            prediction.native_errors += int(result.native_error)
            retrieval_ms += result.latency_ms
            if result.ok:
                sources.add(TOOL_KINDS[tool])
            results.append(result.rows if result.ok else [])
            outcome.turns.append(Turn(call, result.text, generation.tokens_in, generation.tokens_out, result.native_error, result.latency_ms))
            conversation.append({"role": "user", "content": f"TOOL RESULT #{len(results)} ({tool}):\n{result.text}"})
        if not finished:
            prediction.error = "agent_gave_up"
            prediction.plan_valid = False
        prediction.sources = sorted(sources | ({"relational"} if sources and prediction.error is None else set()))
        prediction.retrieval_ms = retrieval_ms
        prediction.latency_ms = (time.perf_counter() - started) * 1000
        prediction.plan = [t.call for t in outcome.turns if t.call and t.call["tool"] in TOOL_KINDS]
        return outcome

    def _dispatch(self, tool: str, arguments: Mapping[str, Any]):
        try:
            if tool == "sql":
                return self.stores.sql(str(arguments.get("query", "")))
            if tool == "vector_search":
                return self.stores.vector_search(str(arguments.get("text", "")), int(arguments.get("top") or 10), arguments.get("where") or None)
            if tool == "cypher":
                return self.stores.cypher(str(arguments.get("query", "")), int(arguments.get("columns") or 1))
        except Exception as error:                            # a malformed argument is the agent's own mistake
            from baselines.tools import ToolResult

            return ToolResult(False, f"error: {type(error).__name__}: {str(error)[:200]}", [], native_error=True)
        from baselines.tools import ToolResult

        return ToolResult(False, f"error: unknown tool {tool!r}", [], native_error=True)

    @staticmethod
    def _finish(prediction: Prediction, arguments: Mapping[str, Any], results: list[list[dict[str, Any]]]) -> str | None:
        """Fill the prediction from a final call; the problem text when a reference does not resolve."""

        try:
            answer, evidence = resolve_final(arguments, results)
        except BadReference as error:
            return f"error: {error}"
        prediction.answer = answer
        prediction.evidence = evidence
        dataset = str(arguments.get("answer_dataset") or "")
        if dataset:
            prediction.evidence[dataset] = list(dict.fromkeys([*prediction.evidence.get(dataset, []), *prediction.answer]))
        prediction.plan_valid = True
        return None


def resolve_final(arguments: Mapping[str, Any], results: list[list[dict[str, Any]]], *, reference_key: str = "result") -> tuple[list[str], dict[str, list[str]]]:
    """The answer ids and evidence of a final answer whose ids may be given by reference to a result's column.

    ``{"<reference_key>": n, "column": c}`` means every value of column ``c`` of result number ``n`` (1-based),
    in order; a list means those ids.  A reference to a result or column that does not exist raises
    :class:`BadReference`, which the agent is told about so it can correct itself.
    """

    def ids_of(value: Any, what: str) -> list[str]:
        if isinstance(value, Mapping):
            number, column = value.get(reference_key), value.get("column")
            if not isinstance(number, int) or isinstance(number, bool) or not 1 <= number <= len(results):
                raise BadReference(f"{what} refers to {reference_key} #{number!r}, but results are numbered 1 to {len(results)}.")
            rows = results[number - 1]
            if not isinstance(column, str) or (rows and not any(column in row for row in rows)):
                columns = sorted({key for row in rows for key in row})
                raise BadReference(f"{what} refers to column {column!r} of {reference_key} #{number}, whose columns are {columns}.")
            return list(dict.fromkeys(str(row[column]) for row in rows if row.get(column) is not None))
        if isinstance(value, list):
            return list(dict.fromkeys(str(i) for i in value if i is not None))
        return []

    answer = ids_of(arguments.get("answer_from") if arguments.get("answer_from") is not None else arguments.get("answer_ids"), "answer_from")
    evidence_raw = arguments.get("evidence") or {}
    evidence = {str(k): ids_of(v, f"evidence[{k!r}]") for k, v in evidence_raw.items()} if isinstance(evidence_raw, Mapping) else {}
    return answer, {k: v for k, v in evidence.items() if v}


class BadReference(ValueError):
    """A final answer refers to a result or column that does not exist."""


def parse_tool_call(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = candidate[4:] if candidate.startswith("json") else candidate
    try:
        call = json.loads(candidate)
    except ValueError:
        match = _JSON_OBJECT.search(text)
        if match is None:
            raise PlanRejected("error: the reply was not a JSON tool call.") from None
        try:
            call = json.loads(match.group(0))
        except ValueError:
            raise PlanRejected("error: the reply was not valid JSON.") from None
    if not isinstance(call, dict) or call.get("tool") not in ("sql", "vector_search", "cypher", "final", "refuse"):
        raise PlanRejected("error: the reply must be one of the five tool calls.")
    if not isinstance(call.get("arguments", {}), dict):
        raise PlanRejected("error: 'arguments' must be an object.")
    return call
