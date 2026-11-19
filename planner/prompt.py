"""The planner's prompt: the query language, the schema, a few examples, and the output contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

from yodb.mcp_server import query_language

SYSTEM = """\
You translate a question about data into a typed, read-only query plan for YoDb, and nothing else.
You see the catalog below: only the dataset, field, relationship and rule names it lists exist. Never
invent a name; never write SQL. If the question cannot be answered from this catalog with the query
language (it asks for a count, a total, a write, or names data that is not there), or it is too
ambiguous to pick a dataset and a condition, answer with a refusal instead of guessing.

OUTPUT: exactly one JSON object, no prose, no code fence:
  {"plan": [<query>, ...], "answer_field": "<field>", "sources": [<kind>, ...]}
or
  {"refuse": {"code": "unsupported" | "ambiguous" | "not_a_read" | "no_such_data" | "other", "reason": "<one sentence>"}}

- "plan" has one query, or two when the question needs a second relationship: the second query may
  take the IDs reached by the first with {"$ids": {"step": 0, "field": "<alias>.id"}} as the value of an
  "in" condition, e.g. {"field": "id", "op": "in", "value": {"$ids": {"step": 0, "field": "business.id"}}}.
- "answer_field" is the row field whose values answer the question: "id" for the "from" dataset, or
  "<alias>.id" for the traversed dataset of the last query.
- "sources" lists the kinds of source the plan uses: "relational" always; "vector" when a query has a
  "similar" condition; "graph" when a traverse step follows a relationship stored as graph edges.
- Ask for the largest page ({"page": {"first": 500}}) and give a deterministic "order_by" unless the
  question asks for the nearest rows (then no "order_by").
- Prefer a rule from the catalog when it says exactly what the question asks.

QUERY LANGUAGE
{language}

CATALOG
{schema}
"""

REPAIR = """\
YoDb refused that plan:
{error}
Fix exactly what the error names and return the whole JSON object again (or a refusal if the question cannot be answered)."""

EXAMPLES: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "Which open businesses in Tucson are rated at least 4.5 stars? Give their names.",
        {"plan": [{"from": {"dataset": "business"}, "select": ["name", "stars"],
                   "where": {"all": [{"field": "city", "op": "eq", "value": "Tucson"}, {"rule": "open_now"}, {"field": "stars", "op": "gte", "value": 4.5}]},
                   "order_by": [{"field": "name", "direction": "asc"}], "page": {"first": 500}}],
         "answer_field": "id", "sources": ["relational"]},
    ),
    (
        "Which categories are the businesses reviewed by the user with id abc listed under?",
        {"plan": [
            {"from": {"dataset": "user"}, "select": ["name"], "where": {"field": "id", "op": "eq", "value": "abc"},
             "traverse": [{"relationship": "user_reviewed_business", "as": "business", "select": ["name"]}], "page": {"first": 500}},
            {"from": {"dataset": "business"}, "select": ["name"], "where": {"field": "id", "op": "in", "value": {"$ids": {"step": 0, "field": "business.id"}}},
             "traverse": [{"relationship": "business_in_category", "as": "category", "select": ["name"]}],
             "order_by": [{"field": "category.name", "direction": "asc"}], "page": {"first": 500}}],
         "answer_field": "category.id", "sources": ["relational", "graph"]},
    ),
    (
        "How many reviews does Tucson have?",
        {"refuse": {"code": "unsupported", "reason": "Counting is not available in the query language; I can list the reviews instead."}},
    ),
)


def system_prompt(schema_text: str, limits: Mapping[str, int], *, rules: bool = True, similar: bool = True) -> str:
    language = query_language(limits, semantic=False, rules=rules, similar=similar)
    return SYSTEM.replace("{language}", language).replace("{schema}", schema_text)


def messages(schema_text: str, question: str, limits: Mapping[str, int], *, examples: Sequence[tuple[str, dict[str, Any]]] = EXAMPLES, **options: Any) -> list[dict[str, str]]:
    """The conversation up to the question: system, the examples as earlier turns, the question."""

    conversation = [{"role": "system", "content": system_prompt(schema_text, limits, **options)}]
    for example_question, example_answer in examples:
        conversation.append({"role": "user", "content": f"Question: {example_question}"})
        conversation.append({"role": "assistant", "content": json.dumps(example_answer, separators=(",", ":"))})
    conversation.append({"role": "user", "content": f"Question: {question}"})
    return conversation


def repair_message(error_text: str) -> dict[str, str]:
    return {"role": "user", "content": REPAIR.replace("{error}", error_text)}
