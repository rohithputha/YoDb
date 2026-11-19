"""The JSON Schema of what the planner must output, for grammar-constrained decoding.

An answer is either a plan (one or two typed queries, the field whose values answer the
question, and the kinds of source used) or a refusal with a code.  The query schema mirrors
YoDb's query model: ``from``, ``select``, ``where`` (predicates, all/any/not, rules, a similarity
search), one ``traverse`` step, ``order_by``, ``page``.  A second step's ``in`` value may be the
``$ids`` reference to the first step's rows.
"""

from __future__ import annotations

import json
from typing import Any

OPERATORS = ["eq", "ne", "in", "not_in", "is_null", "is_not_null", "gt", "gte", "lt", "lte", "contains", "starts_with"]
REFUSAL_CODES = ["unsupported", "ambiguous", "not_a_read", "no_such_data", "other"]

_SCALAR = {"type": ["string", "number", "boolean"]}

CONDITION: dict[str, Any] = {
    "anyOf": [
        {
            "type": "object",
            "properties": {
                "field": {"type": "string"},
                "op": {"type": "string", "enum": OPERATORS},
                "value": {"anyOf": [_SCALAR, {"type": "array", "items": _SCALAR}, {"$ref": "#/$defs/ids"}]},
            },
            "required": ["field", "op"],
            "additionalProperties": False,
        },
        {"type": "object", "properties": {"all": {"type": "array", "items": {"$ref": "#/$defs/condition"}, "minItems": 1}}, "required": ["all"], "additionalProperties": False},
        {"type": "object", "properties": {"any": {"type": "array", "items": {"$ref": "#/$defs/condition"}, "minItems": 1}}, "required": ["any"], "additionalProperties": False},
        {"type": "object", "properties": {"not": {"$ref": "#/$defs/condition"}}, "required": ["not"], "additionalProperties": False},
        {"type": "object", "properties": {"rule": {"type": "string"}}, "required": ["rule"], "additionalProperties": False},
        {
            "type": "object",
            "properties": {
                "similar": {
                    "type": "object",
                    "properties": {"field": {"type": "string"}, "text": {"type": "string"}, "top": {"type": "integer", "minimum": 1, "maximum": 1000}},
                    "required": ["field", "text"],
                    "additionalProperties": False,
                }
            },
            "required": ["similar"],
            "additionalProperties": False,
        },
    ]
}

QUERY: dict[str, Any] = {
    "type": "object",
    "properties": {
        "from": {"type": "object", "properties": {"dataset": {"type": "string"}}, "required": ["dataset"], "additionalProperties": False},
        "select": {"type": "array", "items": {"type": "string"}},
        "where": {"$ref": "#/$defs/condition"},
        "traverse": {
            "type": "array",
            "minItems": 1,
            "maxItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "relationship": {"type": "string"},
                    "as": {"type": "string"},
                    "select": {"type": "array", "items": {"type": "string"}},
                    "where": {"$ref": "#/$defs/condition"},
                    "direction": {"type": "string", "enum": ["forward", "reverse"]},
                    "optional": {"type": "boolean"},
                },
                "required": ["relationship", "as"],
                "additionalProperties": False,
            },
        },
        "order_by": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"field": {"type": "string"}, "direction": {"type": "string", "enum": ["asc", "desc"]}},
                "required": ["field", "direction"],
                "additionalProperties": False,
            },
        },
        "page": {"type": "object", "properties": {"first": {"type": "integer", "minimum": 1, "maximum": 500}}, "required": ["first"], "additionalProperties": False},
    },
    "required": ["from"],
    "additionalProperties": False,
}

OUTPUT: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "PlannerOutput",
    "$defs": {
        "condition": CONDITION,
        "query": QUERY,
        "ids": {
            "type": "object",
            "properties": {
                "$ids": {
                    "type": "object",
                    "properties": {"step": {"type": "integer", "minimum": 0}, "field": {"type": "string"}},
                    "required": ["step", "field"],
                    "additionalProperties": False,
                }
            },
            "required": ["$ids"],
            "additionalProperties": False,
        },
    },
    "anyOf": [
        {
            "type": "object",
            "properties": {
                "plan": {"type": "array", "items": {"$ref": "#/$defs/query"}, "minItems": 1, "maxItems": 2},
                "answer_field": {"type": "string"},
                "sources": {"type": "array", "items": {"type": "string", "enum": ["relational", "vector", "graph"]}},
            },
            "required": ["plan", "answer_field", "sources"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "refuse": {
                    "type": "object",
                    "properties": {"code": {"type": "string", "enum": REFUSAL_CODES}, "reason": {"type": "string"}},
                    "required": ["code", "reason"],
                    "additionalProperties": False,
                }
            },
            "required": ["refuse"],
            "additionalProperties": False,
        },
    ],
}


def output_schema_json() -> str:
    return json.dumps(OUTPUT, separators=(",", ":"))
