"""The natural-language planner: a question becomes a typed YoDb plan, validated before it runs.

    planner/schema.py      the three schema representations an LLM may be shown (RQ3):
                           the complete typed catalog, descriptions only, or retrieved typed fragments (SSSI)
    planner/sssi.py        the Structured Semantic Schema Index: catalog entries embedded and retrieved per question
    planner/grammar.py     the JSON Schema of the planner's output, for grammar-constrained decoding
    planner/prompt.py      the prompt: the query language, the schema, a few examples, the output contract
    planner/llm.py         generators: an OpenAI-compatible endpoint (vLLM, Ollama...), transformers, a scripted one for tests
    planner/planner.py     the loop: generate, validate with YoDb, feed the error back (at most two repairs), execute

The planner is a benchmark *system* (``benchmark/harness.py planner ...``).
"""

from .planner import NaturalLanguagePlanner, PlannerConfig, PlannerOutcome

__all__ = ["NaturalLanguagePlanner", "PlannerConfig", "PlannerOutcome"]
