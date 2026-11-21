"""Ask the typed layer a question in English and watch what happens.

    PYTHONPATH=src python planner/ask.py "Which reviews similar to 'slow service and a long wait' are of open businesses in Tampa, and who wrote them?"
    PYTHONPATH=src python planner/ask.py --schema retrieved --rows 10 "Who are the friends of the user with id ...?"

Prints, in order: the schema representation sent to the model (size), the model's plan (and every repair the typed
layer asked for, with the coded error), the physical plan YoDb chose for each step (which source kind each read goes
to, what was pushed down, the join strategy), the rows, the evidence IDs per dataset, the source kinds used, and the
timings and tokens.  The model and the database come from .env (YODB_LLM_ENDPOINT, YODB_LLM_MODEL, OPENAI_API_KEY,
YODB_CONN_YELP), as for the benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from yodb.dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--catalog", type=Path, default=ROOT / "examples" / "yelp" / "catalog")
    parser.add_argument("--schema", choices=["full", "descriptions", "retrieved"], default="full")
    parser.add_argument("--model", default=os.environ.get("YODB_LLM_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"))
    parser.add_argument("--endpoint", default=os.environ.get("YODB_LLM_ENDPOINT") or None)
    parser.add_argument("--repairs", type=int, default=2)
    parser.add_argument("--rows", type=int, default=5, help="rows of each step to print")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--show-prompt", action="store_true", help="print the schema text the model sees")
    args = parser.parse_args(argv)

    import yodb
    from yodb.embeddings import FastEmbedProvider
    from yodb.errors import YoDbError
    from yodb.mcp_server import error_text

    from benchmark.harness import evidence_of, kinds_of, substitute_ids
    from planner import NaturalLanguagePlanner, PlannerConfig
    from planner.llm import OpenAICompatibleGenerator, TransformersGenerator
    from planner.schema import render
    from planner.sssi import SchemaIndex

    embedder = FastEmbedProvider(args.embedding_model)
    generator = OpenAICompatibleGenerator(args.model, base_url=args.endpoint) if args.endpoint else TransformersGenerator(args.model)
    with yodb.connect(args.catalog, embedder=embedder) as db:
        index = SchemaIndex(db.catalog, embedder) if args.schema == "retrieved" else None
        planner = NaturalLanguagePlanner(db, generator, PlannerConfig(representation=args.schema, repairs=args.repairs), index=index)

        schema_text = render(args.schema, db.catalog, question=args.question, index=index, similar_fields=db.similar_fields())
        section("1. the schema the model sees")
        print(f"representation: {args.schema}; {len(schema_text):,} characters (~{len(schema_text) // 4:,} tokens)")
        if args.show_prompt:
            print(schema_text)

        section("2. the model plans (validated by the typed layer, repaired on a coded error)")
        started = time.perf_counter()
        # Plan without executing: run the loop by hand so each attempt can be shown.
        outcome = _plan_only(planner, args.question)
        for number, attempt in enumerate(outcome["attempts"], 1):
            print(f"attempt {number} ({attempt['tokens_in']} tokens in, {attempt['tokens_out']} out, {attempt['ms']:.0f} ms):")
            print(indent(attempt["text"].strip()[:3000]))
            if attempt["error"]:
                print(f"  -> rejected: {attempt['error']}")
            else:
                print("  -> accepted")
        if outcome["refusal"]:
            print(f"\nThe model refused: {json.dumps(outcome['refusal'])}")
            return 0
        if outcome["plan"] is None:
            print("\nNo valid plan after the repairs; nothing was sent to the stores.")
            return 1
        plan, answer_field = outcome["plan"], outcome["answer_field"]

        section("3. the physical plan YoDb chose (nothing has run yet)")
        for index_, step in enumerate(plan):
            concrete = substitute_ids(step, [[{"id": "sample-id"}]] * index_)
            explanation = db.explain(concrete)
            print(f"step {index_ + 1}: {explanation.plan_kind}")
            for node in explanation.nodes:
                pushed = f", pushed down: {', '.join(node.pushed_filter_fields)}" if node.pushed_filter_fields else ""
                residual = ", residual filter at the coordinator" if node.residual_filter else ""
                limit = f", limit {node.limit}" if node.limit else ""
                print(f"  {node.kind:16} {node.location}{pushed}{residual}{limit}")
                for detail in node.detail:
                    print(f"      {detail}")
            for note in explanation.optimizer:
                print(f"  optimizer: {note}")

        section("4. execution")
        rows_per_step = []
        for index_, step in enumerate(plan):
            concrete = substitute_ids(step, rows_per_step)
            t0 = time.perf_counter()
            try:
                result = db.query(concrete, timeout_seconds=60)
            except YoDbError as error:
                print(f"step {index_ + 1} failed: {error_text(error)}")
                return 1
            rows = [dict(r) for r in result.rows]
            rows_per_step.append(rows)
            activity = getattr(result, "activity", {}) or {}
            print(f"step {index_ + 1}: {len(rows)} rows in {(time.perf_counter() - t0) * 1000:.0f} ms"
                  + (f"; {activity.get('reads', '?')} native reads, {activity.get('rows_read', '?')} rows read from the sources" if activity else ""))
            for row in rows[: args.rows]:
                print("  " + json.dumps(row, ensure_ascii=False, default=str)[:300])
            if len(rows) > args.rows:
                print(f"  ... {len(rows) - args.rows} more")

        section("5. the answer and its provenance")
        final = rows_per_step[-1] if rows_per_step else []
        answers = list(dict.fromkeys(str(r[answer_field]) for r in final if r.get(answer_field) is not None))
        print(f"answer field: {answer_field}; {len(answers)} distinct ids: {answers[:10]}{' ...' if len(answers) > 10 else ''}")
        evidence = evidence_of(db, plan, rows_per_step)
        print("evidence (record ids the answer rests on): " + ", ".join(f"{dataset}: {len(ids)}" for dataset, ids in evidence.items()))
        print(f"source kinds used: {', '.join(kinds_of(db, plan))}")
        print(f"end to end: {(time.perf_counter() - started) * 1000:.0f} ms; model tokens: {sum(a['tokens_in'] for a in outcome['attempts'])} in, {sum(a['tokens_out'] for a in outcome['attempts'])} out")
    return 0


def _plan_only(planner, question: str) -> dict:
    """The planner's loop, without executing: every attempt's text and verdict."""

    from planner.planner import PlanRejected, parse_output
    from planner.prompt import EXAMPLES, messages as build_messages, repair_message

    config = planner.config
    conversation = build_messages(
        planner.schema_for(question), question, planner.db.limits,
        examples=EXAMPLES if config.examples else (), similar=bool(planner.db.similar_fields()),
    )
    attempts, plan, answer_field, refusal = [], None, None, None
    for _ in range(config.repairs + 1):
        generation = planner.generator.generate(conversation, json_schema=None)
        attempt = {"text": generation.text, "tokens_in": generation.tokens_in, "tokens_out": generation.tokens_out, "ms": generation.latency_ms, "error": None}
        attempts.append(attempt)
        try:
            output = parse_output(generation.text)
            if "refuse" in output:
                refusal = output["refuse"]
                break
            planner.validate(output)
            plan, answer_field = output["plan"], output["answer_field"]
            break
        except PlanRejected as rejected:
            attempt["error"] = rejected.message
            conversation = [*conversation, {"role": "assistant", "content": generation.text}, repair_message(rejected.message)]
    return {"attempts": attempts, "plan": plan, "answer_field": answer_field, "refusal": refusal}


def section(title: str) -> None:
    print(f"\n=== {title}")


def indent(text: str) -> str:
    return "\n".join("  " + line for line in text.splitlines())


if __name__ == "__main__":
    sys.exit(main())
