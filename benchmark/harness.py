"""Run a *system* over the benchmark questions and write its predictions.

A system is anything with ``answer(question) -> Prediction``: the gold-plan runner below (the
oracle-plan upper bound, B3), a natural-language planner over YoDb, a monolithic tool agent, a
LlamaIndex router...  The harness records what the scorer needs: the answer set, the evidence
IDs per dataset, the source kinds used, the error raised (if any), latencies, token usage and
repair count.

    PYTHONPATH=src python benchmark/harness.py gold --questions benchmark/questions.jsonl --out runs/gold.jsonl
    PYTHONPATH=src python benchmark/harness.py gold --split held_out --repeat 5 --warm-up
    PYTHONPATH=src python benchmark/harness.py planner --endpoint http://localhost:8000/v1 --schema retrieved

``--repeat N`` runs every question N times after an optional warm-up and keeps every timing
(the proposal's latency protocol); the scorer uses the medians per question.  ``--workers N``
answers N questions at a time (a vLLM endpoint batches them, so a model-bound run gets close to
N times faster); latencies then include queueing at the endpoint, so the timed experiment (E2,
no model) stays sequential.  ``--resume`` keeps the predictions already in the output file and
answers only the questions that are missing.
"""

from __future__ import annotations

import argparse
import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Protocol

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "src"))

from yodb.dotenv import load_dotenv  # noqa: E402

load_dotenv(HERE.parent / ".env")

from benchmark.templates import GRAPH, RELATIONAL, VECTOR  # noqa: E402

MAXIMUM_IDS = 1_000      # the most IDs a second step may carry in one `in` list (YoDb's own limit)


@dataclass
class Prediction:
    """What a system answered for one question."""

    id: str
    plan: list[dict[str, Any]] = field(default_factory=list)   # the typed queries it ran (empty when it refused)
    answer: list[str] = field(default_factory=list)            # the answer set (IDs of the answer dataset)
    evidence: dict[str, list[str]] = field(default_factory=dict)   # dataset -> record IDs the answer rests on
    sources: list[str] = field(default_factory=list)           # kinds of source used: relational, vector, graph
    error: str | None = None                                   # the error code when it refused or failed
    refused: bool = False                                      # it declined to answer (asked, or said it cannot)
    refusal_code: str | None = None                            # the system's own reason code when it refused
    plan_valid: bool | None = None                             # the final query validated (None: no query)
    latency_ms: float | None = None                            # end to end
    retrieval_ms: float | None = None                          # database work only, when the system can tell
    tokens_in: int = 0
    tokens_out: int = 0
    repairs: int = 0
    records_transferred: int = 0                               # intermediate records moved between steps/sources
    native_queries: int = 0                                    # native statements the system sent to a store (baselines)
    native_errors: int = 0                                     # of which the store rejected (H1's "invalid native queries")
    tool_calls: int = 0                                        # tool invocations (baselines)
    run: int = 0                                               # repetition index


class System(Protocol):
    name: str

    def answer(self, question: Mapping[str, Any]) -> Prediction: ...


# --- running a typed plan through YoDb (shared by the gold runner and any planner) ----------------------


def substitute_ids(step: Any, previous_rows: Sequence[Sequence[Mapping[str, Any]]]) -> Any:
    """Replace ``{"$ids": {"step": i, "field": f}}`` by the distinct values of ``f`` in step ``i``'s rows."""

    if isinstance(step, dict):
        if set(step) == {"$ids"}:
            reference = step["$ids"]
            rows = previous_rows[reference["step"]]
            values = list(dict.fromkeys(row.get(reference["field"]) for row in rows if row.get(reference["field"]) is not None))
            return values[:MAXIMUM_IDS]
        return {key: substitute_ids(value, previous_rows) for key, value in step.items()}
    if isinstance(step, list):
        return [substitute_ids(item, previous_rows) for item in step]
    return step


def kinds_of(db, plan: Sequence[Mapping[str, Any]]) -> list[str]:
    """The source kinds a plan touches, from the catalog: a column join is relational, an edge is graph, similar is vector."""

    kinds = {RELATIONAL}
    catalog = db.catalog
    for step in plan:
        if "similar" in json.dumps(step):
            kinds.add(VECTOR)
        for traverse in step.get("traverse", ()):
            relationship = catalog.relationships.get(traverse.get("relationship"))
            if relationship is None:
                relationship = next((r for r in catalog.relationships.values() if traverse.get("relationship") in r.aliases), None)
            if relationship is not None and all(i.edge_type is not None for i in relationship.implementations):
                kinds.add(GRAPH)
    return sorted(kinds)


def run_plan(db, plan: Sequence[Mapping[str, Any]], answer_field: str, answer_dataset: str, *, timeout: float = 60.0) -> Prediction:
    """Execute one or more typed steps through YoDb and collect the answer and evidence."""

    from yodb.errors import YoDbError

    prediction = Prediction(id="", plan=list(plan))
    started = time.perf_counter()
    rows_per_step: list[list[dict[str, Any]]] = []
    transferred = 0
    try:
        for index, step in enumerate(plan):
            concrete = substitute_ids(step, rows_per_step)
            if index > 0:
                in_lists = [v for v in _in_values(concrete)]
                transferred += sum(len(v) for v in in_lists)
                if any(not v for v in in_lists):
                    rows_per_step.append([])          # nothing reached: the chain ends empty
                    continue
            result = db.query(concrete, timeout_seconds=timeout)
            rows_per_step.append([dict(row) for row in result.rows])
        prediction.plan_valid = True
    except YoDbError as error:
        prediction.error = error.code.value
        prediction.plan_valid = False
        prediction.latency_ms = (time.perf_counter() - started) * 1000
        return prediction
    prediction.latency_ms = (time.perf_counter() - started) * 1000
    prediction.retrieval_ms = prediction.latency_ms
    final_rows = rows_per_step[-1] if rows_per_step else []
    prediction.answer = list(dict.fromkeys(str(row[answer_field]) for row in final_rows if row.get(answer_field) is not None))
    prediction.evidence = evidence_of(db, plan, rows_per_step)
    if answer_dataset:
        prediction.evidence.setdefault(answer_dataset, [])
        prediction.evidence[answer_dataset] = list(dict.fromkeys([*prediction.evidence[answer_dataset], *prediction.answer]))
    prediction.sources = kinds_of(db, plan)
    prediction.records_transferred = transferred
    prediction.native_queries = sum(_reads_of(db, step) for step in plan)      # validated before they ran: none invalid
    return prediction


def _reads_of(db, step: Mapping[str, Any]) -> int:
    """Native statements one typed step issues, from its explained plan (scans and edge reads, batches aside)."""

    try:
        explanation = db.explain(substitute_ids(step, [[{"id": "x"}]]))
    except Exception:
        return 1
    return sum(1 for node in explanation.nodes if node.kind in ("remote_scan", "graph_edge_scan")) or 1


def evidence_of(db, plan: Sequence[Mapping[str, Any]], rows_per_step: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, list[str]]:
    """Every record ID that appeared in any step's rows, by dataset (root ids and ``alias.id`` columns)."""

    evidence: dict[str, dict[str, None]] = {}
    catalog = db.catalog
    for step, rows in zip(plan, rows_per_step):
        root = step["from"]["dataset"]
        aliases: dict[str, str] = {}
        for traverse in step.get("traverse", ()):
            relationship = catalog.relationships.get(traverse["relationship"]) or next(
                (r for r in catalog.relationships.values() if traverse["relationship"] in r.aliases), None)
            if relationship is None:
                continue
            other = relationship.to_dataset if traverse.get("direction", "forward") == "forward" else relationship.from_dataset
            aliases[traverse.get("as", traverse["relationship"])] = other
        for row in rows:
            if row.get("id") is not None:
                evidence.setdefault(root, {})[str(row["id"])] = None
            for alias, dataset in aliases.items():
                value = row.get(f"{alias}.id")
                if value is not None:
                    evidence.setdefault(dataset, {})[str(value)] = None
    return {dataset: list(ids) for dataset, ids in evidence.items()}


def _in_values(node: Any) -> Iterable[list]:
    if isinstance(node, dict):
        if node.get("op") == "in" and isinstance(node.get("value"), list):
            yield node["value"]
        for value in node.values():
            yield from _in_values(value)
    elif isinstance(node, list):
        for item in node:
            yield from _in_values(item)


# --- the gold-plan system (B3: the oracle plan, executed by YoDb) ----------------------------------------


class GoldPlanSystem:
    """Runs each question's gold typed plan.  Its answer accuracy measures the engine and the gold
    plans themselves (it should be ~1.0); its latency is the retrieval floor other systems add to."""

    name = "gold"

    def __init__(self, db, *, timeout: float = 60.0) -> None:
        self.db = db
        self.timeout = timeout

    def answer(self, question: Mapping[str, Any]) -> Prediction:
        if not question["plan"]:
            return Prediction(id=question["id"], refused=True, error="refusal")
        prediction = run_plan(self.db, question["plan"], question["answer_field"], question["answer_dataset"], timeout=self.timeout)
        prediction.id = question["id"]
        return prediction


# --- the loop ----------------------------------------------------------------------------------------------


def load_questions(path: Path, *, split: str | None = None, classes: set[str] | None = None, only: Sequence[str] = ()) -> list[dict[str, Any]]:
    questions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if split:
        questions = [q for q in questions if q["split"] == split]
    if classes:
        questions = [q for q in questions if q["class"] in classes]
    if only:
        questions = [q for q in questions if any(o in q["id"] for o in only)]
    return questions


def run(
    system: System | Callable[[], System], questions: Sequence[Mapping[str, Any]], out: Path, *,
    repeat: int = 1, warm_up: bool = False, quiet: bool = False, verbose: bool = False, workers: int = 1, resume: bool = False,
) -> list[Prediction]:
    """Answer every question (``repeat`` times each) and append each prediction to ``out`` as it is made.

    ``system`` is a system, or a factory of systems when ``workers`` > 1: each worker thread answers with
    its own instance (its own store connections), so the systems need not be thread-safe.  With ``resume``,
    the (question, run) pairs already in ``out`` are kept and skipped.
    """

    done: set[tuple[str, int]] = set()
    predictions: list[Prediction] = []
    if resume and out.is_file():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                done.add((record["id"], record.get("run", 0)))
    todo = [(question, repetition) for question in questions for repetition in range(repeat) if (question["id"], repetition) not in done]
    if done and not quiet:
        print(f"resuming: {len(done)} predictions kept, {len(todo)} to go", flush=True)
    is_instance = inspect.ismethod(getattr(system, "answer", None))                   # a system has a bound answer(); a factory does not
    make: Callable[[], System] = (lambda: system) if is_instance else system           # type: ignore[assignment,return-value]
    lock = threading.Lock()
    finished = 0

    def answer_with(worker_system: System, question: Mapping[str, Any], repetition: int) -> Prediction:
        try:
            prediction = worker_system.answer(question)
        except Exception as error:                      # a crashed system call is this question's failure, not the run's
            prediction = Prediction(id=question["id"], error=f"system_error:{type(error).__name__}", plan_valid=False)
            if not quiet:
                print(f"    {question['id']}: {type(error).__name__}: {str(error)[:200]}", flush=True)
        prediction.id = question["id"]
        prediction.run = repetition
        return prediction

    def record(question: Mapping[str, Any], prediction: Prediction, handle) -> None:
        nonlocal finished
        with lock:
            finished += 1
            predictions.append(prediction)
            handle.write(json.dumps(asdict(prediction), ensure_ascii=False) + "\n")
            handle.flush()
            if not quiet and prediction.run == repeat - 1:
                _report(finished, len(todo), question, prediction, verbose)

    with out.open("a" if resume else "w", encoding="utf-8") as handle:
        if workers <= 1 or len(todo) <= 1:
            one = make()
            if warm_up and questions:
                one.answer(questions[0])
            for question, repetition in todo:
                record(question, answer_with(one, question, repetition), handle)
        else:
            local = threading.local()

            def work(item: tuple[Mapping[str, Any], int]) -> None:
                if not hasattr(local, "system"):
                    local.system = make()
                    if warm_up and questions:
                        local.system.answer(questions[0])
                question, repetition = item
                record(question, answer_with(local.system, question, repetition), handle)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(work, todo))
    return predictions


def _report(index: int, total: int, question: Mapping[str, Any], last: Prediction, verbose: bool) -> None:
    outcome = f"error={last.error}" if last.error else (f"refused ({last.refusal_code})" if last.refused else f"{len(last.answer)} answers")
    gold = question.get("gold_answer")
    expected = f", gold {len(gold)}" if gold is not None and question.get("expect_error") is None else (f", expects {question['expect_error']}" if question.get("expect_error") else "")
    print(f"[{index}/{total}] {question['id']}: {outcome}{expected} in {last.latency_ms or 0:.0f} ms"
          + (f", {last.repairs} repair(s)" if last.repairs else ""), flush=True)
    if verbose:
        print(f"    question: {question['question']}")
        for step in last.plan:
            print(f"    plan: {json.dumps(step, ensure_ascii=False)}")
        if last.answer:
            print(f"    answer: {last.answer[:5]}{' ...' if len(last.answer) > 5 else ''}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("system", choices=["gold", "planner", "tool-agent", "router"],
                        help="gold: each question's gold plan through YoDb (B3); planner: the natural-language planner; "
                             "tool-agent: B1, the same model over raw SQL/vector/Cypher tools; router: B2, framework routing")
    parser.add_argument("--questions", type=Path, default=HERE / "questions.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="predictions file (default runs/<system>.jsonl)")
    parser.add_argument("--catalog", type=Path, default=HERE.parent / "examples" / "yelp" / "catalog")
    parser.add_argument("--split", choices=["tune", "held_out"])
    parser.add_argument("--class", dest="classes", action="append", choices=["single", "independent", "dependent", "stress"])
    parser.add_argument("--only", nargs="*", default=(), help="question id substrings")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warm-up", action="store_true")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--sequential", action="store_true", help="forced-sequential execution (joins probe by keys, one read at a time): the RQ2 ablation")
    parser.add_argument("--workers", type=int, default=1, help="questions answered at a time (model-bound runs; latencies then include queueing)")
    parser.add_argument("--resume", action="store_true", help="keep the predictions already in the output file and answer only the missing questions")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true", help="print each question, the plan the system ran and the first answers")
    planner_options = parser.add_argument_group("planner", "options of the natural-language planner")
    planner_options.add_argument("--model", default=os.environ.get("YODB_LLM_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"), help="the model (default: $YODB_LLM_MODEL)")
    planner_options.add_argument("--endpoint", default=os.environ.get("YODB_LLM_ENDPOINT") or None, help="an OpenAI-compatible chat endpoint (vLLM, or a hosted API such as https://openai.rc.asu.edu/v1); without it the model runs in-process with transformers")
    planner_options.add_argument("--api-key", default=None, help="the endpoint's API key (default: $OPENAI_API_KEY)")
    planner_options.add_argument("--llm-timeout", type=float, default=900.0, help="seconds to wait for one model answer (a baseline listing 500 ids can take minutes)")
    planner_options.add_argument("--thinking", action="store_true", help="let a reasoning model (Qwen3...) think before answering (default: off, /no_think)")
    planner_options.add_argument("--schema", choices=["full", "descriptions", "retrieved"], default="full", help="the schema representation shown to the model (RQ3)")
    planner_options.add_argument("--repairs", type=int, default=2)
    planner_options.add_argument("--top-k", type=int, default=12, help="entries the SSSI keeps for the retrieved representation")
    planner_options.add_argument("--grammar", action="store_true", help="constrain decoding with the output JSON Schema (the ablation; off by default)")
    planner_options.add_argument("--no-grammar", action="store_true", help=argparse.SUPPRESS)
    planner_options.add_argument("--no-examples", action="store_true")
    baseline_options = parser.add_argument_group("baselines", "options of the tool agent and the router")
    baseline_options.add_argument("--dsn", default=os.environ.get("YODB_CONN_YELP"), help="the read-only connection the baselines use directly (default: $YODB_CONN_YELP)")
    baseline_options.add_argument("--max-turns", type=int, default=8, help="tool calls the tool agent may make")
    args = parser.parse_args(argv)

    import yodb
    from yodb.embeddings import FastEmbedProvider

    questions = load_questions(args.questions, split=args.split, classes=set(args.classes) if args.classes else None, only=args.only)
    out = args.out or HERE / "runs" / f"{args.system}{'-sequential' if args.sequential else ''}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    needs_embedder = args.system != "gold" or any("similar" in json.dumps(q["plan"]) for q in questions)
    embedder = FastEmbedProvider(args.embedding_model) if needs_embedder else None
    from yodb.execution.operators import FederatedExecutionPolicy
    from yodb.planning import JoinPolicy

    policies = (
        {"join_policy": JoinPolicy(parallel="never"), "execution_policy": FederatedExecutionPolicy(concurrency=1)} if args.sequential else {}
    )
    with yodb.connect(args.catalog, embedder=embedder, pool_size=max(10, args.workers + 2), **policies) as db:
        if args.system == "gold":
            def make() -> System:
                return GoldPlanSystem(db, timeout=args.timeout)
        else:
            from planner.llm import OpenAICompatibleGenerator, TransformersGenerator

            if not args.endpoint:
                in_process = TransformersGenerator(args.model)                 # one model in memory, shared

            def generator():
                return (
                    OpenAICompatibleGenerator(args.model, base_url=args.endpoint, api_key=args.api_key, timeout=args.llm_timeout, thinking=args.thinking)
                    if args.endpoint else in_process
                )

            if args.system == "planner":
                from planner import NaturalLanguagePlanner, PlannerConfig
                from planner.sssi import SchemaIndex

                index = SchemaIndex(db.catalog, embedder) if args.schema == "retrieved" else None
                config = PlannerConfig(representation=args.schema, repairs=args.repairs, grammar=args.grammar, top_k=args.top_k,
                                       examples=not args.no_examples, timeout=args.timeout)

                def make() -> System:
                    return NaturalLanguagePlanner(db, generator(), config, index=index)       # the db pools its connections
            else:
                from baselines.tools import RawStores

                if not args.dsn:
                    parser.error("the baselines need --dsn (or $YODB_CONN_YELP): they read the stores directly")
                schema_text = RawStores(args.dsn, embedder, statement_timeout_seconds=args.timeout).schema_text()

                def make() -> System:
                    stores = RawStores(args.dsn, embedder, statement_timeout_seconds=args.timeout)     # one connection per worker
                    if args.system == "tool-agent":
                        from baselines.tool_agent import ToolAgent

                        return ToolAgent(stores, generator(), max_turns=args.max_turns, grammar=args.grammar, schema_text=schema_text)
                    from baselines.router import RouterSystem

                    return RouterSystem(stores, generator(), grammar=args.grammar, schema_text=schema_text)
            out = args.out or HERE / "runs" / f"{make().name}{'-sequential' if args.sequential else ''}.jsonl"
        predictions = run(make, questions, out, repeat=args.repeat, warm_up=args.warm_up, quiet=args.quiet, verbose=args.verbose,
                          workers=args.workers, resume=args.resume)
    print(f"wrote {len(predictions)} predictions to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
