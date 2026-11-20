"""The planning loop: generate a typed plan, validate it with YoDb, repair at most twice, execute.

    planner = NaturalLanguagePlanner(db, generator, PlannerConfig(representation="retrieved"), index=SchemaIndex(db.catalog, embedder))
    prediction = planner.answer(question)          # a benchmark Prediction

Validation is YoDb's own: every step of the plan is bound and planned (``db.explain``) before
anything runs, so an unknown field, a wrong type, a relationship that does not apply or a
placement the planner refuses comes back as a coded error with a location, which is handed to
the model verbatim.  A plan that still fails after the repairs is recorded with its last error
(the typed layer caught it; nothing ran).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import re
import time
from typing import Any

from yodb.errors import YoDbError
from yodb.mcp_server import error_text

from benchmark.harness import Prediction, kinds_of, run_plan, substitute_ids
from .grammar import OUTPUT
from .llm import Generation, Generator
from .prompt import EXAMPLES, messages as build_messages, repair_message
from .schema import estimate_tokens, render

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class PlannerConfig:
    representation: str = "full"          # full | descriptions | retrieved
    repairs: int = 2                      # most repair rounds after the first attempt
    # Hand the output JSON Schema to the generator as a decoding grammar.  Off by default: on the vLLM behind the
    # evaluation it made the model emit structurally minimal plans (no traverse, no page), so validation + repair
    # does the work instead; "on" is the ablation.
    grammar: bool = False
    top_k: int = 12                       # entries the SSSI keeps (retrieved)
    examples: bool = True                 # include the few-shot examples
    timeout: float = 60.0                 # per executed query


@dataclass
class Attempt:
    text: str
    tokens_in: int
    tokens_out: int
    generation_ms: float
    error: str | None = None              # why this attempt was rejected (None: accepted)


@dataclass
class PlannerOutcome:
    prediction: Prediction
    attempts: list[Attempt] = field(default_factory=list)
    schema_tokens: int = 0
    output: dict[str, Any] | None = None  # the accepted JSON


class PlanRejected(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NaturalLanguagePlanner:
    name = "planner"

    def __init__(self, db, generator: Generator, config: PlannerConfig = PlannerConfig(), *, index=None) -> None:
        self.db = db
        self.generator = generator
        self.config = config
        self.index = index
        self.name = f"planner-{config.representation}-{getattr(generator, 'name', 'llm')}"

    # --- the benchmark system interface ------------------------------------------------------------

    def answer(self, question: Mapping[str, Any]) -> Prediction:
        return self.plan(question["question"], question_id=question.get("id", "")).prediction

    # --- the loop ------------------------------------------------------------------------------------

    def plan(self, question: str, *, question_id: str = "") -> PlannerOutcome:
        started = time.perf_counter()
        schema_text = self.schema_for(question)
        outcome = PlannerOutcome(Prediction(id=question_id), schema_tokens=estimate_tokens(schema_text))
        conversation = build_messages(
            schema_text, question, self.db.limits,
            examples=EXAMPLES if self.config.examples else (), similar=bool(self.db.similar_fields()),
        )
        schema = OUTPUT if self.config.grammar else None
        accepted: dict[str, Any] | None = None
        last_error: str | None = None
        for round_index in range(self.config.repairs + 1):
            generation = self.generator.generate(conversation, json_schema=schema)
            attempt = Attempt(generation.text, generation.tokens_in, generation.tokens_out, generation.latency_ms)
            outcome.attempts.append(attempt)
            try:
                output = parse_output(generation.text)
                if "refuse" not in output:
                    self.validate(output)
                accepted = output
                break
            except PlanRejected as rejected:
                attempt.error = rejected.message
                last_error = rejected.message
                conversation = [*conversation, {"role": "assistant", "content": generation.text}, repair_message(rejected.message)]
        prediction = outcome.prediction
        prediction.tokens_in = sum(a.tokens_in for a in outcome.attempts)
        prediction.tokens_out = sum(a.tokens_out for a in outcome.attempts)
        prediction.repairs = len(outcome.attempts) - 1
        if accepted is None:
            prediction.plan_valid = False
            prediction.error = _code_of(last_error) or "plan_rejected"
            prediction.plan = _last_plan(outcome.attempts)
        elif "refuse" in accepted:
            prediction.refused = True
            prediction.refusal_code = accepted["refuse"].get("code")
            prediction.plan_valid = None
        else:
            outcome.output = accepted
            plan = accepted["plan"]
            answer_field = accepted["answer_field"]
            executed = run_plan(self.db, plan, answer_field, answer_dataset_of(self.db, plan, answer_field), timeout=self.config.timeout)
            for name in ("plan", "answer", "evidence", "sources", "error", "plan_valid", "retrieval_ms", "records_transferred"):
                setattr(prediction, name, getattr(executed, name))
            if not prediction.sources:
                prediction.sources = sorted(set(accepted.get("sources", [])))
        prediction.latency_ms = (time.perf_counter() - started) * 1000
        return outcome

    def schema_for(self, question: str) -> str:
        return render(self.config.representation, self.db.catalog, question=question, index=self.index, similar_fields=self.db.similar_fields(), top_k=self.config.top_k)

    def validate(self, output: Mapping[str, Any]) -> None:
        """Bind and plan every step with YoDb; a second step sees a sample of IDs for its ``$ids``."""

        plan = output.get("plan")
        if not isinstance(plan, list) or not plan or len(plan) > 2 or not all(isinstance(step, dict) for step in plan):
            raise PlanRejected("[plan_shape_invalid]: 'plan' must be a list of one or two query objects.")
        if not isinstance(output.get("answer_field"), str) or not output["answer_field"]:
            raise PlanRejected("[plan_shape_invalid]: 'answer_field' must name the row field whose values answer the question.")
        sample_rows = [[{key: "sample-id" for key in _id_fields(step)}] for step in plan]
        for index, step in enumerate(plan):
            concrete = substitute_ids(step, sample_rows[:index])
            try:
                self.db.explain(concrete)
            except YoDbError as error:
                prefix = f"in plan[{index}] " if len(plan) > 1 else ""
                raise PlanRejected(prefix + error_text(error)) from error
        last = plan[-1]
        field_name = output["answer_field"]
        if field_name != "id" and field_name not in _id_fields(last):
            raise PlanRejected(f"[plan_shape_invalid]: 'answer_field' {field_name!r} is not 'id' or '<alias>.id' of the last query's traverse step.")


# --- helpers -----------------------------------------------------------------------------------------


def parse_output(text: str) -> dict[str, Any]:
    """The one JSON object in the model's text (a code fence or stray prose around it is tolerated)."""

    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = candidate[4:] if candidate.startswith("json") else candidate
    try:
        output = json.loads(candidate)
    except ValueError:
        match = _JSON_OBJECT.search(text)
        if match is None:
            raise PlanRejected("[output_invalid]: the answer was not a JSON object.") from None
        try:
            output = json.loads(match.group(0))
        except ValueError as error:
            raise PlanRejected(f"[output_invalid]: the answer was not valid JSON ({error.msg}).") from None
    if not isinstance(output, dict) or not ({"plan", "refuse"} & set(output)):
        raise PlanRejected("[output_invalid]: the answer must be an object with 'plan' or 'refuse'.")
    refusal = output.get("refuse") if isinstance(output.get("refuse"), dict) else None
    if refusal is not None and str(refusal.get("code", "none")).lower() in ("none", "", "null"):
        refusal = None
    if output.get("plan"):
        output = {k: v for k, v in output.items() if k != "refuse"}          # a plan was given: it counts
    elif refusal is not None:
        output = {"refuse": refusal}
    else:
        raise PlanRejected("[output_invalid]: give a non-empty 'plan', or a 'refuse' with a code and a reason.")
    if "refuse" in output and not (isinstance(output["refuse"], dict) and output["refuse"].get("code")):
        raise PlanRejected("[output_invalid]: a refusal needs a 'code' and a 'reason'.")
    return output


def _id_fields(step: Mapping[str, Any]) -> list[str]:
    fields = ["id"]
    for traverse in step.get("traverse", ()) or ():
        if isinstance(traverse, dict):
            fields.append(f"{traverse.get('as', traverse.get('relationship'))}.id")
    return fields


def answer_dataset_of(db, plan: Sequence[Mapping[str, Any]], answer_field: str) -> str:
    """The dataset the answer IDs belong to: the last step's root, or the traversed dataset of ``alias.id``."""

    last = plan[-1]
    if answer_field == "id":
        return last["from"]["dataset"]
    alias = answer_field[: -len(".id")]
    catalog = db.catalog
    for traverse in last.get("traverse", ()):
        if traverse.get("as", traverse.get("relationship")) != alias:
            continue
        relationship = catalog.relationships.get(traverse["relationship"]) or next(
            (r for r in catalog.relationships.values() if traverse["relationship"] in r.aliases), None)
        if relationship is not None:
            return relationship.to_dataset if traverse.get("direction", "forward") == "forward" else relationship.from_dataset
    return ""


def _code_of(error_message: str | None) -> str | None:
    if not error_message:
        return None
    match = re.search(r"\[([a-z_]+)\]", error_message)
    return match.group(1) if match else None


def _last_plan(attempts: Sequence[Attempt]) -> list[dict[str, Any]]:
    for attempt in reversed(attempts):
        try:
            output = parse_output(attempt.text)
        except PlanRejected:
            continue
        if isinstance(output.get("plan"), list):
            return output["plan"]
    return []
