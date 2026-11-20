"""Score a predictions file against the gold questions.

    PYTHONPATH=src python benchmark/score.py --questions benchmark/questions.jsonl benchmark/runs/gold.jsonl [more runs...]
    PYTHONPATH=src python benchmark/score.py ... --json report.json --split held_out

Per question class and overall, as in the proposal's evaluation plan:

* answer accuracy: exact match and F1 of the answer set (stress questions: 1 when the system
  refused or raised the expected error, else 0);
* source selection: micro- and macro-F1 over the source kinds (relational, vector, graph);
* plan validity: the fraction of questions whose final query validated (stress: refused or the expected error);
* provenance: precision/recall/F1 of the evidence IDs over the gold evidence (dataset-qualified);
* latency: P50/P95 of end-to-end and retrieval latency over the per-question medians (several runs);
* tokens, repairs and transferred records: means.

Quality means come with 95% bootstrap confidence intervals (1,000 resamples over questions).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
CLASSES = ("single", "independent", "dependent", "stress")


def f1(predicted: set, gold: set) -> tuple[float, float, float]:
    """Precision, recall, F1 of two sets (an empty prediction of an empty gold is perfect)."""

    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not predicted or not gold:
        return 0.0, 0.0, 0.0
    hit = len(predicted & gold)
    precision, recall = hit / len(predicted), hit / len(gold)
    score = 0.0 if hit == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, score


def question_scores(question: Mapping[str, Any], prediction: Mapping[str, Any]) -> dict[str, float | None]:
    """Every per-question metric for one (question, prediction) pair."""

    expected_error = question.get("expect_error")
    if expected_error is not None:
        refused = bool(prediction.get("refused")) or prediction.get("error") is not None
        correct = (
            (prediction.get("error") == expected_error)
            or (expected_error == "refusal" and refused)
            or (expected_error != "refusal" and prediction.get("refused") and prediction.get("error") in (None, expected_error))
        )
        return {
            "answer_em": 1.0 if correct else 0.0,
            "answer_f1": 1.0 if correct else 0.0,
            "plan_valid": 1.0 if correct else 0.0,
            "source_f1": None,
            "provenance_precision": None,
            "provenance_recall": None,
            "provenance_f1": None,
        }
    gold_answer = set(map(str, question["gold_answer"]))
    predicted_answer = set(map(str, prediction.get("answer", [])))
    _, _, answer_f1 = f1(predicted_answer, gold_answer)
    gold_sources = set(question.get("sources", []))
    predicted_sources = set(prediction.get("sources", []))
    _, _, source_f1 = f1(predicted_sources, gold_sources)
    gold_evidence = {f"{dataset}:{i}" for dataset, ids in question.get("gold_evidence", {}).items() for i in ids}
    predicted_evidence = {f"{dataset}:{i}" for dataset, ids in prediction.get("evidence", {}).items() for i in ids}
    precision, recall, evidence_f1 = f1(predicted_evidence, gold_evidence)
    return {
        "answer_em": 1.0 if predicted_answer == gold_answer and prediction.get("error") is None else 0.0,
        "answer_f1": answer_f1 if prediction.get("error") is None else 0.0,
        "plan_valid": 1.0 if prediction.get("plan_valid") else 0.0,
        "source_f1": source_f1,
        "provenance_precision": precision,
        "provenance_recall": recall,
        "provenance_f1": evidence_f1,
    }


def source_micro_f1(questions: Sequence[Mapping[str, Any]], predictions: Mapping[str, Mapping[str, Any]]) -> float | None:
    tp = fp = fn = 0
    for question in questions:
        if question.get("expect_error") is not None or question["id"] not in predictions:
            continue
        gold, predicted = set(question.get("sources", [])), set(predictions[question["id"]].get("sources", []))
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    if tp + fp + fn == 0:
        return None
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap(values: Sequence[float], *, resamples: int = 1000, seed: int = 0) -> tuple[float, float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        drawn = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(drawn) / len(drawn))
    means.sort()
    return means[int(0.025 * (resamples - 1))], means[int(0.975 * (resamples - 1))]


def medians_per_question(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
    by_id: dict[str, list[float]] = {}
    for row in rows:
        if row.get(key) is not None:
            by_id.setdefault(row["id"], []).append(float(row[key]))
    return {question_id: statistics.median(values) for question_id, values in by_id.items()}


def score(questions: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], *, resamples: int = 1000) -> dict[str, Any]:
    """The report: per class and overall."""

    # one prediction per question for quality (the first run); every run for latency
    first: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        first.setdefault(row["id"], row)
    latency = medians_per_question(rows, "latency_ms")
    retrieval = medians_per_question(rows, "retrieval_ms")
    report: dict[str, Any] = {"classes": {}, "questions": len(questions), "predicted": sum(1 for q in questions if q["id"] in first)}
    for cls in (*CLASSES, "all"):
        chosen = [q for q in questions if (cls == "all" or q["class"] == cls) and q["id"] in first]
        if not chosen:
            continue
        per_question = [question_scores(q, first[q["id"]]) for q in chosen]
        section: dict[str, Any] = {"questions": len(chosen)}
        for metric in ("answer_em", "answer_f1", "plan_valid", "source_f1", "provenance_precision", "provenance_recall", "provenance_f1"):
            values = [s[metric] for s in per_question if s[metric] is not None]
            if values:
                section[metric] = sum(values) / len(values)
                interval = bootstrap(values, resamples=resamples)
                if interval:
                    section[f"{metric}_ci"] = [round(interval[0], 4), round(interval[1], 4)]
        micro = source_micro_f1(chosen, first)
        if micro is not None:
            section["source_micro_f1"] = micro
        ids = {q["id"] for q in chosen}
        for name, table in (("latency_ms", latency), ("retrieval_ms", retrieval)):
            values = [table[i] for i in ids if i in table]
            if values:
                section[f"{name}_p50"] = percentile(values, 0.5)
                section[f"{name}_p95"] = percentile(values, 0.95)
        for name in ("tokens_in", "tokens_out", "repairs", "records_transferred", "native_queries", "native_errors", "tool_calls"):
            values = [float(first[i].get(name, 0) or 0) for i in ids]
            if values:
                section[f"{name}_mean"] = sum(values) / len(values)
        queries = sum(float(first[i].get("native_queries", 0) or 0) for i in ids)
        if queries:
            section["native_error_rate"] = sum(float(first[i].get("native_errors", 0) or 0) for i in ids) / queries
        section["errors"] = sorted({first[i]["error"] for i in ids if first[i].get("error")})
        report["classes"][cls] = section
    return report


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def print_report(name: str, report: Mapping[str, Any]) -> None:
    print(f"\n{name}: {report['predicted']}/{report['questions']} questions predicted")
    header = f"{'class':<12}{'n':>5}{'EM':>8}{'F1':>8}{'valid':>8}{'srcF1':>8}{'provF1':>8}{'p50ms':>9}{'p95ms':>9}{'tok':>7}{'rep':>5}{'nerr':>6}"
    print(header)
    for cls, section in report["classes"].items():
        def cell(key, width=8, digits=3):
            value = section.get(key)
            return f"{value:>{width}.{digits}f}" if isinstance(value, (int, float)) else f"{'-':>{width}}"
        print(f"{cls:<12}{section['questions']:>5}{cell('answer_em')}{cell('answer_f1')}{cell('plan_valid')}{cell('source_f1')}"
              f"{cell('provenance_f1')}{cell('latency_ms_p50', 9, 0)}{cell('latency_ms_p95', 9, 0)}{cell('tokens_in_mean', 7, 0)}{cell('repairs_mean', 5, 1)}{cell('native_errors_mean', 6, 2)}")
        if section.get("answer_f1_ci"):
            print(f"{'':<12}{'':>5}  F1 95% CI {section['answer_f1_ci'][0]:.3f}-{section['answer_f1_ci'][1]:.3f}"
                  + (f"; errors seen: {', '.join(section['errors'])}" if section.get("errors") else ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="predictions files (one per system)")
    parser.add_argument("--questions", type=Path, default=HERE / "questions.jsonl")
    parser.add_argument("--split", choices=["tune", "held_out"])
    parser.add_argument("--json", type=Path, help="also write the reports as JSON")
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args(argv)
    questions = load_jsonl(args.questions)
    if args.split:
        questions = [q for q in questions if q["split"] == args.split]
    reports = {}
    for path in args.runs:
        report = score(questions, load_jsonl(path), resamples=args.resamples)
        reports[path.stem] = report
        print_report(path.stem, report)
    if args.json:
        args.json.write_text(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
