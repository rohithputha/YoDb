"""Which questions a run got wrong, and how: the gold/predicted answer-set difference per question, grouped by template.

    PYTHONPATH=src python benchmark/diff.py benchmark/runs/gold.jsonl [--questions benchmark/questions.jsonl] [--show 3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from benchmark.score import load_jsonl, question_scores  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--questions", type=Path, default=HERE / "questions.jsonl")
    parser.add_argument("--show", type=int, default=3, help="example ids to print per side of the difference")
    args = parser.parse_args(argv)
    questions = {q["id"]: q for q in load_jsonl(args.questions)}
    first: dict[str, dict] = {}
    for row in load_jsonl(args.run):
        first.setdefault(row["id"], row)
    by_template: dict[str, list[str]] = {}
    for question_id, prediction in first.items():
        question = questions.get(question_id)
        if question is None:
            continue
        scores = question_scores(question, prediction)
        if scores["answer_em"] == 1.0:
            continue
        gold, got = set(map(str, question.get("gold_answer", []))), set(map(str, prediction.get("answer", [])))
        extra, missing = sorted(got - gold), sorted(gold - got)
        line = (
            f"  {question_id}: F1 {scores['answer_f1']:.3f}  gold {len(gold)}  got {len(got)}  "
            f"missing {len(missing)}{(' e.g. ' + ', '.join(missing[:args.show])) if missing else ''}  "
            f"extra {len(extra)}{(' e.g. ' + ', '.join(extra[:args.show])) if extra else ''}"
            + (f"  error={prediction['error']}" if prediction.get("error") else "")
            + ("  (refused)" if prediction.get("refused") else "")
        )
        by_template.setdefault(question["template"], []).append(line)
    if not by_template:
        print("every question matched exactly")
        return 0
    for template, lines in sorted(by_template.items()):
        print(f"{template}: {len(lines)} question(s) not exact")
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
