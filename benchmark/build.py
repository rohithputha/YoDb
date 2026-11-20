"""Instantiate the templates against the loaded Yelp database into ``questions.jsonl``.

    export YELP_DSN="host=localhost port=55434 dbname=yelp user=postgres password=postgres"   # any role that reads yelp.*
    PYTHONPATH=src python benchmark/build.py --out benchmark/questions.jsonl [--seed yodb-bench-v1] [--no-embeddings]

For every template: draw its slot values from parameter pools read from the database (the
cities with the most businesses, the most common categories, prolific users, popular
businesses, fixed thresholds and phrases), fill the question text and the gold plan, run
the oracle SQL for the gold answer set and evidence IDs, and assign the question to the
``tune`` or ``held_out`` split (40 / 200, by hash).  Everything is a function of the seed
and the database content.

One line per question::

    {"id": "S01-003", "template": "S01", "class": "single", "split": "held_out",
     "question": "...", "params": {...}, "plan": [...], "sources": [...],
     "answer_field": "id", "answer_dataset": "business", "gold_answer": [...],
     "gold_evidence": {"business": [...]}, "expect_error": null}
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import itertools
import json
import re
import os
import random
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
ROOT = HERE.parent.parent if HERE.parent.name == "examples" else HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from yodb.dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from benchmark.templates import PAGE, TEMPLATES, TUNE_FRACTION, Template  # noqa: E402

DEFAULT_SEED = "yodb-bench-v1"
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")

# Pools that do not come from the data.
FIXED_POOLS: dict[str, list[Any]] = {
    "stars_hi": [4.0, 4.5],
    "stars_lo": [1.5, 2.0, 2.5],
    "review_count_min": [50, 100, 200],
    "user_reviews_min": [100, 250, 500],
    "useful_min": [3, 5, 10],
    "fans_min": [20, 50, 100],
    "fans_min_hi": [100, 200],
    "fans_min_elite": [200, 300, 500],
    "year": [2017, 2018, 2019],
    "month": [1, 4, 7, 10],
    "word": ["Coffee", "Bar", "Pizza", "Thai", "Mexican", "Vegan"],
    "phrase": [
        "a quiet, relaxed atmosphere, good for conversation",
        "the service was very slow and we waited a long time",
        "amazing tacos and fresh salsa",
        "friendly and attentive staff",
        "overpriced for what you get",
        "excellent coffee and fresh pastries",
        "live music and a lively bar scene",
        "the place was dirty and smelled bad",
        "perfect for a romantic date night",
        "great for kids and families",
        "huge portions, more than enough to share",
        "parking was impossible to find",
    ],
}


# --- pools from the database ----------------------------------------------------------------------------


def read_pools(connection) -> dict[str, list[Any]]:
    """The data-dependent parameter pools: the most useful values of each kind, in a stable order."""

    pools: dict[str, list[Any]] = {name: list(values) for name, values in FIXED_POOLS.items()}
    with connection.cursor() as cursor:
        cursor.execute("SELECT city FROM yelp.business WHERE city IS NOT NULL GROUP BY city ORDER BY count(*) DESC, city LIMIT 6")
        pools["city"] = [row[0] for row in cursor.fetchall()]
        cursor.execute(
            "SELECT bc.category_id, c.name FROM yelp.business_category bc JOIN yelp.category c ON c.category_id = bc.category_id "
            "GROUP BY bc.category_id, c.name ORDER BY count(*) DESC, bc.category_id LIMIT 12"
        )
        pools["category"] = [{"category": row[0], "category_name": row[1]} for row in cursor.fetchall()]
        cursor.execute(
            "SELECT u.user_id FROM yelp.users u WHERE EXISTS (SELECT 1 FROM yelp.friendship f WHERE f.user_id = u.user_id) "
            "AND EXISTS (SELECT 1 FROM yelp.review r WHERE r.user_id = u.user_id) ORDER BY u.review_count DESC, u.user_id LIMIT 40"
        )
        pools["user_id"] = [row[0] for row in cursor.fetchall()]
        cursor.execute(
            "SELECT b.name, b.city FROM yelp.business b WHERE b.review_count >= 100 AND (SELECT count(*) FROM yelp.review r WHERE r.business_id = b.business_id) >= 5 "
            "AND (SELECT count(*) FROM yelp.business b2 WHERE b2.name = b.name AND b2.city = b.city) = 1 "
            "ORDER BY b.review_count DESC, b.business_id LIMIT 40"
        )
        pools["business"] = [{"business_name": row[0], "business_city": row[1]} for row in cursor.fetchall()]
    return pools


# --- instantiation (pure) ----------------------------------------------------------------------------------


def derive(params: dict[str, Any]) -> dict[str, Any]:
    """Slots computed from others: month/year boundaries and names."""

    derived = dict(params)
    if "year" in derived:
        year = int(derived["year"])
        derived["year_start"] = f"{year}-01-01T00:00:00Z"
        derived["year_end"] = f"{year + 1}-01-01T00:00:00Z"
        if "month" in derived:
            month = int(derived["month"])
            derived["month_name"] = MONTHS[month - 1]
            derived["month_start"] = f"{year}-{month:02d}-01T00:00:00Z"
            next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
            derived["month_end"] = f"{next_year}-{next_month:02d}-01T00:00:00Z"
    return derived


def instances_of(template: Template, pools: Mapping[str, list[Any]], seed: str) -> list[dict[str, Any]]:
    """The slot assignments of a template: distinct combinations, in a seeded order, ``template.instances`` of them.

    A pool item that is a dict (``business``, ``category``) contributes several slots at once.
    """

    if not template.slots:
        return [{} for _ in range(template.instances)]
    choices = [list(pools[slot]) for slot in template.slots]
    for slot, values in zip(template.slots, choices):
        if not values:
            raise ValueError(f"template {template.id}: parameter pool '{slot}' is empty")
    combinations = list(itertools.product(*choices))
    rng = random.Random(hashlib.sha256(f"{seed}\0{template.id}".encode()).hexdigest())
    rng.shuffle(combinations)
    chosen = combinations[: template.instances]
    while len(chosen) < template.instances:               # fewer combinations than instances: repeat deterministically
        chosen.append(combinations[len(chosen) % len(combinations)])
    assignments = []
    for combination in chosen:
        params: dict[str, Any] = {}
        for slot, value in zip(template.slots, combination):
            if isinstance(value, dict):
                params.update(value)
            else:
                params[slot] = value
        assignments.append(derive(params))
    return assignments


def fill(value: Any, params: Mapping[str, Any]) -> Any:
    """Substitute ``{slot}`` placeholders in a plan: a bare placeholder keeps the slot's type, text keeps text."""

    if isinstance(value, str):
        if value.startswith("{") and value.endswith("}") and value[1:-1] in params:
            return params[value[1:-1]]
        for slot, replacement in params.items():
            value = value.replace("{" + slot + "}", str(replacement))
        return value
    if isinstance(value, dict):
        return {key: fill(item, params) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [fill(item, params) for item in value]
    return value


def split_of(question_id: str, seed: str) -> str:
    digest = hashlib.sha256(f"{seed}\0split\0{question_id}".encode()).hexdigest()
    return "tune" if int(digest[:8], 16) / 0xFFFFFFFF < TUNE_FRACTION else "held_out"


def question_text(template: Template, params: Mapping[str, Any]) -> str:
    return fill(template.question, {**template.extra_params, **params})


def instantiate(template: Template, index: int, params: dict[str, Any], seed: str) -> dict[str, Any]:
    """One question record without its gold answer (added by :func:`with_gold`)."""

    question_id = f"{template.id}-{index:03d}"
    all_params = {**template.extra_params, **params}
    return {
        "id": question_id,
        "template": template.id,
        "class": template.cls,
        "split": split_of(question_id, seed),
        "question": question_text(template, params),
        "params": params,
        "plan": [fill(step, all_params) for step in template.plan],
        "sources": list(template.sources),
        "answer_field": template.answer,
        "answer_dataset": template.answer_dataset,
        "expect_error": template.expect_error,
        "notes": template.notes,
    }


# --- gold answers from the oracle ----------------------------------------------------------------------


def with_gold(record: dict[str, Any], template: Template, connection, embedder=None) -> dict[str, Any]:
    """Run the oracle SQL and attach the gold answer set and evidence IDs."""

    if template.oracle is None:
        return {**record, "gold_answer": [], "gold_evidence": {}}
    params: dict[str, Any] = {**template.extra_params, **record["params"]}
    if "%(vector)s" in template.oracle:
        if embedder is None:
            raise ValueError(f"template {template.id} needs the embedding model for its oracle; pass --no-embeddings to skip it")
        vector = embedder.embed_texts([params["phrase"]]).__next__()
        params["vector"] = "[" + ",".join(repr(x) for x in vector) + "]"
    with connection.cursor() as cursor:
        cursor.execute(_one_past_the_page(template.oracle), params)
        names = [column.name for column in cursor.description]
        rows = cursor.fetchall()
    complete = len(rows) <= PAGE                  # the page holds the whole result: no row was cut off
    rows = rows[:PAGE]
    answers: dict[str, None] = {}
    evidence: dict[str, dict[str, None]] = {}
    for row in rows:
        values = dict(zip(names, row))
        answers[str(values["answer_id"])] = None
        for name, value in values.items():
            if name.startswith("evidence_") and value is not None:
                evidence.setdefault(name[len("evidence_"):], {})[str(value)] = None
    if template.answer_dataset:
        evidence.setdefault(template.answer_dataset, {}).update(answers)
    return {**record, "gold_answer": list(answers), "gold_evidence": {dataset: list(ids) for dataset, ids in evidence.items()}, "complete": complete}


_PAGE_LIMIT = re.compile(rf"LIMIT {PAGE}\s*$")


def _one_past_the_page(oracle: str) -> str:
    """The oracle asking for one row more than the page, to tell whether the page cut the result."""

    return _PAGE_LIMIT.sub(f"LIMIT {PAGE + 1}", oracle.rstrip())


OVERSAMPLE = 3     # candidate instances drawn per template, so ones with an empty or cut-off gold answer can be passed over


def build(connection, *, seed: str = DEFAULT_SEED, embedder=None, templates=TEMPLATES, pools=None) -> list[dict[str, Any]]:
    """Instantiate every template; an instance whose gold answer is empty, or whose result is larger than
    the page (so the answer set would depend on which 500 rows the page kept), is passed over while a
    later candidate is complete and non-empty.  The ones kept anyway carry ``complete: false``."""

    pools = read_pools(connection) if pools is None else pools
    questions = []
    for template in templates:
        wanted = template.instances
        candidates = instances_of(Template(**{**template.__dict__, "instances": wanted * OVERSAMPLE}), pools, seed)
        kept: list[dict[str, Any]] = []
        spare: list[dict[str, Any]] = []
        for params in candidates:
            if len(kept) >= wanted:
                break
            record = with_gold(instantiate(template, 0, params, seed), template, connection, embedder)
            if template.expect_error is not None or (record["gold_answer"] and record["complete"]):
                kept.append(record)
            else:
                spare.append(record)
        spare.sort(key=lambda r: (not r["complete"], not r["gold_answer"]))     # cut-off before empty is still better than nothing
        kept.extend(spare[: wanted - len(kept)])
        for index, record in enumerate(kept, start=1):
            question_id = f"{template.id}-{index:03d}"
            questions.append({**record, "id": question_id, "split": split_of(question_id, seed)})
    return questions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dsn", default=os.environ.get("YELP_DSN"), help="a connection that can read yelp.* (or env YELP_DSN)")
    parser.add_argument("--out", type=Path, default=HERE / "questions.jsonl")
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--no-embeddings", action="store_true", help="leave out the templates that need the embedding model")
    args = parser.parse_args(argv)
    if not args.dsn:
        parser.error("--dsn (or YELP_DSN) is required")
    import psycopg

    embedder = None
    templates = TEMPLATES
    if args.no_embeddings:
        templates = tuple(t for t in TEMPLATES if t.oracle is None or "%(vector)s" not in t.oracle)
    else:
        from yodb.embeddings import FastEmbedProvider

        embedder = FastEmbedProvider(args.embedding_model)
    with psycopg.connect(args.dsn) as connection:
        questions = build(connection, seed=args.seed, embedder=embedder, templates=templates)
    with args.out.open("w", encoding="utf-8") as handle:
        for question in questions:
            handle.write(json.dumps(question, ensure_ascii=False) + "\n")
    classes: dict[str, int] = {}
    splits: dict[str, int] = {}
    empty = cut = 0
    for question in questions:
        classes[question["class"]] = classes.get(question["class"], 0) + 1
        splits[question["split"]] = splits.get(question["split"], 0) + 1
        if question["expect_error"] is None and not question["gold_answer"]:
            empty += 1
        if not question.get("complete", True):
            cut += 1
    print(f"wrote {len(questions)} questions to {args.out}: {classes}; splits {splits}; {empty} with an empty gold answer; "
          f"{cut} whose result is larger than the {PAGE}-row page" + (" (see benchmark/README.md: those answers depend on the page's order)" if cut else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
