"""Twenty smoke-test queries over the Yelp data layer: one per feature the benchmark will lean on.

    export YODB_CONN_YELP="host=localhost port=5432 dbname=yelp user=yodb_ro password=yodb_ro"
    PYTHONPATH=src python examples/yelp/smoke.py [-q] [name substrings...]

Each query is planned (``explain``) and run; the plan kind, the row count, the first rows and the
timing are printed, and a query that errors is reported.  They are grouped the way the proposal's
workload is: ``S`` single-source, ``I`` independent multi-source (two datasets joined), ``D``
dependent chains (a traversal whose other side is restricted by what the first side returned),
``X`` stress cases that must be refused.  Exit status is non-zero when a query fails unexpectedly.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yodb
from yodb.embeddings import FastEmbedProvider
from yodb.errors import YoDbError

HERE = Path(__file__).parent


def p(field, op, value=None):
    predicate = {"field": field, "op": op}
    if op not in ("is_null", "is_not_null"):
        predicate["value"] = value
    return predicate


def step(relationship, alias, select=None, where=None, optional=False, direction=None):
    s = {"relationship": relationship, "as": alias}
    if select is not None:
        s["select"] = select
    if where is not None:
        s["where"] = where
    if optional:
        s["optional"] = True
    if direction:
        s["direction"] = direction
    return [s]


def similar(text, top=None, field="text"):
    body = {"field": field, "text": text}
    if top is not None:
        body["top"] = top
    return {"similar": body}


QUERIES = [
    # ---------------- S: single source ----------------
    ("S01 highly rated open businesses in a city",
     {"from": {"dataset": "business"}, "select": ["name", "stars", "review_count"],
      "where": {"all": [p("city", "eq", "Philadelphia"), {"rule": "popular_favorite"}]},
      "order_by": [{"field": "review_count", "direction": "desc"}], "page": {"first": 10}}),
    ("S02 businesses by rating range and state",
     {"from": {"dataset": "business"}, "select": ["name", "city", "stars"],
      "where": {"all": [p("state", "eq", "PA"), p("stars", "gte", 4.0), p("stars", "lt", 4.5)]}, "page": {"first": 10}}),
    ("S03 reviews that are negative and helpful",
     {"from": {"dataset": "review"}, "select": ["stars", "useful", "text"],
      "where": {"all": [{"rule": "negative"}, {"rule": "helpful"}]},
      "order_by": [{"field": "useful", "direction": "desc"}], "page": {"first": 5}}),
    ("S04 users who were ever elite, most fans first",
     {"from": {"dataset": "user"}, "select": ["name", "fans", "elite"], "where": {"rule": "ever_elite"},
      "order_by": [{"field": "fans", "direction": "desc"}], "page": {"first": 10}}),
    ("S05 reviews in a date range",
     {"from": {"dataset": "review"}, "select": ["date", "stars"],
      "where": {"all": [p("date", "gte", "2019-01-01T00:00:00Z"), p("date", "lt", "2019-02-01T00:00:00Z")]},
      "order_by": [{"field": "date", "direction": "asc"}], "page": {"first": 5}}),
    ("S06 the reviews nearest to 'quiet, relaxed atmosphere' (vector search)",
     {"from": {"dataset": "review"}, "select": ["stars", "text"], "where": similar("a quiet, relaxed atmosphere, good for conversation", top=5)}),
    ("S07 nearest positive reviews about slow service (vector search under a condition)",
     {"from": {"dataset": "review"}, "select": ["stars", "text"],
      "where": {"all": [similar("the service was very slow and we waited a long time", top=5), {"rule": "positive"}]}}),
    ("S08 categories whose name contains 'Coffee' (coordinator filter)",
     {"from": {"dataset": "category"}, "where": p("name", "contains", "Coffee"), "page": {"first": 10}}),
    # ---------------- I: independent multi-source (joins over columns) ----------------
    ("I01 reviews of businesses in a city, with the business name (column join)",
     {"from": {"dataset": "review"}, "select": ["stars", "text"], "where": {"rule": "positive"},
      "traverse": step("review_of_business", "business", ["name", "city"], p("city", "eq", "Tucson")),
      "order_by": [{"field": "stars", "direction": "desc"}], "page": {"first": 5}}),
    ("I02 businesses and their reviews, reversed column join",
     {"from": {"dataset": "business"}, "select": ["name"], "where": {"rule": "highly_rated"},
      "traverse": step("review_of_business", "review", ["stars", "useful"], {"rule": "helpful"}, direction="reverse"), "page": {"first": 10}}),
    ("I03 active reviewers and their reviews (column join)",
     {"from": {"dataset": "user"}, "select": ["name", "review_count"], "where": {"rule": "active_reviewer"},
      "traverse": step("user_wrote_review", "review", ["stars"], {"rule": "negative"}), "page": {"first": 10}}),
    ("I04 users with no sampled review kept (left join)",
     {"from": {"dataset": "user"}, "select": ["name"], "where": p("fans", "gte", 100),
      "traverse": step("user_wrote_review", "review", ["stars"], optional=True), "page": {"first": 10}}),
    # ---------------- D: dependent chains (graph edges; the other side is read for the IDs reached) ----------------
    ("D01 the friends of users with many fans (graph edge)",
     {"from": {"dataset": "user"}, "select": ["name", "fans"], "where": p("fans", "gte", 200),
      "traverse": step("user_friends_with", "friend", ["name", "review_count"]), "page": {"first": 10}}),
    ("D02 highly rated businesses a prolific user reviewed (graph edge)",
     {"from": {"dataset": "user"}, "select": ["name"], "where": p("review_count", "gte", 1000),
      "traverse": step("user_reviewed_business", "business", ["name", "stars"], {"rule": "highly_rated"}), "page": {"first": 10}}),
    ("D03 who reviewed a popular business (graph edge, reversed)",
     {"from": {"dataset": "business"}, "select": ["name"], "where": {"rule": "popular_favorite"},
      "traverse": step("user_reviewed_business", "reviewer", ["name", "fans"], direction="reverse"), "page": {"first": 10}}),
    ("D04 the categories of highly rated businesses in a city (graph edge)",
     {"from": {"dataset": "business"}, "select": ["name"], "where": {"all": [p("city", "eq", "Nashville"), {"rule": "highly_rated"}]},
      "traverse": step("business_in_category", "category", ["name"]), "page": {"first": 10}}),
    ("D05 businesses in a category (graph edge, reversed)",
     {"from": {"dataset": "category"}, "where": p("id", "eq", "coffee_tea"),
      "traverse": step("business_in_category", "business", ["name", "city", "stars"], {"rule": "open_now"}, direction="reverse"),
      "order_by": [{"field": "business.stars", "direction": "desc"}], "page": {"first": 10}}),
    ("D06 reviews nearest to 'great tacos' and the businesses they are about (vector search then column join)",
     {"from": {"dataset": "review"}, "select": ["stars"], "where": similar("amazing tacos and fresh salsa", top=5),
      "traverse": step("review_of_business", "business", ["name", "city"])}),
    # ---------------- X: stress cases that must be refused ----------------
    ("X01 an unknown field is refused",
     {"from": {"dataset": "business"}, "where": p("rating", "gte", 4)}, "field_not_found"),
    ("X02 a relationship that does not start here is refused",
     {"from": {"dataset": "category"}, "traverse": step("user_friends_with", "x")}, "relationship_not_applicable"),
]


def main(argv: list[str]) -> int:
    quiet = "-q" in argv
    wanted = [a for a in argv if not a.startswith("-")]
    chosen = [q for q in QUERIES if not wanted or any(w in q[0] for w in wanted)]
    failed = 0
    embedder = FastEmbedProvider() if any("similar" in json.dumps(q[1]) for q in chosen) else None
    with yodb.connect(HERE / "catalog", embedder=embedder) as db:
        print(f"catalog {db.status()['catalog']}; sources {db.status()['sources']}; similar fields {db.similar_fields()}")
        for name, query, *expect in chosen:
            expected_error = expect[0] if expect else None
            print(f"\n=== {name}")
            started = time.perf_counter()
            try:
                explanation = db.explain(query)
                result = db.query(query, timeout_seconds=60)
            except YoDbError as error:
                if expected_error and error.code.value == expected_error:
                    print(f"  PASS refused: [{error.code.value}] {error.detail.message}")
                else:
                    failed += 1
                    print(f"  FAIL [{error.code.value}] at {error.detail.location}: {error.detail.message}")
                continue
            elapsed = (time.perf_counter() - started) * 1000
            if expected_error:
                failed += 1
                print(f"  FAIL expected {expected_error}, got {len(result.rows)} rows")
                continue
            print(f"  plan {explanation.plan_kind}, {len(result.rows)} rows in {elapsed:.0f} ms; "
                  + " | ".join(n.kind for n in explanation.nodes))
            if result.applied.get("rules"):
                print("  rules: " + ", ".join(r["name"] for r in result.applied["rules"]))
            if "similar" in result.reports:
                stats = result.reports["similar"].stats
                print(f"  similar: exact={stats.exact} ranked={stats.candidates} returned={stats.returned}")
            if not quiet:
                for row in result.rows[:3]:
                    print("   ", json.dumps({k: (v[:80] + "..." if isinstance(v, str) and len(v) > 80 else v) for k, v in row.items()}, default=str))
    print(f"\n{len(chosen) - failed}/{len(chosen)} ok")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
