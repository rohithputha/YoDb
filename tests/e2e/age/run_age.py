"""End-to-end run against a real PostgreSQL with Apache AGE and pgvector (see README.md next to this file).

Opens the catalog in ``catalog/`` through ``yodb.connect`` (so the live inspector validates the
graph, its labels and edge types), then runs graph traversals and similarity searches and compares
every answer with native SQL over ``shop.links`` and ``shop.tickets`` (the harness's oracle; YoDb
never sees those joins).  Exit status is non-zero when any case fails.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import psycopg

import yodb
from yodb.errors import YoDbError
from yodb.semantic import EmbeddingResult, ProviderInfo

HERE = Path(__file__).parent
CONNINFO = os.environ.get("YODB_E2E_CONNINFO", "host=localhost port=55433 dbname=yodb_age user=yodb_ro password=yodb_ro")
VOCABULARY = ("price", "cancel", "refund", "bug")


class ToyEmbedder:
    """Keyword counts + bias; identical to shop.toy_embed in seed.sql."""

    info = ProviderInfo("toy", "toy-v1", "1")
    dimensions = 5

    def embed(self, request):
        vectors = tuple(tuple(float(t.lower().count(w)) for w in VOCABULARY) + (1.0,) for t in request.texts)
        return EmbeddingResult(vectors, self.info, self.dimensions)


def toy_vector(text: str) -> str:
    return "[" + ",".join(str(float(text.lower().count(w))) for w in VOCABULARY) + ",1.0]"


@dataclass
class Case:
    name: str
    query: dict
    oracle: str | None = None          # native SQL returning the expected rows, in order, with the same column names
    params: tuple = ()
    expect_error: str | None = None    # an error code, when the query must be refused
    check: object = None               # extra assertions on the result: (result) -> str | None
    note: str = ""
    failures: list = field(default_factory=list)


def p(field_, op, value=None):
    predicate = {"field": field_, "op": op}
    if op not in ("is_null", "is_not_null"):
        predicate["value"] = value
    return predicate


def step(relationship="customer_has_ticket", alias="ticket", select=None, where=None, optional=False, direction=None):
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


def similar(text, top=None):
    body = {"field": "body", "text": text}
    if top is not None:
        body["top"] = top
    return {"similar": body}


CUSTOMER_COLUMNS = "c.customer_id AS id, c.name, c.country"
TICKET_JOIN = "JOIN shop.links l ON l.edge_type = 'HAS_TICKET' AND l.from_id = c.customer_id JOIN shop.tickets t ON t.ticket_id = l.to_id"

CASES = [
    # ---------------- graph traversal (Apache AGE) ----------------
    Case("G01 forward traversal, filter on the root",
         {"from": {"dataset": "customer"}, "select": ["name", "country"], "where": p("country", "eq", "US"),
          "traverse": step(select=["subject", "status"])},
         f"SELECT {CUSTOMER_COLUMNS}, t.ticket_id AS \"ticket.id\", t.subject AS \"ticket.subject\", t.status AS \"ticket.status\" "
         f"FROM shop.customers c {TICKET_JOIN} WHERE c.country = 'US' ORDER BY c.customer_id, t.ticket_id"),
    Case("G02 a link that exists only in the graph (c07 -> t10)",
         {"from": {"dataset": "customer"}, "select": ["name"], "where": p("name", "eq", "Gus"), "traverse": step(select=["subject"])},
         "SELECT c.customer_id AS id, c.name, t.ticket_id AS \"ticket.id\", t.subject AS \"ticket.subject\" "
         f"FROM shop.customers c {TICKET_JOIN} WHERE c.name = 'Gus' ORDER BY c.customer_id, t.ticket_id"),
    Case("G03 filters on both sides, order by a joined field, page",
         {"from": {"dataset": "customer"}, "select": ["name"], "where": p("seats", "gte", 5),
          "traverse": step(select=["priority"], where=p("status", "eq", "open")),
          "order_by": [{"field": "ticket.priority", "direction": "desc"}, {"field": "name", "direction": "asc"}], "page": {"first": 4}},
         "SELECT c.customer_id AS id, c.name, t.ticket_id AS \"ticket.id\", t.priority AS \"ticket.priority\" "
         f"FROM shop.customers c {TICKET_JOIN} WHERE c.seats >= 5 AND t.status = 'open' "
         "ORDER BY t.priority DESC, c.name ASC, c.customer_id, t.ticket_id LIMIT 4"),
    Case("G04 left traversal keeps customers without an edge",
         {"from": {"dataset": "customer"}, "select": ["name"], "traverse": step(select=["subject"], optional=True)},
         "SELECT c.customer_id AS id, c.name, t.ticket_id AS \"ticket.id\", t.subject AS \"ticket.subject\" "
         "FROM shop.customers c LEFT JOIN shop.links l ON l.edge_type = 'HAS_TICKET' AND l.from_id = c.customer_id "
         "LEFT JOIN shop.tickets t ON t.ticket_id = l.to_id ORDER BY c.customer_id, t.ticket_id"),
    Case("G05 reverse traversal from the ticket end",
         {"from": {"dataset": "ticket"}, "select": ["subject"], "where": p("priority", "gte", 4),
          "traverse": step(alias="customer", select=["name"], direction="reverse")},
         "SELECT t.ticket_id AS id, t.subject, c.customer_id AS \"customer.id\", c.name AS \"customer.name\" "
         "FROM shop.tickets t JOIN shop.links l ON l.edge_type = 'HAS_TICKET' AND l.to_id = t.ticket_id "
         "JOIN shop.customers c ON c.customer_id = l.from_id WHERE t.priority >= 4 ORDER BY t.ticket_id, c.customer_id"),
    Case("G06 self traversal (referred_by)",
         {"from": {"dataset": "customer"}, "select": ["name"], "traverse": step(relationship="referred_by", alias="referrer", select=["name"])},
         "SELECT c.customer_id AS id, c.name, r.customer_id AS \"referrer.id\", r.name AS \"referrer.name\" "
         "FROM shop.customers c JOIN shop.links l ON l.edge_type = 'REFERRED_BY' AND l.from_id = c.customer_id "
         "JOIN shop.customers r ON r.customer_id = l.to_id ORDER BY c.customer_id, r.customer_id"),
    Case("G07 the traversed side drives when it is far more selective",
         {"from": {"dataset": "customer"}, "select": ["name"], "traverse": step(select=["subject"], where=p("subject", "eq", "Outage"))},
         "SELECT c.customer_id AS id, c.name, t.ticket_id AS \"ticket.id\", t.subject AS \"ticket.subject\" "
         f"FROM shop.customers c {TICKET_JOIN} WHERE t.subject = 'Outage' ORDER BY c.customer_id, t.ticket_id"),
    Case("G08 an unknown relationship is refused",
         {"from": {"dataset": "customer"}, "traverse": step(relationship="nope")}, expect_error="relationship_not_found"),
    # ---------------- similarity search (pgvector) ----------------
    Case("V01 the nearest tickets to a text, nearest first",
         {"from": {"dataset": "ticket"}, "select": ["subject"], "where": similar("refund the price", top=5)},
         "SELECT ticket_id AS id, subject FROM shop.tickets WHERE body_embedding IS NOT NULL "
         "ORDER BY body_embedding <=> %s::vector, ticket_id LIMIT 5", (toy_vector("refund the price"),),
         check=lambda result: None if [r.rank for r in result.reports["similar"].records.values()] and result.reports["similar"].stats.exact else "no ranks"),
    Case("V02 among the open tickets only (the condition is applied before ranking)",
         {"from": {"dataset": "ticket"}, "select": ["subject", "status"], "where": {"all": [similar("cancel because of a bug", top=3), p("status", "eq", "open")]}},
         "SELECT ticket_id AS id, subject, status FROM shop.tickets WHERE body_embedding IS NOT NULL AND status = 'open' "
         "ORDER BY body_embedding <=> %s::vector, ticket_id LIMIT 3", (toy_vector("cancel because of a bug"),)),
    Case("V03 the top rows sorted by the caller's order",
         {"from": {"dataset": "ticket"}, "select": ["subject", "priority"], "where": similar("price", top=4),
          "order_by": [{"field": "priority", "direction": "desc"}]},
         "SELECT * FROM (SELECT ticket_id AS id, subject, priority FROM shop.tickets WHERE body_embedding IS NOT NULL "
         "ORDER BY body_embedding <=> %s::vector, ticket_id LIMIT 4) nearest ORDER BY priority DESC, id", (toy_vector("price"),)),
    Case("V04 top defaults to the page",
         {"from": {"dataset": "ticket"}, "select": ["subject"], "where": similar("bug"), "page": {"first": 2}},
         "SELECT ticket_id AS id, subject FROM shop.tickets WHERE body_embedding IS NOT NULL "
         "ORDER BY body_embedding <=> %s::vector, ticket_id LIMIT 2", (toy_vector("bug"),)),
    Case("V05 a similarity search on the root of a traversal keeps its ranking",
         {"from": {"dataset": "ticket"}, "select": ["subject"], "where": similar("refund", top=3),
          "traverse": step(alias="customer", select=["name"], direction="reverse")},
         "SELECT n.id, n.subject, c.customer_id AS \"customer.id\", c.name AS \"customer.name\" FROM "
         "(SELECT ticket_id AS id, subject, body_embedding <=> %s::vector AS d FROM shop.tickets WHERE body_embedding IS NOT NULL "
         "ORDER BY d, ticket_id LIMIT 3) n JOIN shop.links l ON l.edge_type = 'HAS_TICKET' AND l.to_id = n.id "
         "JOIN shop.customers c ON c.customer_id = l.from_id ORDER BY n.d, n.id, c.customer_id", (toy_vector("refund"),)),
    Case("V06 a search with contains cannot be ranked exactly and is refused",
         {"from": {"dataset": "ticket"}, "where": {"all": [similar("bug"), p("subject", "contains", "a")]}},
         expect_error="query_plan_unsupported"),
    Case("V07 a search in a traverse step is refused",
         {"from": {"dataset": "customer"}, "traverse": step(where=similar("bug"))}, expect_error="query_feature_not_supported"),
]


def run_oracle(connection, sql: str, params: tuple) -> list[dict]:
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        names = [column.name for column in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]


def main(argv: list[str]) -> int:
    quiet = "-q" in argv
    wanted = [a for a in argv if not a.startswith("-")]
    cases = [c for c in CASES if not wanted or any(w in c.name for w in wanted)]
    failed = 0
    started = time.perf_counter()
    with psycopg.connect(CONNINFO, autocommit=True) as oracle, yodb.connect(
        HERE / "catalog", {"e2e-shop": CONNINFO}, embedder=ToyEmbedder()
    ) as db:
        print(f"catalog: {db.status()['sources']}   similar fields: {db.similar_fields()}")
        for case in cases:
            print(f"\n=== {case.name}")
            try:
                explanation = db.explain(case.query)
                if not quiet:
                    print(f"  plan {explanation.plan_kind}: " + " | ".join(
                        f"{n.kind}@{n.location}" + (f" [{'; '.join(n.detail)}]" if n.detail else "") for n in explanation.nodes))
                result = db.query(case.query, timeout_seconds=30)
            except YoDbError as error:
                if case.expect_error and error.code.value == case.expect_error:
                    print(f"  PASS refused as expected: [{error.code.value}] {error.detail.message}")
                    continue
                print(f"  FAIL error: [{error.code.value}] {error.detail.message}")
                failed += 1
                continue
            if case.expect_error:
                print(f"  FAIL expected error {case.expect_error}, got {len(result.rows)} rows")
                failed += 1
                continue
            got = [dict(row) for row in result.rows]
            expected = run_oracle(oracle, case.oracle, case.params)
            if not quiet:
                for row in got:
                    print("   ", json.dumps(row, default=str))
                if "similar" in result.reports:
                    report = result.reports["similar"]
                    print(f"  similar: exact={report.stats.exact} ranked={report.stats.candidates} returned={report.stats.returned} "
                          + " ".join(f"{i}:{r.distance:.3f}" for i, r in sorted(report.records.items(), key=lambda x: x[1].rank)))
            problems = []
            if got != expected:
                problems.append(f"rows differ: got {len(got)}, expected {len(expected)}\n    got      {got}\n    expected {expected}")
            if case.check is not None:
                problem = case.check(result)
                if problem:
                    problems.append(problem)
            if problems:
                failed += 1
                print("  FAIL " + "\n  FAIL ".join(problems))
            else:
                print(f"  PASS ({len(got)} rows)")
    print(f"\n{len(cases) - failed}/{len(cases)} passed in {time.perf_counter() - started:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
