"""The shop with its customer-ticket relationship stored as edges of an Apache AGE graph (no database needed).

``graph_world`` serves the relational reads from in-memory SQL and answers the compiled
Cypher-in-SQL edge reads from an in-memory edge list, returning agtype text exactly as
AGE would (``"c1"``), so the executor's decoding is exercised too.
"""

from __future__ import annotations

import json
import re

from yodb.catalog import SourceKind

from support.shop import ASSIGNMENTS, CUSTOMERS, DATASETS, PLANS, TICKETS
from support.sql_world import SqlWorld, evaluation

SOURCES = """\
api_version: yodb/v0.2
sources:
  crm:
    kind: postgres
    connection_ref: crm
    read_only: true
    datasets:
      customer:
        resource: crm.customers
        identity: [id]
        fields:
          id: {physical_name: customer_id}
          name: {physical_name: name}
          country: {physical_name: country}
          referrer_id: {physical_name: referrer_id}
          seats: {physical_name: seats}
  billing:
    kind: postgres
    connection_ref: billing
    read_only: true
    datasets:
      customer:
        resource: billing.plans
        identity: [id]
        fields:
          id: {physical_name: customer_id}
          plan: {physical_name: plan}
  desk:
    kind: postgres
    connection_ref: desk
    read_only: true
    datasets:
      ticket:
        resource: desk.tickets
        identity: [id]
        fields:
          id: {physical_name: ticket_id}
          customer_id: {physical_name: customer_id}
          subject: {physical_name: subject}
          status: {physical_name: status}
          priority: {physical_name: priority}
          legacy_key: {physical_name: legacy_key}
          body: {physical_name: body}
  triage:
    kind: postgres
    connection_ref: triage
    read_only: true
    datasets:
      ticket:
        resource: triage.assignments
        identity: [id]
        fields:
          id: {physical_name: ticket_id}
          assignee: {physical_name: assignee}
  graphdb:
    kind: postgres
    connection_ref: graphdb
    read_only: true
    graph:
      name: shop_graph
      vertices:
        customer: {label: Customer, id_property: id}
        ticket: {label: Ticket, id_property: id}
resolution:
  customer:
    identity_source: crm
    field_sources: {id: crm, name: crm, country: crm, referrer_id: crm, seats: crm, plan: billing}
  ticket:
    identity_source: desk
    field_sources: {id: desk, customer_id: desk, subject: desk, status: desk, priority: desk, legacy_key: desk, body: desk, assignee: triage}
"""

RELATIONS = """\
api_version: yodb/v0.2
relationships:
  customer_has_ticket:
    from: customer
    to: ticket
    description: A ticket belongs to a customer (stored as graph edges only).
    aliases: [tickets]
    cardinality: one_to_many
    direction: bi
    implementations:
      - from: {source: graphdb, field: id}
        to: {source: graphdb, field: id}
        edge_type: HAS_TICKET
  referred_by:
    from: customer
    to: customer
    description: A customer was referred by another (graph edges only).
    cardinality: many_to_one
    direction: uni
    implementations:
      - from: {source: graphdb, field: id}
        to: {source: graphdb, field: id}
        edge_type: REFERRED_BY
  customer_has_ticket_by_field:
    from: customer
    to: ticket
    description: The same link, also declared as a field join (the field join is preferred).
    cardinality: one_to_many
    direction: uni
    implementations:
      - from: {source: graphdb, field: id}
        to: {source: graphdb, field: id}
        edge_type: HAS_TICKET
      - from: {source: crm, field: id}
        to: {source: desk, field: customer_id}
"""

# The edges, as the graph holds them: HAS_TICKET follows ``customer_id`` of the tickets (with one extra edge to a
# ticket whose row says another customer, so the graph, not the column, decides), REFERRED_BY follows ``referrer_id``.
HAS_TICKET = [(customer, ticket) for ticket, customer, *_ in TICKETS if customer is not None] + [("c7", "t10")]
REFERRED_BY = [(customer, referrer) for customer, _, _, referrer, _ in CUSTOMERS if referrer is not None]
EDGES = {"HAS_TICKET": HAS_TICKET, "REFERRED_BY": REFERRED_BY}

_PATTERN = re.compile(r"\((a|b):(\w+)\)-\[:(\w+)\]->\((a|b):(\w+)\)")


def graph_active():
    return evaluation(DATASETS, SOURCES, RELATIONS)


class GraphWorld:
    """SQL reads on SQLite; edge reads from ``EDGES``, answered as AGE answers them."""

    source_kind = SourceKind.POSTGRES

    def __init__(self, edges: dict[str, list[tuple[str, str]]] | None = None) -> None:
        self.sql = SqlWorld({
            "crm.customers": (("customer_id", "name", "country", "referrer_id", "seats"), CUSTOMERS),
            "billing.plans": (("customer_id", "plan"), PLANS),
            "desk.tickets": (("ticket_id", "customer_id", "subject", "status", "priority", "legacy_key", "body"), TICKETS),
            "triage.assignments": (("ticket_id", "assignee"), ASSIGNMENTS),
        })
        self.edges = EDGES if edges is None else edges
        self.edge_queries = []

    @property
    def queries(self):
        return self.sql.queries

    def execute(self, query, *, timeout_seconds=None):
        if query.requires_extension != "age":
            return self.sql.execute(query, timeout_seconds=timeout_seconds)
        self.edge_queries.append(query)
        match = _PATTERN.search(query.sql)
        assert match, query.sql
        left_var, _, edge_type, right_var, _ = match.groups()
        ids = json.loads(query.parameters[0])["ids"]
        anchored_from = left_var == "a"             # (a)-[]->(b): the given IDs are the "from" end
        rows = []
        for start, end in self.edges[edge_type]:
            given, other = (start, end) if anchored_from else (end, start)
            if given in ids:
                rows.append({"start_id": json.dumps(given), "end_id": json.dumps(other)})
        limit = query.parameters[1] if len(query.parameters) > 1 else None
        return tuple(rows if limit is None else rows[:limit])
