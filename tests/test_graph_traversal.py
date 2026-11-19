"""Traversing a relationship stored as edges of an Apache AGE graph.

The relational sides run on in-memory SQL; the edge reads are answered from an in-memory
edge list as AGE answers them (agtype text).  Every join result is compared with a join
computed in plain Python over the same edges.
"""

from __future__ import annotations

import json
import unittest

from yodb.catalog import CatalogValidationError, SourceKind, load_catalog
from yodb.compilation import PostgresQueryCompiler, QueryCompilerRegistry
from yodb.errors import ErrorCode, QueryError, QueryExecutionError
from yodb.execution import QueryExecutionAdapterRegistry, QueryExecutionEngine
from yodb.execution.operators.edges import decode_agtype
from yodb.inspection import PostgresCatalogValidator, SourceInspection
from yodb.mcp_server import describe_catalog
from yodb.operators import OPERATORS, OperatorKind
from yodb.planning import (
    EdgeJoin,
    FederatedPhysicalPlanner,
    GraphEdgeScan,
    JoinPolicy,
    POSTGRES_CAPABILITIES,
    PostgresPlanningAdapter,
    SourcePlanningRegistry,
    SourceStatistics,
    StatisticsService,
    operator_support,
)

from support.catalogs import StaticRuntime
from support.graph import DATASETS, EDGES, RELATIONS, SOURCES, GraphWorld, graph_active
from support.shop import customer_rows, ticket_rows
from support.statistics import MapProvider
from support.sql_world import evaluation
from test_joins import p, query, rows_of, traverse

ACTIVE = graph_active()


def engine(world, *, statistics=None, join_policy=JoinPolicy(), active=ACTIVE):
    planner = FederatedPhysicalPlanner(SourcePlanningRegistry([PostgresPlanningAdapter()]), statistics=statistics)
    return QueryExecutionEngine(
        StaticRuntime(active),
        QueryCompilerRegistry([PostgresQueryCompiler()]),
        QueryExecutionAdapterRegistry([world]),
        planner=planner,
        statistics=statistics,
        join_policy=join_policy,
    )


def edge_oracle(left_rows, right_rows, edges, *, left_ok=lambda r: True, right_ok=lambda r: True, optional=False,
                alias="ticket", left_columns, right_columns, order=(), first=100):
    """The plain-Python join over an edge list: a pair matches when (left id, right id) is an edge."""

    from functools import cmp_to_key

    from test_joins import _compare

    linked = {}
    for start, end in edges:
        linked.setdefault(start, []).append(end)
    by_id = {row["id"]: row for row in right_rows}
    pairs = []
    for left in left_rows:
        if not left_ok(left):
            continue
        matches = [by_id[end] for end in dict.fromkeys(linked.get(left["id"], ())) if end in by_id and right_ok(by_id[end])]
        if not matches and optional:
            pairs.append((left, None))
        pairs.extend((left, m) for m in matches)

    def value(pair, column):
        left, right = pair
        if column.startswith(alias + "."):
            return None if right is None else right.get(column[len(alias) + 1:])
        return left.get(column)

    terms = [*order, ("id", False), (f"{alias}.id", False)]

    def cmp(a, b):
        for column, descending in terms:
            result = _compare(value(a, column), value(b, column), descending)
            if result:
                return result
        return 0

    pairs.sort(key=cmp_to_key(cmp))
    columns = [*left_columns, *(f"{alias}.{c}" for c in right_columns)]
    return [{c: value(pair, c) for c in columns} for pair in pairs[:first]]


def catalog_from(sources: str = SOURCES, relations: str = RELATIONS, datasets: str = DATASETS):
    return evaluation(datasets, sources, relations).catalog


class CatalogTests(unittest.TestCase):
    def test_a_graph_source_with_no_tables_loads_and_the_edge_implementation_is_a_graph_one(self) -> None:
        from yodb.catalog import graph_implementation

        catalog = catalog_from()
        self.assertEqual(catalog.sources["graphdb"].datasets, {})
        self.assertEqual(catalog.sources["graphdb"].graph.vertices["ticket"].label, "Ticket")
        self.assertTrue(graph_implementation(catalog, catalog.relationships["customer_has_ticket"].implementations[0]))

    def test_an_edge_end_must_be_the_dataset_identity(self) -> None:
        bad = RELATIONS.replace("to: {source: graphdb, field: id}\n        edge_type: HAS_TICKET", "to: {source: graphdb, field: customer_id}\n        edge_type: HAS_TICKET", 1)
        with self.assertRaisesRegex(CatalogValidationError, "must be 'id' for a graph edge"):
            catalog_from(relations=bad)

    def test_both_datasets_of_an_edge_need_a_vertex_label(self) -> None:
        bad = SOURCES.replace("        ticket: {label: Ticket, id_property: id}\n", "")
        with self.assertRaisesRegex(CatalogValidationError, "declares no graph vertex for dataset 'ticket'"):
            catalog_from(sources=bad)

    def test_graph_names_must_be_plain_identifiers(self) -> None:
        for bad, message in (
            (SOURCES.replace("name: shop_graph", "name: shop-graph"), "letters, digits and underscores"),
            (SOURCES.replace("label: Ticket", "label: 'Ticket; DROP'"), "letters, digits and underscores"),
            (SOURCES.replace("id_property: id}", "id_property: 'a b'}"), "letters, digits and underscores"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(CatalogValidationError, message):
                catalog_from(sources=bad)
        with self.assertRaisesRegex(CatalogValidationError, "edge_type"):
            catalog_from(relations=RELATIONS.replace("edge_type: HAS_TICKET", "edge_type: 'HAS TICKET'"))

    def test_a_graph_is_only_declared_on_a_postgres_source_and_its_datasets_must_exist(self) -> None:
        with self.assertRaisesRegex(CatalogValidationError, "only supported on a postgres source"):
            catalog_from(sources=SOURCES.replace("  graphdb:\n    kind: postgres", "  graphdb:\n    kind: neo4j"))
        with self.assertRaisesRegex(CatalogValidationError, "unknown dataset 'invoice'"):
            catalog_from(sources=SOURCES.replace("        ticket: {label: Ticket, id_property: id}", "        invoice: {label: Ticket, id_property: id}"))
        with self.assertRaisesRegex(CatalogValidationError, "stands for both"):
            catalog_from(sources=SOURCES.replace("label: Ticket", "label: Customer"))

    def test_a_source_without_a_graph_still_needs_datasets(self) -> None:
        bad = SOURCES.replace("    graph:\n      name: shop_graph\n      vertices:\n        customer: {label: Customer, id_property: id}\n        ticket: {label: Ticket, id_property: id}\n", "")
        with self.assertRaisesRegex(CatalogValidationError, "datasets must not be empty"):
            catalog_from(sources=bad)

    def test_the_fingerprint_covers_the_graph(self) -> None:
        from yodb.query.fingerprint import catalog_fingerprint

        self.assertNotEqual(
            catalog_fingerprint(catalog_from()), catalog_fingerprint(catalog_from(sources=SOURCES.replace("label: Ticket", "label: Tickets")))
        )


class CompilerTests(unittest.TestCase):
    def scan(self, **changes):
        base = dict(
            source_name="graphdb", source_kind=SourceKind.POSTGRES, connection_ref="graphdb", graph="shop_graph",
            edge_type="HAS_TICKET", from_label="Customer", from_id_property="id", to_label="Ticket", to_id_property="tid",
            anchored="from", ids=("c1", "c2"), maximum_rows=50,
        )
        return GraphEdgeScan(**{**base, **changes})

    def test_the_cypher_runs_inside_sql_with_the_ids_as_its_one_parameter(self) -> None:
        compiled = PostgresQueryCompiler().compile_edges(self.scan())
        self.assertEqual(compiled.requires_extension, "age")
        self.assertIn("FROM ag_catalog.cypher('shop_graph', $yodb_cypher$ UNWIND $ids AS yodb_id MATCH (a:Customer)-[:HAS_TICKET]->(b:Ticket) WHERE a.id = yodb_id RETURN a.id, b.tid $yodb_cypher$, %s)", compiled.sql)
        self.assertIn("AS (start_id ag_catalog.agtype, end_id ag_catalog.agtype)", compiled.sql)
        self.assertTrue(compiled.sql.endswith("LIMIT %s"))
        self.assertEqual(compiled.parameters, (json.dumps({"ids": ["c1", "c2"]}), 51))
        self.assertEqual([c.logical_field for c in compiled.output_columns], ["start_id", "end_id"])
        self.assertNotIn("c1", compiled.sql)

    def test_anchored_at_the_to_end_matches_the_edge_backwards(self) -> None:
        compiled = PostgresQueryCompiler().compile_edges(self.scan(anchored="to", maximum_rows=None))
        self.assertIn("MATCH (b:Customer)-[:HAS_TICKET]->(a:Ticket) WHERE a.tid = yodb_id RETURN a.tid, b.id", compiled.sql)
        self.assertNotIn("LIMIT", compiled.sql)
        self.assertEqual(len(compiled.parameters), 1)

    def test_an_edge_scan_needs_ids_and_plain_identifiers(self) -> None:
        with self.assertRaises(QueryError):
            PostgresQueryCompiler().compile_edges(self.scan(ids=()))
        with self.assertRaises(QueryError):
            PostgresQueryCompiler().compile_edges(self.scan(edge_type="HAS TICKET"))

    def test_agtype_scalars_decode_to_text_ids(self) -> None:
        self.assertEqual(decode_agtype('"c01"'), "c01")
        self.assertEqual(decode_agtype("7"), "7")
        self.assertEqual(decode_agtype(b'"x"'), "x")
        self.assertEqual(decode_agtype(7), "7")
        self.assertEqual(decode_agtype(7.0), "7")
        self.assertIsNone(decode_agtype(None))
        self.assertIsNone(decode_agtype("null"))
        self.assertIsNone(decode_agtype('{"id": 1}::vertex'))
        self.assertIsNone(decode_agtype("[1, 2]"))
        self.assertEqual(decode_agtype("not json"), "not json")


class TraversalResultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = GraphWorld()
        self.db = engine(self.world)
        self.customers, self.tickets = customer_rows(), ticket_rows()

    def run_join(self, **kw):
        return rows_of(self.db.execute(query(**kw)))

    def test_an_inner_traversal_follows_the_graph_edges_not_the_column(self) -> None:
        got = self.run_join(steps=traverse(select=["subject", "status"]), where=p("country", "eq", "UK"))
        expected = edge_oracle(
            self.customers, self.tickets, EDGES["HAS_TICKET"], left_ok=lambda r: r["country"] == "UK",
            left_columns=["id", "name", "country"], right_columns=["id", "subject", "status"],
        )
        self.assertEqual(got, expected)
        # c7 (UK) has no ticket by column but one by edge (t10): the graph decides
        self.assertIn({"id": "c7", "name": "Gus", "country": "UK", "ticket.id": "t10", "ticket.subject": "Thanks", "ticket.status": "closed"}, got)
        self.assertEqual(list(got[0]), ["id", "name", "country", "ticket.id", "ticket.subject", "ticket.status"])

    def test_filters_on_both_sides_and_fields_from_several_sources_work(self) -> None:
        got = self.run_join(
            select=["name", "plan"], where=p("plan", "eq", "pro"),
            steps=traverse(select=["subject", "assignee"], where=p("status", "eq", "open")),
        )
        expected = edge_oracle(
            self.customers, self.tickets, EDGES["HAS_TICKET"],
            left_ok=lambda r: r["plan"] == "pro", right_ok=lambda r: r["status"] == "open",
            left_columns=["id", "name", "plan"], right_columns=["id", "subject", "assignee"],
        )
        self.assertEqual(got, expected)
        self.assertTrue(got)

    def test_a_left_traversal_keeps_customers_with_no_edge(self) -> None:
        got = self.run_join(steps=traverse(select=["subject"], optional=True))
        expected = edge_oracle(
            self.customers, self.tickets, EDGES["HAS_TICKET"], optional=True,
            left_columns=["id", "name", "country"], right_columns=["id", "subject"],
        )
        self.assertEqual(got, expected)
        self.assertIn({"id": "c8", "name": "Hal", "country": "DE", "ticket.id": None, "ticket.subject": None}, got)

    def test_ordering_and_paging_apply_to_the_joined_rows(self) -> None:
        got = self.run_join(
            steps=traverse(select=["priority"]), order_by=[{"field": "ticket.priority", "direction": "desc"}, {"field": "name", "direction": "asc"}], first=3,
        )
        expected = edge_oracle(
            self.customers, self.tickets, EDGES["HAS_TICKET"],
            left_columns=["id", "name", "country"], right_columns=["id", "priority"],
            order=[("ticket.priority", True), ("name", False)], first=3,
        )
        self.assertEqual(got, expected)
        self.assertEqual([r["ticket.priority"] for r in got], [5, 5, 5])

    def test_a_bidirectional_edge_can_be_walked_from_the_ticket_end(self) -> None:
        got = self.run_join(
            dataset="ticket", select=["subject"], where=p("priority", "gte", 4),
            steps=traverse(alias="customer", select=["name"], direction="reverse"),
        )
        reversed_edges = [(ticket, customer) for customer, ticket in EDGES["HAS_TICKET"]]
        expected = edge_oracle(
            self.tickets, self.customers, reversed_edges, left_ok=lambda r: r["priority"] >= 4, alias="customer",
            left_columns=["id", "subject"], right_columns=["id", "name"],
        )
        self.assertEqual(got, expected)
        self.assertTrue(got)

    def test_a_dataset_can_be_joined_to_itself_over_edges(self) -> None:
        got = self.run_join(select=["name"], steps=traverse(relationship="referred_by", alias="referrer", select=["name"]))
        expected = edge_oracle(
            self.customers, self.customers, EDGES["REFERRED_BY"], alias="referrer",
            left_columns=["id", "name"], right_columns=["id", "name"],
        )
        self.assertEqual(got, expected)
        self.assertIn({"id": "c2", "name": "Bob", "referrer.id": "c1", "referrer.name": "Ann"}, got)

    def test_a_field_implementation_is_preferred_over_the_edges(self) -> None:
        explanation = self.db.explain(query(steps=traverse(relationship="customer_has_ticket_by_field")))
        self.assertEqual(explanation.plan_kind, "hash_join")
        self.assertEqual(self.world.edge_queries, [])

    def test_the_edges_are_read_for_the_driving_side_only_and_the_other_side_for_the_ids_reached(self) -> None:
        self.run_join(where=p("name", "eq", "Ann"), steps=traverse(select=["subject"]))
        (edge_query,) = self.world.edge_queries
        self.assertEqual(json.loads(edge_query.parameters[0]), {"ids": ["c1"]})
        self.assertIn("(a:Customer)-[:HAS_TICKET]->(b:Ticket)", edge_query.sql)
        desk_reads = self.world.sql.reads_of("desk")
        self.assertEqual(len(desk_reads), 1)
        self.assertEqual(set(desk_reads[0].parameters[:-1]), {"t1", "t2", "t3", "t13"})

    def test_when_the_traversed_side_drives_the_edges_are_read_backwards(self) -> None:
        statistics = StatisticsService({SourceKind.POSTGRES: MapProvider({
            "crm.customers": SourceStatistics(row_count=1_000_000), "billing.plans": SourceStatistics(row_count=1_000_000),
            "desk.tickets": SourceStatistics(row_count=10), "triage.assignments": SourceStatistics(row_count=10),
        })})
        world = GraphWorld()
        db = engine(world, statistics=statistics)
        raw = query(steps=traverse(select=["subject"], where=p("status", "eq", "pending")))
        self.assertIn("driver=right", db.explain(raw).optimizer)
        got = rows_of(db.execute(raw))
        (edge_query,) = world.edge_queries
        self.assertIn("(b:Customer)-[:HAS_TICKET]->(a:Ticket) WHERE a.id = yodb_id", edge_query.sql)
        self.assertEqual(got, self.run_join(steps=traverse(select=["subject"], where=p("status", "eq", "pending"))))

    def test_edges_to_rows_the_step_filter_drops_do_not_match(self) -> None:
        got = self.run_join(where=p("name", "eq", "Ann"), steps=traverse(select=["subject"], where=p("status", "eq", "closed")))
        self.assertEqual([r["ticket.id"] for r in got], ["t2"])

    def test_a_duplicate_edge_matches_once(self) -> None:
        world = GraphWorld({"HAS_TICKET": [("c1", "t1"), ("c1", "t1")], "REFERRED_BY": []})
        got = rows_of(engine(world).execute(query(where=p("name", "eq", "Ann"), steps=traverse(select=["subject"]))))
        self.assertEqual([r["ticket.id"] for r in got], ["t1"])

    def test_the_result_and_plan_carry_fingerprints_that_see_the_edge(self) -> None:
        raw = query(steps=traverse(select=["subject"]))
        by_field = query(steps=traverse(relationship="customer_has_ticket_by_field", select=["subject"]))
        self.assertNotEqual(self.db.execute(raw).query_fingerprint, self.db.execute(by_field).query_fingerprint)
        self.assertNotEqual(self.db.explain(raw).plan_fingerprint, self.db.explain(by_field).plan_fingerprint)


class GuardAndErrorTests(unittest.TestCase):
    def test_the_edges_of_one_batch_are_bounded(self) -> None:
        world = GraphWorld({"HAS_TICKET": [("c1", f"t{i}") for i in range(30)], "REFERRED_BY": []})
        db = engine(world, join_policy=JoinPolicy(maximum_probe_rows=20))
        with self.assertRaises(QueryExecutionError) as caught:
            db.execute(query(where=p("name", "eq", "Ann"), steps=traverse(select=["subject"])))
        self.assertEqual(caught.exception.code, ErrorCode.QUERY_COORDINATOR_LIMIT_EXCEEDED)

    def test_ids_go_to_the_graph_in_batches(self) -> None:
        world = GraphWorld()
        db = engine(world, join_policy=JoinPolicy(batch_size=3))
        got = rows_of(db.execute(query(steps=traverse(select=["subject"]))))
        self.assertEqual(len(world.edge_queries), 3)      # 8 customers, batches of 3
        self.assertEqual(got, rows_of(engine(GraphWorld()).execute(query(steps=traverse(select=["subject"])))))

    def test_edges_in_a_source_without_a_graph_engine_are_refused_in_plain_words(self) -> None:
        sources = SOURCES.replace("  graphdb:\n    kind: postgres\n    connection_ref: graphdb\n    read_only: true\n    graph:\n      name: shop_graph\n      vertices:\n        customer: {label: Customer, id_property: id}\n        ticket: {label: Ticket, id_property: id}\n",
                                  "  graphdb:\n    kind: neo4j\n    connection_ref: graphdb\n    read_only: true\n    datasets:\n      customer:\n        resource: Customer\n        identity: [id]\n        fields:\n          id: {physical_name: id}\n      ticket:\n        resource: Ticket\n        identity: [id]\n        fields:\n          id: {physical_name: id}\n")
        active = evaluation(DATASETS, sources, RELATIONS)
        with self.assertRaises(QueryError) as caught:
            engine(GraphWorld(), active=active).execute(query(steps=traverse()))
        self.assertEqual(caught.exception.code, ErrorCode.QUERY_FEATURE_NOT_SUPPORTED)
        self.assertIn("no graph engine", caught.exception.detail.message)
        self.assertFalse(describe_catalog(active.catalog)["relationships"]["customer_has_ticket"]["traversable"])


class ExplainAndDescribeTests(unittest.TestCase):
    def test_explain_shows_the_driver_the_edge_read_and_the_probe(self) -> None:
        explanation = engine(GraphWorld()).explain(query(where=p("country", "eq", "US"), steps=traverse(select=["subject"])))
        self.assertEqual(explanation.plan_kind, "edge_join")
        kinds = [node.kind for node in explanation.nodes]
        self.assertIn("graph_edge_scan", kinds)
        self.assertEqual(kinds[-1], "edge_join")
        edge = next(node for node in explanation.nodes if node.kind == "graph_edge_scan")
        self.assertEqual(edge.location, "graphdb")
        self.assertIn("graph=shop_graph edge_type=HAS_TICKET", edge.detail)
        self.assertIn("(Customer) -[edge]-> (Ticket), for one batch of Customer IDs", edge.detail)
        self.assertTrue(any("over graph edges HAS_TICKET" in d for d in explanation.nodes[-1].detail))

    def test_the_mcp_description_says_the_relationship_is_traversed_over_edges(self) -> None:
        described = describe_catalog(ACTIVE.catalog)["relationships"]["customer_has_ticket"]
        self.assertTrue(described["traversable"])
        self.assertEqual(described["on"], "graph edge HAS_TICKET (customer.id -> ticket.id)")
        self.assertTrue(described["reversible"])

    def test_the_traverse_operator_is_implemented_and_postgres_offers_it(self) -> None:
        self.assertTrue(OPERATORS.get(OperatorKind.TRAVERSE).implemented)
        view = {item.kind: item for item in operator_support(POSTGRES_CAPABILITIES)}
        self.assertEqual(view[OperatorKind.TRAVERSE].strategies, ("edge_join",))

    def test_an_edge_join_node_exposes_its_inputs(self) -> None:
        planner = engine(GraphWorld())._joins
        planned = planner.plan(query(steps=traverse(select=["subject"])), ACTIVE)
        self.assertIsInstance(planned.plan, EdgeJoin)
        self.assertEqual(len(planned.plan.inputs()), 3)
        self.assertIs(planned.plan.with_inputs(planned.plan.inputs()).edges, planned.plan.edges)


class ValidatorTests(unittest.TestCase):
    """Validating a declared graph against what the inspector saw in Apache AGE's catalog."""

    def inspection(self, *, extensions=None, graphs=None):
        from datetime import UTC, datetime

        return SourceInspection(
            source_name="graphdb", source_kind=SourceKind.POSTGRES, inspected_at=datetime.now(UTC),
            extensions={"age": "1.5.0"} if extensions is None else extensions,
            provider_metadata={} if graphs is None else {"graphs": graphs},
        )

    def codes(self, inspection):
        report = PostgresCatalogValidator().validate(ACTIVE.catalog, inspection)
        return [finding.code for finding in report.findings]

    def test_a_graph_with_its_labels_and_edge_types_validates(self) -> None:
        graphs = {"shop_graph": {"vertex_labels": ["Customer", "Ticket"], "edge_types": ["HAS_TICKET", "REFERRED_BY"]}}
        self.assertEqual(self.codes(self.inspection(graphs=graphs)), [])

    def test_missing_extension_graph_label_or_edge_type_is_reported(self) -> None:
        self.assertEqual(self.codes(self.inspection(extensions={})), ["extension_not_installed"])
        self.assertEqual(self.codes(self.inspection(graphs={})), ["graph_not_found"])
        graphs = {"shop_graph": {"vertex_labels": ["Customer"], "edge_types": ["HAS_TICKET"]}}
        self.assertEqual(sorted(self.codes(self.inspection(graphs=graphs))), ["edge_type_not_found", "vertex_label_not_found"])

    def test_the_inspector_builds_the_graph_metadata_from_age_catalog_rows(self) -> None:
        from yodb.inspection.postgres import _build_inspection
        from yodb.inspection import InspectionRequest

        request = InspectionRequest(source_name="graphdb", source=ACTIVE.catalog.sources["graphdb"])
        built = _build_inspection(
            request, [{"server_version": "16.4"}], [{"extname": "age", "extversion": "1.5.0"}], [], [], [], [], [],
            [
                {"graph_name": "shop_graph", "label_name": "_ag_label_vertex", "label_kind": "v"},
                {"graph_name": "shop_graph", "label_name": "Customer", "label_kind": "v"},
                {"graph_name": "shop_graph", "label_name": "HAS_TICKET", "label_kind": "e"},
            ],
        )
        self.assertEqual(built.provider_metadata["graphs"], {"shop_graph": {"vertex_labels": ["Customer"], "edge_types": ["HAS_TICKET"]}})
        self.assertIn("graph_schema", {c.value for c in built.capabilities})


if __name__ == "__main__":
    unittest.main()
