"""The similarity search: parsing, planning, SQL and execution of ``{"similar": ...}``.

The sources are in-memory SQL (``SqlWorld``) with the vectors stored as JSON text and the
pgvector distance operators as SQLite functions, so the ranked read really filters, restricts
by keys, ranks by cosine distance and returns each row's distance.
"""

from __future__ import annotations

import json
import unittest

from yodb.compilation import PostgresQueryCompiler, QueryCompilerRegistry
from yodb.compilation.postgres import DISTANCE_COLUMN
from yodb.errors import ErrorCode, QueryError, QueryExecutionError
from yodb.execution import QueryExecutionAdapterRegistry, QueryExecutionEngine
from yodb.mcp_server import describe_catalog, query_language
from yodb.operators import OPERATORS, OperatorKind
from yodb.planning import (
    CoordinatorSortPage,
    FederatedPhysicalPlanner,
    PlannerPolicy,
    PostgresPlanningAdapter,
    RecordAssembly,
    RemoteScan,
    SourcePlanningRegistry,
    StepRole,
)
from yodb.query import bind_query, parse_query, resolve_query_sources
from yodb.query.similar import BoundSimilarPredicate
from yodb.semantic import ProviderInfo, SemanticExtension, SemanticPolicy, SemanticRuntime
from yodb.similar import SimilarExtension, SimilarPolicy, SimilarQueryReport, SimilarRank

from support.catalogs import StaticRuntime
from support.sql_world import SqlWorld
from support.tickets import EMBEDDER_INFO, FakeEmbedder, KeywordVerifier, ticket_catalog
from support.vector_store import store_catalog

ACTIVE = ticket_catalog()
STORE_ACTIVE = store_catalog()

# The query vector the fake embedder returns is (1, 0, 0.5): nearest first is t1, t2, t4, t3, t5; t6 has no vector.
VECTORS = {"t1": [1, 0, 0.5], "t2": [1, 0, 0], "t3": [0, 1, 0], "t4": [0.5, 0, 1], "t5": [-1, 0, -0.5], "t6": None}
NEAREST = ["t1", "t2", "t4", "t3", "t5"]
TICKETS = [
    # id, subject, body, priority
    ("t1", "Zeta", "price too high", 1), ("t2", "Alpha", "love it", 5), ("t3", "Gamma", "crash", 5),
    ("t4", "Beta", "price hike", 5), ("t5", "Delta", "fine", 2), ("t6", "Eta", "no vector", 5),
]
OWNERS = [("t2", "ann", None), ("t3", "ann", None), ("t5", "ann", None), ("t1", "bob", None)]


def world(*, store: bool = False) -> SqlWorld:
    tables = {
        "public.tickets": (("id", "subject", "body", "priority", "body_embedding"),
                           [(i, s, b, p, None if VECTORS[i] is None else json.dumps(VECTORS[i])) for i, s, b, p in TICKETS]),
        "public.owners": (("ticket_id", "owner", "memo"), OWNERS),
    }
    if store:
        tables["public.tickets"] = (("id", "subject", "body", "priority"), [(i, s, b, p) for i, s, b, p in TICKETS])
        tables["vec.ticket_vectors"] = (("ticket_id", "embedding"), [(i, json.dumps(v)) for i, v in VECTORS.items() if v is not None])
    return SqlWorld(tables)


def engine(world, *, embedder=FakeEmbedder(), active=ACTIVE, policy=PlannerPolicy(), semantic=None, similar_policy=None):
    extensions = []
    if semantic is not None:
        extensions.append(semantic)
    if embedder is not None or similar_policy is not None:
        extensions.append(SimilarExtension(embedder, policy=similar_policy))
    planner = FederatedPhysicalPlanner(SourcePlanningRegistry([PostgresPlanningAdapter()]), policy=policy, extensions=extensions)
    return QueryExecutionEngine(
        StaticRuntime(active), QueryCompilerRegistry([PostgresQueryCompiler()]), QueryExecutionAdapterRegistry([world]),
        planner=planner, extensions=extensions,
    )


def similar(text="anything", field="body", top=None):
    body = {"field": field, "text": text}
    if top is not None:
        body["top"] = top
    return {"similar": body}


def q(where, select=("subject",), first=10, order_by=None, **extra):
    raw = {"from": {"dataset": "ticket"}, "select": list(select), "where": where, "page": {"first": first}, **extra}
    if order_by:
        raw["order_by"] = order_by
    return raw


def ids(result):
    return [row["id"] for row in result.rows]


class ParseAndBindTests(unittest.TestCase):
    def bind(self, where):
        return bind_query(parse_query(q(where)), ACTIVE)

    def test_a_similarity_search_binds_to_a_semantic_eligible_text_field(self) -> None:
        bound = self.bind(similar("late delivery", top=5))
        term = bound.where
        self.assertIsInstance(term, BoundSimilarPredicate)
        self.assertEqual((term.field.name, term.text, term.top), ("body", "late delivery", 5))
        self.assertFalse(bound.explicit_order)
        self.assertTrue(bind_query(parse_query(q(similar(), order_by=[{"field": "subject", "direction": "asc"}])), ACTIVE).explicit_order)

    def test_the_shape_is_checked(self) -> None:
        for where, code, location in (
            ({"similar": {"field": "body"}}, ErrorCode.QUERY_SHAPE_INVALID, "where.similar.text"),
            ({"similar": {"field": "body", "text": "x", "top": 0}}, ErrorCode.QUERY_LIMIT_INVALID, "where.similar.top"),
            ({"similar": {"field": "body", "text": "x", "top": True}}, ErrorCode.QUERY_LIMIT_INVALID, "where.similar.top"),
            ({"similar": {"field": "body", "text": "x", "limit": 3}}, ErrorCode.QUERY_SHAPE_INVALID, "where.similar"),
            ({"similar": {"field": "body", "text": "x", "top": 5_000}}, ErrorCode.QUERY_LIMIT_INVALID, "where.similar.top"),
            ({"similar": {"field": "body", "text": "   "}}, ErrorCode.QUERY_SHAPE_INVALID, "where.similar.text"),
            ({"similar": {"field": "priority", "text": "x"}}, ErrorCode.QUERY_OPERATOR_NOT_SUPPORTED, "where.similar.field"),
            ({"similar": {"field": "subject", "text": "x"}}, ErrorCode.QUERY_OPERATOR_NOT_SUPPORTED, "where.similar.field"),
            ({"any": [similar(), {"field": "priority", "op": "eq", "value": 1}]}, ErrorCode.QUERY_EXPRESSION_INVALID, "where.any[0]"),
            ({"all": [similar("a"), similar("b")]}, ErrorCode.QUERY_LIMIT_INVALID, "where"),
        ):
            with self.subTest(where=where), self.assertRaises(QueryError) as caught:
                self.bind(where)
            self.assertEqual(caught.exception.code, code)
            self.assertEqual(caught.exception.detail.location, location)

    def test_the_text_and_top_appear_in_the_query_fingerprint(self) -> None:
        a, b, c = (self.bind(similar(text, top=top)).query_fingerprint for text, top in (("x", 3), ("y", 3), ("x", 4)))
        self.assertEqual(len({a, b, c}), 3)


class PlanTests(unittest.TestCase):
    def plan(self, raw, *, embedder=FakeEmbedder(), active=ACTIVE, policy=PlannerPolicy()):
        planner = FederatedPhysicalPlanner(
            SourcePlanningRegistry([PostgresPlanningAdapter()]), policy=policy, extensions=(SimilarExtension(embedder),)
        )
        bound = bind_query(parse_query(raw), active)
        return planner.plan(resolve_query_sources(bound, active))

    @staticmethod
    def rank_node(planned) -> SimilarRank:
        node = planned.plan
        while not isinstance(node, SimilarRank):
            (node,) = node.inputs()
        return node

    def test_the_anchor_read_is_ranked_and_cut_to_top_and_the_rank_node_pages_by_relevance(self) -> None:
        planned = self.plan(q(similar(top=3)))
        rank = self.rank_node(planned)
        self.assertIsInstance(rank.input, RemoteScan)
        search = rank.input.vector_search
        self.assertEqual((search.column, search.shortlist_size, search.return_distance, rank.input.limit), ("body_embedding", 3, True, 3))
        self.assertIsNone(rank.input.maximum_rows)
        self.assertEqual((rank.top, rank.first, rank.relevance_order, rank.via_store, rank.exact), (3, 10, True, False, True))
        self.assertTrue(rank.orders_and_pages)
        self.assertNotIsInstance(planned.plan.input, CoordinatorSortPage)          # no coordinator sort over the ranking
        kinds = [node.kind for node in planned.explain.nodes]
        self.assertEqual(kinds, ["remote_scan", "similar_rank", "result_project"])
        self.assertIn("distance asc", planned.explain.nodes[1].ordering)

    def test_with_a_caller_order_the_top_rows_are_sorted_by_it(self) -> None:
        planned = self.plan(q(similar(top=3), order_by=[{"field": "subject", "direction": "asc"}]))
        rank = self.rank_node(planned)
        self.assertFalse(rank.relevance_order)
        sort = planned.plan.input
        self.assertIsInstance(sort, CoordinatorSortPage)
        self.assertEqual([t.field.name for t in sort.order_by], ["subject", "id"])

    def test_top_defaults_to_the_page_and_is_capped_by_the_policy(self) -> None:
        self.assertEqual(self.rank_node(self.plan(q(similar(), first=7))).top, 7)
        planner = FederatedPhysicalPlanner(
            SourcePlanningRegistry([PostgresPlanningAdapter()]),
            extensions=(SimilarExtension(FakeEmbedder(), policy=SimilarPolicy(maximum_top=2, embedder=EMBEDDER_INFO, embedder_dimensions=3)),),
        )
        bound = bind_query(parse_query(q(similar(top=50))), ACTIVE)
        rank = self.rank_node(planner.plan(resolve_query_sources(bound, ACTIVE)))
        self.assertEqual(rank.top, 2)
        self.assertTrue(any("top reduced from 50 to 2" in reason for reason in rank.choice_reasons))

    def test_other_conditions_are_pushed_under_the_ranking(self) -> None:
        planned = self.plan(q({"all": [similar(top=2), {"field": "priority", "op": "gte", "value": 3}]}))
        scan = self.rank_node(planned).input
        self.assertIsNotNone(scan.pushed_filter)
        self.assertEqual(planned.explain.nodes[0].pushed_filter_fields, ("priority",))

    def test_a_contributor_filter_restricts_the_anchor_before_it_ranks(self) -> None:
        planned = self.plan(q({"all": [similar(top=2), {"field": "owner", "op": "eq", "value": "ann"}]}))
        assembly = self.rank_node(planned).input
        self.assertIsInstance(assembly, RecordAssembly)
        self.assertIsNotNone(assembly.anchor.vector_search)
        self.assertEqual([(s.role, s.source_name) for s in assembly.schedule], [(StepRole.REQUIRED, "directory"), (StepRole.ANCHOR, "helpdesk")])

    def test_a_separate_vector_store_ranks_and_the_anchor_is_read_for_its_ids(self) -> None:
        planned = self.plan(q(similar(top=3)), active=STORE_ACTIVE)
        rank = self.rank_node(planned)
        assembly = rank.input
        self.assertIsInstance(assembly, RecordAssembly)
        store = next(c for c in assembly.contributors if c.source.source_name == "vectors")
        self.assertTrue(store.vector_search.return_distance)
        self.assertEqual([(s.role, s.source_name) for s in assembly.schedule], [(StepRole.SHORTLIST, "vectors"), (StepRole.ANCHOR, "helpdesk")])
        self.assertTrue(rank.via_store)
        self.assertTrue(rank.exact)
        with_filter = self.rank_node(self.plan(q({"all": [similar(top=3), {"field": "priority", "op": "gte", "value": 3}]}), active=STORE_ACTIVE))
        self.assertFalse(with_filter.exact)
        self.assertIn("approximate", " ".join(with_filter.describe().detail))

    def test_searches_that_cannot_run_exactly_are_refused_with_the_reasons(self) -> None:
        other_model = FakeEmbedder(info=ProviderInfo("fake", "embed-v2", "1"))
        for raw, embedder, active, policy, expected in (
            (q(similar()), None, ACTIVE, PlannerPolicy(), "no embedding provider is configured"),
            (q(similar()), other_model, ACTIVE, PlannerPolicy(), "does not match the stored embeddings"),
            (q({"all": [similar(), {"field": "subject", "op": "contains", "value": "x"}]}), FakeEmbedder(), ACTIVE, PlannerPolicy(), "not enforced by its source"),
            (q(similar(field="memo")), FakeEmbedder(), ACTIVE, PlannerPolicy(), "no source holds embeddings"),
            (q(similar()), FakeEmbedder(), STORE_ACTIVE, PlannerPolicy(maximum_transfer_keys=0), "key transfer is disabled"),
        ):
            with self.subTest(expected=expected), self.assertRaises(QueryError) as caught:
                self.plan(raw, embedder=embedder, active=active, policy=policy)
            self.assertEqual(caught.exception.code, ErrorCode.QUERY_PLAN_UNSUPPORTED)
            self.assertIn(expected, caught.exception.detail.message)
            self.assertEqual(caught.exception.detail.location, "where")

    def test_without_the_extension_the_term_is_refused_in_plain_words(self) -> None:
        planner = FederatedPhysicalPlanner(SourcePlanningRegistry([PostgresPlanningAdapter()]))
        bound = bind_query(parse_query(q(similar())), ACTIVE)
        with self.assertRaises(QueryError) as caught:
            planner.plan(resolve_query_sources(bound, ACTIVE))
        self.assertEqual(caught.exception.code, ErrorCode.QUERY_FEATURE_NOT_SUPPORTED)
        self.assertIn("The query has a similarity search, but nothing is configured", caught.exception.detail.message)

    def test_a_similarity_search_and_a_semantic_condition_do_not_mix(self) -> None:
        runtime = SemanticRuntime(KeywordVerifier(), FakeEmbedder())
        planner = FederatedPhysicalPlanner(
            SourcePlanningRegistry([PostgresPlanningAdapter()]),
            extensions=(SemanticExtension(runtime), SimilarExtension(FakeEmbedder())),
        )
        bound = bind_query(parse_query(q({"all": [similar(), {"semantic": {"field": "body", "proposition": "mentions price"}}]})), ACTIVE)
        with self.assertRaises(QueryError) as caught:
            planner.plan(resolve_query_sources(bound, ACTIVE))
        self.assertEqual(caught.exception.code, ErrorCode.QUERY_FEATURE_NOT_SUPPORTED)

    def test_the_plan_fingerprint_sees_top_and_the_distance_projection(self) -> None:
        a = self.plan(q(similar(top=3))).plan_fingerprint
        b = self.plan(q(similar(top=4))).plan_fingerprint
        self.assertNotEqual(a, b)
        self.assertEqual(a, self.plan(q(similar("other text", top=3))).plan_fingerprint)      # values never change the structure

    def test_the_operator_is_in_the_catalog_and_postgres_offers_it(self) -> None:
        from yodb.planning import POSTGRES_CAPABILITIES, operator_support

        self.assertTrue(OPERATORS.get(OperatorKind.SIMILARITY_SEARCH).implemented)
        view = {item.kind: item for item in operator_support(POSTGRES_CAPABILITIES)}
        self.assertEqual(view[OperatorKind.SIMILARITY_SEARCH].strategies, ("ranked_read", "ranked_store"))


class CompilerTests(unittest.TestCase):
    def test_the_distance_is_selected_and_ordered_by(self) -> None:
        planner = FederatedPhysicalPlanner(SourcePlanningRegistry([PostgresPlanningAdapter()]), extensions=(SimilarExtension(FakeEmbedder()),))
        bound = bind_query(parse_query(q({"all": [similar(top=3), {"field": "priority", "op": "gte", "value": 3}]})), ACTIVE)
        planned = planner.plan(resolve_query_sources(bound, ACTIVE))
        from dataclasses import replace

        scan = PlanTests.rank_node(planned).input
        scan = replace(scan, vector_search=replace(scan.vector_search, query_vector=(1.0, 0.0, 0.5)), key_filter=("t1", "t2"))
        compiled = PostgresQueryCompiler().compile_scan(scan)
        self.assertIn('"body_embedding" <=> %s::vector AS "_distance"', compiled.sql)
        self.assertIn('ORDER BY "body_embedding" <=> %s::vector, "id" ASC', compiled.sql)
        self.assertIn('"body_embedding" IS NOT NULL', compiled.sql)
        self.assertEqual(compiled.parameters, ("[1.0,0.0,0.5]", 3, "t1", "t2", "[1.0,0.0,0.5]", 3))
        self.assertEqual(compiled.output_columns[-1].logical_field, DISTANCE_COLUMN)


class ExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = world()
        self.db = engine(self.world)

    def test_the_nearest_rows_come_back_nearest_first_with_their_distances(self) -> None:
        result = self.db.execute(q(similar(top=3)))
        self.assertEqual(ids(result), ["t1", "t2", "t4"])
        self.assertEqual(list(result.rows[0]), ["id", "subject"])                      # no helper field leaks
        report = result.reports["similar"]
        self.assertIsInstance(report, SimilarQueryReport)
        self.assertEqual([report.records[i].rank for i in ids(result)], [1, 2, 3])
        self.assertAlmostEqual(report.records["t1"].distance, 0.0, places=9)
        self.assertLess(report.records["t2"].distance, report.records["t4"].distance)
        self.assertEqual((report.stats.candidates, report.stats.returned, report.stats.top, report.stats.exact), (3, 3, 3, True))
        self.assertEqual(report.stats.embedding_model, "embed-v1")

    def test_rows_without_a_vector_are_never_ranked(self) -> None:
        self.assertEqual(ids(self.db.execute(q(similar(top=10)))), NEAREST)

    def test_the_page_is_the_nearest_rows_and_top_defaults_to_it(self) -> None:
        result = self.db.execute(q(similar(), first=2))
        self.assertEqual(ids(result), ["t1", "t2"])
        self.assertEqual(result.reports["similar"].stats.top, 2)

    def test_other_conditions_narrow_the_rows_that_are_ranked(self) -> None:
        result = self.db.execute(q({"all": [similar(top=2), {"field": "priority", "op": "gte", "value": 3}]}))
        self.assertEqual(ids(result), ["t2", "t4"])           # t1 (priority 1) is not a candidate
        self.assertEqual(result.reports["similar"].records["t2"].rank, 1)

    def test_a_caller_order_sorts_the_top_rows(self) -> None:
        result = self.db.execute(q(similar(top=3), order_by=[{"field": "subject", "direction": "asc"}]))
        self.assertEqual(ids(result), ["t2", "t4", "t1"])    # the 3 nearest (t1, t2, t4) by subject: Alpha, Beta, Zeta
        self.assertEqual(result.reports["similar"].records["t1"].rank, 1)

    def test_a_contributor_condition_restricts_the_ranked_read_to_its_ids(self) -> None:
        result = self.db.execute(q({"all": [similar(top=2), {"field": "owner", "op": "eq", "value": "ann"}]}, select=("subject", "owner")))
        self.assertEqual(ids(result), ["t2", "t3"])          # ann owns t2, t3, t5: nearest among them
        self.assertEqual(result.rows[0]["owner"], "ann")
        helpdesk = self.world.reads_of("helpdesk")
        self.assertEqual(len(helpdesk), 1)
        self.assertIn("IN (", helpdesk[0].sql)

    def test_a_restriction_too_large_to_transfer_is_refused_rather_than_answered_wrongly(self) -> None:
        db = engine(world(), policy=PlannerPolicy(maximum_transfer_keys=2, maximum_key_batches=1))
        with self.assertRaises(QueryExecutionError) as caught:
            db.execute(q({"all": [similar(top=2), {"field": "owner", "op": "eq", "value": "ann"}]}))
        self.assertEqual(caught.exception.code, ErrorCode.QUERY_ROW_LIMIT_EXCEEDED)
        self.assertIn("could not be narrowed", caught.exception.detail.message)

    def test_the_vector_store_ranks_and_the_anchor_is_read_only_for_the_ids_it_returns(self) -> None:
        store_world = world(store=True)
        result = engine(store_world, active=STORE_ACTIVE).execute(q(similar(top=3)))
        self.assertEqual(ids(result), ["t1", "t2", "t4"])
        self.assertTrue(result.reports["similar"].stats.via_store)
        self.assertEqual([r.rank for r in result.reports["similar"].records.values()], [1, 2, 3])
        (helpdesk,) = store_world.reads_of("helpdesk")
        self.assertIn("IN (", helpdesk.sql)
        self.assertEqual(set(helpdesk.parameters[:-1]), {"t1", "t2", "t4"})

    def test_a_store_ranking_before_a_filter_is_reported_as_approximate(self) -> None:
        result = engine(world(store=True), active=STORE_ACTIVE).execute(q({"all": [similar(top=3), {"field": "priority", "op": "gte", "value": 3}]}))
        self.assertEqual(ids(result), ["t2", "t4"])          # t1 was among the 3 nearest, then dropped by the filter
        self.assertFalse(result.reports["similar"].stats.exact)

    def test_no_embedder_at_run_time_is_a_clear_error(self) -> None:
        db = engine(world(), embedder=None, similar_policy=SimilarPolicy(embedder=EMBEDDER_INFO, embedder_dimensions=3))
        with self.assertRaises(QueryExecutionError) as caught:
            db.execute(q(similar()))
        self.assertEqual(caught.exception.code, ErrorCode.SEMANTIC_PROVIDER_UNAVAILABLE)

    def test_a_failing_or_mismatching_embedder_is_reported(self) -> None:
        with self.assertRaises(QueryExecutionError) as caught:
            engine(world(), embedder=FakeEmbedder(fail=True)).execute(q(similar()))
        self.assertEqual(caught.exception.code, ErrorCode.SEMANTIC_PROVIDER_FAILED)

    def test_explain_runs_nothing(self) -> None:
        self.db.explain(q(similar(top=3)))
        self.assertEqual(self.world.queries, [])


class JoinTests(unittest.TestCase):
    """A similarity search in the root filter of a traverse query ranks the root rows first."""

    def setUp(self) -> None:
        self.world = world()
        self.db = engine(self.world)

    def test_the_root_side_drives_and_the_joined_rows_keep_the_ranking(self) -> None:
        raw = {
            "from": {"dataset": "ticket"}, "select": ["subject"], "where": similar(top=3),
            "traverse": [{"relationship": "ticket_reference", "as": "ref", "select": ["subject"]}], "page": {"first": 10},
        }
        explanation = self.db.explain(raw)
        self.assertIn("driver=left", explanation.optimizer)
        self.assertTrue(any("similarity search" in note for note in explanation.optimizer))
        result = self.db.execute(raw)
        self.assertEqual([row["id"] for row in result.rows], ["t1", "t2", "t4"])
        self.assertEqual([row["ref.id"] for row in result.rows], ["t1", "t2", "t4"])
        self.assertEqual([r.rank for _, r in sorted(result.reports["similar"].records.items(), key=lambda i: i[1].rank)], [1, 2, 3])

    def test_without_top_the_join_page_is_the_default(self) -> None:
        raw = {
            "from": {"dataset": "ticket"}, "where": similar(),
            "traverse": [{"relationship": "ticket_reference", "as": "ref"}], "page": {"first": 2},
        }
        self.assertEqual([row["id"] for row in self.db.execute(raw).rows], ["t1", "t2"])

    def test_a_search_in_the_step_is_refused(self) -> None:
        raw = {
            "from": {"dataset": "ticket"},
            "traverse": [{"relationship": "ticket_reference", "as": "ref", "where": similar()}],
        }
        with self.assertRaises(QueryError) as caught:
            self.db.execute(raw)
        self.assertEqual(caught.exception.code, ErrorCode.QUERY_FEATURE_NOT_SUPPORTED)
        self.assertEqual(caught.exception.detail.location, "traverse[0].where")


class ClientAndMcpTests(unittest.TestCase):
    def test_describe_marks_the_fields_embedded_with_the_configured_model(self) -> None:
        from yodb.client import YoDb

        db = YoDb(StaticRuntime(ACTIVE), engine(world()), embedding_model="embed-v1")
        self.assertTrue(db.similar_enabled)
        self.assertEqual(db.similar_fields(), {"ticket": ("body",)})
        self.assertEqual(db.describe()["ticket"]["fields"]["body"], {"type": "text", "semantic": True, "similar": True})
        self.assertNotIn("similar", db.describe()["ticket"]["fields"]["memo"])
        self.assertEqual(YoDb(StaticRuntime(ACTIVE), engine(world()), embedding_model="other").similar_fields(), {})
        described = describe_catalog(ACTIVE.catalog, similar=db.similar_fields())
        self.assertTrue(described["datasets"]["ticket"]["fields"]["body"]["similar"])
        self.assertNotIn("similar", described["datasets"]["ticket"]["fields"]["memo"])

    def test_the_semantic_runtimes_embedder_serves_similarity_searches_too(self) -> None:
        from yodb.client import _extensions

        extensions = _extensions(SemanticRuntime(KeywordVerifier(), FakeEmbedder()), None)
        self.assertEqual([type(e).__name__ for e in extensions], ["SemanticExtension", "SimilarExtension"])
        self.assertIs(extensions[1].embedder, extensions[0].runtime.embedder)
        self.assertEqual([type(e).__name__ for e in _extensions(SemanticRuntime(KeywordVerifier()), None)], ["SemanticExtension"])
        self.assertEqual([type(e).__name__ for e in _extensions(None, FakeEmbedder())], ["SimilarExtension"])

    def test_the_guide_describes_the_search_only_when_an_embedder_is_configured(self) -> None:
        limits = engine(world()).limits
        with_it = query_language(limits, similar=True)
        self.assertIn('"similar"', with_it)
        self.assertIn("similarity searches (no embedding model is configured on this server)", query_language(limits, semantic=True))
        self.assertIn("semantic conditions (none is configured on this server), and similarity searches (no embedding model", query_language(limits))
        self.assertNotIn("none is configured", query_language(limits, semantic=True, similar=True))

    def test_the_mcp_answer_lists_each_rows_distance_and_rank(self) -> None:
        from yodb.mcp_server import _similar_report

        result = engine(world()).execute(q(similar(top=2)))
        reported = _similar_report(result.reports["similar"])
        self.assertEqual(reported["records_returned"], 2)
        self.assertEqual([entry["id"] for entry in reported["ranked"]], ["t1", "t2"])
        self.assertEqual(reported["ranked"][0]["rank"], 1)
        self.assertTrue(reported["exact"])


if __name__ == "__main__":
    unittest.main()
