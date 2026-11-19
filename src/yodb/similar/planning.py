"""SIMILARITY_SEARCH planning: the K nearest records by embedding distance.

The ranking is done by the source's vector search (pgvector), among the rows that
satisfy the query's other conditions, so the result is exact whenever the source
can combine ranking with those conditions.  The vectors sit either in the
dataset's identity source (the ranked read *is* the anchor read) or in a separate
vector-store source (ranked there, then the anchor is read for the K IDs).  There
is no model judging rows and nothing to repair: a search that cannot run exactly is
refused at planning time with the reasons, never answered approximately in silence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

from ..errors import ErrorCode, ErrorDetail, QueryError
from ..operators import OperatorKind
from ..query.models import BoundFilterExpression, BoundQuery
from ..query.resolution import QuerySourceShape, ResolvedField, SingleSourceQueryBinding, SourceResolvedQuery
from ..query.semantic import semantic_predicates
from ..query.similar import BoundSimilarPredicate, similar_predicates, without_similar_terms
from ..semantic.contracts import ProviderCost, ProviderInfo
from ..planning.capabilities import OperatorSupport, SourceCapabilities, SupportLevel, support_view
from ..planning.contracts import (
    PhysicalNode,
    PhysicalPlan,
    PlanExplanationNode,
    PlanProperties,
    RemoteScan,
    ResultCompleteness,
    ResultShape,
    UnaryNode,
    VectorSearch,
    coordinator_location,
    properties_from,
    remote_location,
)
from ..planning.optimizer import Estimate, RankedRead, Variant
from ..planning.operators.base import Claim, ExtensionPlan, PlanningServices, Strategy, effective_limit


@dataclass(frozen=True)
class SimilarPolicy:
    """Planner limits for similarity searches, and what the configured embedder produces."""

    maximum_top: int = 1_000                  # most records a search may rank and return
    embedder: ProviderInfo | None = None
    embedder_dimensions: int | None = None
    # What one embedding call is expected to cost (for the cost model, not billing).
    embedding_cost: ProviderCost = ProviderCost(money_per_call=0.0001, latency_ms_per_call=80.0)
    vector_ranking_per_row_ms: float = 0.002

    def __post_init__(self) -> None:
        if self.maximum_top < 1:
            raise ValueError("maximum_top must be positive")


@dataclass(frozen=True)
class SimilarRank(UnaryNode):
    """Keep the ``top`` input rows nearest to the query text, nearest first.

    The input is the ranked read (the anchor, or the anchor narrowed to a store's
    shortlist), which carries each row's distance.  When the caller gave no order,
    the node also pages the result, so the page is the nearest ``first`` rows.
    """

    operator: ClassVar[OperatorKind] = OperatorKind.SIMILARITY_SEARCH

    input: PhysicalPlan
    field: ResolvedField
    text: str
    top: int
    first: int
    relevance_order: bool            # no caller order: the output is ordered by distance and paged here
    via_store: bool
    exact: bool
    embedding_model: str
    embedding_dimensions: int
    properties: PlanProperties
    choice_reasons: tuple[str, ...] = ()

    @property
    def orders_and_pages(self) -> bool:
        return self.relevance_order

    def shape(self) -> dict[str, object]:
        return {
            "kind": "similar_rank",
            "input": self.input.shape(),
            "field": self.field.field.name,
            "top": self.top,
            "first": self.first if self.relevance_order else None,
            "via_store": self.via_store,
        }

    def describe(self) -> PlanExplanationNode:
        return PlanExplanationNode(
            "similar_rank",
            "coordinator",
            tuple(field.field.name for field in self.properties.output_fields),
            ordering=("distance asc",) if self.relevance_order else (),
            limit=self.first if self.relevance_order else self.top,
            detail=(
                f"similar_field={self.field.field.name}",
                f"top={self.top}",
                "ranked by " + ("the vector store" if self.via_store else "the identity source"),
                "exact" if self.exact else "approximate: ranked before a later condition, so fewer than top rows may qualify",
                *(f"note: {reason}" for reason in self.choice_reasons),
            ),
        )


class SimilarOperator:
    kind = OperatorKind.SIMILARITY_SEARCH

    def __init__(self, services: PlanningServices, policy: SimilarPolicy = SimilarPolicy()) -> None:
        self._services = services
        self._policy = policy

    def claim(self, where: BoundFilterExpression | None) -> Claim | None:
        terms = similar_predicates(where)
        return Claim(terms, without_similar_terms(where)) if terms else None

    def plan(self, claim: Claim, core: SourceResolvedQuery, scans, query: BoundQuery) -> ExtensionPlan:
        if len(claim.terms) > 1:
            raise QueryError(
                ErrorDetail(
                    code=ErrorCode.QUERY_FEATURE_NOT_SUPPORTED,
                    message="Execution supports one similarity search per query.",
                    retryable=False,
                    location="where",
                )
            )
        term: BoundSimilarPredicate = claim.terms[0]  # type: ignore[assignment]
        if semantic_predicates(query.where):          # the original filter: another extension may have claimed its term already
            raise QueryError(
                ErrorDetail(
                    code=ErrorCode.QUERY_FEATURE_NOT_SUPPORTED,
                    message="A similarity search and a semantic condition cannot be combined in one query.",
                    retryable=False,
                    location="where",
                )
            )
        binding, store = self._binding(core, term)
        reasons = self._refusals(core, term, binding, store, scans.fully_pushed)
        if reasons:
            raise QueryError(
                ErrorDetail(
                    code=ErrorCode.QUERY_PLAN_UNSUPPORTED,
                    message="A similarity search cannot run here: " + "; ".join(reasons) + ".",
                    retryable=False,
                    location="where",
                )
            )
        first = effective_limit(query)
        wanted = term.top if term.top is not None else first
        cap, notes = self._cap(core, store)
        top = min(wanted, cap)
        if top < wanted:
            notes = (*notes, f"top reduced from {wanted} to {top} by the limits in force")
        search = VectorSearch(
            column=binding.column,
            metric=binding.metric,
            model=binding.model,
            dimensions=binding.dimensions,
            shortlist_size=top,
            fallback_maximum_rows=None if store is not None else self._services.row_cap(core.identity_source.source_kind),
            return_distance=True,
        )
        field = next(f for source in core.sources for f in source.fields if f.field.name == term.field.name)
        # Ranking in a separate store happens before the anchor applies its own conditions: with any condition
        # left, some of the K nearest may be dropped afterwards, so fewer than ``top`` rows can come back.
        exact = store is None or core.query.where is None
        store_scan = self._store_scan(store, search, query)
        policy = self._policy
        variant = Variant(
            f"similar:{top}",
            cost=lambda candidates: Estimate(policy.embedding_cost.latency_ms_per_call, policy.embedding_cost.money_per_call),
            demand=RankedRead(top, policy.vector_ranking_per_row_ms, via_store=store is not None),
            approximate=not exact,
            payload=top,
        )

        def prepare(scans: tuple[RemoteScan, ...]) -> tuple[RemoteScan, ...]:
            if store_scan is not None:
                return (*scans, store_scan)
            anchor, *rest = scans
            return (replace(anchor, vector_search=search, limit=top, maximum_rows=None), *rest)

        def build(node: PhysicalNode, choice_notes: tuple[str, ...]) -> SimilarRank:
            return SimilarRank(
                input=node,
                field=field,
                text=term.text,
                top=top,
                first=first,
                relevance_order=not query.explicit_order,
                via_store=store is not None,
                exact=exact,
                embedding_model=search.model,
                embedding_dimensions=search.dimensions,
                properties=properties_from(
                    node.properties,
                    location=coordinator_location(),
                    ordering=None,
                    completeness=ResultCompleteness.EXACT if exact else ResultCompleteness.APPROXIMATE,
                ),
                choice_reasons=choice_notes,
            )

        strategy = Strategy(variant=variant, prepare=prepare, build=build)
        return ExtensionPlan(self.kind, (strategy,), strategy, lambda chosen: notes)

    # --- where the vectors are, and whether the search can run exactly -------------------

    @staticmethod
    def _binding(core: SourceResolvedQuery, term: BoundSimilarPredicate):
        binding = core.identity_source.embeddings.get(term.field.name)
        if binding is not None:
            return binding, None
        for store in core.extension_sources:
            binding = store.embeddings.get(term.field.name)
            if binding is not None:
                return binding, store
        return None, None

    def _refusals(self, core, term, binding, store, fully_pushed: bool) -> list[str]:
        services, policy = self._services, self._policy
        reasons: list[str] = []
        anchor = core.identity_source
        if binding is None:
            reasons.append(f"no source holds embeddings for field '{term.field.name}'")
            return reasons
        if policy.embedder is None or policy.embedder_dimensions is None:
            reasons.append("no embedding provider is configured")
        elif policy.embedder.model != binding.model or policy.embedder_dimensions != binding.dimensions:
            reasons.append(
                f"the embedding provider ({policy.embedder.model}, {policy.embedder_dimensions} dimensions) does not match "
                f"the stored embeddings ({binding.model}, {binding.dimensions} dimensions)"
            )
        ranking_source = anchor if store is None else store
        vector_caps = services.capabilities(ranking_source.source_kind).vector_search
        if vector_caps is None:
            reasons.append(f"source '{ranking_source.source_name}' cannot do vector search")
        else:
            if binding.metric not in vector_caps.metrics:
                reasons.append(f"source '{ranking_source.source_name}' cannot rank by '{binding.metric.value}' distance")
            needs_combining = core.query.where is not None or core.shape is QuerySourceShape.MULTI_SOURCE
            if store is None and needs_combining and not vector_caps.combines_with_filters:
                reasons.append("the source cannot combine vector ranking with the other conditions")
            if store is not None and core.shape is QuerySourceShape.MULTI_SOURCE and not vector_caps.combines_with_filters:
                reasons.append("the vector store cannot combine ranking with key restrictions")
        if store is not None:
            if services.capabilities(anchor.source_kind).key_lookup is None:
                reasons.append("the identity source cannot be restricted by IDs, so the ranked IDs could not narrow its read")
            if not services.policy.maximum_transfer_keys:
                reasons.append("key transfer is disabled, so the ranked IDs could not narrow the identity source's read")
        if not fully_pushed:
            reasons.append(
                "a condition is not enforced by its source (contains/starts_with, or one mixing sources), "
                "so the rows could not be ranked among its matches"
            )
        return reasons

    def _cap(self, core: SourceResolvedQuery, store: SingleSourceQueryBinding | None) -> tuple[int, tuple[str, ...]]:
        services = self._services
        limits = [self._policy.maximum_top]
        ranking_source = store or core.identity_source
        vector_caps = services.capabilities(ranking_source.source_kind).vector_search
        if vector_caps is not None and vector_caps.maximum_shortlist is not None:
            limits.append(vector_caps.maximum_shortlist)
        if store is not None:
            anchor_caps = services.capabilities(core.identity_source.source_kind)
            limits.append(services.policy.maximum_transfer_keys)
            if anchor_caps.key_lookup is not None:
                limits.append(anchor_caps.key_lookup.maximum_keys)
        return min(limit for limit in limits if limit), ()

    def _store_scan(self, store: SingleSourceQueryBinding | None, search: VectorSearch, query: BoundQuery) -> RemoteScan | None:
        if store is None:
            return None
        caps = self._services.capabilities(store.source_kind)
        projection = (store.logical_id,)
        return RemoteScan(
            source=store,
            projection=projection,
            pushed_filter=None,
            order_by=(),
            limit=search.shortlist_size,
            maximum_rows=None,
            key_lookup_limit=None if caps.key_lookup is None else caps.key_lookup.maximum_keys,
            vector_search=search,
            properties=PlanProperties(
                output_fields=projection,
                logical_id=store.logical_id,
                ids_are_unique=True,
                ordering=None,
                location=remote_location(store.source_name),
                completeness=ResultCompleteness.EXACT,
                result_shape=ResultShape.RECORDS,
                catalog_fingerprint=query.catalog_fingerprint,
            ),
        )


@support_view(OperatorKind.SIMILARITY_SEARCH)
def _similar_support(caps: SourceCapabilities) -> OperatorSupport:
    if caps.vector_search is None:
        return OperatorSupport(OperatorKind.SIMILARITY_SEARCH, SupportLevel.UNAVAILABLE, (), "no vector search")
    return OperatorSupport(
        OperatorKind.SIMILARITY_SEARCH,
        SupportLevel.SOURCE,
        ("ranked_read", "ranked_store"),
        "the source ranks by vector distance; YoDb keeps the K nearest and their distances",
    )
