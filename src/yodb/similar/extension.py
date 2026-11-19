"""The similarity search as one pluggable extension.

    extension = SimilarExtension(embedder)
    engine = QueryExecutionEngine(catalog, compilers, executors, extensions=[extension])

The query term (``{"similar": ...}``) is registered by default in
:mod:`yodb.query.similar`; this supplies the planning operator and the execution
handler.  A result's report is ``result.reports["similar"]``.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

from ..execution.operators.base import Handler
from ..planning.operators.base import PlanningServices
from ..semantic.contracts import EmbeddingProvider
from .execution import NAME, execute
from .planning import SimilarOperator, SimilarPolicy, SimilarRank


class SimilarExtension:
    """Plan and run similarity searches with the given embedding provider.

    ``embedder`` may be omitted to plan only (or to get a clear error when a query
    asks for a search with no provider).  Unless given, the policy learns the
    embedding space from the embedder and the cost from its ``cost_hint``.
    """

    name = NAME

    def __init__(self, embedder: EmbeddingProvider | None = None, *, policy: SimilarPolicy | None = None) -> None:
        self.embedder = embedder
        self.policy = policy or _policy_from(embedder)

    def planning_operator(self, services: PlanningServices) -> SimilarOperator:
        return SimilarOperator(services, self.policy)

    def handlers(self) -> Mapping[type, Handler]:
        return {SimilarRank: partial(execute, self.embedder)}


def _policy_from(embedder: EmbeddingProvider | None) -> SimilarPolicy:
    if embedder is None:
        return SimilarPolicy()
    hint = getattr(embedder, "cost_hint", None)
    return SimilarPolicy(
        embedder=embedder.info,
        embedder_dimensions=embedder.dimensions,
        **({"embedding_cost": hint} if hint is not None else {}),
    )
