"""The similarity search: the K nearest records by embedding distance, answered by pgvector.

``SimilarExtension`` is the interface to the engine; everything else here is what
it is built from.  The embedding provider contract is shared with the semantic
filter (:class:`yodb.semantic.EmbeddingProvider`).
"""

from .contracts import SimilarExecutionStats, SimilarQueryReport, SimilarRecord
from .execution import execute as execute_similar_rank
from .extension import SimilarExtension
from .planning import SimilarOperator, SimilarPolicy, SimilarRank

__all__ = [
    "SimilarExecutionStats",
    "SimilarExtension",
    "SimilarOperator",
    "SimilarPolicy",
    "SimilarQueryReport",
    "SimilarRank",
    "SimilarRecord",
    "execute_similar_rank",
]
