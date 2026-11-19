"""Execution of the SIMILARITY_SEARCH operator: embed the text, run the ranked read, keep the K nearest."""

from __future__ import annotations

from dataclasses import replace

from ..compilation.postgres import DISTANCE_COLUMN
from ..errors import ErrorCode, ErrorDetail, QueryExecutionError, YoDbError
from ..execution.contracts import LogicalRow
from ..execution.operators.base import ExecutionContext, Run, fail
from ..planning import RemoteScan, transform_plan
from ..semantic.contracts import EmbeddingProvider, EmbeddingRequest
from .contracts import SimilarExecutionStats, SimilarQueryReport, SimilarRecord
from .planning import SimilarRank

NAME = "similar"


def execute(embedder: EmbeddingProvider | None, ctx: ExecutionContext, node: SimilarRank, run: Run) -> tuple[LogicalRow, ...]:
    if embedder is None:
        fail(ErrorCode.SEMANTIC_PROVIDER_UNAVAILABLE, "No embedding provider is configured, so a similarity search cannot run.")
    vector = _embed(embedder, node, run)
    rows = ctx.execute(_with_query_vector(node.input, vector), run)
    if run.fell_back:
        # The ranked read would have been cut before the other conditions narrowed it (too many IDs to
        # transfer), so a plain read ran instead; without the vectors nothing can be ranked here.
        fail(
            ErrorCode.QUERY_ROW_LIMIT_EXCEEDED,
            "The rows to rank could not be narrowed to the matches of the other conditions (too many IDs would "
            "have to be transferred between sources); add a more selective filter.",
        )
    ranked = sorted(
        (row for row in rows if isinstance(row.get(DISTANCE_COLUMN), (int, float))),
        key=lambda row: (float(row[DISTANCE_COLUMN]), str(row.get("id"))),
    )[: node.top]
    records = {
        row["id"]: SimilarRecord(distance=float(row[DISTANCE_COLUMN]), rank=rank) for rank, row in enumerate(ranked, start=1)
    }
    if run.trace is not None:
        run.trace.reports[NAME] = SimilarQueryReport(
            stats=SimilarExecutionStats(
                field=node.field.field.name,
                text=node.text,
                top=node.top,
                candidates=len(rows),
                returned=len(ranked),
                via_store=node.via_store,
                embedding_model=node.embedding_model,
                exact=node.exact,
            ),
            records=records,
        ).combined_with(run.trace.reports.get(NAME))
    kept = ranked[: node.first] if node.relevance_order else ranked
    return tuple({name: value for name, value in row.items() if name != DISTANCE_COLUMN} for row in kept)


def _embed(embedder: EmbeddingProvider, node: SimilarRank, run: Run) -> tuple[float, ...]:
    try:
        result = embedder.embed(EmbeddingRequest((node.text,), run.remaining()))
    except YoDbError:
        raise
    except Exception as error:
        raise QueryExecutionError(
            ErrorDetail(code=ErrorCode.SEMANTIC_PROVIDER_FAILED, message="The embedding provider failed.", retryable=False)
        ) from error
    if result.info.model != node.embedding_model or result.dimensions != node.embedding_dimensions or len(result.vectors) != 1:
        fail(
            ErrorCode.QUERY_PLAN_INVARIANT_VIOLATION,
            "The embedding provider does not match the stored embedding model and dimensions.",
        )
    return result.vectors[0]


def _with_query_vector(plan, vector: tuple[float, ...]):
    """Fill the embedded query vector into the one scan that searches by vector."""

    def fill(node):
        if isinstance(node, RemoteScan) and node.vector_search is not None and node.vector_search.query_vector is None:
            return replace(node, vector_search=replace(node.vector_search, query_vector=vector))
        return node

    return transform_plan(plan, fill)
