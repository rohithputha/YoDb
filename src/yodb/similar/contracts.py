"""What a similarity search reports: how it ran, and the distance and rank of every record it returned."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimilarRecord:
    """Why one record is in the answer: its distance to the query text, and its rank (1 is nearest)."""

    distance: float
    rank: int


@dataclass(frozen=True)
class SimilarExecutionStats:
    """Actual work of one similarity search."""

    field: str
    text: str
    top: int                        # the K asked for
    candidates: int                 # rows the ranking chose from, as far as the source reported them
    returned: int                   # rows kept (at most ``top``)
    via_store: bool                 # the vectors were read from a separate vector-store source
    embedding_model: str
    embedding_model_calls: int = 1
    # False when the rows were ranked before some other condition was applied (a later restriction may
    # have dropped some of the K nearest, so fewer than ``top`` rows can come back): see the planner.
    exact: bool = True

    def __post_init__(self) -> None:
        if min(self.top, self.candidates, self.returned) < 0 or self.returned > self.top:
            raise ValueError("returned <= top and no negative counts")


@dataclass(frozen=True)
class SimilarQueryReport:
    """What a similarity search did and, per returned record, its distance and rank."""

    stats: SimilarExecutionStats
    records: dict[object, SimilarRecord]

    def restricted_to(self, ids) -> "SimilarQueryReport":
        """The same report keeping per-record entries only for the returned rows."""

        return SimilarQueryReport(self.stats, {key: record for key, record in self.records.items() if key in ids})

    def combined_with(self, earlier: "SimilarQueryReport | None") -> "SimilarQueryReport":
        """This report added to an earlier one of the same query (a join may run the search in batches)."""

        if earlier is None:
            return self
        mine, theirs = self.stats, earlier.stats
        records = {**earlier.records, **self.records}
        ranked = sorted(records.items(), key=lambda item: (item[1].distance, str(item[0])))
        return SimilarQueryReport(
            SimilarExecutionStats(
                field=mine.field,
                text=mine.text,
                top=mine.top,
                candidates=mine.candidates + theirs.candidates,
                returned=min(mine.top, len(ranked)),
                via_store=mine.via_store,
                embedding_model=mine.embedding_model,
                embedding_model_calls=mine.embedding_model_calls + theirs.embedding_model_calls,
                exact=mine.exact and theirs.exact,
            ),
            {key: SimilarRecord(record.distance, rank) for rank, (key, record) in enumerate(ranked, start=1)},
        )
