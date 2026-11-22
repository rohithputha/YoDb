"""SQLite statistics from ``sqlite_stat1``, the table ``ANALYZE`` fills.

Only what ``ANALYZE`` already collected is read: the table's row count, and for the leading column
of each index the number of distinct values (rows divided by the average rows per value).  A database
that was never analyzed has no ``sqlite_stat1``; that, or any failure, is simply "unknown" and the
planner falls back to its fixed defaults.  There are no histograms or most-common values.
"""

from __future__ import annotations

from collections.abc import Callable
import time
from threading import Lock
from typing import Any

from ..connections import SourceConnectionAdapter
from ..query.resolution import SingleSourceQueryBinding
from .statistics import ColumnStatistics, SourceStatistics

_STAT_SQL = "SELECT idx, stat FROM sqlite_stat1 WHERE tbl = ?"
_LEADING_COLUMN_SQL = "SELECT name FROM pragma_index_info(?) WHERE seqno = 0"


class SqliteStatisticsProvider:
    def __init__(
        self,
        connections: SourceConnectionAdapter[Any],
        *,
        ttl_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must not be negative")
        self._connections = connections
        self._ttl = ttl_seconds
        self._clock = clock
        # (connection_ref, resource) -> (expires, row count, distinct count by physical column) or None
        self._cache: dict[tuple[str, str], tuple[float, tuple[float, dict[str, float]] | None]] = {}
        self._lock = Lock()
        self.last_error: str | None = None

    def statistics(self, source: SingleSourceQueryBinding) -> SourceStatistics | None:
        key = (source.connection_ref, source.resource)
        now = self._clock()
        with self._lock:
            cached = self._cache.get(key)
        if cached is None or cached[0] <= now:
            try:
                entry = self._read(source)
            except Exception as error:  # noqa: BLE001 - unknown, never a query failure
                entry = None
                self.last_error = type(error).__name__
            with self._lock:
                self._cache[key] = (now + self._ttl, entry)
        else:
            entry = cached[1]
        if entry is None:
            return None
        row_count, distinct = entry
        columns = {
            field.field.name: ColumnStatistics(distinct_count=min(distinct[field.physical_name], row_count))
            for field in (source.logical_id, *source.fields)
            if field.physical_name in distinct
        }
        return SourceStatistics(row_count=row_count, columns=columns)

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()

    def _read(self, source: SingleSourceQueryBinding) -> tuple[float, dict[str, float]] | None:
        with self._connections.acquire(source.connection_ref) as connection:
            try:
                stats = connection.execute(_STAT_SQL, (source.resource,)).fetchall()
            except Exception:  # noqa: BLE001 - no sqlite_stat1: never analyzed
                return None
            if not stats:
                return None
            row_count = 0.0
            distinct: dict[str, float] = {}
            for index, stat in stats:
                numbers = [float(part) for part in str(stat).split() if part.replace(".", "", 1).isdigit()]
                if not numbers:
                    continue
                row_count = max(row_count, numbers[0])
                if index is None or len(numbers) < 2 or numbers[1] <= 0:
                    continue
                leading = connection.execute(_LEADING_COLUMN_SQL, (index,)).fetchall()
                if leading and leading[0][0] is not None:
                    distinct[leading[0][0]] = max(1.0, numbers[0] / numbers[1])
        return row_count, distinct
