"""A small in-memory 'database world' for tests: real filtering, key lists and limits, no server.

The compiler's SQL is run on SQLite (``%s`` becomes ``?``; each schema is an attached database),
so a source returns exactly the rows its query asks for.  Use it where a test must see what a
predicate or an ``IN`` list really selects; use the plain fakes where it does not matter.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from math import sqrt
from pathlib import Path
import re
import sqlite3
import threading
from tempfile import TemporaryDirectory
import weakref

from yodb.catalog import SourceKind, load_catalog
from yodb.inspection import SourceInspection, SourceValidationReport
from yodb.runtime import CatalogEvaluation, SourceRuntimeState, SourceRuntimeStatus

_NOW = datetime(2026, 10, 3, tzinfo=UTC)


def evaluation(datasets: str, sources: str, relations: str, rules: str | None = None) -> CatalogEvaluation:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        files = (("datasets", datasets), ("sources", sources), ("relations", relations), ("rules", rules))
        for name, text in files:
            if text is not None:
                (root / f"{name}.yaml").write_text(text, encoding="utf-8")
        catalog = load_catalog(root)
    return CatalogEvaluation(
        catalog=catalog,
        evaluated_at=_NOW,
        sources={
            name: SourceRuntimeState(
                source_name=name,
                status=SourceRuntimeStatus.VALID,
                inspection=SourceInspection(source_name=name, source_kind=source.kind, inspected_at=_NOW),
                validation=SourceValidationReport(source_name=name, inspected_at=_NOW),
            )
            for name, source in catalog.sources.items()
        },
    )


def _vector(text):
    return [float(x) for x in json.loads(text)]


def _cosine(a, b):
    a, b = _vector(a), _vector(b)
    dot = sum(x * y for x, y in zip(a, b))
    norm = sqrt(sum(x * x for x in a)) * sqrt(sum(y * y for y in b))
    return 1.0 - (dot / norm if norm else 0.0)


def _l2(a, b):
    return sqrt(sum((x - y) ** 2 for x, y in zip(_vector(a), _vector(b))))


def _inner(a, b):
    return -sum(x * y for x, y in zip(_vector(a), _vector(b)))


# pgvector's distance operators, as SQLite functions over vectors stored as JSON text.
_DISTANCE_OPERATORS = {"<=>": "yodb_cosine", "<->": "yodb_l2", "<#>": "yodb_inner"}
_DISTANCE = re.compile(r'("[^"]+") (<=>|<->|<#>) %s::vector')


class SqlWorld:
    """Executes compiled PostgreSQL-dialect scans against in-memory tables.

    A ranked scan (``ORDER BY "col" <=> %s::vector``) works too: a vector column holds
    each vector as JSON text, and the distance operators become SQLite functions.
    """

    source_kind = SourceKind.POSTGRES

    def __init__(self, tables: dict[str, tuple[tuple[str, ...], list[tuple]]]) -> None:
        """``tables``: ``"schema.table" -> (column names, rows)``."""

        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        weakref.finalize(self, self._db.close)          # closed when the world is garbage, so tests need no cleanup
        self._db.create_function("yodb_cosine", 2, _cosine, deterministic=True)
        self._db.create_function("yodb_l2", 2, _l2, deterministic=True)
        self._db.create_function("yodb_inner", 2, _inner, deterministic=True)
        attached = set()
        for resource, (columns, rows) in tables.items():
            schema, table = resource.split(".")
            if schema not in attached:
                self._db.execute(f'ATTACH DATABASE ":memory:" AS "{schema}"')
                attached.add(schema)
            self._db.execute(f'CREATE TABLE "{schema}"."{table}" ({", ".join(f"{c}" for c in columns)})')
            self._db.executemany(f'INSERT INTO "{schema}"."{table}" VALUES ({", ".join("?" for _ in columns)})', rows)
        self.queries = []
        self._lock = threading.Lock()      # one SQLite connection: a server that runs tools in threads must take turns

    def execute(self, query, *, timeout_seconds=None):
        with self._lock:
            self.queries.append(query)
            sql = _DISTANCE.sub(lambda m: f"{_DISTANCE_OPERATORS[m.group(2)]}({m.group(1)}, %s)", query.sql)
            sql = sql.replace('COLLATE "C"', "COLLATE BINARY")          # SQLite's byte-order collation
            rows = self._db.execute(sql.replace("%s", "?"), query.parameters).fetchall()
        names = [column.logical_field for column in query.output_columns]
        return tuple(dict(zip(names, row)) for row in rows)

    def reads_of(self, source: str) -> list:
        return [q for q in self.queries if q.source_name == source]
