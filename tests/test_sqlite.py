"""SQLite as a source: the real client against a temporary SQLite file, compared with plain SQL.  No Docker."""

from __future__ import annotations

from datetime import UTC, datetime
import os
import sqlite3
import tempfile
import threading
import unittest

import yaml

import yodb
from yodb.catalog import CatalogValidationError, SourceKind, load_catalog
from yodb.connections import EnvSqliteConnectionResolver, SqliteConnectionAdapter, SqliteConnectionSettings, MappingSqliteConnectionResolver
from yodb.connections.sqlite import settings_from_text
from yodb.errors import CatalogRuntimeError, ErrorCode, SourceConnectionError, YoDbError
from yodb.inspection import InspectionRequest, SqliteSourceInspector
from yodb.inspection.sqlite import sqlite_type_family

FIELDS = {
    "id": ("id", "i"), "name": ("string", "n"), "seats": ("int", "s"), "score": ("float", "x"),
    "active": ("bool", "a"), "created": ("timestamp", "c"),
}
ROWS = [
    (1, "bravo", 10, 1.5, 1, "2026-01-02 03:04:05"),
    (2, "alpha", 5, None, 0, 1767225600),                 # epoch seconds
    (3, None, None, 2.5, 1, None),
    (4, "Zulu", 10, 0.5, 0, "2026-03-01T00:00:00Z"),
    (5, "alpha", 7, 3.5, 1, "2026-02-01 00:00:00"),
]
TABLE = "create table acct(id integer primary key, name text collate nocase, seats integer, score real, active boolean, created datetime)"


def write_catalog(directory: str, kind: str = "sqlite", extra: dict | None = None) -> str:
    root = os.path.join(directory, "catalog")
    os.makedirs(root, exist_ok=True)
    names = list(FIELDS)
    source = {"kind": kind, "connection_ref": "local", "read_only": True,
              "datasets": {"acct": {"resource": "acct", "identity": ["id"], "fields": {n: {"physical_name": n} for n in names}}}}
    source.update(extra or {})
    docs = {
        "datasets": {"api_version": "yodb/v0.1", "catalog": {"name": "t", "version": 1},
                     "datasets": {"acct": {"description": "d", "fields": {n: {"type": t, "description": d} for n, (t, d) in FIELDS.items()}}}},
        "sources": {"api_version": "yodb/v0.1", "sources": {"s": source},
                    "resolution": {"acct": {"identity_source": "s", "field_sources": {n: "s" for n in names}}}},
        "relations": {"api_version": "yodb/v0.1", "relationships": {}},
    }
    for name, doc in docs.items():
        with open(os.path.join(root, f"{name}.yaml"), "w") as handle:
            yaml.safe_dump(doc, handle)
    return root


class SqliteSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = os.path.join(self._dir.name, "acct.db")
        raw = sqlite3.connect(self.path)
        raw.execute(TABLE)
        raw.executemany("insert into acct values (?,?,?,?,?,?)", ROWS)
        raw.commit()
        raw.close()
        self.catalog = write_catalog(self._dir.name)
        self.db = yodb.connect(self.catalog, {"local": self.path})
        self.addCleanup(self.db.close)
        self.oracle = sqlite3.connect(self.path)
        self.addCleanup(self.oracle.close)

    def ask(self, select, where=None, order=None, first=50):
        query = {"from": {"dataset": "acct"}, "select": select, "page": {"first": first}}
        if where is not None:
            query["where"] = where
        if order is not None:
            query["order_by"] = [{"field": f, "direction": d} for f, d in order]
        return [dict(row) for row in self.db.query(query).rows]

    def test_every_field_is_returned_in_its_logical_type(self) -> None:
        rows = {row["id"]: row for row in self.ask(list(FIELDS))}
        self.assertIs(rows[1]["active"], True)
        self.assertIs(rows[2]["active"], False)
        self.assertEqual(rows[1]["created"], datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC))
        self.assertEqual(rows[2]["created"], datetime(2026, 1, 1, tzinfo=UTC))        # an epoch number
        self.assertEqual(rows[4]["created"], datetime(2026, 3, 1, tzinfo=UTC))        # ISO with T and Z
        self.assertIsNone(rows[3]["created"])
        self.assertIsInstance(rows[1]["score"], float)

    def test_filters_match_plain_sql(self) -> None:
        cases = [
            ({"field": "seats", "op": "gte", "value": 7}, "seats >= 7"),
            ({"field": "seats", "op": "in", "value": [5, 7]}, "seats in (5, 7)"),
            ({"field": "seats", "op": "not_in", "value": [5, 7]}, "seats not in (5, 7)"),
            ({"field": "name", "op": "is_null"}, "name is null"),
            ({"field": "name", "op": "ne", "value": "alpha"}, "name collate binary <> 'alpha'"),
            ({"field": "active", "op": "eq", "value": True}, "active = 1"),
            ({"any": [{"field": "seats", "op": "lt", "value": 6}, {"field": "score", "op": "gt", "value": 3}]}, "seats < 6 or score > 3"),
            ({"not": {"field": "seats", "op": "eq", "value": 10}}, "not (seats = 10)"),
        ]
        for where, sql in cases:
            with self.subTest(sql):
                expected = sorted(r[0] for r in self.oracle.execute(f"select id from acct where {sql}"))
                self.assertEqual(sorted(r["id"] for r in self.ask(["name"], where)), expected)

    def test_text_compares_in_byte_order_whatever_the_column_collation(self) -> None:
        # the column is COLLATE NOCASE; YoDb's text equality is case-sensitive
        self.assertEqual([r["id"] for r in self.ask(["name"], {"field": "name", "op": "eq", "value": "zulu"})], [])
        self.assertEqual([r["id"] for r in self.ask(["name"], {"field": "name", "op": "eq", "value": "Zulu"})], [4])

    def test_order_matches_the_coordinators_null_and_text_order(self) -> None:
        asc = [r["id"] for r in self.ask(["name"], order=[("name", "asc"), ("id", "asc")])]
        desc = [r["id"] for r in self.ask(["name"], order=[("name", "desc"), ("id", "asc")])]
        self.assertEqual(asc, [4, 2, 5, 1, 3])      # "Zulu" < "alpha" < "bravo" by code point; NULL last
        self.assertEqual(desc, [3, 1, 2, 5, 4])     # NULL first

    def test_timestamp_filters_and_order_run_in_yodb_over_mixed_storage(self) -> None:
        where = {"field": "created", "op": "gte", "value": "2026-01-15T00:00:00Z"}
        self.assertEqual(sorted(r["id"] for r in self.ask(["name"], where)), [4, 5])
        ordered = [r["id"] for r in self.ask(["name"], {"field": "created", "op": "is_not_null"}, order=[("created", "asc")])]
        self.assertEqual(ordered, [2, 1, 5, 4])

    def test_paging_limits_rows(self) -> None:
        self.assertEqual(len(self.ask(["name"], first=2)), 2)

    def test_status_and_describe(self) -> None:
        self.assertEqual(self.db.status()["sources"], {"s": "valid"})
        self.assertIn("acct", self.db.describe())

    def test_the_source_is_read_only(self) -> None:
        with self.db._engine._executors.adapter_for(SourceKind.SQLITE)._connections.acquire("local") as connection:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("insert into acct(id) values (99)")


class SqliteCatalogRuleTests(unittest.TestCase):
    def test_graph_and_embeddings_are_rejected_on_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            graph = {"graph": {"name": "g", "vertices": {"acct": {"label": "Acct"}}}}
            with self.assertRaisesRegex(CatalogValidationError, "graph is not supported on a sqlite"):
                load_catalog(write_catalog(directory, extra=graph))
        with tempfile.TemporaryDirectory() as directory:
            root = write_catalog(directory)
            path = os.path.join(root, "sources.yaml")
            datasets = os.path.join(root, "datasets.yaml")
            with open(path) as handle:
                doc = yaml.safe_load(handle)
            doc["sources"]["s"]["datasets"]["acct"]["embeddings"] = {
                "name": {"column": "v", "model": "m", "dimensions": 3}}
            with open(path, "w") as handle:
                yaml.safe_dump(doc, handle)
            with open(datasets) as handle:
                d = yaml.safe_load(handle)
            d["datasets"]["acct"]["fields"]["name"]["semantic_eligible"] = True
            with open(datasets, "w") as handle:
                yaml.safe_dump(d, handle)
            with self.assertRaisesRegex(CatalogValidationError, "embeddings is not supported on a sqlite"):
                load_catalog(root)


class SqliteInspectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = os.path.join(self._dir.name, "x.db")
        raw = sqlite3.connect(self.path)
        raw.executescript(
            "create table acct(id integer primary key, name text not null, ref text unique);"
            "create table pet(id text, owner integer references acct(id), primary key (id));"
            "create view v as select id from acct; create index pet_owner on pet(owner);"
        )
        raw.close()

    def inspect(self):
        adapter = SqliteConnectionAdapter(MappingSqliteConnectionResolver({"local": SqliteConnectionSettings(self.path)}))
        self.addCleanup(adapter.close)
        catalog = load_catalog(write_catalog(self._dir.name))
        return SqliteSourceInspector(adapter).inspect(InspectionRequest(source_name="s", source=catalog.sources["s"]))

    def test_resources_keys_and_foreign_keys(self) -> None:
        resources = self.inspect().resources
        self.assertEqual(set(resources), {"acct", "pet", "v"})
        self.assertEqual(resources["acct"].primary_key.fields, ("id",))
        self.assertEqual([u.fields for u in resources["acct"].unique_constraints], [("ref",)])
        self.assertEqual(resources["pet"].primary_key.fields, ("id",))
        fk = resources["pet"].foreign_keys[0]
        self.assertEqual((fk.fields, fk.target_resource, fk.target_fields), (("owner",), "acct", ("id",)))
        self.assertEqual(resources["v"].kind.value, "view")

    def test_type_families(self) -> None:
        cases = {"INTEGER": "int", "VARCHAR(20)": "string", "TEXT": "string", "REAL": "float", "NUMERIC": "float",
                 "BOOLEAN": "bool", "DATETIME": "timestamp", "TIMESTAMP": "timestamp", "UUID": "uuid",
                 "JSON": "json", "BLOB": "bytes", "": "other"}
        for declared, family in cases.items():
            self.assertEqual(sqlite_type_family(declared), family, declared)

    def test_a_catalog_that_does_not_match_is_refused(self) -> None:
        root = write_catalog(self._dir.name)          # binds acct.seats, which this file lacks
        with self.assertRaises(CatalogRuntimeError) as caught:
            yodb.connect(root, {"local": self.path})
        self.assertIn("does not match", caught.exception.detail.message)


class SqliteConnectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = os.path.join(self._dir.name, "c.db")
        sqlite3.connect(self.path).execute("create table t(a)").connection.close()

    def adapter(self, path=None):
        adapter = SqliteConnectionAdapter(MappingSqliteConnectionResolver({"r": SqliteConnectionSettings(path or self.path)}))
        self.addCleanup(adapter.close)
        return adapter

    def test_one_connection_per_thread_reused_within_a_thread(self) -> None:
        adapter = self.adapter()
        seen = []

        def grab():
            with adapter.acquire("r") as first, adapter.acquire("r") as second:
                seen.append((first, second))

        threads = [threading.Thread(target=grab) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertIs(seen[0][0], seen[0][1])
        self.assertIsNot(seen[0][0], seen[1][0])

    def test_a_missing_file_is_a_coded_error_and_is_not_created(self) -> None:
        missing = os.path.join(self._dir.name, "nope.db")
        with self.assertRaises(SourceConnectionError) as caught:
            self.adapter(missing).acquire("r").__enter__()
        self.assertEqual(caught.exception.code, ErrorCode.SOURCE_UNAVAILABLE)
        self.assertFalse(os.path.exists(missing))

    def test_closed_adapter_refuses(self) -> None:
        adapter = self.adapter()
        adapter.close()
        with self.assertRaises(SourceConnectionError) as caught:
            adapter.acquire("r")
        self.assertEqual(caught.exception.code, ErrorCode.CONNECTION_POOL_CLOSED)

    def test_env_resolver_and_value_formats(self) -> None:
        resolver = EnvSqliteConnectionResolver(environ={"YODB_CONN_LOCAL_DB": "/tmp/a.db"})
        self.assertEqual(resolver.resolve("local-db").path, "/tmp/a.db")
        with self.assertRaises(SourceConnectionError) as caught:
            resolver.resolve("other")
        self.assertIn("YODB_CONN_OTHER", caught.exception.detail.message)
        self.assertEqual(settings_from_text("sqlite:////abs/x.db").path, "/abs/x.db")
        self.assertEqual(settings_from_text("sqlite:///rel/x.db").path, "rel/x.db")
        self.assertEqual(settings_from_text("file:/abs/x.db?mode=rw").path, "/abs/x.db")


class SqliteTimeoutTests(unittest.TestCase):
    def test_a_slow_statement_is_stopped_and_coded_as_a_timeout(self) -> None:
        from yodb.compilation import CompiledSqliteQuery
        from yodb.execution import SqliteQueryExecutionAdapter

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "t.db")
            sqlite3.connect(path).close()
            connections = SqliteConnectionAdapter(MappingSqliteConnectionResolver({"r": SqliteConnectionSettings(path)}))
            self.addCleanup(connections.close)
            slow = CompiledSqliteQuery(
                source_name="s", connection_ref="r", parameters=(), output_columns=(),
                sql="WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT count(*) FROM n",
            )
            with self.assertRaises(YoDbError) as caught:
                SqliteQueryExecutionAdapter(connections).execute(slow, timeout_seconds=0.2)
            self.assertEqual(caught.exception.code, ErrorCode.QUERY_TIMEOUT)


if __name__ == "__main__":
    unittest.main()
