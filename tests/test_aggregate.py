"""Aggregates (``aggregate`` / ``group_by``): validation, planning, compilation, and results compared with plain SQL.

The end-to-end half runs the real client against a temporary SQLite file (no Docker) twice: once with the source
computing the groups, once with the source's aggregate capability removed so YoDb groups the rows itself.  Both must
equal what SQLite itself says.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import math
import os
import sqlite3
import tempfile
import unittest

import yaml

import yodb
from yodb.errors import ErrorCode, QueryError, YoDbError
from yodb.execution.operators.aggregate import aggregate_rows
from yodb.execution.postgres import _logical_row
from yodb.catalog import LogicalType
from yodb.compilation import PostgresQueryCompiler
from yodb.planning import (
    AggregateMeasure,
    AggregateSpec,
    CapabilityPlanningAdapter,
    FederatedPhysicalPlanner,
    PlannerPolicy,
    PostgresPlanningAdapter,
    SQLITE_CAPABILITIES,
    SourcePlanningRegistry,
    SqlitePlanningAdapter,
)
from yodb.query import AggregateFunction, bind_query, parse_query, resolve_query_sources

from support.catalogs import multi_source_catalog

FIELDS = {
    "id": ("id", "i"), "region": ("string", "r"), "status": ("string", "s"), "amount": ("int", "a"),
    "price": ("float", "p"), "paid": ("bool", "b"), "placed": ("timestamp", "t"), "note": ("json", "n"),
}
ROWS = [
    (1, "north", "open", 10, 1.5, 1, "2026-01-01 00:00:00", None),
    (2, "north", "open", 20, 2.5, 0, "2026-01-01T00:00:00Z", None),       # the same instant as row 1, stored differently
    (3, "north", "done", 30, None, 1, 1767225600, None),
    (4, "south", "open", 5, 4.0, 0, "2026-02-01 00:00:00", None),
    (5, "south", "done", None, 0.5, 1, None, None),
    (6, None, "open", 7, 3.5, 1, "2026-03-01 00:00:00", None),
    (7, None, "done", 7, 2.0, 0, "2026-03-01 00:00:00", None),
    (8, "East", "open", 1, 1.0, 1, "2026-04-01 00:00:00", None),
]
TABLE = (
    "create table orders(id integer primary key, region text collate nocase, status text, amount integer, "
    "price real, paid boolean, placed datetime, note json)"
)


def write_catalog(directory: str) -> str:
    root = os.path.join(directory, "catalog")
    os.makedirs(root, exist_ok=True)
    names = list(FIELDS)
    docs = {
        "datasets": {"api_version": "yodb/v0.1", "catalog": {"name": "t", "version": 1},
                     "datasets": {"order": {"description": "d", "fields": {n: {"type": t, "description": d} for n, (t, d) in FIELDS.items()}}}},
        "sources": {"api_version": "yodb/v0.1",
                    "sources": {"s": {"kind": "sqlite", "connection_ref": "local", "read_only": True,
                                      "datasets": {"order": {"resource": "orders", "identity": ["id"],
                                                             "fields": {n: {"physical_name": n} for n in names}}}}},
                    "resolution": {"order": {"identity_source": "s", "field_sources": {n: "s" for n in names}}}},
        "relations": {"api_version": "yodb/v0.1", "relationships": {}},
    }
    for name, doc in docs.items():
        with open(os.path.join(root, f"{name}.yaml"), "w") as handle:
            yaml.safe_dump(doc, handle)
    return root


def code_of(call) -> ErrorCode:
    try:
        call()
    except QueryError as error:
        return error.detail.code
    raise AssertionError("no error was raised")


def agg(fn, field=None, alias="x"):
    term = {"fn": fn, "as": alias}
    if field is not None:
        term["field"] = field
    return term


class AggregateWorld(unittest.TestCase):
    """A SQLite file, the client over it, and an oracle connection to the same file."""

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = os.path.join(self._dir.name, "orders.db")
        raw = sqlite3.connect(self.path)
        raw.execute(TABLE)
        raw.executemany("insert into orders values (?,?,?,?,?,?,?,?)", ROWS)
        raw.commit()
        raw.close()
        self.db = yodb.connect(write_catalog(self._dir.name), {"local": self.path})
        self.addCleanup(self.db.close)
        self.oracle = sqlite3.connect(self.path)
        self.addCleanup(self.oracle.close)

    def source_without_aggregates(self) -> None:
        """The same engine with a SQLite whose aggregate capability is removed, so YoDb must group the rows."""

        adapters = SourcePlanningRegistry(
            [PostgresPlanningAdapter(), CapabilityPlanningAdapter(replace(SQLITE_CAPABILITIES, aggregate=None))]
        )
        self.db._engine._planner = FederatedPhysicalPlanner(adapters)

    def ask(self, raw: dict) -> list[dict]:
        return [dict(row) for row in self.db.query({"from": {"dataset": "order"}, **raw}).rows]

    def kinds(self, raw: dict) -> list[str]:
        return [node.kind for node in self.db.explain({"from": {"dataset": "order"}, **raw}).nodes]


def same(test: unittest.TestCase, actual: list[dict], names: list[str], expected: list[tuple]) -> None:
    rows = [tuple(row[name] for name in names) for row in actual]
    test.assertEqual(len(rows), len(expected), (rows, expected))
    for got, want in zip(rows, expected):
        for one, other in zip(got, want):
            if isinstance(other, float) and one is not None:
                test.assertTrue(math.isclose(one, other), (got, want))
            else:
                test.assertEqual(one, other, (got, want))


# (name, query, SQL over SQLite, the output names).  Text is compared in byte order, as YoDb compares it.
ORACLE = [
    ("whole dataset", {"aggregate": [agg("count", alias="n"), agg("sum", "amount", "total"), agg("min", "amount", "lo"),
                                       agg("max", "amount", "hi"), agg("avg", "amount", "mean")]},
     "select count(*), sum(amount), min(amount), max(amount), avg(amount) from orders", ["n", "total", "lo", "hi", "mean"]),
    ("group by status", {"group_by": ["status"], "aggregate": [agg("count", alias="n"), agg("sum", "amount", "total")]},
     "select status, count(*), sum(amount) from orders group by status order by status", ["status", "n", "total"]),
    ("a text key declared NOCASE groups in byte order",
     {"group_by": ["region"], "aggregate": [agg("count", alias="n")]},
     "select region collate binary as r, count(*) from orders group by region collate binary "
     "order by r is null, r", ["region", "n"]),
    ("two keys, NULL keys form one group",
     {"group_by": ["region", "status"], "aggregate": [agg("count", alias="n")]},
     "select region collate binary as r, status, count(*) from orders group by r, status "
     "order by r is null, r, status", ["region", "status", "n"]),
    ("count of a field skips NULL, count(*) does not",
     {"group_by": ["status"], "aggregate": [agg("count", alias="rows"), agg("count", "amount", "amounts"),
                                              agg("count_distinct", "amount", "kinds")]},
     "select status, count(*), count(amount), count(distinct amount) from orders group by status order by status",
     ["status", "rows", "amounts", "kinds"]),
    ("sum and avg of a float", {"group_by": ["status"], "aggregate": [agg("sum", "price", "total"), agg("avg", "price", "mean")]},
     "select status, sum(price), avg(price) from orders group by status order by status", ["status", "total", "mean"]),
    ("min and max of text compare in byte order",
     {"group_by": ["status"], "aggregate": [agg("min", "region", "first"), agg("max", "region", "last")]},
     "select status, min(region collate binary), max(region collate binary) from orders group by status order by status",
     ["status", "first", "last"]),
    ("a filter before grouping",
     {"where": {"field": "amount", "op": "gte", "value": 7}, "group_by": ["status"],
      "aggregate": [agg("count", alias="n"), agg("max", "amount", "hi")]},
     "select status, count(*), max(amount) from orders where amount >= 7 group by status order by status",
     ["status", "n", "hi"]),
    ("a boolean key is a boolean", {"group_by": ["paid"], "aggregate": [agg("count", alias="n")]},
     "select paid, count(*) from orders group by paid order by paid", ["paid", "n"]),
    ("order by an alias, ties broken by the keys",
     {"group_by": ["region", "status"], "aggregate": [agg("count", alias="n")], "order_by": [{"field": "n", "direction": "desc"}]},
     "select region collate binary as r, status, count(*) as n from orders group by r, status "
     "order by n desc, r is null, r, status", ["region", "status", "n"]),
    ("order by a key, descending: NULLs first",
     {"group_by": ["region"], "aggregate": [agg("count", alias="n")], "order_by": [{"field": "region", "direction": "desc"}]},
     "select region collate binary as r, count(*) from orders group by r order by r is not null, r desc", ["region", "n"]),
    ("a page of groups",
     {"group_by": ["region", "status"], "aggregate": [agg("count", alias="n")], "page": {"first": 3}},
     "select region collate binary as r, status, count(*) from orders group by r, status "
     "order by r is null, r, status limit 3", ["region", "status", "n"]),
    ("order by a text aggregate",
     {"group_by": ["status"], "aggregate": [agg("min", "region", "first")], "order_by": [{"field": "first", "direction": "desc"}]},
     "select status, min(region collate binary) as f from orders group by status order by f desc, status", ["status", "first"]),
    ("distinct is group by with nothing to compute", {"group_by": ["status", "paid"]},
     "select distinct status, paid from orders order by status, paid", ["status", "paid"]),
    ("no rows without a key is one row", {"where": {"field": "amount", "op": "gt", "value": 1000},
                                          "aggregate": [agg("count", alias="n"), agg("sum", "amount", "total"), agg("min", "amount", "lo")]},
     "select count(*), sum(amount), min(amount) from orders where amount > 1000", ["n", "total", "lo"]),
    ("no rows with a key is no rows", {"where": {"field": "amount", "op": "gt", "value": 1000}, "group_by": ["status"],
                                       "aggregate": [agg("count", alias="n")]},
     "select status, count(*) from orders where amount > 1000 group by status", ["status", "n"]),
]


class OracleTests(AggregateWorld):
    def check(self) -> None:
        for name, query, sql, names in ORACLE:
            with self.subTest(name):
                expected = [tuple(row) for row in self.oracle.execute(sql)]
                if "paid" in names:
                    expected = [tuple(bool(v) if n == "paid" else v for n, v in zip(names, row)) for row in expected]
                same(self, self.ask(query), names, expected)

    def test_the_source_computes_the_groups_exactly_as_sql_does(self) -> None:
        self.check()
        self.assertEqual(self.kinds({"group_by": ["status"], "aggregate": [agg("count", alias="n")]}), ["remote_scan"])

    def test_yodb_computes_the_same_groups_when_the_source_cannot(self) -> None:
        self.source_without_aggregates()
        self.check()
        kinds = self.kinds({"group_by": ["status"], "aggregate": [agg("count", alias="n")]})
        self.assertEqual(kinds, ["remote_scan", "coordinator_aggregate", "coordinator_sort_page"])

    def test_a_timestamp_key_is_grouped_by_the_instant_not_by_how_it_is_stored(self) -> None:
        # rows 1, 2 and 3 are the same instant stored as text, as ISO with a Z, and as an epoch number
        query = {"group_by": ["placed"], "aggregate": [agg("count", alias="n")],
                 "where": {"field": "amount", "op": "in", "value": [10, 20, 30]}}
        for fallback in (False, True):
            with self.subTest(fallback=fallback):
                if fallback:
                    self.source_without_aggregates()
                groups = self.ask(query)
                self.assertEqual([(g["placed"], g["n"]) for g in groups], [(datetime(2026, 1, 1, tzinfo=UTC), 3)])

    def test_a_timestamp_extreme_runs_in_yodb_and_is_a_datetime(self) -> None:
        rows = self.ask({"aggregate": [agg("min", "placed", "first"), agg("max", "placed", "last")]})
        self.assertEqual(rows, [{"first": datetime(2026, 1, 1, tzinfo=UTC), "last": datetime(2026, 4, 1, tzinfo=UTC)}])
        self.assertIn("coordinator_aggregate", self.kinds({"aggregate": [agg("min", "placed", "first")]}))

    def test_a_filter_yodb_applies_keeps_the_aggregate_in_yodb(self) -> None:
        query = {"where": {"field": "region", "op": "starts_with", "value": "no"}, "group_by": ["status"],
                 "aggregate": [agg("count", alias="n")]}
        self.assertEqual(self.kinds(query), ["remote_scan", "coordinator_filter", "coordinator_aggregate", "coordinator_sort_page"])
        self.assertEqual([(r["status"], r["n"]) for r in self.ask(query)], [("done", 1), ("open", 2)])

    def test_rows_yodb_must_group_are_bounded_by_the_row_guard_but_groups_the_source_computes_are_not(self) -> None:
        adapters = SourcePlanningRegistry(
            [PostgresPlanningAdapter(), CapabilityPlanningAdapter(replace(SQLITE_CAPABILITIES, aggregate=None))]
        )
        query = {"group_by": ["status"], "aggregate": [agg("count", alias="n")]}
        self.db._engine._planner = FederatedPhysicalPlanner(adapters, policy=PlannerPolicy(maximum_rows_per_source=3))
        with self.assertRaises(YoDbError) as caught:
            self.ask(query)
        self.assertEqual(caught.exception.detail.code, ErrorCode.QUERY_ROW_LIMIT_EXCEEDED)
        pushed = SourcePlanningRegistry([PostgresPlanningAdapter(), SqlitePlanningAdapter()])
        self.db._engine._planner = FederatedPhysicalPlanner(pushed, policy=PlannerPolicy(maximum_rows_per_source=3))
        self.assertEqual(len(self.ask(query)), 2)

    def test_the_result_has_no_id_and_exactly_the_keys_and_aliases(self) -> None:
        rows = self.ask({"group_by": ["status"], "aggregate": [agg("count", alias="n")]})
        self.assertEqual([list(row) for row in rows], [["status", "n"], ["status", "n"]])

    def test_the_result_is_deterministic_without_an_order(self) -> None:
        query = {"group_by": ["region", "status"], "aggregate": [agg("count", alias="n")]}
        self.assertEqual(self.ask(query), self.ask(query))
        self.assertEqual([r["region"] for r in self.ask(query)], ["East", "north", "north", "south", "south", None, None])


class ValidationTests(AggregateWorld):
    def code(self, raw: dict) -> ErrorCode:
        return code_of(lambda: self.ask(raw))

    def test_sum_and_avg_are_numeric_only(self) -> None:
        for fn in ("sum", "avg"):
            for field in ("region", "paid", "placed"):
                with self.subTest(fn=fn, field=field):
                    self.assertEqual(self.code({"aggregate": [agg(fn, field)]}), ErrorCode.QUERY_AGGREGATE_INVALID)

    def test_min_and_max_take_orderable_fields_only(self) -> None:
        for field in ("paid", "note"):
            self.assertEqual(self.code({"aggregate": [agg("min", field)]}), ErrorCode.QUERY_AGGREGATE_INVALID)

    def test_only_count_may_omit_its_field(self) -> None:
        self.assertEqual(self.code({"aggregate": [agg("sum")]}), ErrorCode.QUERY_AGGREGATE_INVALID)
        self.assertEqual(self.ask({"aggregate": [agg("count", alias="n")]}), [{"n": 8}])

    def test_group_keys_are_public_and_groupable(self) -> None:
        self.assertEqual(self.code({"group_by": ["nope"], "aggregate": [agg("count")]}), ErrorCode.FIELD_NOT_FOUND)
        self.assertEqual(self.code({"group_by": ["note"], "aggregate": [agg("count")]}), ErrorCode.QUERY_AGGREGATE_INVALID)

    def test_a_private_field_cannot_be_grouped_by(self) -> None:
        from yodb.catalog import Visibility

        active = self.db._engine._catalog_runtime.require_active()
        spec = active.catalog.datasets["order"].fields["region"]
        object.__setattr__(spec, "visibility", Visibility.INTERNAL)
        self.addCleanup(object.__setattr__, spec, "visibility", Visibility.PUBLIC)
        self.assertEqual(self.code({"group_by": ["region"], "aggregate": [agg("count")]}), ErrorCode.FIELD_NOT_ACCESSIBLE)

    def test_order_by_may_name_only_keys_and_aliases(self) -> None:
        raw = {"group_by": ["status"], "aggregate": [agg("count", alias="n")]}
        self.assertEqual(self.code({**raw, "order_by": [{"field": "amount", "direction": "asc"}]}), ErrorCode.QUERY_AGGREGATE_INVALID)
        self.assertEqual(self.code({**raw, "order_by": [{"field": "id", "direction": "asc"}]}), ErrorCode.QUERY_AGGREGATE_INVALID)

    def test_select_cannot_be_combined_with_an_aggregate(self) -> None:
        self.assertEqual(self.code({"select": ["status"], "aggregate": [agg("count")]}), ErrorCode.QUERY_AGGREGATE_INVALID)

    def test_aliases_are_their_own_names(self) -> None:
        self.assertEqual(self.code({"aggregate": [agg("count", alias="status")]}), ErrorCode.QUERY_AGGREGATE_INVALID)
        self.assertEqual(self.code({"group_by": ["status"], "aggregate": [agg("count", alias="status")]}), ErrorCode.QUERY_AGGREGATE_INVALID)
        self.assertEqual(self.code({"aggregate": [agg("count", alias="n"), agg("sum", "amount", "n")]}), ErrorCode.QUERY_AGGREGATE_INVALID)
        self.assertEqual(self.code({"aggregate": [agg("count", alias="not valid")]}), ErrorCode.QUERY_SHAPE_INVALID)

    def test_the_parser_refuses_what_it_does_not_know(self) -> None:
        for raw in (
            {"aggregate": [{"fn": "median", "field": "amount", "as": "m"}]},
            {"aggregate": [{"fn": "count", "as": "n", "extra": 1}]},
            {"aggregate": [{"fn": "count"}]},
            {"aggregate": []},
            {"group_by": []},
            {"group_by": ["status", "status"], "aggregate": [agg("count")]},
            {"aggregate": {"fn": "count"}},
        ):
            with self.subTest(raw):
                self.assertIn(self.code(raw), {ErrorCode.QUERY_SHAPE_INVALID, ErrorCode.QUERY_AGGREGATE_INVALID})

    def test_aggregates_do_not_traverse(self) -> None:
        raw = {"from": {"dataset": "order"}, "aggregate": [agg("count")], "traverse": [{"relationship": "r", "as": "x"}]}
        self.assertEqual(code_of(lambda: self.db.query(raw)), ErrorCode.QUERY_AGGREGATE_UNSUPPORTED)

    def test_the_fingerprint_follows_what_is_computed_not_how_much_is_shown(self) -> None:
        def fingerprint(raw):
            active = self.db._engine._catalog_runtime.require_active()
            return bind_query(parse_query({"from": {"dataset": "order"}, **raw}), active).query_fingerprint

        base = {"group_by": ["status"], "aggregate": [agg("count", alias="n")]}
        self.assertEqual(fingerprint(base), fingerprint({**base, "page": {"first": 1}}))
        for other in (
            {"group_by": ["region"], "aggregate": [agg("count", alias="n")]},
            {"group_by": ["status"], "aggregate": [agg("count", alias="m")]},
            {"group_by": ["status"], "aggregate": [agg("sum", "amount", "n")]},
            {"group_by": ["status"]},
            {"aggregate": [agg("count", alias="n")]},
        ):
            self.assertNotEqual(fingerprint(base), fingerprint(other), other)


class MultiSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.active = multi_source_catalog()

    def resolve(self, raw: dict):
        return resolve_query_sources(bind_query(parse_query({"from": {"dataset": "customer"}, **raw}), self.active), self.active)

    def test_an_aggregate_over_fields_of_one_source_resolves_as_one_source(self) -> None:
        resolved = self.resolve({"group_by": ["status"], "aggregate": [agg("count", alias="n")]})
        self.assertEqual(sorted(f.field.name for f in resolved.identity_source.fields), ["id", "status"])   # id is the identity anchor, never selected

    def test_an_aggregate_over_fields_of_two_sources_is_refused_with_a_code(self) -> None:
        for raw in (
            {"group_by": ["status"], "aggregate": [agg("count_distinct", "plan", "plans")]},
            {"group_by": ["plan"], "aggregate": [agg("count", alias="n")]},
            {"where": {"field": "plan", "op": "eq", "value": "pro"}, "group_by": ["status"], "aggregate": [agg("count", alias="n")]},
        ):
            with self.subTest(raw):
                self.assertEqual(code_of(lambda: self.resolve(raw)), ErrorCode.QUERY_AGGREGATE_UNSUPPORTED)

    def test_postgres_receives_a_group_by_statement_without_the_logical_id(self) -> None:
        planner = FederatedPhysicalPlanner(SourcePlanningRegistry([PostgresPlanningAdapter()]))
        planned = planner.plan(self.resolve({
            "where": {"field": "status", "op": "in", "value": ["a", "b"]},
            "group_by": ["status"],
            "aggregate": [agg("count", alias="n"), agg("count_distinct", "name", "names"), agg("max", "name", "last")],
            "order_by": [{"field": "n", "direction": "desc"}],
            "page": {"first": 5},
        }))
        scan = planned.plan
        self.assertIsNotNone(scan.aggregate)
        compiled = PostgresQueryCompiler().compile_scan(scan)
        self.assertEqual(
            compiled.sql,
            '\n'.join([
                'SELECT "status" COLLATE "C" AS "status", COUNT(*) AS "n", COUNT(DISTINCT "name" COLLATE "C") AS "names", '
                'MAX("name" COLLATE "C") AS "last"',
                'FROM "crm"."accounts"',
                'WHERE "status" IN (%s, %s)',
                'GROUP BY "status" COLLATE "C"',
                'ORDER BY "n" DESC, "status" COLLATE "C" ASC',
                'LIMIT %s',
            ]),
        )
        self.assertEqual(compiled.parameters, ("a", "b", 5))
        self.assertEqual([c.logical_field for c in compiled.output_columns], ["status", "n", "names", "last"])

    def test_postgres_whole_dataset_aggregate_has_no_group_by_clause(self) -> None:
        planner = FederatedPhysicalPlanner(SourcePlanningRegistry([PostgresPlanningAdapter()]))
        compiled = PostgresQueryCompiler().compile_scan(planner.plan(self.resolve({"aggregate": [agg("count", alias="n")]})).plan)
        self.assertEqual(compiled.sql, 'SELECT COUNT(*) AS "n"\nFROM "crm"."accounts"\nORDER BY\nLIMIT %s'.replace("\nORDER BY", ""))

    def test_a_filter_postgres_does_not_run_keeps_the_aggregate_in_yodb(self) -> None:
        planner = FederatedPhysicalPlanner(SourcePlanningRegistry([PostgresPlanningAdapter()]))
        planned = planner.plan(self.resolve({
            "where": {"field": "name", "op": "contains", "value": "x"},
            "group_by": ["status"], "aggregate": [agg("count", alias="n")],
        }))
        self.assertEqual([n.kind for n in planned.explain.nodes], ["remote_scan", "coordinator_filter", "coordinator_aggregate", "coordinator_sort_page"])
        self.assertTrue(planned.explain.nodes[0].detail == ())

    def test_explain_names_where_the_aggregate_runs(self) -> None:
        planner = FederatedPhysicalPlanner(SourcePlanningRegistry([PostgresPlanningAdapter()]))
        pushed = planner.plan(self.resolve({"group_by": ["status"], "aggregate": [agg("count", alias="n")]})).explain.nodes
        self.assertEqual(pushed[0].fields, ("status", "n"))
        self.assertIn("aggregate at source: count(*) as n by status", pushed[0].detail)


class InYodbSemanticsTests(unittest.TestCase):
    """The coordinator's aggregate follows SQL, whatever the data."""

    def spec(self, *measures, keys=()):
        from yodb.query import ResolvedField  # noqa: F401  (documents what a key is)

        return AggregateSpec(group_by=keys, measures=measures)

    def measure(self, function, name, alias="x", result=LogicalType.INT):
        field = None if name is None else _Field(name)
        return AggregateMeasure(function=function, field=field, alias=alias, result_type=result)

    def test_empty_input_without_keys_is_one_row(self) -> None:
        rows = aggregate_rows(
            self.spec(self.measure(AggregateFunction.COUNT, None, "n"), self.measure(AggregateFunction.SUM, "v", "s"),
                      self.measure(AggregateFunction.AVG, "v", "a"), self.measure(AggregateFunction.MIN, "v", "lo"),
                      self.measure(AggregateFunction.COUNT_DISTINCT, "v", "d")),
            (),
        )
        self.assertEqual(rows, ({"n": 0, "s": None, "a": None, "lo": None, "d": 0},))

    def test_nulls_are_skipped_and_avg_is_a_float(self) -> None:
        rows = aggregate_rows(
            self.spec(self.measure(AggregateFunction.COUNT, None, "n"), self.measure(AggregateFunction.COUNT, "v", "c"),
                      self.measure(AggregateFunction.SUM, "v", "s"), self.measure(AggregateFunction.AVG, "v", "a")),
            ({"v": 1}, {"v": 2}, {"v": None}),
        )
        self.assertEqual(rows, ({"n": 3, "c": 2, "s": 3, "a": 1.5},))
        self.assertIsInstance(rows[0]["s"], int)

    def test_an_all_null_group_sums_to_null(self) -> None:
        rows = aggregate_rows(self.spec(self.measure(AggregateFunction.SUM, "v", "s"), keys=(_Field("k"),)), ({"k": 1, "v": None},))
        self.assertEqual(rows, ({"k": 1, "s": None},))


class _Named:
    def __init__(self, name):
        self.name = name


class _Field:
    """What the in-YoDb aggregate reads of a resolved field: its logical name."""

    def __init__(self, name):
        self.field = _Named(name)


class PostgresNumericTests(unittest.TestCase):
    def test_a_numeric_sum_or_average_comes_back_as_the_logical_type(self) -> None:
        from decimal import Decimal

        row = _logical_row(("n", "total", "mean", "key"), (3, Decimal("12"), Decimal("4.5"), "a"), "crm",
                           (LogicalType.INT, LogicalType.INT, LogicalType.FLOAT, LogicalType.STRING))
        self.assertEqual(row, {"n": 3, "total": 12, "mean": 4.5, "key": "a"})
        self.assertIs(type(row["total"]), int)
        self.assertIs(type(row["mean"]), float)

    def test_a_plain_scan_value_is_left_alone(self) -> None:
        from decimal import Decimal

        self.assertEqual(_logical_row(("price",), (Decimal("1.50"),), "crm"), {"price": Decimal("1.50")})


if __name__ == "__main__":
    unittest.main()
