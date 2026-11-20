"""Raw access to the three stores for the baselines: SQL, vector search and Cypher, read-only.

This is what a developer gives an agent when there is no typed layer: the real tables, a
vector column and the graph, each behind a thin tool.  Every call is bounded (rows, time) and
read-only (the session is ``default_transaction_read_only``), and every failure is returned as
text for the model to read, as a tool would.  Nothing here validates anything: that is the point.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
from typing import Any

MAXIMUM_ROWS = 500
MAXIMUM_CELL = 300            # characters of one value shown to the model
_FORBIDDEN = re.compile(r"^\s*(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|vacuum|call|do)\b", re.IGNORECASE)


@dataclass
class ToolResult:
    ok: bool
    text: str                                   # what the model sees
    rows: list[dict[str, Any]]                  # the raw rows, for evidence collection
    native_error: bool = False                  # the store rejected the native query (the H1 count)
    latency_ms: float = 0.0


class RawStores:
    """SQL, vector search and Cypher over one PostgreSQL (with pgvector and Apache AGE)."""

    def __init__(self, dsn: str, embedder=None, *, graph: str = "yelp_graph", statement_timeout_seconds: float = 30.0) -> None:
        import psycopg

        self.graph = graph
        self.embedder = embedder
        self.connection = psycopg.connect(dsn, autocommit=True, options="-c default_transaction_read_only=on")
        with self.connection.cursor() as cursor:
            cursor.execute("SET statement_timeout = %s", (str(int(statement_timeout_seconds * 1000)),))
            try:
                cursor.execute("LOAD 'age'")
            except psycopg.Error:
                pass                                                    # preloaded for the role, or absent

    # --- what the agent is told -------------------------------------------------------------------

    def schema_text(self, schema: str = "yelp") -> str:
        """The raw schemas: table DDL from information_schema, the vector column, the graph's labels and edge types."""

        lines = [f"POSTGRESQL TABLES (schema {schema}; query with SELECT only; ids are text):"]
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = %s ORDER BY table_name, ordinal_position",
                (schema,),
            )
            tables: dict[str, list[str]] = {}
            for table, column, data_type in cursor.fetchall():
                tables.setdefault(table, []).append(f"{column} {data_type}")
            for table, columns in tables.items():
                lines.append(f"  {schema}.{table}({', '.join(columns)})")
            vector_columns = [(t, c.split()[0]) for t, cols in tables.items() for c in cols if "USER-DEFINED" in c and "embedding" in c]
            if vector_columns:
                lines.append("VECTOR COLUMNS (pgvector; use the vector_search tool, which embeds your text with the same model as the data):")
                for table, column in vector_columns:
                    lines.append(f"  {schema}.{table}.{column}")
            try:
                cursor.execute(
                    "SELECT l.name::text, l.kind::text FROM ag_catalog.ag_label l JOIN ag_catalog.ag_graph g ON g.graphid = l.graph WHERE g.name = %s ORDER BY l.kind, l.name",
                    (self.graph,),
                )
                labels = cursor.fetchall()
            except Exception:
                labels = []
        if labels:
            vertices = [name for name, kind in labels if kind == "v" and not name.startswith("_")]
            edges = [name for name, kind in labels if kind == "e" and not name.startswith("_")]
            lines.append(f"GRAPH {self.graph} (Apache AGE; use the cypher tool): vertex labels {', '.join(vertices)} (each vertex has one property, id, the same text id as the table row); "
                         f"edge types {', '.join(edges)}")
        return "\n".join(lines)

    # --- the tools -----------------------------------------------------------------------------------

    def sql(self, query: str) -> ToolResult:
        if _FORBIDDEN.match(query or ""):
            return ToolResult(False, "refused: only SELECT statements are allowed", [], native_error=True)
        return self._run(query, ())

    def vector_search(self, text: str, top: int = 10, where: str | None = None, table: str = "yelp.review", column: str = "text_embedding", id_column: str = "review_id") -> ToolResult:
        if self.embedder is None:
            return ToolResult(False, "error: no embedding model is configured", [], native_error=True)
        if where and _FORBIDDEN.match(where):
            return ToolResult(False, "refused: 'where' must be a condition", [], native_error=True)
        top = max(1, min(int(top or 10), MAXIMUM_ROWS))
        vector = next(iter(self.embedder.embed_texts([text])))
        literal = "[" + ",".join(repr(x) for x in vector) + "]"
        condition = f"{column} IS NOT NULL" + (f" AND ({where})" if where else "")
        query = f"SELECT {id_column} AS id, {column} <=> %s::vector AS distance FROM {table} WHERE {condition} ORDER BY distance, {id_column} LIMIT {top}"
        return self._run(query, (literal,))

    def cypher(self, query: str, columns: int = 1) -> ToolResult:
        if not query or "$$" in query:
            return ToolResult(False, "error: a Cypher query is required (without $$)", [], native_error=True)
        if re.search(r"\b(create|delete|set|remove|merge|drop)\b", query, re.IGNORECASE):
            return ToolResult(False, "refused: read-only Cypher only (MATCH ... RETURN)", [], native_error=True)
        columns = max(1, min(int(columns or 1), 10))
        names = ", ".join(f"c{i} ag_catalog.agtype" for i in range(columns))
        sql = f"SELECT * FROM ag_catalog.cypher('{self.graph}', $yodb$ {query} $yodb$) AS ({names}) LIMIT {MAXIMUM_ROWS + 1}"
        result = self._run(sql, ())
        if result.ok:
            for row in result.rows:
                for key, value in list(row.items()):
                    row[key] = _agtype_text(value)
            result.text = render_rows(result.rows)
        return result

    def _run(self, query: str, params: tuple) -> ToolResult:
        started = time.perf_counter()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                names = [c.name for c in cursor.description] if cursor.description else []
                fetched = cursor.fetchmany(MAXIMUM_ROWS + 1) if names else []
        except Exception as error:                                  # the store's own error, verbatim, for the model
            return ToolResult(False, f"error: {str(error).strip().splitlines()[0][:400]}", [], native_error=True, latency_ms=(time.perf_counter() - started) * 1000)
        rows = [dict(zip(names, row)) for row in fetched[:MAXIMUM_ROWS]]
        text = render_rows(rows) + (f"\n(truncated at {MAXIMUM_ROWS} rows)" if len(fetched) > MAXIMUM_ROWS else "")
        return ToolResult(True, text, rows, latency_ms=(time.perf_counter() - started) * 1000)

    def close(self) -> None:
        self.connection.close()


def render_rows(rows: list[dict[str, Any]], limit: int = 60) -> str:
    """Rows as JSON lines the model can read; long values are cut, many rows are summarized."""

    if not rows:
        return "(no rows)"
    shown = [json.dumps({k: _short(v) for k, v in row.items()}, default=str) for row in rows[:limit]]
    extra = f"\n... {len(rows) - limit} more rows (all {len(rows)} are kept for your final answer if you reference them)" if len(rows) > limit else ""
    return "\n".join(shown) + extra


def _short(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAXIMUM_CELL:
        return value[:MAXIMUM_CELL] + "..."
    if isinstance(value, float):
        return round(value, 4)
    return value


def _agtype_text(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        text = value.strip()
        for suffix in ("::vertex", "::edge", "::path"):
            if text.endswith(suffix):
                text = text[: -len(suffix)]
        try:
            return json.loads(text)
        except ValueError:
            return text
    return value
