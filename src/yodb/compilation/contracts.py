"""Backend-neutral compiler contracts and backend-native compiled queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..catalog import LogicalType, SourceKind
from ..planning import GraphEdgeScan, RemoteScan
from ..query.resolution import SourceResolvedQuery


@dataclass(frozen=True)
class CompiledOutputColumn:
    """One selected SQL output column and its logical result-field name."""

    sql_alias: str
    logical_field: str
    # The field's logical type, for a backend whose driver returns values that need coercing (SQLite
    # stores timestamps as text or numbers and booleans as 0/1).  None: the driver's values are used as is.
    logical_type: LogicalType | None = None


@runtime_checkable
class CompiledQuery(Protocol):
    """A backend-native command that has not yet been executed.

    The command's native payload remains owned by its compiler and execution
    adapter.  The engine uses only this small common envelope to route it
    safely and to normalize the returned fields.
    """

    @property
    def source_kind(self) -> SourceKind:
        """The backend that can execute this command."""

    source_name: str
    connection_ref: str
    output_columns: tuple[CompiledOutputColumn, ...]


@runtime_checkable
class QueryCompilerAdapter(Protocol):
    """Compile one source-local resolved query for exactly one backend kind."""

    @property
    def source_kind(self) -> SourceKind:
        """The configured source kind handled by this compiler."""

    def compile(self, query: SourceResolvedQuery) -> CompiledQuery:
        """Return a backend-native, parameterized command or a typed error."""

    def compile_scan(self, scan: RemoteScan) -> CompiledQuery:
        """Compile one planner-validated physical remote scan fragment."""

    def compile_edges(self, scan: GraphEdgeScan) -> CompiledQuery:
        """Compile one read of graph edges (a :class:`GraphEdgeScan` with its IDs filled in)."""


@dataclass(frozen=True)
class CompiledPostgresQuery:
    """A PostgreSQL statement with catalog-derived identifiers and bound values.

    ``sql`` uses Psycopg's ``%s`` placeholders. User values appear only in
    ``parameters``; SQL identifiers are derived exclusively from the resolved,
    already-validated source binding.
    """

    source_name: str
    connection_ref: str
    sql: str
    parameters: tuple[object, ...]
    output_columns: tuple[CompiledOutputColumn, ...]
    # A server extension the statement needs loaded in the session (``"age"`` for a Cypher call);
    # the executor loads it when it may, and names it when the statement fails for its absence.
    requires_extension: str | None = None

    @property
    def source_kind(self) -> SourceKind:
        """PostgreSQL is the only backend that can execute this statement."""

        return SourceKind.POSTGRES


@dataclass(frozen=True)
class CompiledSqliteQuery:
    """A SQLite statement with catalog-derived identifiers and bound values.

    ``sql`` uses ``?`` placeholders; user values appear only in ``parameters``.
    """

    source_name: str
    connection_ref: str
    sql: str
    parameters: tuple[object, ...]
    output_columns: tuple[CompiledOutputColumn, ...]

    @property
    def source_kind(self) -> SourceKind:
        return SourceKind.SQLITE
