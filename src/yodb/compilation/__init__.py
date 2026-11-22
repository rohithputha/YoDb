"""Physical query-compilation contracts and backend-specific compilers."""

from .contracts import CompiledOutputColumn, CompiledPostgresQuery, CompiledQuery, CompiledSqliteQuery, QueryCompilerAdapter
from .postgres import PostgresQueryCompiler
from .sqlite import SqliteQueryCompiler
from .registry import QueryCompilerRegistry

__all__ = [
    "CompiledOutputColumn",
    "CompiledPostgresQuery",
    "CompiledQuery",
    "CompiledSqliteQuery",
    "SqliteQueryCompiler",
    "PostgresQueryCompiler",
    "QueryCompilerAdapter",
    "QueryCompilerRegistry",
]
