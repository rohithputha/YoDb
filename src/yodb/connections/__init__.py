"""Reusable source connection resolution and pooling."""

from .contracts import ConnectionReferenceResolver, SourceConnectionAdapter
from .postgres import (
    EnvPostgresConnectionResolver,
    MappingPostgresConnectionResolver,
    PostgresConnectionAdapter,
    PostgresConnectionReferenceResolver,
    PostgresConnectionSettings,
)
from .registry import ConnectionAdapterRegistry
from .sqlite import (
    EnvSqliteConnectionResolver,
    MappingSqliteConnectionResolver,
    SqliteConnectionAdapter,
    SqliteConnectionReferenceResolver,
    SqliteConnectionSettings,
)

__all__ = [
    "EnvSqliteConnectionResolver",
    "MappingSqliteConnectionResolver",
    "SqliteConnectionAdapter",
    "SqliteConnectionReferenceResolver",
    "SqliteConnectionSettings",
    "EnvPostgresConnectionResolver",
    "ConnectionReferenceResolver",
    "ConnectionAdapterRegistry",
    "MappingPostgresConnectionResolver",
    "PostgresConnectionAdapter",
    "PostgresConnectionReferenceResolver",
    "PostgresConnectionSettings",
    "SourceConnectionAdapter",
]
