"""Central query execution engine and backend execution adapters."""

from .contracts import (
    ActiveCatalogProvider,
    LogicalRow,
    QueryExecutionAdapter,
    QueryExecutionResult,
)
from .engine import QueryExecutionEngine
from .federated import ExecutionTrace, FederatedExecutionPolicy, FederatedPlanExecutor, ScanActual
from .postgres import PostgresQueryExecutionAdapter
from .registry import QueryExecutionAdapterRegistry
from .sqlite import SqliteQueryExecutionAdapter

__all__ = [
    "ActiveCatalogProvider",
    "LogicalRow",
    "PostgresQueryExecutionAdapter",
    "SqliteQueryExecutionAdapter",
    "QueryExecutionAdapter",
    "QueryExecutionAdapterRegistry",
    "QueryExecutionEngine",
    "QueryExecutionResult",
    "ExecutionTrace",
    "ScanActual",
    "FederatedExecutionPolicy",
    "FederatedPlanExecutor",
]
