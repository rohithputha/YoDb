"""SQLite execution adapter for parameterized compiled YoDb statements.

SQLite is dynamically typed: a timestamp is text, an epoch number or a Julian day, a boolean is 0/1,
JSON is text.  The compiler records each output column's logical type, and rows are coerced to it
here, so the rest of YoDb sees the same Python values it sees from PostgreSQL.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import sqlite3
import time
from typing import Any
from uuid import UUID

from ..catalog import LogicalType, SourceKind
from ..compilation import CompiledOutputColumn, CompiledQuery, CompiledSqliteQuery
from ..connections import SourceConnectionAdapter
from ..errors import ErrorCode, ErrorDetail, QueryExecutionError, YoDbError
from .contracts import LogicalRow

# How many SQLite VM instructions run between two looks at the clock.
_PROGRESS_INTERVAL = 1_000
# An epoch number this large is milliseconds, not seconds (year 5138 in seconds, 1973 in milliseconds).
_MILLISECONDS_THRESHOLD = 1e11


class SqliteQueryExecutionAdapter:
    """Execute :class:`CompiledSqliteQuery` through a supplied connection adapter."""

    source_kind = SourceKind.SQLITE

    def __init__(
        self,
        connections: SourceConnectionAdapter[Any],
        *,
        default_statement_timeout_seconds: float | None = 60.0,
    ) -> None:
        if connections.source_kind is not SourceKind.SQLITE:
            raise ValueError("SqliteQueryExecutionAdapter requires a SQLite connection adapter")
        if default_statement_timeout_seconds is not None and default_statement_timeout_seconds <= 0:
            raise ValueError("default_statement_timeout_seconds must be positive")
        self._connections = connections
        self._default_statement_timeout = default_statement_timeout_seconds

    def execute(
        self,
        query: CompiledQuery,
        *,
        timeout_seconds: float | None = None,
    ) -> tuple[LogicalRow, ...]:
        if not isinstance(query, CompiledSqliteQuery):
            raise QueryExecutionError(
                ErrorDetail(
                    code=ErrorCode.QUERY_EXECUTION_UNSUPPORTED,
                    message="The SQLite executor requires a SQLite compiled query.",
                    retryable=False,
                )
            )
        limit = timeout_seconds if timeout_seconds is not None else self._default_statement_timeout
        try:
            with self._connections.acquire(query.connection_ref) as connection:
                rows = _run(connection, query, limit)
        except YoDbError:
            raise
        except _Timeout as error:
            raise QueryExecutionError(
                ErrorDetail(
                    code=ErrorCode.QUERY_TIMEOUT,
                    message="The SQLite source did not answer within the time limit.",
                    retryable=True,
                    source_name=query.source_name,
                )
            ) from error
        except Exception as error:
            raise QueryExecutionError(
                ErrorDetail(
                    code=ErrorCode.QUERY_EXECUTION_FAILED,
                    message="The SQLite source could not execute the compiled query.",
                    retryable=False,
                    source_name=query.source_name,
                    details={"exception_type": type(error).__name__},
                )
            ) from error
        return tuple(_logical_row(query.output_columns, row, query.source_name) for row in rows)


class _Timeout(Exception):
    pass


def _run(connection: sqlite3.Connection, query: CompiledSqliteQuery, limit: float | None) -> list[Any]:
    """Run the statement, stopping it from inside SQLite once ``limit`` seconds have passed.

    SQLite has no ``statement_timeout``; a progress handler that returns true aborts the statement
    (it then fails with "interrupted").  The connection is this thread's own, so the handler needs no lock.
    """

    expired = False
    if limit is not None:
        deadline = time.monotonic() + limit

        def stop() -> bool:
            nonlocal expired
            if time.monotonic() >= deadline:
                expired = True
            return expired

        connection.set_progress_handler(stop, _PROGRESS_INTERVAL)
    try:
        return connection.execute(query.sql, query.parameters).fetchall()
    except sqlite3.OperationalError as error:
        if expired or "interrupted" in str(error).lower():
            raise _Timeout() from error
        raise
    finally:
        if limit is not None:
            connection.set_progress_handler(None, 0)


def _logical_row(columns: tuple[CompiledOutputColumn, ...], row: Any, source_name: str) -> LogicalRow:
    if not isinstance(row, (tuple, list)) or len(row) != len(columns):
        raise _failure("The SQLite source returned an unexpected result shape.", source_name)
    return {column.logical_field: _coerce(column, value, source_name) for column, value in zip(columns, row, strict=True)}


def _coerce(column: CompiledOutputColumn, value: Any, source_name: str) -> Any:
    coerce = None if column.logical_type is None else _COERCIONS.get(column.logical_type)
    if value is None or coerce is None:
        return value
    try:
        return coerce(value)
    except (ValueError, TypeError, OverflowError, OSError):
        raise _failure(
            f"A value of field '{column.logical_field}' could not be read as {column.logical_type.value}.",
            source_name,
        ) from None


def _failure(message: str, source_name: str) -> QueryExecutionError:
    return QueryExecutionError(
        ErrorDetail(
            code=ErrorCode.QUERY_EXECUTION_FAILED, message=message, retryable=False, source_name=source_name
        )
    )


def _timestamp(value: Any) -> datetime:
    """Text (ISO 8601, ``T`` or space, optional ``Z``/offset) or an epoch number; naive means UTC."""

    if isinstance(value, bool):
        raise TypeError("a boolean is not a timestamp")
    if isinstance(value, (int, float)):
        seconds = value / 1000 if abs(value) >= _MILLISECONDS_THRESHOLD else value
        return datetime.fromtimestamp(seconds, UTC)
    if isinstance(value, str):
        moment = datetime.fromisoformat(value.strip())
        return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)
    raise TypeError("not a timestamp")


def _boolean(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "t", "true", "y", "yes"):
            return True
        if text in ("0", "f", "false", "n", "no"):
            return False
    raise ValueError("not a boolean")


def _integer(value: Any) -> int:
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("not an integer")
    return int(value)


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, (str, bytes, bytearray)) else value


def _uuid(value: Any) -> str:
    return str(UUID(bytes=bytes(value))) if isinstance(value, (bytes, bytearray)) else str(UUID(str(value)))


_COERCIONS = {
    LogicalType.TIMESTAMP: _timestamp,
    LogicalType.BOOL: _boolean,
    LogicalType.INT: _integer,
    LogicalType.FLOAT: float,
    LogicalType.JSON: _json,
    LogicalType.UUID: _uuid,
}
