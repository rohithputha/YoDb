"""PostgreSQL execution adapter for parameterized compiled YoDb statements."""

from __future__ import annotations

from datetime import UTC, datetime
import math
import re
from typing import Any

from ..catalog import SourceKind
from ..compilation import CompiledPostgresQuery, CompiledQuery
from ..connections import SourceConnectionAdapter
from ..errors import ErrorCode, ErrorDetail, QueryExecutionError, YoDbError
from .contracts import LogicalRow


class PostgresQueryExecutionAdapter:
    """Execute :class:`CompiledPostgresQuery` through a supplied connection adapter.

    The adapter deliberately accepts the provider-neutral connection protocol,
    rather than constructing a pool itself.  That keeps pooling and secret
    resolution reusable by inspection and future PostgreSQL query features.
    """

    source_kind = SourceKind.POSTGRES

    def __init__(
        self,
        connections: SourceConnectionAdapter[Any],
        *,
        default_statement_timeout_seconds: float | None = 60.0,
    ) -> None:
        if connections.source_kind is not SourceKind.POSTGRES:
            raise ValueError("PostgresQueryExecutionAdapter requires a PostgreSQL connection adapter")
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
        """Run a PostgreSQL statement and normalize rows to logical aliases."""

        if not isinstance(query, CompiledPostgresQuery):
            raise QueryExecutionError(
                ErrorDetail(
                    code=ErrorCode.QUERY_EXECUTION_UNSUPPORTED,
                    message="The PostgreSQL executor requires a PostgreSQL compiled query.",
                    retryable=False,
                )
            )
        limit = timeout_seconds if timeout_seconds is not None else self._default_statement_timeout
        try:
            with self._connections.acquire(
                query.connection_ref,
                timeout_seconds=timeout_seconds,
            ) as connection:
                if query.requires_extension is not None:
                    _load_extension(connection, query.requires_extension)
                with connection.cursor() as cursor:
                    if limit is not None:
                        # The database stops the statement itself; the pool timeout only bounds waiting.
                        cursor.execute(
                            "SELECT set_config('statement_timeout', %s, true)", (str(max(1, math.ceil(limit * 1000))),)
                        )
                    if query.requires_extension == "age":
                        # AGE looks its own functions up unqualified (age_unnest behind UNWIND, ...); the catalog's
                        # tables are always schema-qualified, so the path change touches nothing else.  Local to
                        # this transaction, so the pooled session is unchanged afterwards.
                        cursor.execute("SELECT set_config('search_path', %s, true)", ('ag_catalog, "$user", public',))
                    cursor.execute(query.sql, query.parameters)
                    rows = cursor.fetchall()
        except YoDbError:
            raise
        except Exception as error:
            if type(error).__name__ == "QueryCanceled":
                raise QueryExecutionError(
                    ErrorDetail(
                        code=ErrorCode.QUERY_TIMEOUT,
                        message="The PostgreSQL source did not answer within the time limit.",
                        retryable=True,
                        source_name=query.source_name,
                    )
                ) from error
            message = "The PostgreSQL source could not execute the compiled query."
            if query.requires_extension is not None and _extension_missing(error, query.requires_extension):
                message = (
                    f"The PostgreSQL source could not run the statement because the '{query.requires_extension}' "
                    "extension is not available in the session: install it (CREATE EXTENSION) and preload it for "
                    f"YoDb's role (ALTER ROLE ... SET session_preload_libraries = '{query.requires_extension}', or "
                    "shared_preload_libraries on the server)."
                )
            raise QueryExecutionError(
                ErrorDetail(
                    code=ErrorCode.QUERY_EXECUTION_FAILED,
                    message=message,
                    retryable=False,
                    source_name=query.source_name,
                )
            ) from error

        fields = tuple(column.logical_field for column in query.output_columns)
        return tuple(_logical_row(fields, row, query.source_name) for row in rows)


_LOADABLE = re.compile(r"^[a-z_][a-z0-9_]*$")


def _load_extension(connection: Any, name: str) -> None:
    """``LOAD`` a server library into the session, if the role may.

    Apache AGE's Cypher call only works once its library is loaded.  A superuser
    can ``LOAD`` it; an ordinary read-only role cannot, and relies on the server
    or the role preloading it (``session_preload_libraries``).  A refused ``LOAD``
    is therefore not an error here: the transaction it aborted is rolled back and
    the statement runs; if the library is really absent the statement fails with
    a message that says so.
    """

    if not _LOADABLE.fullmatch(name):
        raise ValueError(f"not a loadable extension name: {name!r}")
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"LOAD '{name}'")
    except Exception:
        connection.rollback()


def _extension_missing(error: Exception, name: str) -> bool:
    text = str(error).lower()
    return name in text or "cypher" in text


def _logical_row(
    fields: tuple[str, ...],
    row: Any,
    source_name: str,
) -> LogicalRow:
    if not isinstance(row, (tuple, list)) or len(row) != len(fields):
        raise QueryExecutionError(
            ErrorDetail(
                code=ErrorCode.QUERY_EXECUTION_FAILED,
                message="The PostgreSQL source returned an unexpected result shape.",
                retryable=False,
                source_name=source_name,
            )
        )
    return {name: _normalize(value) for name, value in zip(fields, row, strict=True)}


def _normalize(value: Any) -> Any:
    """Return timestamps as UTC-aware, matching how query values are bound.

    A ``timestamp`` (without time zone) column arrives naive; comparing it with
    an aware query value would raise and the coordinator would drop every row.
    Naive source timestamps are interpreted as UTC.
    """

    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return value
