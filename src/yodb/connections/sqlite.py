"""SQLite connection resolution and per-thread read-only connections.

A SQLite connection is cheap and must not be shared between concurrently running statements, and
execution runs scans on up to four threads.  So there is no pool: each thread gets its own connection
per connection reference, opened read-only (``mode=ro`` and ``PRAGMA query_only=ON``) and kept until
the adapter is closed or the thread has ended.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sqlite3
import threading
from typing import Protocol
from urllib.parse import quote, unquote, urlparse

from ..catalog import SourceKind
from ..errors import ErrorCode, ErrorDetail, SourceConnectionError


@dataclass(frozen=True)
class SqliteConnectionSettings:
    """Private SQLite settings returned by a connection-reference resolver: the database file."""

    path: str


class SqliteConnectionReferenceResolver(Protocol):
    def resolve(self, connection_ref: str) -> SqliteConnectionSettings:
        """Return the database file for ``connection_ref``."""


def settings_from_text(value: str) -> SqliteConnectionSettings:
    """A plain path, ``sqlite:///relative.db`` / ``sqlite:////absolute.db`` (SQLAlchemy's convention), or a
    ``file:`` URI.  A URI's query is ignored: YoDb always opens read-only."""

    value = value.strip()
    if value.startswith("sqlite:///"):
        return SqliteConnectionSettings(path=unquote(value[len("sqlite:///"):].split("?", 1)[0]))
    if value.startswith("file:"):
        return SqliteConnectionSettings(path=unquote(urlparse(value).path or value[len("file:"):].split("?", 1)[0]))
    return SqliteConnectionSettings(path=value)


class MappingSqliteConnectionResolver:
    """A simple resolver for local development and deterministic tests."""

    def __init__(self, settings_by_ref: Mapping[str, SqliteConnectionSettings]) -> None:
        self._settings_by_ref = dict(settings_by_ref)

    def resolve(self, connection_ref: str) -> SqliteConnectionSettings:
        try:
            return self._settings_by_ref[connection_ref]
        except KeyError as error:
            raise SourceConnectionError(
                ErrorDetail(
                    code=ErrorCode.CONNECTION_REFERENCE_NOT_FOUND,
                    message="No SQLite connection settings were found for the configured reference.",
                    retryable=False,
                )
            ) from error


class EnvSqliteConnectionResolver:
    """Resolve a connection reference from an environment variable holding the database file's path.

    ``connection_ref: local-orders`` reads ``YODB_CONN_LOCAL_ORDERS`` (the same naming as the PostgreSQL
    resolver).
    """

    def __init__(self, *, prefix: str = "YODB_CONN_", environ: Mapping[str, str] | None = None) -> None:
        self._prefix = prefix
        self._environ = os.environ if environ is None else environ

    def variable_for(self, connection_ref: str) -> str:
        return self._prefix + re.sub(r"[^A-Za-z0-9]", "_", connection_ref).upper()

    def resolve(self, connection_ref: str) -> SqliteConnectionSettings:
        value = self._environ.get(self.variable_for(connection_ref))
        if not value:
            raise SourceConnectionError(
                ErrorDetail(
                    code=ErrorCode.CONNECTION_REFERENCE_NOT_FOUND,
                    message=(
                        f"No connection settings for reference '{connection_ref}': "
                        f"set the environment variable {self.variable_for(connection_ref)} to the SQLite file's path."
                    ),
                    retryable=False,
                )
            )
        return settings_from_text(value)


class SqliteConnectionAdapter:
    """Lease the calling thread's read-only connection to a SQLite file.

    ``acquire`` never waits (there is nothing to wait for), so ``timeout_seconds`` is accepted for
    the connection-adapter contract and unused: a statement's time limit is enforced by the executor.
    """

    source_kind = SourceKind.SQLITE

    def __init__(self, resolver: SqliteConnectionReferenceResolver) -> None:
        self._resolver = resolver
        self._lock = threading.Lock()
        self._connections: dict[tuple[int, str], sqlite3.Connection] = {}
        self._closed = False

    def acquire(self, connection_ref: str, *, timeout_seconds: float | None = None):
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        return self._lease(self._connection_for(connection_ref))

    @contextmanager
    def _lease(self, connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
        try:
            yield connection
        finally:
            if connection.in_transaction:
                connection.rollback()

    def _connection_for(self, connection_ref: str) -> sqlite3.Connection:
        key = (threading.get_ident(), connection_ref)
        with self._lock:
            self._raise_if_closed()
            connection = self._connections.get(key)
            if connection is not None:
                return connection
            settings = self._resolver.resolve(connection_ref)
            self._close_connections_of_ended_threads()
        connection = _open_read_only(settings, connection_ref)
        with self._lock:
            if self._closed:
                connection.close()
                self._raise_if_closed()
            self._connections[key] = connection
        return connection

    def _close_connections_of_ended_threads(self) -> None:
        alive = {thread.ident for thread in threading.enumerate()}
        for key in [key for key in self._connections if key[0] not in alive]:
            self._connections.pop(key).close()

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise SourceConnectionError(
                ErrorDetail(
                    code=ErrorCode.CONNECTION_POOL_CLOSED,
                    message="The SQLite connection adapter is closed.",
                    retryable=False,
                )
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            connections = tuple(self._connections.values())
            self._connections.clear()
        for connection in connections:
            connection.close()


def _open_read_only(settings: SqliteConnectionSettings, connection_ref: str) -> sqlite3.Connection:
    path = Path(settings.path).expanduser()
    uri = f"file:{quote(str(path.resolve()))}?mode=ro"
    try:
        # check_same_thread=False only so ``close()`` may run on another thread; a connection is only ever used
        # by the thread that opened it.
        connection = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=5.0, isolation_level=None)
        connection.execute("PRAGMA query_only = ON")
    except sqlite3.Error as error:
        # The driver's message can carry the file's path: say only what happened.
        raise SourceConnectionError(
            ErrorDetail(
                code=ErrorCode.SOURCE_UNAVAILABLE,
                message=f"The SQLite file for connection reference '{connection_ref}' could not be opened read-only "
                "(does it exist, and may this process read it?).",
                retryable=False,
            )
        ) from error
    return connection
