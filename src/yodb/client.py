"""The front door: open a catalog and ask questions.

    import yodb

    with yodb.connect("catalog/") as db:                       # connections from YODB_CONN_<REF> variables
        result = db.query({"from": {"dataset": "customer"}, "select": ["name"], "page": {"first": 10}})
        print(db.explain(query).nodes)

``connect`` builds everything a query needs from one catalog directory: the
connections of every source kind (a PostgreSQL pool of read-only sessions, a read-only SQLite
connection per thread), the catalog runtime (which inspects the real
sources and refuses to activate a catalog that does not match them), the planner
with statistics, and the executor.  Anything else can still be assembled by hand
from the lower-level pieces; this is only the short path.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from .catalog import Catalog, SourceKind
from .compilation import PostgresQueryCompiler, QueryCompilerRegistry, SqliteQueryCompiler
from .connections import (
    EnvPostgresConnectionResolver,
    MappingPostgresConnectionResolver,
    PostgresConnectionAdapter,
    PostgresConnectionReferenceResolver,
    PostgresConnectionSettings,
    EnvSqliteConnectionResolver,
    MappingSqliteConnectionResolver,
    SqliteConnectionAdapter,
    SqliteConnectionReferenceResolver,
    SqliteConnectionSettings,
)
from .connections.sqlite import settings_from_text
from .errors import CatalogRuntimeError, ErrorCode, ErrorDetail, QueryError
from .execution import (
    PostgresQueryExecutionAdapter,
    SqliteQueryExecutionAdapter,
    QueryExecutionAdapterRegistry,
    QueryExecutionEngine,
    QueryExecutionResult,
)
from .execution.operators import FederatedExecutionPolicy
from .inspection import (
    InspectionAdapterBinding,
    PostgresCatalogValidator,
    PostgresSourceInspector,
    SourceInspectionRegistry,
    SqliteCatalogValidator,
    SqliteSourceInspector,
)
from .planning import (
    FederatedPhysicalPlanner,
    ObservationStore,
    JoinPolicy,
    PlannerPolicy,
    PlanExplanation,
    PostgresPlanningAdapter,
    PostgresStatisticsProvider,
    SourcePlanningRegistry,
    SqlitePlanningAdapter,
    SqliteStatisticsProvider,
    StatisticsService,
)
from .runtime import CatalogRefreshResult, InMemoryCatalogRuntime
from .semantic import EmbeddingProvider, SemanticExtension, SemanticRuntime
from .similar import SimilarExtension

# A mapping of connection reference to a libpq conninfo (postgres source) or a file path (sqlite source), or
# to the settings objects of either; or a resolver object; or None to read ``YODB_CONN_<REF>`` variables.
Connections = (
    Mapping[str, "str | PostgresConnectionSettings | SqliteConnectionSettings"]
    | PostgresConnectionReferenceResolver
    | None
)


class YoDb:
    """An open, validated catalog and the engine that answers queries against it."""

    def __init__(
        self,
        runtime: InMemoryCatalogRuntime,
        engine: QueryExecutionEngine,
        *,
        close=None,
        semantic: bool = False,
        embedding_model: str | None = None,
    ) -> None:
        self._runtime = runtime
        self._engine = engine
        self._close = close
        self._semantic = semantic
        self._embedding_model = embedding_model

    @classmethod
    def connect(
        cls,
        catalog: str | Path,
        connections: Connections = None,
        *,
        semantic: SemanticExtension | SemanticRuntime | None = None,
        embedder: EmbeddingProvider | SimilarExtension | None = None,
        statistics: bool = True,
        pool_size: int = 10,
        acquire_timeout_seconds: float = 30.0,
        statement_timeout_seconds: float | None = 60.0,
        planner_policy: PlannerPolicy = PlannerPolicy(),
        execution_policy: FederatedExecutionPolicy = FederatedExecutionPolicy(),
        join_policy: JoinPolicy = JoinPolicy(),
        pool_factory=None,
    ) -> "YoDb":
        """Load ``catalog`` (a directory of the three YAML files), inspect every source, and open it.

        ``connections`` maps each catalog ``connection_ref`` to a libpq conninfo string (a postgres source)
        or the path of a SQLite file (a sqlite source), or is a resolver object; left out, each
        reference is read from ``YODB_CONN_<REF>``.  Raises a
        :class:`CatalogRuntimeError` listing what is wrong when the catalog does not match the sources.

        ``semantic`` enables semantic conditions (a verifier, with an optional embedder for the
        shortlist).  ``embedder`` enables similarity searches (``{"similar": ...}``); left out, the
        semantic runtime's embedder serves them when it has one.
        """

        postgres_resolver, sqlite_resolver = _resolvers(connections)
        postgres = PostgresConnectionAdapter(
            postgres_resolver,
            max_size=pool_size,
            acquire_timeout_seconds=acquire_timeout_seconds,
            pool_factory=pool_factory,
        )
        sqlite = SqliteConnectionAdapter(sqlite_resolver)
        try:
            postgres_inspector = PostgresSourceInspector(postgres)
            sqlite_inspector = SqliteSourceInspector(sqlite)
            runtime = InMemoryCatalogRuntime(
                catalog,
                SourceInspectionRegistry(
                    [
                        InspectionAdapterBinding(
                            source_kind=postgres_inspector.source_kind,
                            inspector=postgres_inspector,
                            validator=PostgresCatalogValidator(),
                        ),
                        InspectionAdapterBinding(
                            source_kind=sqlite_inspector.source_kind,
                            inspector=sqlite_inspector,
                            validator=SqliteCatalogValidator(),
                        ),
                    ]
                ),
            )
            result = runtime.refresh()
            if runtime.active is None:
                raise CatalogRuntimeError(_activation_failure(result))
            stats = (
                StatisticsService(
                    {
                        SourceKind.POSTGRES: PostgresStatisticsProvider(postgres),
                        SourceKind.SQLITE: SqliteStatisticsProvider(sqlite),
                    },
                    observations=ObservationStore(),
                )
                if statistics
                else None
            )
            extensions = _extensions(semantic, embedder)
            planner = FederatedPhysicalPlanner(
                SourcePlanningRegistry([PostgresPlanningAdapter(), SqlitePlanningAdapter()]),
                policy=planner_policy,
                statistics=stats,
                extensions=extensions,
            )
            engine = QueryExecutionEngine(
                runtime,
                QueryCompilerRegistry([PostgresQueryCompiler(), SqliteQueryCompiler()]),
                QueryExecutionAdapterRegistry(
                    [
                        PostgresQueryExecutionAdapter(
                            postgres, default_statement_timeout_seconds=statement_timeout_seconds
                        ),
                        SqliteQueryExecutionAdapter(
                            sqlite, default_statement_timeout_seconds=statement_timeout_seconds
                        ),
                    ]
                ),
                planner=planner,
                statistics=stats,
                extensions=extensions,
                execution_policy=execution_policy,
                join_policy=join_policy,
            )
        except BaseException:
            postgres.close()
            sqlite.close()
            raise
        similar = next((e for e in extensions if isinstance(e, SimilarExtension)), None)
        return cls(
            runtime,
            engine,
            close=_close_all(postgres, sqlite),
            semantic=any(isinstance(e, SemanticExtension) for e in extensions),
            embedding_model=None if similar is None or similar.embedder is None else similar.embedder.info.model,
        )

    # --- asking -------------------------------------------------------------------------

    def query(self, query: Mapping[str, Any] | str, *, timeout_seconds: float | None = None) -> QueryExecutionResult:
        """Run one logical query.  ``timeout_seconds`` bounds the whole query, every read included."""

        return self._engine.execute(_as_query(query), timeout_seconds=timeout_seconds)

    def explain(self, query: Mapping[str, Any] | str) -> PlanExplanation:
        """The physical plan (sources, pushdown, read order, strategy) without running it."""

        return self._engine.explain(_as_query(query))

    # --- the catalog --------------------------------------------------------------------

    @property
    def catalog(self) -> Catalog:
        return self._runtime.require_active().catalog

    @property
    def semantic_enabled(self) -> bool:
        """Whether a semantic filter was configured, so a query may hold a semantic condition."""

        return self._semantic

    @property
    def similar_enabled(self) -> bool:
        """Whether an embedding provider was configured, so a query may hold a similarity search."""

        return self._embedding_model is not None

    @property
    def embedding_model(self) -> str | None:
        """The model of the configured embedding provider (a field is searchable when its embeddings use it)."""

        return self._embedding_model

    def similar_fields(self) -> dict[str, tuple[str, ...]]:
        """Per dataset, the fields a similarity search can rank: embedded with the configured model."""

        if self._embedding_model is None:
            return {}
        found: dict[str, list[str]] = {}
        for source in self.catalog.sources.values():
            for dataset_name, representation in source.datasets.items():
                for field_name, binding in representation.embeddings.items():
                    if binding.model == self._embedding_model:
                        found.setdefault(dataset_name, []).append(field_name)
        return {dataset: tuple(sorted(fields)) for dataset, fields in found.items()}

    @property
    def limits(self) -> dict[str, int]:
        """The bounds in force: page sizes, list sizes and the row guards."""

        return self._engine.limits

    def describe(self) -> dict[str, Any]:
        """What can be queried: each dataset's description and fields with their types."""

        catalog = self.catalog
        similar = self.similar_fields()
        return {
            name: {
                "description": dataset.description,
                "fields": {
                    field: {
                        "type": spec.type.value,
                        **({"semantic": True} if spec.semantic_eligible else {}),
                        **({"similar": True} if field in similar.get(name, ()) else {}),
                    }
                    for field, spec in dataset.fields.items()
                },
                **self._rules_of(name),
            }
            for name, dataset in catalog.datasets.items()
        }

    def _rules_of(self, dataset: str) -> dict[str, Any]:
        rules = {rule.name: rule.definition for rule in self.catalog.rules.get(dataset, ()) if rule.visible}
        return {"rules": rules} if rules else {}

    def status(self) -> dict[str, Any]:
        """The active catalog and each source's validation status."""

        active = self._runtime.require_active()
        return {
            "catalog": {"name": active.catalog.metadata.name, "version": active.catalog.metadata.version},
            "evaluated_at": active.evaluated_at.isoformat(),
            "sources": {name: state.status.value for name, state in active.sources.items()},
        }

    def refresh(self) -> CatalogRefreshResult:
        """Re-read the catalog files and re-inspect the sources; the old catalog stays active if the new one fails."""

        return self._runtime.refresh()

    # --- lifecycle ----------------------------------------------------------------------

    def close(self) -> None:
        if self._close is not None:
            self._close()
            self._close = None

    def __enter__(self) -> "YoDb":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


connect = YoDb.connect


def _close_all(*adapters):
    def close() -> None:
        for adapter in adapters:
            adapter.close()

    return close


class _SqliteFromGenericResolver:
    """A resolver object written for PostgreSQL, asked for a SQLite reference: its conninfo is read as the path."""

    def __init__(self, resolver: PostgresConnectionReferenceResolver) -> None:
        self._resolver = resolver

    def resolve(self, connection_ref: str) -> SqliteConnectionSettings:
        settings = self._resolver.resolve(connection_ref)
        if isinstance(settings, SqliteConnectionSettings):
            return settings
        return settings_from_text(getattr(settings, "conninfo", str(settings)))


def _resolver(connections: Connections) -> PostgresConnectionReferenceResolver:
    return _resolvers(connections)[0]


def _resolvers(connections: Connections) -> tuple[PostgresConnectionReferenceResolver, SqliteConnectionReferenceResolver]:
    """The resolver of each backend.  A reference is only ever resolved by the backend of its source's kind."""

    if connections is None:
        return EnvPostgresConnectionResolver(), EnvSqliteConnectionResolver()
    if isinstance(connections, Mapping):
        postgres = {
            ref: value if isinstance(value, PostgresConnectionSettings) else PostgresConnectionSettings(conninfo=str(value))
            for ref, value in connections.items()
            if not isinstance(value, SqliteConnectionSettings)
        }
        sqlite = {
            ref: value if isinstance(value, SqliteConnectionSettings) else settings_from_text(str(value))
            for ref, value in connections.items()
            if not isinstance(value, PostgresConnectionSettings)
        }
        return MappingPostgresConnectionResolver(postgres), MappingSqliteConnectionResolver(sqlite)
    return connections, _SqliteFromGenericResolver(connections)


def _extensions(
    semantic: SemanticExtension | SemanticRuntime | None,
    embedder: EmbeddingProvider | SimilarExtension | None,
) -> tuple[SemanticExtension | SimilarExtension, ...]:
    extensions: list[SemanticExtension | SimilarExtension] = []
    if semantic is not None:
        extensions.append(SemanticExtension(semantic) if isinstance(semantic, SemanticRuntime) else semantic)
    if embedder is None and extensions and extensions[0].runtime is not None:
        embedder = extensions[0].runtime.embedder            # one embedder serves both
    if embedder is not None:
        extensions.append(embedder if isinstance(embedder, SimilarExtension) else SimilarExtension(embedder))
    return tuple(extensions)


def _as_query(query: Mapping[str, Any] | str) -> Mapping[str, Any]:
    if isinstance(query, str):
        try:
            query = json.loads(query)
        except json.JSONDecodeError as error:
            raise QueryError(
                ErrorDetail(
                    code=ErrorCode.QUERY_SHAPE_INVALID,
                    message=f"The query is not valid JSON ({error.msg} at line {error.lineno}, column {error.colno}).",
                    retryable=False,
                )
            ) from error
        except (ValueError, RecursionError) as error:        # e.g. a 100,000-digit number, or absurd nesting
            raise QueryError(
                ErrorDetail(
                    code=ErrorCode.QUERY_SHAPE_INVALID,
                    message="The query text is too large or too deeply nested to read.",
                    retryable=False,
                )
            ) from error
    return query


def _activation_failure(result: CatalogRefreshResult) -> ErrorDetail:
    """A safe, specific account of why the catalog did not activate."""

    if result.candidate is None:
        return ErrorDetail(
            code=ErrorCode.CATALOG_LOAD_FAILED,
            message="The catalog YAML files could not be loaded or statically validated "
            "(run `yodb catalog <dir>` for the specific error).",
            retryable=False,
        )
    mismatched: dict[str, list[str]] = {}      # the source answered, and the catalog does not match it
    uninspected: dict[str, list[str]] = {}     # the source could not be inspected at all
    for name, state in result.candidate.sources.items():
        if state.error is not None:
            uninspected[name] = [f"{state.error.code.value}: {state.error.message}"]
        elif state.validation is not None and not state.validation.is_valid:
            mismatched[name] = [f"{f.severity.value}: {f.message}" for f in state.validation.findings]
    parts = []
    if mismatched:
        parts.append("the catalog does not match " + ", ".join(mismatched))
    if uninspected:
        parts.append("could not inspect " + ", ".join(uninspected))
    return ErrorDetail(
        code=ErrorCode.SOURCE_VALIDATION_FAILED,
        message=f"Cannot open the catalog: {'; '.join(parts) or 'a source did not validate'}.",
        retryable=False,
        details={"sources": {**mismatched, **uninspected}},
    )
