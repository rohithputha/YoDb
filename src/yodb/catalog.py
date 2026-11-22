"""Three-file YAML catalog loader for YoDb V0.1.

The public YAML shape is deliberately backend-neutral:
datasets -> sources/resources/physical fields -> relationships.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
import re
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator


# The catalog format this YoDb reads: the same major version, and any minor version up to this one.
# It moves only when a file format changes, not with every release of the package.
FORMAT_VERSION = (0, 2)
# rules.yaml first appeared in this format version, so it cannot declare an earlier one.
RULES_FORMAT_VERSION = (0, 2)
API_VERSION = f"yodb/v{FORMAT_VERSION[0]}.{FORMAT_VERSION[1]}"
_API_VERSION = re.compile(r"yodb/v(\d+)\.(\d+)")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
# A graph name, vertex label, edge type or property name as it is written into Cypher: letters, digits
# and underscores only, so a catalog value can never carry Cypher or SQL syntax into a statement.
_GRAPH_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class CatalogValidationError(ValueError):
    """Raised when catalog YAML cannot form a valid static catalog."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class LogicalType(str, Enum):
    ID = "id"
    STRING = "string"
    TEXT = "text"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    TIMESTAMP = "timestamp"
    UUID = "uuid"
    JSON = "json"
    BYTES = "bytes"


class Visibility(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"


class SourceKind(str, Enum):
    POSTGRES = "postgres"
    NEO4J = "neo4j"
    SQLITE = "sqlite"


class Cardinality(str, Enum):
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


class FieldSpec(StrictModel):
    type: LogicalType
    description: str
    aliases: tuple[str, ...] = ()
    example_values: tuple[Any, ...] = ()
    visibility: Visibility = Visibility.PUBLIC
    semantic_eligible: bool = False


class DatasetSpec(StrictModel):
    description: str
    aliases: tuple[str, ...] = ()
    fields: dict[str, FieldSpec]

    @model_validator(mode="after")
    def requires_public_id(self) -> "DatasetSpec":
        identifier = self.fields.get("id")
        if identifier is None or identifier.type is not LogicalType.ID:
            raise ValueError("must declare an 'id' field with type 'id'")
        if identifier.visibility is not Visibility.PUBLIC:
            raise ValueError("the 'id' field must have public visibility")
        return self


class CatalogMetadata(StrictModel):
    name: str
    version: int = Field(gt=0)


class DatasetsDocument(StrictModel):
    api_version: str
    catalog: CatalogMetadata
    datasets: dict[str, DatasetSpec]


class SourceFieldSpec(StrictModel):
    physical_name: str


class VectorMetric(str, Enum):
    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "inner_product"


class EmbeddingBinding(StrictModel):
    """A vector column holding precomputed embeddings of one logical text field.

    The column sits either in the source that owns the text, or in a separate
    *vector store* source that represents the same dataset (identity only).

    The embedding is a physical retrieval aid for ``SemanticFilter``; callers
    never name it.  ``model``, ``dimensions``, ``metric`` and ``version`` must
    describe how the stored vectors were produced so a query embedding is only
    ever compared with vectors of the same space.
    """

    column: str
    model: str
    dimensions: int = Field(gt=0)
    metric: VectorMetric = VectorMetric.COSINE
    version: str = "1"


class SourceDatasetSpec(StrictModel):
    resource: str
    identity: tuple[str, ...] = Field(min_length=1)
    fields: dict[str, SourceFieldSpec]
    # Logical text field name -> the vector column embedding it in this source.
    # A binding may sit in a *vector store* representation that maps only the identity: the
    # logical text field is then owned by another source, and this source holds only its vectors.
    embeddings: dict[str, EmbeddingBinding] = Field(default_factory=dict)


class VertexSpec(StrictModel):
    """The vertex label of an AGE graph that stands for one dataset.

    ``id_property`` is the vertex property holding the dataset's logical ``id``
    (as a string, exactly as the dataset's identity source stores it).
    """

    label: str
    id_property: str = "id"


class GraphSpec(StrictModel):
    """An Apache AGE graph inside a PostgreSQL source, used to traverse relationships.

    ``name`` is the graph as created with ``create_graph``.  ``vertices`` says which
    vertex label stands for which dataset; a relationship implementation with an
    ``edge_type`` on this source then traverses the edges of that type between the
    two labels.  The graph only has to hold identities and edges: the rows of the
    datasets stay in their own tables (of this or another source).
    """

    name: str
    vertices: dict[str, VertexSpec]


class SourceSpec(StrictModel):
    kind: SourceKind
    connection_ref: str
    read_only: Literal[True]
    # A source that only holds a graph (no table of any dataset) may leave this empty.
    datasets: dict[str, SourceDatasetSpec] = Field(default_factory=dict)
    graph: GraphSpec | None = None


class DatasetResolution(StrictModel):
    identity_source: str
    field_sources: dict[str, str]


class SourcesDocument(StrictModel):
    api_version: str
    sources: dict[str, SourceSpec]
    resolution: dict[str, DatasetResolution]


class RelationEndpoint(StrictModel):
    source: str
    field: str


class RelationshipImplementation(StrictModel):
    from_endpoint: RelationEndpoint = Field(alias="from")
    to_endpoint: RelationEndpoint = Field(alias="to")
    edge_type: str | None = None


class RelationshipSpec(StrictModel):
    from_dataset: str = Field(alias="from")
    to_dataset: str = Field(alias="to")
    description: str
    aliases: tuple[str, ...] = ()
    cardinality: Cardinality
    direction: Literal["uni", "bi"]
    implementations: tuple[RelationshipImplementation, ...] = Field(min_length=1)


class RelationsDocument(StrictModel):
    api_version: str
    relationships: dict[str, RelationshipSpec]


class RuleSpec(StrictModel):
    """A named filter on one dataset: what a business word means.

    ``definition`` is for the agent, ``condition`` is the filter YoDb runs (the query filter
    language, with ``{"rule": ...}`` to build on other rules), and a rule that is not
    ``visible`` is left out of discovery and cannot be named in a query.
    """

    name: str
    definition: str = Field(min_length=1)
    condition: dict[str, Any]
    visible: bool = True


class RulesDocument(StrictModel):
    api_version: str
    rules: dict[str, tuple[RuleSpec, ...]]


class Catalog(StrictModel):
    metadata: CatalogMetadata
    datasets: dict[str, DatasetSpec]
    sources: dict[str, SourceSpec]
    resolution: dict[str, DatasetResolution]
    relationships: dict[str, RelationshipSpec]
    rules: dict[str, tuple[RuleSpec, ...]] = Field(default_factory=dict)


def load_catalog(directory: str | Path) -> Catalog:
    """Load and statically validate ``datasets.yaml``, ``sources.yaml``, ``relations.yaml`` and an optional ``rules.yaml``.

    The function performs no network I/O and never resolves connection
    references. It is safe to run during local development and CI.
    """

    root = Path(directory)
    datasets_document = _load_document(root / "datasets.yaml", DatasetsDocument)
    sources = _load_document(root / "sources.yaml", SourcesDocument)
    relations = _load_document(root / "relations.yaml", RelationsDocument)
    rules = _load_document(root / "rules.yaml", RulesDocument, required=False, minimum=RULES_FORMAT_VERSION)

    catalog = Catalog(
        metadata=datasets_document.catalog,
        datasets=datasets_document.datasets,
        sources=sources.sources,
        resolution=sources.resolution,
        relationships=relations.relationships,
        rules=rules.rules if rules is not None else {},
    )
    _validate_catalog(catalog)
    return catalog


def _load_document(
    path: Path,
    model: type[StrictModel],
    *,
    required: bool = True,
    minimum: tuple[int, int] = (0, 0),
) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except FileNotFoundError as error:
        if not required:
            return None
        raise CatalogValidationError(f"Required catalog file is missing: {path.name}") from error
    except yaml.YAMLError as error:
        raise CatalogValidationError(f"Invalid YAML in {path.name}: {error}") from error

    if not isinstance(raw, dict):
        raise CatalogValidationError(f"{path.name} must contain a YAML mapping at its root")
    _check_api_version(path.name, raw.get("api_version"), minimum)

    try:
        return model.model_validate(raw)
    except ValidationError as error:
        raise CatalogValidationError(f"Invalid {path.name}: {error}") from error


def _check_api_version(file_name: str, declared: Any, minimum: tuple[int, int] = (0, 0)) -> None:
    """Refuse a file written for another format major or a newer minor than this YoDb reads.

    A missing key is left to the model, which reports it like any other required key.
    """

    if declared is None:
        return
    match = _API_VERSION.fullmatch(declared) if isinstance(declared, str) else None
    if match is None:
        raise CatalogValidationError(f"{file_name}: api_version must look like '{API_VERSION}', not {declared!r}")
    major, minor = int(match[1]), int(match[2])
    if (major, minor) > FORMAT_VERSION and major == FORMAT_VERSION[0]:
        raise CatalogValidationError(
            f"{file_name}: api_version {declared} is newer than the {API_VERSION} this YoDb reads; "
            "use a YoDb that supports it."
        )
    if major != FORMAT_VERSION[0]:
        raise CatalogValidationError(
            f"{file_name}: api_version {declared} is a different major version from the {API_VERSION} this YoDb reads."
        )
    if (major, minor) < minimum:
        raise CatalogValidationError(
            f"{file_name}: api_version must be yodb/v{minimum[0]}.{minimum[1]} or later, not {declared}."
        )


def _validate_catalog(catalog: Catalog) -> None:
    _validate_identifier("catalog.name", catalog.metadata.name)
    _validate_named_mapping("datasets", catalog.datasets)
    _validate_named_mapping("sources", catalog.sources)
    _validate_named_mapping("relationships", catalog.relationships, allow_empty=True)

    for dataset_name, dataset in catalog.datasets.items():
        _validate_named_mapping(f"datasets.{dataset_name}.fields", dataset.fields)

    for source_name, source in catalog.sources.items():
        if not source.connection_ref.strip():
            raise CatalogValidationError(f"sources.{source_name}.connection_ref must not be blank")
        _validate_named_mapping(f"sources.{source_name}.datasets", source.datasets, allow_empty=source.graph is not None)
        if source.graph is not None:
            _validate_graph(catalog, source_name, source)
        for dataset_name, representation in source.datasets.items():
            if dataset_name not in catalog.datasets:
                raise CatalogValidationError(
                    f"sources.{source_name}.datasets references unknown dataset '{dataset_name}'"
                )
            if not representation.resource.strip():
                raise CatalogValidationError(
                    f"sources.{source_name}.datasets.{dataset_name}.resource must not be blank"
                )
            _validate_named_mapping(
                f"sources.{source_name}.datasets.{dataset_name}.fields",
                representation.fields,
            )
            for field_name, binding in representation.fields.items():
                if field_name not in catalog.datasets[dataset_name].fields:
                    raise CatalogValidationError(
                        f"sources.{source_name}.datasets.{dataset_name}.fields references "
                        f"unknown field '{field_name}'"
                    )
                if not binding.physical_name.strip():
                    raise CatalogValidationError(
                        f"sources.{source_name}.datasets.{dataset_name}.fields.{field_name}.physical_name "
                        "must not be blank"
                    )
            for identity_field in representation.identity:
                if identity_field not in representation.fields:
                    raise CatalogValidationError(
                        f"sources.{source_name}.datasets.{dataset_name}.identity field "
                        f"'{identity_field}' is not mapped"
                    )
            if source.kind is SourceKind.SQLITE and representation.embeddings:
                raise CatalogValidationError(
                    f"sources.{source_name}.datasets.{dataset_name}.embeddings is not supported on a sqlite source "
                    "(SQLite has no vector search)"
                )
            for field_name, embedding in representation.embeddings.items():
                location = f"sources.{source_name}.datasets.{dataset_name}.embeddings.{field_name}"
                logical_field = catalog.datasets[dataset_name].fields.get(field_name)
                if logical_field is None:
                    raise CatalogValidationError(f"{location} must name a logical field of the dataset")
                if field_name in representation.fields:
                    pass                      # the vectors sit next to the text
                elif "id" not in representation.identity:
                    raise CatalogValidationError(
                        f"{location}: a vector store representation must map the dataset identity 'id'"
                    )
                if not logical_field.semantic_eligible or logical_field.type not in {
                    LogicalType.STRING,
                    LogicalType.TEXT,
                }:
                    raise CatalogValidationError(
                        f"{location} requires a string/text field with semantic_eligible: true"
                    )
                if not embedding.column.strip() or not embedding.model.strip() or not embedding.version.strip():
                    raise CatalogValidationError(f"{location} column, model and version must not be blank")
    _validate_embedding_owners(catalog)

    _validate_resolutions(catalog)
    _validate_relationships(catalog)

    from .rules import validate_rules      # the rule checks bind filters, which needs the query package

    validate_rules(catalog)


def _validate_embedding_owners(catalog: Catalog) -> None:
    """One logical text field has at most one embedding declaration across all sources."""

    seen: dict[tuple[str, str], str] = {}
    for source_name, source in catalog.sources.items():
        for dataset_name, representation in source.datasets.items():
            for field_name in representation.embeddings:
                key = (dataset_name, field_name)
                if key in seen:
                    raise CatalogValidationError(
                        f"embeddings for '{dataset_name}.{field_name}' are declared by both "
                        f"'{seen[key]}' and '{source_name}'; declare them once"
                    )
                seen[key] = source_name


def _validate_resolutions(catalog: Catalog) -> None:
    if set(catalog.resolution) != set(catalog.datasets):
        missing = sorted(set(catalog.datasets) - set(catalog.resolution))
        extra = sorted(set(catalog.resolution) - set(catalog.datasets))
        details = []
        if missing:
            details.append(f"missing resolutions for: {', '.join(missing)}")
        if extra:
            details.append(f"unknown dataset resolutions: {', '.join(extra)}")
        raise CatalogValidationError("resolution must cover every dataset (" + "; ".join(details) + ")")

    for dataset_name, resolution in catalog.resolution.items():
        _validate_source_field(catalog, dataset_name, resolution.identity_source, "id", "identity_source")
        identity_representation = catalog.sources[resolution.identity_source].datasets[dataset_name]
        if "id" not in identity_representation.identity:
            raise CatalogValidationError(
                f"resolution.{dataset_name}.identity_source '{resolution.identity_source}' must use 'id' "
                "as an identity field"
            )

        expected_fields = set(catalog.datasets[dataset_name].fields)
        actual_fields = set(resolution.field_sources)
        if expected_fields != actual_fields:
            missing = sorted(expected_fields - actual_fields)
            extra = sorted(actual_fields - expected_fields)
            details = []
            if missing:
                details.append(f"missing field sources: {', '.join(missing)}")
            if extra:
                details.append(f"unknown fields: {', '.join(extra)}")
            raise CatalogValidationError(
                f"resolution.{dataset_name}.field_sources must cover every field "
                f"({'; '.join(details)})"
            )

        if resolution.field_sources["id"] != resolution.identity_source:
            raise CatalogValidationError(
                f"resolution.{dataset_name}.field_sources.id must match identity_source"
            )

        for field_name, source_name in resolution.field_sources.items():
            _validate_source_field(catalog, dataset_name, source_name, field_name, "field_sources")


def _validate_relationships(catalog: Catalog) -> None:
    for relationship_name, relationship in catalog.relationships.items():
        if relationship.from_dataset not in catalog.datasets:
            raise CatalogValidationError(
                f"relationships.{relationship_name}.from references unknown dataset "
                f"'{relationship.from_dataset}'"
            )
        if relationship.to_dataset not in catalog.datasets:
            raise CatalogValidationError(
                f"relationships.{relationship_name}.to references unknown dataset "
                f"'{relationship.to_dataset}'"
            )

        for implementation in relationship.implementations:
            if implementation.edge_type is not None:
                _validate_graph_implementation(catalog, relationship_name, relationship, implementation)
                continue
            _validate_source_field(
                catalog,
                relationship.from_dataset,
                implementation.from_endpoint.source,
                implementation.from_endpoint.field,
                f"relationships.{relationship_name}.implementations.from",
            )
            _validate_source_field(
                catalog,
                relationship.to_dataset,
                implementation.to_endpoint.source,
                implementation.to_endpoint.field,
                f"relationships.{relationship_name}.implementations.to",
            )
            from_type = catalog.datasets[relationship.from_dataset].fields[implementation.from_endpoint.field].type
            to_type = catalog.datasets[relationship.to_dataset].fields[implementation.to_endpoint.field].type
            if from_type is not to_type:
                raise CatalogValidationError(
                    f"relationships.{relationship_name}.implementations compares "
                    f"'{relationship.from_dataset}.{implementation.from_endpoint.field}' ({from_type.value}) with "
                    f"'{relationship.to_dataset}.{implementation.to_endpoint.field}' ({to_type.value}); "
                    "the two fields must have the same type"
                )


def _validate_source_field(
    catalog: Catalog,
    dataset_name: str,
    source_name: str,
    field_name: str,
    location: str,
) -> None:
    source = catalog.sources.get(source_name)
    if source is None:
        raise CatalogValidationError(f"{location} references unknown source '{source_name}'")
    representation = source.datasets.get(dataset_name)
    if representation is None:
        raise CatalogValidationError(
            f"{location} source '{source_name}' does not represent dataset '{dataset_name}'"
        )
    if field_name not in representation.fields:
        raise CatalogValidationError(
            f"{location} field '{field_name}' is not mapped for dataset '{dataset_name}' "
            f"in source '{source_name}'"
        )


def _validate_graph(catalog: Catalog, source_name: str, source: SourceSpec) -> None:
    """A graph declaration: an AGE graph in a PostgreSQL source whose vertex labels stand for datasets."""

    location = f"sources.{source_name}.graph"
    if source.kind is SourceKind.SQLITE:
        raise CatalogValidationError(f"{location} is not supported on a sqlite source (SQLite has no graph engine)")
    if source.kind is not SourceKind.POSTGRES:
        raise CatalogValidationError(f"{location} is only supported on a postgres source (Apache AGE)")
    _validate_graph_identifier(f"{location}.name", source.graph.name)
    if not source.graph.vertices:
        raise CatalogValidationError(f"{location}.vertices must name at least one dataset")
    labels: dict[str, str] = {}
    for dataset_name, vertex in source.graph.vertices.items():
        if dataset_name not in catalog.datasets:
            raise CatalogValidationError(f"{location}.vertices references unknown dataset '{dataset_name}'")
        _validate_graph_identifier(f"{location}.vertices.{dataset_name}.label", vertex.label)
        _validate_graph_identifier(f"{location}.vertices.{dataset_name}.id_property", vertex.id_property)
        if vertex.label in labels:
            raise CatalogValidationError(
                f"{location}.vertices: label '{vertex.label}' stands for both '{labels[vertex.label]}' and '{dataset_name}'"
            )
        labels[vertex.label] = dataset_name


def _validate_graph_identifier(location: str, value: str) -> None:
    if not _GRAPH_IDENTIFIER.fullmatch(value):
        raise CatalogValidationError(
            f"{location} '{value}' must be letters, digits and underscores, starting with a letter or underscore"
        )


def graph_implementation(catalog: Catalog, implementation: RelationshipImplementation) -> bool:
    """Whether an ``edge_type`` implementation runs on an AGE graph (a postgres source with ``graph``)."""

    source = catalog.sources.get(implementation.from_endpoint.source)
    return source is not None and source.kind is SourceKind.POSTGRES and source.graph is not None


def _validate_graph_implementation(
    catalog: Catalog,
    relationship_name: str,
    relationship: RelationshipSpec,
    implementation: RelationshipImplementation,
) -> None:
    location = f"relationships.{relationship_name}.implementations"
    _validate_graph_identifier(f"{location}.edge_type", implementation.edge_type)

    from_source = implementation.from_endpoint.source
    to_source = implementation.to_endpoint.source
    if from_source != to_source:
        raise CatalogValidationError(f"{location} with edge_type must use one graph source")

    source = catalog.sources.get(from_source)
    if source is None:
        raise CatalogValidationError(f"{location}.from references unknown source '{from_source}'")

    if source.kind is SourceKind.POSTGRES and source.graph is not None:
        # An AGE edge joins the two datasets' identities: each end must be the dataset's ``id``, and the
        # graph must have a vertex label for both datasets.
        for end, dataset_name, endpoint in (
            ("from", relationship.from_dataset, implementation.from_endpoint),
            ("to", relationship.to_dataset, implementation.to_endpoint),
        ):
            if endpoint.field != "id":
                raise CatalogValidationError(
                    f"{location}.{end}.field must be 'id' for a graph edge (an edge links dataset identities)"
                )
            if dataset_name not in source.graph.vertices:
                raise CatalogValidationError(
                    f"{location} edge source '{from_source}' declares no graph vertex for dataset '{dataset_name}'"
                )
        return

    if source.kind is not SourceKind.NEO4J:
        raise CatalogValidationError(
            f"{location}.edge_type source '{from_source}' must be a postgres source with a 'graph' (Apache AGE) or neo4j"
        )

    for dataset_name, endpoint in ((relationship.from_dataset, implementation.from_endpoint), (relationship.to_dataset, implementation.to_endpoint)):
        _validate_source_field(catalog, dataset_name, endpoint.source, endpoint.field, f"{location}.edge")


def _validate_named_mapping(location: str, values: dict[str, Any], *, allow_empty: bool = False) -> None:
    if not values and not allow_empty:
        raise CatalogValidationError(f"{location} must not be empty")
    for name in values:
        _validate_identifier(location, name)


def _validate_identifier(location: str, value: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise CatalogValidationError(
            f"{location} identifier '{value}' must use lower_snake_case"
        )
