"""The schema a planner is shown, in three representations (the subject of RQ3).

* ``full``          every dataset, field (with type, description, examples), relationship (with its
                    ends, cardinality, direction and whether it is a column join or graph edges) and
                    rule (with its definition): exact and complete, and the most tokens;
* ``descriptions``  dataset and field names with their descriptions only: no types, no keys, no
                    relationship ends or directions, no rule conditions (what a loose tool
                    description gives an agent);
* ``retrieved``     the typed fragments the SSSI retrieves for the question (see ``sssi.py``): the
                    ``full`` rendering restricted to the entries judged relevant.

All three render to plain text for the prompt; ``tokens`` estimates their cost.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from typing import Any

from yodb.catalog import Catalog, RelationshipSpec, Visibility, graph_implementation

REPRESENTATIONS = ("full", "descriptions", "retrieved")


def estimate_tokens(text: str) -> int:
    """A tokenizer-free estimate (about 4 characters per token for English and JSON)."""

    return max(1, len(text) // 4)


def _examples(spec) -> str:
    return f" e.g. {', '.join(json.dumps(v) for v in spec.example_values[:3])}" if spec.example_values else ""


def relationship_line(name: str, relationship: RelationshipSpec, catalog: Catalog) -> str:
    field_join = next((i for i in relationship.implementations if i.edge_type is None), None)
    if field_join is not None:
        how = f"on {relationship.from_dataset}.{field_join.from_endpoint.field} = {relationship.to_dataset}.{field_join.to_endpoint.field}"
    else:
        edge = next((i for i in relationship.implementations if graph_implementation(catalog, i)), None)
        how = f"over graph edge {edge.edge_type}" if edge else "not traversable"
    direction = "reversible" if relationship.direction == "bi" else "one way"
    aliases = f" (aliases: {', '.join(relationship.aliases)})" if relationship.aliases else ""
    return f"- {name}{aliases}: {relationship.from_dataset} -> {relationship.to_dataset}, {relationship.cardinality.value}, {direction}, {how}. {relationship.description}"


def render_full(
    catalog: Catalog,
    *,
    datasets: Iterable[str] | None = None,
    fields: Mapping[str, Iterable[str]] | None = None,
    relationships: Iterable[str] | None = None,
    similar_fields: Mapping[str, Iterable[str]] | None = None,
) -> str:
    """The typed rendering; restricted to the given datasets/fields/relationships when they are given."""

    chosen = list(datasets) if datasets is not None else list(catalog.datasets)
    lines = ["DATASETS (query them by these names; fields are typed; 'id' is always returned)"]
    for name in chosen:
        dataset = catalog.datasets[name]
        aliases = f" (aliases: {', '.join(dataset.aliases)})" if dataset.aliases else ""
        lines.append(f"{name}{aliases}: {dataset.description}")
        wanted = set(fields[name]) if fields is not None and name in fields else None
        for field_name, spec in dataset.fields.items():
            if spec.visibility is not Visibility.PUBLIC or (wanted is not None and field_name not in wanted and field_name != "id"):
                continue
            flags = ""
            if similar_fields and field_name in similar_fields.get(name, ()):
                flags = " [similar: vector search on this text]"
            elif spec.semantic_eligible:
                flags = " [text]"
            lines.append(f"  {field_name}: {spec.type.value}{flags} - {spec.description}{_examples(spec)}")
        rules = [r for r in catalog.rules.get(name, ()) if r.visible]
        if rules:
            lines.append("  rules (use as {\"rule\": name}): " + "; ".join(f"{r.name} = {r.definition}" for r in rules))
    chosen_relationships = list(relationships) if relationships is not None else [
        n for n, r in catalog.relationships.items() if r.from_dataset in chosen and r.to_dataset in chosen
    ]
    if chosen_relationships:
        lines.append("RELATIONSHIPS (a traverse step names one; 'reverse' starts at its -> end when reversible)")
        for name in chosen_relationships:
            lines.append(relationship_line(name, catalog.relationships[name], catalog))
    return "\n".join(lines)


def render_descriptions(catalog: Catalog) -> str:
    """Names and descriptions only: what a tool description tends to give an agent."""

    lines = ["DATASETS"]
    for name, dataset in catalog.datasets.items():
        lines.append(f"{name}: {dataset.description}")
        for field_name, spec in dataset.fields.items():
            if spec.visibility is Visibility.PUBLIC:
                lines.append(f"  {field_name}: {spec.description}")
        rules = [r for r in catalog.rules.get(name, ()) if r.visible]
        if rules:
            lines.append("  rules: " + "; ".join(f"{r.name} = {r.definition}" for r in rules))
    lines.append("RELATIONSHIPS")
    for name, relationship in catalog.relationships.items():
        lines.append(f"- {name}: {relationship.description}")
    return "\n".join(lines)


def render(representation: str, catalog: Catalog, *, question: str | None = None, index=None, similar_fields=None, **retrieval: Any) -> str:
    if representation == "full":
        return render_full(catalog, similar_fields=similar_fields)
    if representation == "descriptions":
        return render_descriptions(catalog)
    if representation == "retrieved":
        if index is None or question is None:
            raise ValueError("the retrieved representation needs the SSSI index and the question")
        fragment = index.retrieve(question, **retrieval)
        return render_full(catalog, datasets=fragment.datasets, fields=fragment.fields, relationships=fragment.relationships, similar_fields=similar_fields)
    raise ValueError(f"unknown schema representation {representation!r}; use one of {REPRESENTATIONS}")
