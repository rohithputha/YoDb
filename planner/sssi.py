"""The Structured Semantic Schema Index (SSSI): typed catalog entries, retrieved per question.

Every dataset, field, relationship and rule of the catalog is an *entry* with a short text
(name, aliases, description, type) and a vector (the configured embedding provider, the same
BGE-small model that embeds the data).  For a question, entries are scored by a hybrid of
lexical overlap and vector similarity, the best are kept, and the kept set is *closed*: a
field brings its dataset, a relationship brings both its datasets and both datasets bring the
relationships between them.  The result is a typed fragment (datasets, their relevant fields,
relationships) that ``schema.render_full`` renders exactly as it would the whole catalog.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math
import re

from yodb.catalog import Catalog, Visibility
from yodb.semantic.contracts import EmbeddingProvider, EmbeddingRequest

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("the a an of in on at to for and or with which who what that this is are was were be by from as has have had its it their there how many list find show give me do does".split())


def tokens(text: str) -> list[str]:
    words = [w for w in _WORD.findall(text.lower()) if w not in _STOP]
    # split snake_case and keep simple stems (plural -> singular, -ed, -ing)
    out = []
    for word in words:
        for part in word.split("_"):
            if not part:
                continue
            if len(part) > 4 and part.endswith("ies"):
                part = part[:-3] + "y"
            elif len(part) > 3 and part.endswith("s") and not part.endswith("ss"):
                part = part[:-1]
            out.append(part)
    return out


@dataclass(frozen=True)
class Entry:
    kind: str                 # dataset | field | relationship | rule
    key: str                  # "business", "business.stars", "user_friends_with", "business.highly_rated"
    text: str                 # what is embedded and matched
    dataset: str | None = None
    datasets: tuple[str, ...] = ()     # a relationship's two ends


@dataclass
class Fragment:
    datasets: list[str] = field(default_factory=list)
    fields: dict[str, list[str]] = field(default_factory=dict)
    relationships: list[str] = field(default_factory=list)
    entries: list[tuple[float, Entry]] = field(default_factory=list)     # the scored entries kept, best first


def entries_of(catalog: Catalog) -> list[Entry]:
    entries: list[Entry] = []
    for name, dataset in catalog.datasets.items():
        entries.append(Entry("dataset", name, f"{name} {' '.join(dataset.aliases)}: {dataset.description}", dataset=name))
        for field_name, spec in dataset.fields.items():
            if spec.visibility is not Visibility.PUBLIC:
                continue
            examples = " ".join(str(v) for v in spec.example_values[:3])
            entries.append(Entry("field", f"{name}.{field_name}", f"{name} {field_name} {' '.join(spec.aliases)} ({spec.type.value}): {spec.description} {examples}", dataset=name))
        for rule in catalog.rules.get(name, ()):
            if rule.visible:
                entries.append(Entry("rule", f"{name}.{rule.name}", f"{name} rule {rule.name}: {rule.definition}", dataset=name))
    for name, relationship in catalog.relationships.items():
        entries.append(Entry(
            "relationship", name,
            f"{name} {' '.join(relationship.aliases)}: {relationship.from_dataset} to {relationship.to_dataset}. {relationship.description}",
            datasets=(relationship.from_dataset, relationship.to_dataset),
        ))
    return entries


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class SchemaIndex:
    """The entries with their vectors, and hybrid retrieval over them."""

    def __init__(self, catalog: Catalog, embedder: EmbeddingProvider | None = None, *, lexical_weight: float = 0.5) -> None:
        self.catalog = catalog
        self.entries = entries_of(catalog)
        self._tokens = [set(tokens(entry.text)) for entry in self.entries]
        self._document_frequency: dict[str, int] = {}
        for words in self._tokens:
            for word in words:
                self._document_frequency[word] = self._document_frequency.get(word, 0) + 1
        self.embedder = embedder
        self._vectors: list[Sequence[float]] | None = None
        if embedder is not None:
            self._vectors = list(embedder.embed(EmbeddingRequest(tuple(entry.text for entry in self.entries))).vectors)
        self.lexical_weight = lexical_weight if embedder is not None else 1.0

    def scores(self, question: str) -> list[float]:
        words = tokens(question)
        total = len(self.entries)
        lexical = []
        for entry_words in self._tokens:
            score = 0.0
            for word in set(words):
                if word in entry_words:
                    score += math.log(1 + total / self._document_frequency[word])
            lexical.append(score)
        top = max(lexical) or 1.0
        lexical = [s / top for s in lexical]
        if self._vectors is None:
            return lexical
        vector = self.embedder.embed(EmbeddingRequest((question,))).vectors[0]
        semantic = [_cosine(vector, v) for v in self._vectors]
        low, high = min(semantic), max(semantic)
        semantic = [(s - low) / (high - low) if high > low else 0.0 for s in semantic]
        w = self.lexical_weight
        return [w * l + (1 - w) * s for l, s in zip(lexical, semantic)]

    def retrieve(self, question: str, *, top_k: int = 12, minimum_score: float = 0.15) -> Fragment:
        """The best entries, closed over datasets and relationships, as a typed fragment."""

        scored = sorted(zip(self.scores(question), self.entries), key=lambda item: -item[0])
        kept = [(s, e) for s, e in scored[:top_k] if s >= minimum_score]
        if not kept:
            kept = scored[:3]
        fragment = Fragment(entries=kept)
        datasets: dict[str, None] = {}
        fields: dict[str, dict[str, None]] = {}
        for _, entry in kept:
            if entry.kind == "dataset":
                datasets[entry.key] = None
            elif entry.kind in ("field", "rule"):
                datasets[entry.dataset] = None
                if entry.kind == "field":
                    fields.setdefault(entry.dataset, {})[entry.key.split(".", 1)[1]] = None
            elif entry.kind == "relationship":
                for name in entry.datasets:
                    datasets[name] = None
        # a dataset with no field kept shows all of its fields; relationships between kept datasets come along
        relationships = [
            name for name, relationship in self.catalog.relationships.items()
            if relationship.from_dataset in datasets and relationship.to_dataset in datasets
        ]
        for _, entry in kept:
            if entry.kind == "relationship" and entry.key not in relationships:
                relationships.append(entry.key)
        fragment.datasets = list(datasets)
        fragment.fields = {name: list(names) for name, names in fields.items()}
        fragment.relationships = relationships
        return fragment
