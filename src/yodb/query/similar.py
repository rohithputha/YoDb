"""The similarity search as a registered query term.

``{"similar": {"field": ..., "text": ..., "top": K}}`` keeps the ``top`` records
whose text ``field`` is nearest to ``text`` in embedding space, among the records
that satisfy the other conditions, nearest first.  Unlike a semantic condition it
does not ask whether anything is *true* of a record: it is a ranking, answered by
the source's vector search with no model judging the rows.  It is a conjunct
(one per query, never under ``any``/``not``) on a public text field the catalog
marks ``semantic_eligible`` and holds embeddings for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..catalog import LogicalType
from ..errors import ErrorCode
from .extensions import TermExtension, strip_terms, terms_of_type
from .fields import bind_public_field
from .models import BoundDataset, BoundExtensionTerm, BoundField, BoundFilterExpression, ExtensionTerm, FieldUse
from .shape import fail, mapping, non_empty_string, reject_unknown

_KEYS = frozenset({"field", "text", "top"})
MAXIMUM_TEXT_CHARS = 1_000          # policy.limits["similar.text_chars"] overrides
MAXIMUM_TOP = 1_000                 # policy.limits["similar.top"] overrides


@dataclass(frozen=True)
class SimilarPredicate(ExtensionTerm):
    """The ``top`` records nearest to ``text`` by ``field`` (before binding)."""

    field: str
    text: str
    top: int | None


@dataclass(frozen=True)
class BoundSimilarPredicate(BoundExtensionTerm):
    """A validated similarity search on a public, semantic-eligible text field.

    ``top`` is None when the caller left it out: the page size then applies.
    """

    field: BoundField
    text: str
    top: int | None


def _parse(body: Any, location: str) -> SimilarPredicate:
    body = mapping(body, location)
    reject_unknown(body, _KEYS, location)
    top = body.get("top")
    if top is not None and (isinstance(top, bool) or not isinstance(top, int) or top < 1):
        fail(ErrorCode.QUERY_LIMIT_INVALID, "'top' must be a positive integer.", f"{location}.top")
    return SimilarPredicate(
        field=non_empty_string(body.get("field"), f"{location}.field"),
        text=non_empty_string(body.get("text"), f"{location}.text"),
        top=top,
    )


def _bind(term: SimilarPredicate, root: BoundDataset, location: str, policy: Any) -> BoundSimilarPredicate:
    field = bind_public_field(root, term.field, f"{location}.similar.field")
    if field.spec.type not in {LogicalType.STRING, LogicalType.TEXT}:
        fail(
            ErrorCode.QUERY_OPERATOR_NOT_SUPPORTED,
            f"Field '{field.name}' of type '{field.spec.type.value}' cannot take a similarity search.",
            f"{location}.similar.field",
        )
    if not field.spec.semantic_eligible:
        fail(
            ErrorCode.QUERY_OPERATOR_NOT_SUPPORTED,
            f"Field '{field.name}' is not marked semantic_eligible in the catalog, so it has no embeddings to search.",
            f"{location}.similar.field",
        )
    text = term.text.strip()
    if not text:
        fail(ErrorCode.QUERY_VALUE_TYPE_INVALID, "The text to search for must not be blank.", f"{location}.similar.text")
    limit = policy.limit("similar.text_chars", MAXIMUM_TEXT_CHARS)
    if len(text) > limit:
        fail(ErrorCode.QUERY_LIMIT_INVALID, f"The text may not exceed {limit} characters.", f"{location}.similar.text")
    top_limit = policy.limit("similar.top", MAXIMUM_TOP)
    if term.top is not None and term.top > top_limit:
        fail(ErrorCode.QUERY_LIMIT_INVALID, f"'top' may be at most {top_limit}.", f"{location}.similar.top")
    return BoundSimilarPredicate(field=field, text=text, top=term.top)


def _vector_stores(term: BoundSimilarPredicate, catalog, dataset: str) -> tuple[str, ...]:
    """Sources that hold only the vectors of the term's field (the text lives elsewhere)."""

    return tuple(
        name
        for name, source in catalog.sources.items()
        if (representation := source.datasets.get(dataset)) is not None
        and term.field.name in representation.embeddings
        and term.field.name not in representation.fields
    )


SIMILAR_TERM = TermExtension(
    key="similar",
    parsed_type=SimilarPredicate,
    bound_type=BoundSimilarPredicate,
    parse=_parse,
    bind=_bind,
    describe=lambda term: {"similar": {"field": term.field.name, "text": term.text, "top": term.top}},
    uses=lambda term: ((term.field, FieldUse.EXTENSION),),
    noun="similarity search",
    extra_sources=_vector_stores,
)


def similar_predicates(expression: BoundFilterExpression | None) -> tuple[BoundSimilarPredicate, ...]:
    """Every similarity search in ``expression``, in document order."""

    return terms_of_type(expression, BoundSimilarPredicate)  # type: ignore[return-value]


def without_similar_terms(expression: BoundFilterExpression | None) -> BoundFilterExpression | None:
    """The filter with its similarity conjuncts removed."""

    return strip_terms(expression, BoundSimilarPredicate)
