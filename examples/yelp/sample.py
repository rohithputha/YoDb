"""Deterministic sampling of the Yelp Open Dataset (pure Python, no database).

The target subset is the proposal's: up to 10,000 businesses, 100,000 reviews, 50,000
users and 200,000 friendship edges.  Every choice is a function of a seed string and
the record IDs (a SHA-256 ordering), never of file order or time, so two machines
with the same files and seed produce the same subset.

Order of selection:

1. businesses: those in the wanted cities/states (all when none is given), the first
   ``businesses`` in hash order;
2. reviews: reviews of those businesses, the ``reviews`` with the smallest hashes
   (a bounded heap over a streamed file, so the 5 GB review file never sits in memory);
3. users: the authors of those reviews, by how many sampled reviews they wrote, then
   hash order, the first ``users``; a sampled review whose author is not sampled keeps
   ``user_id`` NULL;
4. friendships: pairs of sampled users that Yelp lists as friends, both directions,
   the first ``friend_edges`` pairs in hash order of the unordered pair;
5. categories: every category of a sampled business.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
import hashlib
import heapq
import json
import re
from typing import Any

DEFAULT_SEED = "yodb-yelp-v1"

_NON_WORD = re.compile(r"[^a-z0-9]+")


def hash_key(seed: str, *parts: str) -> str:
    """The sort key of one record: a hex digest of the seed and the record's identity."""

    return hashlib.sha256(("\0".join((seed, *parts))).encode("utf-8")).hexdigest()


def category_id(name: str) -> str:
    """``"Coffee & Tea"`` -> ``"coffee_tea"``: a stable, graph-safe id for a category name."""

    return _NON_WORD.sub("_", name.strip().lower()).strip("_")


def split_categories(text: str | None) -> list[str]:
    return [part.strip() for part in (text or "").split(",") if part.strip()]


def split_friends(text: str | None) -> list[str]:
    if not text or text.strip().lower() == "none":
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def read_jsonl(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    for line in lines:
        line = line.strip()
        if line:
            yield json.loads(line)


@dataclass
class Sample:
    businesses: dict[str, dict[str, Any]] = field(default_factory=dict)      # business_id -> record
    reviews: dict[str, dict[str, Any]] = field(default_factory=dict)         # review_id -> record (user_id may be None)
    users: dict[str, dict[str, Any]] = field(default_factory=dict)           # user_id -> record
    friendships: list[tuple[str, str]] = field(default_factory=list)         # (user_id, friend_id), both directions
    categories: dict[str, str] = field(default_factory=dict)                 # category_id -> name
    business_categories: list[tuple[str, str]] = field(default_factory=list)  # (business_id, category_id)

    def summary(self) -> dict[str, int]:
        return {
            "businesses": len(self.businesses),
            "reviews": len(self.reviews),
            "reviews_without_user": sum(1 for r in self.reviews.values() if r.get("user_id") is None),
            "users": len(self.users),
            "friendship_edges": len(self.friendships),
            "categories": len(self.categories),
            "business_categories": len(self.business_categories),
        }


def sample_businesses(
    records: Iterable[dict[str, Any]],
    *,
    limit: int,
    seed: str = DEFAULT_SEED,
    cities: set[str] | None = None,
    states: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """The ``limit`` businesses of the wanted places with the smallest hashes."""

    cities = {c.lower() for c in cities} if cities else None
    states = {s.upper() for s in states} if states else None
    heap: list[tuple[str, str, dict[str, Any]]] = []
    for record in records:
        if cities is not None and str(record.get("city", "")).lower() not in cities:
            continue
        if states is not None and str(record.get("state", "")).upper() not in states:
            continue
        key = hash_key(seed, "business", record["business_id"])
        _keep_smallest(heap, (key, record["business_id"], record), limit)
    return {business_id: record for _, business_id, record in _in_key_order(heap)}


def sample_reviews(
    records: Iterable[dict[str, Any]], business_ids: set[str], *, limit: int, seed: str = DEFAULT_SEED
) -> dict[str, dict[str, Any]]:
    """The ``limit`` reviews of the sampled businesses with the smallest hashes."""

    heap: list[tuple[str, str, dict[str, Any]]] = []
    for record in records:
        if record.get("business_id") not in business_ids:
            continue
        key = hash_key(seed, "review", record["review_id"])
        _keep_smallest(heap, (key, record["review_id"], record), limit)
    return {review_id: record for _, review_id, record in _in_key_order(heap)}


def choose_users(reviews: dict[str, dict[str, Any]], *, limit: int, seed: str = DEFAULT_SEED) -> list[str]:
    """The authors of the sampled reviews: most sampled reviews first, then hash order."""

    counts: dict[str, int] = {}
    for review in reviews.values():
        user_id = review.get("user_id")
        if user_id:
            counts[user_id] = counts.get(user_id, 0) + 1
    ordered = sorted(counts, key=lambda user_id: (-counts[user_id], hash_key(seed, "user", user_id)))
    return ordered[:limit]


def sample_users(records: Iterable[dict[str, Any]], user_ids: set[str]) -> dict[str, dict[str, Any]]:
    """The user records of the chosen users (the only ones kept from the user file)."""

    found = {record["user_id"]: record for record in records if record.get("user_id") in user_ids}
    return {user_id: found[user_id] for user_id in sorted(found)}          # file order never leaks into the result


def sample_friendships(users: dict[str, dict[str, Any]], *, limit: int, seed: str = DEFAULT_SEED) -> list[tuple[str, str]]:
    """Friend pairs among the sampled users, both directions, the first ``limit`` unordered pairs in hash order.

    Yelp lists friendships on both users, so a pair is seen twice; it is counted once and stored twice.
    """

    pairs: set[tuple[str, str]] = set()
    for user_id, record in users.items():
        for friend in split_friends(record.get("friends")):
            if friend in users and friend != user_id:
                pairs.add((user_id, friend) if user_id < friend else (friend, user_id))
    chosen = sorted(pairs, key=lambda pair: hash_key(seed, "friendship", pair[0], pair[1]))[:limit]
    edges: list[tuple[str, str]] = []
    for a, b in chosen:
        edges.append((a, b))
        edges.append((b, a))
    return edges


def collect_categories(businesses: dict[str, dict[str, Any]]) -> tuple[dict[str, str], list[tuple[str, str]]]:
    categories: dict[str, str] = {}
    links: list[tuple[str, str]] = []
    for business_id, record in businesses.items():
        seen: set[str] = set()
        for name in split_categories(record.get("categories")):
            identifier = category_id(name)
            if not identifier or identifier in seen:
                continue
            seen.add(identifier)
            categories.setdefault(identifier, name)
            links.append((business_id, identifier))
    return categories, links


def build_sample(
    business_lines: Iterable[str],
    review_lines: Iterable[str],
    user_lines: Iterable[str],
    *,
    businesses: int = 10_000,
    reviews: int = 100_000,
    users: int = 50_000,
    friend_edges: int = 200_000,
    cities: set[str] | None = None,
    states: set[str] | None = None,
    seed: str = DEFAULT_SEED,
) -> Sample:
    """Read the three JSON-lines files (as line iterables) and build the whole subset."""

    sample = Sample()
    sample.businesses = sample_businesses(read_jsonl(business_lines), limit=businesses, seed=seed, cities=cities, states=states)
    sample.reviews = sample_reviews(read_jsonl(review_lines), set(sample.businesses), limit=reviews, seed=seed)
    chosen = set(choose_users(sample.reviews, limit=users, seed=seed))
    sample.users = sample_users(read_jsonl(user_lines), chosen)
    present = set(sample.users)
    for review in sample.reviews.values():
        if review.get("user_id") not in present:
            review["user_id"] = None
    sample.friendships = sample_friendships(sample.users, limit=friend_edges // 2, seed=seed)
    sample.categories, sample.business_categories = collect_categories(sample.businesses)
    return sample


def _keep_smallest(heap: list, item: tuple, limit: int) -> None:
    """Keep the ``limit`` items with the smallest keys: a bounded heap whose top is the largest key kept."""

    if limit <= 0:
        return
    entry = (_Invert(item[0]), item[1], item[2])
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif entry[0] > heap[0][0]:            # a smaller key than the largest kept
        heapq.heapreplace(heap, entry)


def _in_key_order(heap: list) -> list:
    """The kept items, smallest key first."""

    return sorted(heap, key=lambda entry: entry[0].key)


class _Invert:
    """Orders strings in reverse, so a min-heap keeps the largest key at the top."""

    __slots__ = ("key",)

    def __init__(self, key: str) -> None:
        self.key = key

    def __lt__(self, other: "_Invert") -> bool:
        return self.key > other.key

    def __gt__(self, other: "_Invert") -> bool:
        return self.key < other.key

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Invert) and self.key == other.key
