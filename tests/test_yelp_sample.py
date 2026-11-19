"""The Yelp sampler (examples/yelp/sample.py): deterministic, bounded, and consistent across the three files."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "yelp"))

from sample import build_sample, category_id, choose_users, hash_key, sample_businesses, split_friends  # noqa: E402

from yodb.catalog import load_catalog  # noqa: E402


def business(i: int) -> str:
    return json.dumps({
        "business_id": f"b{i:03d}", "name": f"Place {i}", "city": "Tucson" if i % 2 else "Reno", "state": "AZ" if i % 2 else "NV",
        "stars": 3.5 + (i % 4) * 0.5, "review_count": i, "is_open": i % 3 != 0,
        "categories": "Restaurants, Coffee & Tea" if i % 3 else "Bars, Nightlife", "attributes": {"WiFi": "free"}, "hours": None,
    })


def review(i: int, businesses: int = 60, users: int = 40) -> str:
    return json.dumps({"review_id": f"r{i:04d}", "business_id": f"b{i % businesses:03d}", "user_id": f"u{i % users:03d}",
                       "stars": 1 + i % 5, "useful": i % 7, "text": f"review {i}", "date": "2019-03-04 05:06:07"})


def user(i: int, users: int = 40) -> str:
    friends = [f"u{j:03d}" for j in range(users) if j != i and (i + j) % 5 == 0]
    return json.dumps({"user_id": f"u{i:03d}", "name": f"U{i}", "review_count": 10 * i, "yelping_since": "2015-01-01 00:00:00",
                       "friends": ", ".join(friends) or "None", "fans": i, "average_stars": 3.7, "elite": "2019,2020" if i % 4 == 0 else ""})


BUSINESSES = [business(i) for i in range(60)]
REVIEWS = [review(i) for i in range(500)]
USERS = [user(i) for i in range(40)]


class SamplerTests(unittest.TestCase):
    def test_the_sample_is_a_function_of_the_seed_and_the_ids_only(self) -> None:
        first = build_sample(BUSINESSES, REVIEWS, USERS, businesses=20, reviews=100, users=15, friend_edges=30)
        shuffled = build_sample(reversed(BUSINESSES), reversed(REVIEWS), reversed(USERS), businesses=20, reviews=100, users=15, friend_edges=30)
        self.assertEqual(list(first.businesses), list(shuffled.businesses))
        self.assertEqual(list(first.reviews), list(shuffled.reviews))
        self.assertEqual(list(first.users), list(shuffled.users))
        self.assertEqual(first.friendships, shuffled.friendships)
        other_seed = build_sample(BUSINESSES, REVIEWS, USERS, businesses=20, reviews=100, users=15, friend_edges=30, seed="other")
        self.assertNotEqual(list(first.businesses), list(other_seed.businesses))

    def test_sizes_are_respected_and_the_subset_is_consistent(self) -> None:
        sample = build_sample(BUSINESSES, REVIEWS, USERS, businesses=20, reviews=100, users=15, friend_edges=30)
        summary = sample.summary()
        self.assertEqual((summary["businesses"], summary["reviews"], summary["users"]), (20, 100, 15))
        self.assertLessEqual(summary["friendship_edges"], 30)
        self.assertTrue(all(r["business_id"] in sample.businesses for r in sample.reviews.values()))
        self.assertTrue(all(r["user_id"] is None or r["user_id"] in sample.users for r in sample.reviews.values()))
        self.assertGreater(summary["reviews_without_user"], 0)               # 40 authors, 15 kept
        self.assertTrue(all(a in sample.users and b in sample.users for a, b in sample.friendships))
        self.assertEqual(len(set(sample.friendships)), len(sample.friendships))
        self.assertTrue(all((b, a) in set(sample.friendships) for a, b in sample.friendships))   # both directions
        self.assertTrue(all(b in sample.businesses and c in sample.categories for b, c in sample.business_categories))

    def test_users_are_chosen_by_how_many_sampled_reviews_they_wrote(self) -> None:
        reviews = {f"r{i}": {"user_id": "heavy" if i < 5 else f"u{i}"} for i in range(10)}
        self.assertEqual(choose_users(reviews, limit=1), ["heavy"])

    def test_places_can_be_restricted(self) -> None:
        from sample import read_jsonl

        tucson = sample_businesses(read_jsonl(BUSINESSES), limit=100, cities={"tucson"})
        self.assertTrue(tucson and all(b["city"] == "Tucson" for b in tucson.values()))
        nevada = sample_businesses(read_jsonl(BUSINESSES), limit=100, states={"nv"})
        self.assertTrue(nevada and all(b["state"] == "NV" for b in nevada.values()))

    def test_helpers(self) -> None:
        self.assertEqual(category_id("Coffee & Tea"), "coffee_tea")
        self.assertEqual(category_id("  Bars "), "bars")
        self.assertEqual(split_friends("None"), [])
        self.assertEqual(split_friends("a, b"), ["a", "b"])
        self.assertEqual(hash_key("s", "x"), hash_key("s", "x"))
        self.assertNotEqual(hash_key("s", "x"), hash_key("s", "y"))


class CatalogTests(unittest.TestCase):
    def test_the_yelp_catalog_loads_with_its_graph_edges_and_rules(self) -> None:
        catalog = load_catalog(Path(__file__).resolve().parent.parent / "examples" / "yelp" / "catalog")
        self.assertEqual(set(catalog.datasets), {"business", "review", "user", "category"})
        self.assertEqual(catalog.sources["yelp"].graph.name, "yelp_graph")
        edges = {name: rel.implementations[0].edge_type for name, rel in catalog.relationships.items()}
        self.assertEqual(edges["user_friends_with"], "FRIENDS_WITH")
        self.assertIsNone(edges["review_of_business"])
        self.assertIn("popular_favorite", [r.name for r in catalog.rules["business"]])


if __name__ == "__main__":
    unittest.main()
