"""The parameterized question templates.

Each template is one *shape* of question: a natural-language sentence with slots, the gold
typed query (one or two YoDb steps; a second step reads the IDs the first returned), the
kinds of source the answer needs, and the native SQL that computes the gold answer set and
evidence IDs directly over the Yelp tables (the oracle: the harness only, systems never see it).

Classes, as in the proposal's workload:

* ``single``       one dataset: filters, rules, ordering, a similarity search;
* ``independent``  two datasets joined over a column, each side filtered on its own (both reads
                   can run at once; RQ2's parallel case);
* ``dependent``    chains: a graph traversal whose other side is read for the IDs reached, or a
                   second step over the IDs of the first (two/three sources);
* ``stress``       questions a correct system must refuse or push back on (unknown field or
                   dataset, wrong type, unsupported aggregation, a write, ambiguity).

Slots are filled by ``build.py`` from parameter pools drawn from the database (cities with
the most businesses, popular categories, prolific users...).  A plan value written ``"{slot}"``
takes the slot's value with its type; text with a slot inside takes it as text.  A value
``{"$ids": {"step": 0, "field": "business.id"}}`` in the second step is replaced at run time by
the distinct values of that field in the first step's rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PAGE = 500          # every gold query asks for the largest page; the oracle takes the same LIMIT in the same order
TOP = 10            # similarity searches keep this many

RELATIONAL, VECTOR, GRAPH = "relational", "vector", "graph"
TUNE_FRACTION = 40 / 240


@dataclass(frozen=True)
class Template:
    id: str
    cls: str
    question: str                                  # with {slot}s
    slots: tuple[str, ...]                         # parameter pools the question draws from, in order
    instances: int                                 # how many questions to make from it
    plan: tuple[dict[str, Any], ...] = ()          # gold steps (empty for a stress case with no valid query)
    sources: tuple[str, ...] = ()                  # kinds of source the gold plan touches
    oracle: str | None = None                      # SQL: answer_id, and evidence_<dataset> columns; %(slot)s params
    answer: str = "id"                             # the row field whose distinct values are the answer set
    answer_dataset: str = ""
    expect_error: str | None = None                # stress: the error a system must raise (or "refusal" for any)
    notes: str = ""
    extra_params: dict[str, Any] = field(default_factory=dict)   # fixed values added to the oracle's params


def p(field_: str, op: str, value: Any = None) -> dict[str, Any]:
    predicate: dict[str, Any] = {"field": field_, "op": op}
    if op not in ("is_null", "is_not_null"):
        predicate["value"] = value
    return predicate


def q(dataset: str, select=None, where=None, traverse=None, order_by=None, first: int = PAGE) -> dict[str, Any]:
    query: dict[str, Any] = {"from": {"dataset": dataset}, "page": {"first": first}}
    if select is not None:
        query["select"] = list(select)
    if where is not None:
        query["where"] = where
    if traverse is not None:
        query["traverse"] = [traverse]
    if order_by is not None:
        query["order_by"] = [{"field": f, "direction": d} for f, d in order_by]
    return query


def step(relationship: str, alias: str, select=None, where=None, direction: str | None = None, optional: bool = False) -> dict[str, Any]:
    s: dict[str, Any] = {"relationship": relationship, "as": alias}
    if select is not None:
        s["select"] = list(select)
    if where is not None:
        s["where"] = where
    if direction:
        s["direction"] = direction
    if optional:
        s["optional"] = True
    return s


def similar(text: str, top: int = TOP) -> dict[str, Any]:
    return {"similar": {"field": "text", "text": text, "top": top}}


def ids_from(step_index: int, field_: str) -> dict[str, Any]:
    return {"$ids": {"step": step_index, "field": field_}}


# --- oracle fragments (native SQL over the Yelp tables; the harness only) ------------------------------

_BUSINESS_CITY_STARS = "b.city = %(city)s AND b.stars >= %(stars_hi)s"
_VEC = "ORDER BY r.text_embedding <=> %(vector)s::vector, r.review_id LIMIT %(top)s"

TEMPLATES: tuple[Template, ...] = (
    # ================= single source (60) =================
    Template(
        "S01", "single", "Which businesses in {city} are rated at least {stars_hi} stars? Give their names.",
        ("city", "stars_hi"), 8,
        plan=(q("business", ["name", "stars"], {"all": [p("city", "eq", "{city}"), p("stars", "gte", "{stars_hi}")]}, order_by=[("name", "asc")]),),
        sources=(RELATIONAL,), answer_dataset="business",
        oracle=f"SELECT b.business_id AS answer_id FROM yelp.business b WHERE {_BUSINESS_CITY_STARS} ORDER BY b.name, b.business_id LIMIT {PAGE}",
    ),
    Template(
        "S02", "single", "List the businesses in {city} that are still open and have at least {review_count_min} reviews, most reviewed first.",
        ("city", "review_count_min"), 8,
        plan=(q("business", ["name", "review_count"], {"all": [p("city", "eq", "{city}"), {"rule": "open_now"}, p("review_count", "gte", "{review_count_min}")]},
               order_by=[("review_count", "desc")]),),
        sources=(RELATIONAL,), answer_dataset="business",
        oracle=f"SELECT b.business_id AS answer_id FROM yelp.business b WHERE b.city = %(city)s AND b.is_open AND b.review_count >= %(review_count_min)s "
               f"ORDER BY b.review_count DESC, b.business_id LIMIT {PAGE}",
    ),
    Template(
        "S03", "single", "Find reviews of at most {stars_lo} stars that at least {useful_min} people found useful.",
        ("stars_lo", "useful_min"), 6,
        plan=(q("review", ["stars", "useful"], {"all": [p("stars", "lte", "{stars_lo}"), p("useful", "gte", "{useful_min}")]}, order_by=[("useful", "desc")]),),
        sources=(RELATIONAL,), answer_dataset="review",
        oracle=f"SELECT r.review_id AS answer_id FROM yelp.review r WHERE r.stars <= %(stars_lo)s AND r.useful >= %(useful_min)s ORDER BY r.useful DESC, r.review_id LIMIT {PAGE}",
    ),
    Template(
        "S04", "single", "Which five-star reviews were written in {month_name} {year}?",
        ("year", "month"), 6,
        plan=(q("review", ["stars", "date"], {"all": [p("stars", "eq", 5), p("date", "gte", "{month_start}"), p("date", "lt", "{month_end}")]}, order_by=[("date", "asc")]),),
        sources=(RELATIONAL,), answer_dataset="review",
        oracle=f"SELECT r.review_id AS answer_id FROM yelp.review r WHERE r.stars = 5 AND r.date >= %(month_start)s::timestamp AND r.date < %(month_end)s::timestamp "
               f"ORDER BY r.date, r.review_id LIMIT {PAGE}",
    ),
    Template(
        "S05", "single", "Which users who have ever been Yelp Elite have at least {fans_min} fans?",
        ("fans_min",), 3,
        plan=(q("user", ["name", "fans"], {"all": [{"rule": "ever_elite"}, p("fans", "gte", "{fans_min}")]}, order_by=[("fans", "desc")]),),
        sources=(RELATIONAL,), answer_dataset="user",
        oracle=f"SELECT u.user_id AS answer_id FROM yelp.users u WHERE u.elite IS NOT NULL AND u.fans >= %(fans_min)s ORDER BY u.fans DESC, u.user_id LIMIT {PAGE}",
    ),
    Template(
        "S06", "single", "Which users have written at least {user_reviews_min} reviews and give {stars_hi} stars or more on average?",
        ("user_reviews_min", "stars_hi"), 5,
        plan=(q("user", ["name", "review_count", "average_stars"], {"all": [p("review_count", "gte", "{user_reviews_min}"), p("average_stars", "gte", "{stars_hi}")]},
               order_by=[("review_count", "desc")]),),
        sources=(RELATIONAL,), answer_dataset="user",
        oracle=f"SELECT u.user_id AS answer_id FROM yelp.users u WHERE u.review_count >= %(user_reviews_min)s AND u.average_stars >= %(stars_hi)s "
               f"ORDER BY u.review_count DESC, u.user_id LIMIT {PAGE}",
    ),
    Template(
        "S07", "single", "Which categories have '{word}' in their name?",
        ("word",), 4,
        plan=(q("category", ["name"], p("name", "contains", "{word}"), order_by=[("name", "asc")]),),
        sources=(RELATIONAL,), answer_dataset="category",
        oracle=f"SELECT c.category_id AS answer_id FROM yelp.category c WHERE position(%(word)s in c.name) > 0 ORDER BY c.name, c.category_id LIMIT {PAGE}",
    ),
    Template(
        "S08", "single", "Find the {top} reviews that most talk about {phrase}.",
        ("phrase",), 12,
        plan=(q("review", ["stars", "text"], similar("{phrase}")),),
        sources=(VECTOR, RELATIONAL), answer_dataset="review", extra_params={"top": TOP},
        oracle=f"SELECT r.review_id AS answer_id FROM yelp.review r WHERE r.text_embedding IS NOT NULL {_VEC}",
    ),
    Template(
        "S09", "single", "Among the positive reviews (4 stars or more), which {top} most talk about {phrase}?",
        ("phrase",), 8,
        plan=(q("review", ["stars", "text"], {"all": [{"rule": "positive"}, similar("{phrase}")]}),),
        sources=(VECTOR, RELATIONAL), answer_dataset="review", extra_params={"top": TOP},
        oracle=f"SELECT r.review_id AS answer_id FROM yelp.review r WHERE r.text_embedding IS NOT NULL AND r.stars >= 4 {_VEC}",
    ),
    # ================= independent multi-source (60): column joins, both sides filtered =================
    Template(
        "I01", "independent", "Show the positive reviews (4 stars or more) of businesses in {city} rated at least {stars_hi} stars, with the business name.",
        ("city", "stars_hi"), 12,
        plan=(q("review", ["stars"], {"rule": "positive"}, step("review_of_business", "business", ["name"], {"all": [p("city", "eq", "{city}"), p("stars", "gte", "{stars_hi}")]}),
               order_by=[("stars", "desc")]),),
        sources=(RELATIONAL,), answer_dataset="review",
        oracle=f"SELECT r.review_id AS answer_id, b.business_id AS evidence_business FROM yelp.review r JOIN yelp.business b ON b.business_id = r.business_id "
               f"WHERE r.stars >= 4 AND {_BUSINESS_CITY_STARS} ORDER BY r.stars DESC, r.review_id, b.business_id LIMIT {PAGE}",
    ),
    Template(
        "I02", "independent", "Which helpful negative reviews (2 stars or less, at least 5 useful votes) were written about open businesses in {city} with at least {review_count_min} reviews?",
        ("city", "review_count_min"), 12,
        plan=(q("review", ["stars", "useful"], {"all": [{"rule": "negative"}, {"rule": "helpful"}]},
               step("review_of_business", "business", ["name"], {"all": [p("city", "eq", "{city}"), {"rule": "open_now"}, p("review_count", "gte", "{review_count_min}")]}),
               order_by=[("useful", "desc")]),),
        sources=(RELATIONAL,), answer_dataset="review",
        oracle=f"SELECT r.review_id AS answer_id, b.business_id AS evidence_business FROM yelp.review r JOIN yelp.business b ON b.business_id = r.business_id "
               f"WHERE r.stars <= 2 AND r.useful >= 5 AND b.city = %(city)s AND b.is_open AND b.review_count >= %(review_count_min)s "
               f"ORDER BY r.useful DESC, r.review_id, b.business_id LIMIT {PAGE}",
    ),
    Template(
        "I03", "independent", "Which five-star reviews written in {year} are by active reviewers (50 reviews or more)?",
        ("year",), 6,
        plan=(q("review", ["stars", "date"], {"all": [p("stars", "eq", 5), p("date", "gte", "{year_start}"), p("date", "lt", "{year_end}")]},
               step("user_wrote_review", "author", ["name", "review_count"], {"rule": "active_reviewer"}, direction="reverse"), order_by=[("date", "asc")]),),
        sources=(RELATIONAL,), answer_dataset="review",
        oracle=f"SELECT r.review_id AS answer_id, u.user_id AS evidence_user FROM yelp.review r JOIN yelp.users u ON u.user_id = r.user_id "
               f"WHERE r.stars = 5 AND r.date >= %(year_start)s::timestamp AND r.date < %(year_end)s::timestamp AND u.review_count >= 50 "
               f"ORDER BY r.date, r.review_id, u.user_id LIMIT {PAGE}",
    ),
    Template(
        "I04", "independent", "List the negative reviews (2 stars or less) written by users who were ever Elite and have at least {fans_min} fans.",
        ("fans_min",), 6,
        plan=(q("user", ["name", "fans"], {"all": [{"rule": "ever_elite"}, p("fans", "gte", "{fans_min}")]},
               step("user_wrote_review", "review", ["stars", "useful"], {"rule": "negative"}), order_by=[("fans", "desc")]),),
        sources=(RELATIONAL,), answer="review.id", answer_dataset="review",
        oracle=f"SELECT r.review_id AS answer_id, u.user_id AS evidence_user FROM yelp.users u JOIN yelp.review r ON r.user_id = u.user_id "
               f"WHERE u.elite IS NOT NULL AND u.fans >= %(fans_min)s AND r.stars <= 2 ORDER BY u.fans DESC, u.user_id, r.review_id LIMIT {PAGE}",
    ),
    Template(
        "I05", "independent", "Which five-star reviews does {business_name} in {business_city} have?",
        ("business",), 8,
        plan=(q("business", ["name"], {"all": [p("name", "eq", "{business_name}"), p("city", "eq", "{business_city}")]},
               step("review_of_business", "review", ["stars", "useful"], p("stars", "eq", 5), direction="reverse"), order_by=[("name", "asc")]),),
        sources=(RELATIONAL,), answer="review.id", answer_dataset="review",
        oracle=f"SELECT r.review_id AS answer_id, b.business_id AS evidence_business FROM yelp.business b JOIN yelp.review r ON r.business_id = b.business_id "
               f"WHERE b.name = %(business_name)s AND b.city = %(business_city)s AND r.stars = 5 ORDER BY b.name, b.business_id, r.review_id LIMIT {PAGE}",
    ),
    Template(
        "I06", "independent", "For businesses in {city} with at least {review_count_min} reviews, list their helpful reviews (at least 5 useful votes).",
        ("city", "review_count_min"), 8,
        plan=(q("business", ["name"], {"all": [p("city", "eq", "{city}"), p("review_count", "gte", "{review_count_min}")]},
               step("review_of_business", "review", ["useful"], {"rule": "helpful"}, direction="reverse"), order_by=[("review.useful", "desc")]),),
        sources=(RELATIONAL,), answer="review.id", answer_dataset="review",
        oracle=f"SELECT r.review_id AS answer_id, b.business_id AS evidence_business FROM yelp.business b JOIN yelp.review r ON r.business_id = b.business_id "
               f"WHERE b.city = %(city)s AND b.review_count >= %(review_count_min)s AND r.useful >= 5 ORDER BY r.useful DESC, b.business_id, r.review_id LIMIT {PAGE}",
    ),
    Template(
        "I07", "independent", "Which reviews with at least {useful_min} useful votes were written by users with at least {fans_min} fans?",
        ("useful_min", "fans_min"), 8,
        plan=(q("user", ["name", "fans"], p("fans", "gte", "{fans_min}"), step("user_wrote_review", "review", ["useful"], p("useful", "gte", "{useful_min}")),
               order_by=[("review.useful", "desc")]),),
        sources=(RELATIONAL,), answer="review.id", answer_dataset="review",
        oracle=f"SELECT r.review_id AS answer_id, u.user_id AS evidence_user FROM yelp.users u JOIN yelp.review r ON r.user_id = u.user_id "
               f"WHERE u.fans >= %(fans_min)s AND r.useful >= %(useful_min)s ORDER BY r.useful DESC, u.user_id, r.review_id LIMIT {PAGE}",
    ),
    # ================= dependent chains (80): graph edges and two-step plans =================
    Template(
        "D01", "dependent", "Who are the friends of the user with id {user_id}?",
        ("user_id",), 8,
        plan=(q("user", ["name"], p("id", "eq", "{user_id}"), step("user_friends_with", "friend", ["name", "fans"]), order_by=[("friend.name", "asc")]),),
        sources=(GRAPH, RELATIONAL), answer="friend.id", answer_dataset="user",
        oracle=f"SELECT f.friend_id AS answer_id, u.user_id AS evidence_user FROM yelp.users u JOIN yelp.friendship f ON f.user_id = u.user_id "
               f"JOIN yelp.users fr ON fr.user_id = f.friend_id WHERE u.user_id = %(user_id)s ORDER BY fr.name, u.user_id, f.friend_id LIMIT {PAGE}",
    ),
    Template(
        "D02", "dependent", "Which friends of users with at least {fans_min_hi} fans have at least {fans_min} fans themselves?",
        ("fans_min_hi", "fans_min"), 8,
        plan=(q("user", ["name", "fans"], p("fans", "gte", "{fans_min_hi}"), step("user_friends_with", "friend", ["name", "fans"], p("fans", "gte", "{fans_min}")),
               order_by=[("fans", "desc")]),),
        sources=(GRAPH, RELATIONAL), answer="friend.id", answer_dataset="user",
        oracle=f"SELECT f.friend_id AS answer_id, u.user_id AS evidence_user FROM yelp.users u JOIN yelp.friendship f ON f.user_id = u.user_id "
               f"JOIN yelp.users fr ON fr.user_id = f.friend_id WHERE u.fans >= %(fans_min_hi)s AND fr.fans >= %(fans_min)s "
               f"ORDER BY u.fans DESC, u.user_id, f.friend_id LIMIT {PAGE}",
    ),
    Template(
        "D03", "dependent", "Which businesses rated at least {stars_hi} stars has the user with id {user_id} reviewed?",
        ("user_id", "stars_hi"), 8,
        plan=(q("user", ["name"], p("id", "eq", "{user_id}"), step("user_reviewed_business", "business", ["name", "stars"], p("stars", "gte", "{stars_hi}")),
               order_by=[("business.name", "asc")]),),
        sources=(GRAPH, RELATIONAL), answer="business.id", answer_dataset="business",
        oracle=f"SELECT DISTINCT b.business_id AS answer_id, r.user_id AS evidence_user FROM yelp.review r JOIN yelp.business b ON b.business_id = r.business_id "
               f"WHERE r.user_id = %(user_id)s AND b.stars >= %(stars_hi)s ORDER BY b.business_id LIMIT {PAGE}",
    ),
    Template(
        "D04", "dependent", "Which users with at least {fans_min} fans have reviewed {business_name} in {business_city}?",
        ("business", "fans_min"), 8,
        plan=(q("business", ["name"], {"all": [p("name", "eq", "{business_name}"), p("city", "eq", "{business_city}")]},
               step("user_reviewed_business", "reviewer", ["name", "fans"], p("fans", "gte", "{fans_min}"), direction="reverse"), order_by=[("reviewer.fans", "desc")]),),
        sources=(GRAPH, RELATIONAL), answer="reviewer.id", answer_dataset="user",
        oracle=f"SELECT DISTINCT u.user_id AS answer_id, b.business_id AS evidence_business FROM yelp.business b JOIN yelp.review r ON r.business_id = b.business_id "
               f"JOIN yelp.users u ON u.user_id = r.user_id WHERE b.name = %(business_name)s AND b.city = %(business_city)s AND u.fans >= %(fans_min)s "
               f"ORDER BY u.user_id, b.business_id LIMIT {PAGE}",
    ),
    Template(
        "D05", "dependent", "Which categories do the businesses in {city} rated at least {stars_hi} stars belong to?",
        ("city", "stars_hi"), 8,
        plan=(q("business", ["name"], {"all": [p("city", "eq", "{city}"), p("stars", "gte", "{stars_hi}")]}, step("business_in_category", "category", ["name"]),
               order_by=[("category.name", "asc")]),),
        sources=(GRAPH, RELATIONAL), answer="category.id", answer_dataset="category",
        oracle=f"SELECT bc.category_id AS answer_id, b.business_id AS evidence_business, c.name FROM yelp.business b "
               f"JOIN yelp.business_category bc ON bc.business_id = b.business_id JOIN yelp.category c ON c.category_id = bc.category_id "
               f"WHERE {_BUSINESS_CITY_STARS} ORDER BY c.name, b.business_id, bc.category_id LIMIT {PAGE}",
    ),
    Template(
        "D06", "dependent", "Which open businesses in {city} are listed under {category_name} and rated at least {stars_hi} stars?",
        ("category", "city", "stars_hi"), 8,
        plan=(q("category", ["name"], p("id", "eq", "{category}"),
               step("business_in_category", "business", ["name", "stars"], {"all": [p("city", "eq", "{city}"), {"rule": "open_now"}, p("stars", "gte", "{stars_hi}")]}, direction="reverse"),
               order_by=[("business.stars", "desc")]),),
        sources=(GRAPH, RELATIONAL), answer="business.id", answer_dataset="business",
        oracle=f"SELECT b.business_id AS answer_id, bc.category_id AS evidence_category FROM yelp.business_category bc JOIN yelp.business b ON b.business_id = bc.business_id "
               f"WHERE bc.category_id = %(category)s AND b.city = %(city)s AND b.is_open AND b.stars >= %(stars_hi)s ORDER BY b.stars DESC, bc.category_id, b.business_id LIMIT {PAGE}",
    ),
    Template(
        "D07", "dependent", "Which categories are the businesses reviewed by the user with id {user_id} listed under?",
        ("user_id",), 8,
        plan=(q("user", ["name"], p("id", "eq", "{user_id}"), step("user_reviewed_business", "business", ["name"])),
              q("business", ["name"], p("id", "in", ids_from(0, "business.id")), step("business_in_category", "category", ["name"]), order_by=[("category.name", "asc")])),
        sources=(GRAPH, RELATIONAL), answer="category.id", answer_dataset="category",
        oracle=f"SELECT DISTINCT bc.category_id AS answer_id, r.business_id AS evidence_business FROM yelp.review r JOIN yelp.business_category bc ON bc.business_id = r.business_id "
               f"WHERE r.user_id = %(user_id)s ORDER BY bc.category_id, r.business_id LIMIT {PAGE}",
        notes="two steps: the businesses reached over REVIEWED, then their categories over IN_CATEGORY",
    ),
    Template(
        "D08", "dependent", "Which businesses in {city} share a category with {business_name} in {business_city}?",
        ("business", "city"), 8,
        plan=(q("business", ["name"], {"all": [p("name", "eq", "{business_name}"), p("city", "eq", "{business_city}")]}, step("business_in_category", "category", ["name"])),
              q("category", ["name"], p("id", "in", ids_from(0, "category.id")),
                step("business_in_category", "business", ["name", "stars"], p("city", "eq", "{city}"), direction="reverse"), order_by=[("business.name", "asc")])),
        sources=(GRAPH, RELATIONAL), answer="business.id", answer_dataset="business",
        oracle=f"SELECT DISTINCT b2.business_id AS answer_id, bc1.category_id AS evidence_category FROM yelp.business b1 "
               f"JOIN yelp.business_category bc1 ON bc1.business_id = b1.business_id JOIN yelp.business_category bc2 ON bc2.category_id = bc1.category_id "
               f"JOIN yelp.business b2 ON b2.business_id = bc2.business_id WHERE b1.name = %(business_name)s AND b1.city = %(business_city)s AND b2.city = %(city)s "
               f"ORDER BY b2.business_id, bc1.category_id LIMIT {PAGE}",
        notes="two steps over the category vertices (business -> category -> business)",
    ),
    Template(
        "D09", "dependent", "Which businesses rated at least {stars_hi} stars have the friends of the user with id {user_id} reviewed?",
        ("user_id", "stars_hi"), 8,
        plan=(q("user", ["name"], p("id", "eq", "{user_id}"), step("user_friends_with", "friend", ["name"])),
              q("user", ["name"], p("id", "in", ids_from(0, "friend.id")), step("user_reviewed_business", "business", ["name", "stars"], p("stars", "gte", "{stars_hi}")),
                order_by=[("business.name", "asc")])),
        sources=(GRAPH, RELATIONAL), answer="business.id", answer_dataset="business",
        oracle=f"SELECT DISTINCT b.business_id AS answer_id, f.friend_id AS evidence_user FROM yelp.friendship f JOIN yelp.review r ON r.user_id = f.friend_id "
               f"JOIN yelp.business b ON b.business_id = r.business_id WHERE f.user_id = %(user_id)s AND b.stars >= %(stars_hi)s ORDER BY b.business_id, f.friend_id LIMIT {PAGE}",
        notes="two steps over the user vertices (user -> friends -> businesses)",
    ),
    Template(
        "D10", "dependent", "Which businesses are the {top} reviews that most talk about {phrase} about?",
        ("phrase",), 8,
        plan=(q("review", ["stars"], similar("{phrase}"), step("review_of_business", "business", ["name", "city"])),),
        sources=(VECTOR, RELATIONAL), answer="business.id", answer_dataset="business", extra_params={"top": TOP},
        oracle=f"SELECT n.business_id AS answer_id, n.review_id AS evidence_review FROM (SELECT r.review_id, r.business_id FROM yelp.review r "
               f"WHERE r.text_embedding IS NOT NULL {_VEC}) n ORDER BY n.business_id, n.review_id",
    ),
    # ================= stress (40): must be refused or pushed back on =================
    Template(
        "X01", "stress", "Which businesses in {city} have a rating above {stars_hi}?",
        ("city", "stars_hi"), 5, plan=(q("business", ["name"], {"all": [p("city", "eq", "{city}"), p("rating", "gt", "{stars_hi}")]}),),
        expect_error="field_not_found", notes="the natural question names a field that does not exist; the gold is the refusal",
    ),
    Template(
        "X02", "stress", "List the menus of businesses in {city}.", ("city",), 5,
        plan=(q("menu", None, p("city", "eq", "{city}")),), expect_error="dataset_not_found",
    ),
    Template(
        "X03", "stress", "Which categories are friends with the category {category_name}?", ("category",), 5,
        plan=(q("category", ["name"], p("id", "eq", "{category}"), step("user_friends_with", "friend")),), expect_error="relationship_not_applicable",
    ),
    Template(
        "X04", "stress", "How many businesses in {city} are rated at least {stars_hi} stars?", ("city", "stars_hi"), 5,
        expect_error="refusal", notes="counting is not supported: a correct system says so (or fetches and counts, which the harness also accepts)",
    ),
    Template(
        "X05", "stress", "Which businesses in {city} have excellent stars?", ("city",), 5,
        plan=(q("business", ["name"], {"all": [p("city", "eq", "{city}"), p("stars", "gte", "excellent")]}),), expect_error="query_value_type_invalid",
    ),
    Template(
        "X06", "stress", "What is the best place?", (), 5, expect_error="refusal",
        notes="ambiguous: no dataset, place or criterion; a correct system asks rather than guesses",
    ),
    Template(
        "X07", "stress", "Delete the reviews of {business_name} in {business_city}.", ("business",), 5, expect_error="refusal",
        notes="a write; YoDb has no write path and an agent must not pretend to",
    ),
    Template(
        "X08", "stress", "Which users with at least {fans_min} fans reviewed the user with id {user_id}?", ("fans_min", "user_id"), 5,
        plan=(q("user", ["name"], p("id", "eq", "{user_id}"), step("user_reviewed_business", "reviewer", ["name"], p("fans", "gte", "{fans_min}"), direction="reverse")),),
        expect_error="relationship_not_applicable", notes="the relationship exists but does not end at a user",
    ),
)


def by_class() -> dict[str, int]:
    counts: dict[str, int] = {}
    for template in TEMPLATES:
        counts[template.cls] = counts.get(template.cls, 0) + template.instances
    return counts
