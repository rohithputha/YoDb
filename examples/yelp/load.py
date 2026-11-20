"""Load a deterministic sample of the Yelp Open Dataset into PostgreSQL + pgvector + Apache AGE.

    python examples/yelp/load.py --data ~/yelp --dsn "host=localhost port=5432 dbname=yelp user=postgres password=..."

Steps (each can be skipped or run alone):

    --schema         apply schema.sql (drops and recreates the ``yelp`` schema and the graph)
    tables           sample the three JSON-lines files and COPY the rows in
    embeddings       embed every review text with the configured model into ``yelp.review.text_embedding``
    graph            load the vertices and edges of ``yelp_graph`` (friendships, who reviewed what, categories)

``--data`` is the directory holding ``yelp_academic_dataset_business.json``, ``..._review.json``
and ``..._user.json``.  The DSN must be an owner of the database (it creates tables and the graph);
YoDb itself only ever uses the read-only role ``yodb_ro`` that schema.sql creates.

Sizes default to the proposal's target (10,000 businesses, 100,000 reviews, 50,000 users, 200,000
friendship edges); ``--preset fallback`` is the preregistered fallback (25,000 reviews).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import psycopg

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent.parent if HERE.parent.name == "examples" else HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from yodb.dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from sample import DEFAULT_SEED, Sample, build_sample  # noqa: E402

PRESETS = {
    "target": dict(businesses=10_000, reviews=100_000, users=50_000, friend_edges=200_000),
    "fallback": dict(businesses=10_000, reviews=25_000, users=50_000, friend_edges=200_000),
    "smoke": dict(businesses=300, reviews=2_000, users=1_000, friend_edges=4_000),
}
FILES = {
    "business": "yelp_academic_dataset_business.json",
    "review": "yelp_academic_dataset_review.json",
    "user": "yelp_academic_dataset_user.json",
}
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
GRAPH = "yelp_graph"


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


# --- tables ------------------------------------------------------------------------------------------


def _timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _elite(value: str | None) -> str | None:
    """Yelp writes the year 2020 as ``20,20`` in the elite list; store it as ``2020``."""

    if not value:
        return None
    years = [year.strip() for year in value.replace("20,20", "2020").split(",") if year.strip()]
    return ",".join(dict.fromkeys(years)) or None


def _jsonb(value) -> str | None:
    return None if value is None else json.dumps(value)


def copy_rows(cursor, table: str, columns: tuple[str, ...], rows) -> int:
    count = 0
    with cursor.copy(f"COPY {table} ({', '.join(columns)}) FROM STDIN") as copy:
        for row in rows:
            copy.write_row(row)
            count += 1
    return count


def load_tables(connection, sample: Sample) -> None:
    with connection.cursor() as cursor:
        n = copy_rows(cursor, "yelp.business",
                      ("business_id", "name", "address", "city", "state", "postal_code", "latitude", "longitude",
                       "stars", "review_count", "is_open", "categories", "attributes", "hours"),
                      ((b["business_id"], b.get("name") or "", b.get("address"), b.get("city"), b.get("state"),
                        b.get("postal_code"), b.get("latitude"), b.get("longitude"), b.get("stars"), b.get("review_count"),
                        bool(b.get("is_open")) if b.get("is_open") is not None else None, b.get("categories"),
                        _jsonb(b.get("attributes")), _jsonb(b.get("hours")))
                       for b in sample.businesses.values()))
        log(f"  yelp.business: {n:,} rows")
        n = copy_rows(cursor, "yelp.category", ("category_id", "name"), sample.categories.items())
        log(f"  yelp.category: {n:,} rows")
        n = copy_rows(cursor, "yelp.business_category", ("business_id", "category_id"), sample.business_categories)
        log(f"  yelp.business_category: {n:,} rows")
        n = copy_rows(cursor, "yelp.users",
                      ("user_id", "name", "review_count", "yelping_since", "useful", "funny", "cool", "fans", "elite", "average_stars"),
                      ((u["user_id"], u.get("name"), u.get("review_count"), _timestamp(u.get("yelping_since")), u.get("useful"),
                        u.get("funny"), u.get("cool"), u.get("fans"), _elite(u.get("elite")), u.get("average_stars"))
                       for u in sample.users.values()))
        log(f"  yelp.users: {n:,} rows")
        n = copy_rows(cursor, "yelp.friendship", ("user_id", "friend_id"), sample.friendships)
        log(f"  yelp.friendship: {n:,} rows")
        n = copy_rows(cursor, "yelp.review",
                      ("review_id", "user_id", "business_id", "stars", "useful", "funny", "cool", "text", "date"),
                      ((r["review_id"], r.get("user_id"), r["business_id"], r.get("stars"), r.get("useful"), r.get("funny"),
                        r.get("cool"), r.get("text"), _timestamp(r.get("date")))
                       for r in sample.reviews.values()))
        log(f"  yelp.review: {n:,} rows")
    connection.commit()


# --- embeddings --------------------------------------------------------------------------------------


def load_embeddings(connection, model_name: str, batch_size: int, threads: int | None = None) -> None:
    from yodb.embeddings import FastEmbedProvider

    provider = FastEmbedProvider(model_name, threads=threads)
    log(f"  embedding model {provider.info.model}: {provider.dimensions} dimensions")
    with connection.cursor() as cursor:
        cursor.execute("SELECT review_id, text FROM yelp.review WHERE text IS NOT NULL AND btrim(text) <> '' ORDER BY review_id")
        rows = cursor.fetchall()
        dimensions = provider.dimensions
        cursor.execute("SELECT atttypmod FROM pg_attribute WHERE attrelid = 'yelp.review'::regclass AND attname = 'text_embedding'")
        declared = cursor.fetchone()[0]
        if declared is not None and declared > 0 and declared != dimensions:
            raise SystemExit(f"yelp.review.text_embedding is vector({declared}) but the model has {dimensions} dimensions; "
                             "change schema.sql (and the catalog's dimensions) to match the model")
        cursor.execute("CREATE TEMP TABLE review_vectors (review_id text PRIMARY KEY, embedding vector) ON COMMIT DROP")
        started = time.perf_counter()
        done, last_report = 0, started
        log(f"  embedding {len(rows):,} review texts on this machine (a laptop CPU does roughly 20-60 per second)")
        with cursor.copy("COPY review_vectors (review_id, embedding) FROM STDIN") as copy:
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                vectors = provider.embed_texts([text for _, text in batch], batch_size=batch_size)
                for (review_id, _), vector in zip(batch, vectors):
                    copy.write_row((review_id, "[" + ",".join(repr(x) for x in vector) + "]"))
                done += len(batch)
                now = time.perf_counter()
                if now - last_report >= 15 or done == len(rows):
                    rate = done / max(now - started, 1e-9)
                    remaining = (len(rows) - done) / rate if rate else 0
                    log(f"  embedded {done:,}/{len(rows):,} reviews ({rate:,.0f}/s, about {remaining / 60:.0f} min left)")
                    last_report = now
        cursor.execute("UPDATE yelp.review r SET text_embedding = v.embedding FROM review_vectors v WHERE v.review_id = r.review_id")
        log(f"  stored {cursor.rowcount:,} embeddings")
        connection.commit()                      # the vectors are safe before the index is attempted
        cursor.execute("DROP INDEX IF EXISTS yelp.review_embedding_hnsw")
        try:
            cursor.execute("CREATE INDEX review_embedding_hnsw ON yelp.review USING hnsw (text_embedding vector_cosine_ops)")
            connection.commit()
            log("  built the HNSW index (cosine)")
        except psycopg.Error as error:
            connection.rollback()
            log(f"  no HNSW index ({type(error).__name__}); searches will scan (fine at this size)")


# --- graph -------------------------------------------------------------------------------------------


def cypher(cursor, query: str, rows: list[dict], *, columns: str = "v ag_catalog.agtype") -> None:
    """Run one Cypher statement over a batch of rows passed as the ``$rows`` parameter."""

    cursor.execute(
        f"SELECT * FROM ag_catalog.cypher('{GRAPH}', $yodb$ {query} $yodb$, %s) AS ({columns})",
        (json.dumps({"rows": rows}),),
    )


def load_graph(connection, batch_size: int) -> None:
    with connection.cursor() as cursor:
        try:
            cursor.execute("LOAD 'age'")
        except psycopg.Error:
            connection.rollback()
        # AGE's own functions (age_unnest behind UNWIND, ...) are looked up unqualified: the session needs ag_catalog on its path.
        cursor.execute('SET search_path = ag_catalog, "$user", public')
        cursor.execute(f"SELECT * FROM ag_catalog.cypher('{GRAPH}', $$ MATCH (n) DETACH DELETE n $$) AS (v ag_catalog.agtype)")
        for label, sql in (
            ("Business", "SELECT business_id FROM yelp.business ORDER BY 1"),
            ("User", "SELECT user_id FROM yelp.users ORDER BY 1"),
            ("Category", "SELECT category_id FROM yelp.category ORDER BY 1"),
        ):
            cursor.execute(sql)
            ids = [row[0] for row in cursor.fetchall()]
            for start in range(0, len(ids), batch_size):
                cypher(cursor, f"UNWIND $rows AS r CREATE (:{label} {{id: r.id}})", [{"id": i} for i in ids[start:start + batch_size]])
            log(f"  {label}: {len(ids):,} vertices")
        for edge, from_label, to_label, sql in (
            ("FRIENDS_WITH", "User", "User", "SELECT user_id, friend_id FROM yelp.friendship ORDER BY 1, 2"),
            ("REVIEWED", "User", "Business", "SELECT DISTINCT user_id, business_id FROM yelp.review WHERE user_id IS NOT NULL ORDER BY 1, 2"),
            ("IN_CATEGORY", "Business", "Category", "SELECT business_id, category_id FROM yelp.business_category ORDER BY 1, 2"),
        ):
            cursor.execute(sql)
            pairs = cursor.fetchall()
            for start in range(0, len(pairs), batch_size):
                cypher(
                    cursor,
                    f"UNWIND $rows AS r MATCH (a:{from_label}), (b:{to_label}) WHERE a.id = r.a AND b.id = r.b CREATE (a)-[:{edge}]->(b)",
                    [{"a": a, "b": b} for a, b in pairs[start:start + batch_size]],
                )
            log(f"  {edge}: {len(pairs):,} edges")
        connection.commit()
        cursor.execute("ANALYZE")
    connection.commit()


# --- main --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("YELP_DATA", "~/yelp")),
                        help="directory with the Yelp JSON-lines files (default ~/yelp or $YELP_DATA); fetched when missing (see --url/--archive/--kaggle)")
    parser.add_argument("--url", help="download link from the Yelp licence form, used when the files are missing (or YELP_DATASET_URL in .env)")
    parser.add_argument("--archive", type=Path, help="an already downloaded archive to extract when the files are missing")
    parser.add_argument("--kaggle", action="store_true", help="fetch the Kaggle mirror when the files are missing (kaggle CLI + credentials)")
    parser.add_argument("--dsn", default=os.environ.get("YELP_DSN"), help="owner connection string (or env YELP_DSN)")
    # (checked by hand: argparse's ``choices`` with ``nargs="*"`` rejects the empty/default list in some Python versions)
    parser.add_argument("steps", nargs="*", metavar="STEP", help="which of tables, embeddings, graph to run (default: all three)")
    parser.add_argument("--schema", action="store_true", help="apply schema.sql first (drops the yelp schema and the graph)")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="target")
    for name in ("businesses", "reviews", "users", "friend-edges"):
        parser.add_argument(f"--{name}", type=int, help=f"override the preset's {name.replace('-', ' ')}")
    parser.add_argument("--cities", help="comma-separated city names to sample from (default: all)")
    parser.add_argument("--states", help="comma-separated state codes to sample from (default: all)")
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=256, help="rows per embedding call and per graph statement")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("YODB_EMBED_THREADS", "0")) or None,
                        help="CPU threads the embedding runtime may use (default: the runtime's choice; try 1 or 4 if it hangs)")
    parser.add_argument("--dry-run", action="store_true", help="sample and print the counts; touch no database")
    parser.add_argument("--save-sample", type=Path, help="also write the sampled IDs as JSON (for reproducibility notes)")
    args = parser.parse_args(argv)
    args.steps = args.steps or ["tables", "embeddings", "graph"]
    unknown = [step for step in args.steps if step not in ("tables", "embeddings", "graph")]
    if unknown:
        parser.error(f"steps must be among tables, embeddings, graph (not {unknown[0]!r})")

    sizes = {**PRESETS[args.preset]}
    for name in ("businesses", "reviews", "users", "friend_edges"):
        value = getattr(args, name)
        if value is not None:
            sizes[name] = value

    sample = None
    if "tables" in args.steps or args.dry_run:
        from fetch import ensure

        args.data = ensure(args.data, url=args.url, archive=args.archive, kaggle=args.kaggle)   # downloads/extracts when missing
        log(f"sampling with seed {args.seed!r}: {sizes}")
        started = time.perf_counter()
        with (args.data / FILES["business"]).open(encoding="utf-8") as b, \
             (args.data / FILES["review"]).open(encoding="utf-8") as r, \
             (args.data / FILES["user"]).open(encoding="utf-8") as u:
            sample = build_sample(
                b, r, u, seed=args.seed,
                cities={c.strip() for c in args.cities.split(",")} if args.cities else None,
                states={s.strip() for s in args.states.split(",")} if args.states else None,
                **sizes,
            )
        log(f"sampled in {time.perf_counter() - started:.0f}s: {sample.summary()}")
        if args.save_sample:
            args.save_sample.write_text(json.dumps({
                "seed": args.seed, "sizes": sizes, "cities": args.cities, "states": args.states,
                "businesses": sorted(sample.businesses), "reviews": sorted(sample.reviews), "users": sorted(sample.users),
            }))
        if args.dry_run:
            return 0

    if not args.dsn:
        parser.error("--dsn (or YELP_DSN) is required")
    with psycopg.connect(args.dsn) as connection:
        if args.schema:
            log("applying schema.sql")
            connection.execute((HERE / "schema.sql").read_text(encoding="utf-8"))
            connection.commit()
        if "tables" in args.steps:
            log("loading tables")
            load_tables(connection, sample)
        if "embeddings" in args.steps:
            log("embedding reviews")
            load_embeddings(connection, args.embedding_model, args.batch_size, args.threads)
        if "graph" in args.steps:
            log("loading the graph")
            load_graph(connection, args.batch_size)
        with connection.cursor() as cursor:
            cursor.execute("SELECT (SELECT count(*) FROM yelp.business), (SELECT count(*) FROM yelp.review), "
                           "(SELECT count(*) FROM yelp.users), (SELECT count(*) FROM yelp.friendship), "
                           "(SELECT count(*) FROM yelp.review WHERE text_embedding IS NOT NULL)")
            businesses, reviews, users, friendships, embedded = cursor.fetchone()
    log(f"done: {businesses:,} businesses, {reviews:,} reviews ({embedded:,} embedded), {users:,} users, {friendships:,} friendship edges")
    print("\nYoDb reads this database as the read-only role; in your shell:")
    print('  export YODB_CONN_YELP="host=<host> port=<port> dbname=<db> user=yodb_ro password=yodb_ro"')
    print(f"  yodb validate examples/yelp/catalog && yodb query --embedding-model {args.embedding_model} examples/yelp/catalog @examples/yelp/queries/nearest_quiet_reviews.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
