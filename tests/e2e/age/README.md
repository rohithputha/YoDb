# Graph traversal and similarity search against a real PostgreSQL (Apache AGE + pgvector)

Builds a throwaway PostgreSQL 16 image with the Apache AGE and pgvector extensions, seeds a
small shop (customers, tickets, an AGE graph holding the customer-ticket and referral edges,
toy ticket embeddings), opens the catalog in `catalog/` through the live inspector (which
checks the graph, its labels and edge types), and runs traversals and similarity searches.
Every answer is compared with native SQL over the seed's edge table (`shop.links`) and
`pgvector` ordering; YoDb never sees those joins.

Nothing is persisted: data lives on `tmpfs`.

## Run

```bash
# 1. build the image (once) and start it on port 55433, loopback only
docker build -t yodb-age tests/e2e/age
docker run -d --name yodb-age-pg --tmpfs /var/lib/postgresql/data \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=yodb_age \
  -p 127.0.0.1:55433:5432 yodb-age
until docker exec yodb-age-pg pg_isready -U postgres -d yodb_age; do sleep 1; done; sleep 2

# 2. seed tables, embeddings, the graph and the read-only role yodb_ro (the only role YoDb uses)
docker exec -i yodb-age-pg psql -U postgres -d yodb_age -v ON_ERROR_STOP=1 -q < tests/e2e/age/seed.sql

# 3. run (or pass name substrings: `... run_age.py G0 V05`; `-q` hides rows)
PYTHONPATH=src .venv/bin/python tests/e2e/age/run_age.py

# 4. tear down (removes data too)
docker rm -f yodb-age-pg
```

Override the server with `YODB_E2E_CONNINFO="host=... dbname=... user=yodb_ro password=..."`.
Exit code is non-zero when any case fails.

## What the seed shows

- `yodb_ro` cannot `LOAD 'age'` itself (only a superuser can), so the seed sets
  `session_preload_libraries = 'age'` and `search_path = ag_catalog, "$user", public`
  on the role. A production role needs the same two settings (or the server needs
  `shared_preload_libraries = 'age'`), plus `SELECT` on `ag_catalog` and on the graph's
  schema. YoDb still tries `LOAD 'age'` first and ignores a refusal.
- The graph holds only identities (`{id: ...}` on each vertex) and edges. The rows stay
  in `shop.customers` and `shop.tickets`. One link (`c07 -> t10`) exists only in the graph.
- The ticket embeddings are a toy 5-dimensional keyword count computed in SQL
  (`shop.toy_embed`); `run_age.py`'s `ToyEmbedder` computes the same vector for a query
  text, so the catalog declares `model: toy-v1, dimensions: 5`.

## Cases

`G` graph traversals (forward, reverse, left, self, filters on both sides, driving side),
`V` similarity searches (nearest first, under a condition, with a caller's order, the default
`top`, on the root of a traversal, and two refusals).
