# The Yelp data layer

The Yelp Open Dataset, sampled deterministically into one PostgreSQL with the pgvector and
Apache AGE extensions, and described by a YoDb catalog. It is the data layer of the
"Typed Abstract Query Layer" project: relational tables for the structured facts, a vector
column for review text, and a graph for the relationships that have no column.

| Where | What |
| --- | --- |
| `yelp.business`, `yelp.review`, `yelp.users`, `yelp.category` | the sampled rows (plus `yelp.business_category` and `yelp.friendship`, the relational copies of the graph edges, kept for oracles and baselines) |
| `yelp.review.text_embedding` | `vector(384)`, BAAI/bge-small-en-v1.5, cosine, HNSW index |
| graph `yelp_graph` | vertices `Business`, `User`, `Category` (holding only `id`); edges `FRIENDS_WITH` (both directions), `REVIEWED` (one per user-business pair), `IN_CATEGORY` |
| `catalog/` | four datasets, five relationships (two column joins, three graph edges), nine rules |

The default sample is the proposal's target: 10,000 businesses, 100,000 reviews, up to
50,000 users and 200,000 friendship edges; `--preset fallback` is the preregistered fallback
(25,000 reviews), `--preset smoke` a few hundred rows for trying things out.

## 1. Get the data

Download the Yelp Open Dataset (https://business.yelp.com/data/resources/open-dataset/,
free for academic use) and unpack it; you need `yelp_academic_dataset_business.json`,
`..._review.json` (about 5 GB) and `..._user.json` (about 3 GB) in one directory.

## 2. A PostgreSQL with both extensions

Any PostgreSQL 16 with `vector` and `age` installed works. For a local one:

```bash
docker build -t yodb-age tests/e2e/age            # PostgreSQL 16 + Apache AGE + pgvector
docker run -d --name yelp-pg -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=yelp \
  -p 127.0.0.1:55434:5432 -v yelp-data:/var/lib/postgresql/data yodb-age
```

On the research node, point `--dsn` at its PostgreSQL instead (the connecting role must own the
database: it creates the schema, the graph and the read-only role `yodb_ro`).

## 3. Load

```bash
pip install -e ".[semantic]"                        # fastembed, for the embeddings
export YELP_DSN="host=localhost port=55434 dbname=yelp user=postgres password=postgres"

python examples/yelp/load.py --data ~/yelp --dry-run                 # just the counts (reads the three files, ~10 min)
python examples/yelp/load.py --data ~/yelp --schema                  # schema + tables + embeddings + graph
```

Steps can be run alone (`load.py tables`, `load.py embeddings`, `load.py graph`) and the sample
can be narrowed (`--cities Philadelphia,Tucson`, `--states PA`, `--reviews 25000`). Everything is
a function of `--seed` (default `yodb-yelp-v1`) and the record IDs, so the same files give the same
subset on every machine; `--save-sample sample.json` writes the chosen IDs for the paper's
reproducibility notes.

Embedding 100,000 reviews with bge-small on a laptop CPU takes about 15–25 minutes
(`--batch-size` trades memory for speed); on the GPU node install `onnxruntime-gpu` and fastembed
uses it. Loading the graph takes a few minutes.

## 4. Query

```bash
export YODB_CONN_YELP="host=localhost port=55434 dbname=yelp user=yodb_ro password=yodb_ro"
yodb validate examples/yelp/catalog                 # checks tables, columns, the graph, its labels and edge types
yodb catalog  examples/yelp/catalog

yodb query examples/yelp/catalog @examples/yelp/queries/friends_of_popular_users.json
yodb query examples/yelp/catalog @examples/yelp/queries/coffee_places_by_rating.json
yodb query --embedding-model BAAI/bge-small-en-v1.5 examples/yelp/catalog @examples/yelp/queries/nearest_quiet_reviews.json

PYTHONPATH=src python examples/yelp/smoke.py        # the 20 smoke-test queries: S single-source, I joins, D graph chains, X refusals
yodb mcp --embedding-model BAAI/bge-small-en-v1.5 examples/yelp/catalog      # serve it to an agent
```

The proposal's running example, *highly rated Phoenix restaurants whose reviews emphasize a
quiet ambiance and whose categories are connected to restaurants this user likes*, decomposes
into the pieces the smoke queries exercise: a similarity search over `review.text` joined to its
business (D06), the businesses a user reviewed over the `REVIEWED` edges (D02), and the
categories of those businesses over `IN_CATEGORY` (D04). A query follows one relationship at a
time; the agent (or the planner being built on top) composes the chain.

## Notes

- Ids are Yelp's 22-character strings; the graph vertices hold the same strings, so a vertex
  matches its row by `id`.
- A sampled review whose author was not among the sampled users keeps `user_id` NULL; the
  `REVIEWED` edges exist only for sampled authors.
- Friendship is stored in both directions, so `user_friends_with` is a one-way relationship
  that already lists every friend.
- The newer Yelp releases do not include Phoenix; `--cities` picks any city in the release you have.
