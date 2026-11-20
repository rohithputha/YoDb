# Runbook: from nothing to results

Everything below runs on a laptop (macOS or Linux). You need: **Docker Desktop**, **Python 3.11+**,
**git**, about **15 GB of disk** (the Yelp download is ~4.5 GB compressed, ~9 GB unpacked) and the
**ASU model endpoint key**. Expect one afternoon: most of the time is the Yelp download and
embedding 100k reviews on a CPU.

## 0. The code and its Python environment

```bash
cd ~/Fall26/diml/YoDb
python3 -m venv .venv
source .venv/bin/activate                      # every later command assumes this
pip install -e ".[semantic,mcp]" pytest        # yodb + fastembed (embeddings) + the MCP server
PYTHONPATH=src python -m unittest discover -s tests          # 740 tests, no database needed, ~15 s
```

## 1. Secrets and connections: `.env`

Open `.env` (there is one already; `.env.example` is the template) and paste your key:

```
OPENAI_API_KEY=<your ASU key>
```

Leave the rest as it is for the local Docker setup below (ports 55434).

## 2. PostgreSQL with pgvector and Apache AGE (Docker)

```bash
docker build -t yodb-age tests/e2e/age                                   # ~5 min the first time
docker run -d --name yelp-pg -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=yelp \
  -p 127.0.0.1:55434:5432 -v yelp-data:/var/lib/postgresql/data yodb-age
until docker exec yelp-pg pg_isready -U postgres -d yelp; do sleep 1; done
```

The data persists in the `yelp-data` volume across restarts (`docker start yelp-pg`).

### 2a. Check the engine against the real extensions (5 minutes, recommended once)

```bash
docker run -d --name yodb-age-pg --tmpfs /var/lib/postgresql/data -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=yodb_age -p 127.0.0.1:55433:5432 yodb-age
until docker exec yodb-age-pg pg_isready -U postgres -d yodb_age; do sleep 1; done; sleep 2
docker exec -i yodb-age-pg psql -U postgres -d yodb_age -v ON_ERROR_STOP=1 -q < tests/e2e/age/seed.sql
PYTHONPATH=src python tests/e2e/age/run_age.py          # 30 cases (15 under each join strategy), all must PASS
docker rm -f yodb-age-pg
```

This is the first time the Cypher-in-SQL path, the `agtype` decoding and the parallel join run
against a real AGE. If a `G` case fails, stop here and send the output.

## 3. The Yelp data

1. The Yelp Open Dataset sits behind a licence form: https://business.yelp.com/data/resources/open-dataset/
   Accept it, copy the download link, and let the fetcher download and extract it into `~/yelp`:

```bash
python examples/yelp/fetch.py --url "<the download link>"          # ~4.5 GB; resumes if rerun
```

   (or `--archive` for a file you already downloaded, or `--kaggle` for the Kaggle mirror; or put the
   link in `.env` as `YELP_DATASET_URL` and `load.py` fetches it itself when the files are missing).

2. Load (the owner connection comes from `YELP_DSN` in `.env`):

```bash
python examples/yelp/load.py --dry-run                 # counts only: ~10 min, reads the 3 files
python examples/yelp/load.py --schema                  # schema + tables + embeddings + graph
```

`--schema` drops and recreates everything, so it is safe to rerun. Embedding 100,000 reviews on a
laptop CPU takes 15–25 minutes; `--preset fallback` loads 25,000 reviews instead (the proposal's
fallback), `--preset smoke` a few hundred rows to try the pipeline end to end in a minute.

3. Check it:

```bash
yodb validate examples/yelp/catalog                                  # tables, columns, graph, labels, edge types
PYTHONPATH=src python examples/yelp/smoke.py                         # 20 queries: S single, I joins, D graph chains, X refusals
yodb query --embedding-model BAAI/bge-small-en-v1.5 examples/yelp/catalog @examples/yelp/queries/nearest_quiet_reviews.json
```

## 4. The benchmark questions

```bash
PYTHONPATH=src python benchmark/build.py                            # -> benchmark/questions.jsonl (240 questions)
PYTHONPATH=src python benchmark/harness.py gold                     # the gold plans through YoDb (B3)
PYTHONPATH=src python benchmark/score.py benchmark/runs/gold.jsonl  # expect F1 ~ 1.00 in every class
```

A class below ~1.0 for `gold` means a gold plan or an oracle disagrees with the engine on some
template: look at the failing ids in `benchmark/runs/gold.jsonl` against `questions.jsonl`.

## 5. The model: a first smoke test (needs the key)

```bash
PYTHONPATH=src python benchmark/harness.py planner --split tune --only S01 D01 X01
PYTHONPATH=src python benchmark/harness.py tool-agent --split tune --only S01 D01 X01
PYTHONPATH=src python benchmark/harness.py router --split tune --only S01 D01 X01
```

Each prints one line per question (answers or the error). If the planner's answers look wrong on
the tune split, adjust the prompt wording in `planner/prompt.py` — never using held-out questions.

## 6. The experiments

```bash
PYTHONPATH=src python benchmark/experiments.py all --repeat 5       # E1, E2 (5 timed runs), E3, ablations
cat benchmark/results/results.md                                    # the tables; results.json; resources.json
```

E1/E3/ablations call the model for 200 questions each (budget the endpoint's rate limits: ~200
× 4 systems × ~1–3 calls); E2 calls no model. `experiments.py e2` runs one experiment;
`experiments.py tables` re-makes the tables from the runs on disk. Paste the tables into
`REPORT.md`.

## 7. Give it to an agent (optional)

```bash
yodb mcp --embedding-model BAAI/bge-small-en-v1.5 examples/yelp/catalog
```

## Teardown

```bash
docker rm -f yelp-pg && docker volume rm yelp-data
```
