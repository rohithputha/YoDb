# Runbook: from nothing to results

Everything below runs on a laptop (macOS or Linux). You need: **Docker Desktop**, **Python 3.11+**,
**git**, about **15 GB of disk** (the Yelp download is ~4.5 GB compressed, ~9 GB unpacked) and the
**ASU model endpoint key**. Expect one afternoon: most of the time is the Yelp download and
embedding 100k reviews on a CPU.

## The one-command version (laptop or ASU Sol)

`scripts/run_all.sh` does sections 0–6 below in one go and can be rerun after an interruption (every
step is skipped once its result exists; the experiments continue from the predictions already on disk):

```bash
git clone <repo> YoDb && cd YoDb
cp .env.example .env      # put OPENAI_API_KEY and YELP_DATA (the folder with the yelp_academic_dataset_*.json files) in it
scripts/run_all.sh --preset target --workers 8
```

It finds a container runtime itself: Docker or Podman on a laptop, Apptainer on an HPC node (no root
needed; the server runs as you, its data under `$SCRATCH/yodb-pg`). `scripts/pg.sh up|status|psql|logs|down|destroy`
manages that server afterwards. `--workers 8` answers eight questions at a time in the model-bound runs,
which is what makes the sweep take about an hour instead of most of a day against the shared endpoint.

### On ASU Sol

```bash
# 1. a compute session: 16 cores and 64 GB for the embeddings; 12 hours covers the target preset comfortably
interactive -p public -q public -c 16 --mem=64G -t 0-12:00      # -p htc also works but is capped at 4 hours

# 2. Python >= 3.11 (Sol's default python is older): one of
module load mamba/latest && mamba create -n yodb python=3.12 -y && source activate yodb

# 3. the dataset: copy it to scratch once (from your laptop: scp -r ~/Downloads/YelpJSON/yelp_dataset sol.asu.edu:/scratch/$USER/yelp_dataset)
#    and set YELP_DATA=/scratch/<asurite>/yelp_dataset in .env

# 4. the image: Sol allows `apptainer build --fakeroot`, which run_all.sh tries first; if it is refused, build
#    the image on your laptop and bring the archive (docker save yodb-age -o yodb-age.tar; scp it to
#    /scratch/<asurite>/yodb-pg/yodb-age.tar) and set YODB_PG_IMAGE_TAR to that path in .env

# 5.
scripts/run_all.sh --preset target --workers 8
```

Keep the session open while it runs (`tmux` on the login node before `interactive` survives a dropped
connection). The server lives only inside the job: when the job ends, `scripts/pg.sh up` in the next
one starts it again on the same data directory, and `run_all.sh` continues.

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
