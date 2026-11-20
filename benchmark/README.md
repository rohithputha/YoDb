# The benchmark

> Connection strings, the model endpoint and the API key can live in a `.env` at the repository root
> (copy `.env.example`); every script here reads it, and a variable already in the environment wins.

240 parameterized questions over the [Yelp data layer](../examples/yelp), in the four classes of
the proposal's workload, with executable reference answers; a harness that runs any *system* over
them; and a scorer for the proposal's metrics. Gold answers come from native SQL over the Yelp
tables (the oracle), which systems never see.

| Class | Questions | What they need |
| --- | --- | --- |
| `single` | 60 | one dataset: filters, rules, ordering, a similarity search over review text |
| `independent` | 60 | two datasets joined over a column, each side filtered on its own (both reads can run at once) |
| `dependent` | 80 | chains: a graph traversal whose other side is read for the IDs reached, or a second step over the first step's IDs (two or three sources) |
| `stress` | 40 | must be refused or pushed back on: unknown field or dataset, wrong value type, a relationship that does not apply, counting, a write, an ambiguous ask |

40 questions form the `tune` split (thresholds, prompts), 200 are `held_out`.

Every template (`templates.py`) carries the natural-language question with slots, the gold
typed plan (one or two YoDb queries; a second step reads the IDs of the first through
`{"$ids": {"step": 0, "field": "business.id"}}`), the kinds of source the plan touches
(`relational`, `vector`, `graph`), the oracle SQL and, for stress cases, the error a system
must raise. Slots are drawn from pools read from the database (the cities with the most
businesses, the most common categories, prolific users, popular businesses) and fixed lists
(thresholds, months, phrases), deterministically from a seed.

## Build the questions

```bash
export YELP_DSN="host=localhost port=55434 dbname=yelp user=postgres password=postgres"   # any role that reads yelp.*
PYTHONPATH=src python benchmark/build.py --out benchmark/questions.jsonl
```

This prints the class counts, how many questions have an empty gold answer (a few are
fine — an empty set is a valid answer — but many mean the slot pools do not fit the sample;
`--seed` reshuffles) and how many have a result larger than the 500-row page. The builder draws
three candidates per instance and prefers the ones whose whole result fits the page and is not
empty, because a cut-off result makes the answer set depend on which 500 rows the page kept —
a system that computes the full set (a `SELECT DISTINCT`, say) would then be scored against a
prefix of it. The questions kept anyway carry `complete: false`; the proposal's sample sizes
(100k reviews, 50k users) make them rare. The templates with a similarity search embed their phrase with the same
model as the data (`--embedding-model`, `--no-embeddings` leaves them out).

## Run a system

```bash
export YODB_CONN_YELP="host=localhost port=55434 dbname=yelp user=yodb_ro password=yodb_ro"
PYTHONPATH=src python benchmark/harness.py gold --out benchmark/runs/gold.jsonl
PYTHONPATH=src python benchmark/harness.py gold --split held_out --repeat 5 --warm-up     # the latency protocol
```

`gold` runs each question's gold plan through YoDb: the oracle-plan pipeline (**B3**). Its answer
F1 checks the engine and the gold plans against the oracle (expect ~1.0; a template below that is
a bug in one of them), and its latency is the retrieval floor every planner adds to.

A new system is a class with `name` and `answer(question) -> Prediction` (`harness.py`); the
natural-language planner, the monolithic tool agent (**B1**) and the LlamaIndex router (**B2**)
plug in there. `run_plan(db, plan, ...)` executes a typed plan (one or two steps) and fills in
the answer set, evidence IDs, source kinds, validity, latency and transferred records, so a
planner only has to produce the plan and report its tokens and repairs.

## Score

```bash
PYTHONPATH=src python benchmark/score.py benchmark/runs/gold.jsonl benchmark/runs/planner.jsonl --split held_out --json report.json
```

Per class and overall: answer-set exact match and F1 (stress: refused / the expected error),
source-selection micro- and macro-F1, plan validity, provenance precision/recall/F1 over the
gold evidence IDs, P50/P95 of end-to-end and retrieval latency (medians over repeated runs),
mean tokens, repairs and transferred records, with 95% bootstrap intervals (1,000 resamples)
on the quality means. The experiments map onto the classes: **E1** (RQ1) compares systems per
class, **E2** (RQ2) compares the DAG and forced-sequential executors on the `independent`
questions, **E3** (RQ3) compares schema representations on the planner's validity and tokens.

## The whole protocol

```bash
PYTHONPATH=src python benchmark/experiments.py all --endpoint http://localhost:8000/v1 --repeat 5
```

runs E1 (planner, B1, B2, B3 on the held-out split), E2 (the gold plans with the DAG executor and
with `--sequential`, five timed runs after a warm-up), E3 (the three schema representations) and the
ablations (no grammar, no repairs), records peak memory per run, and writes `results/results.md`
and `results/results.json`. `experiments.py e2` runs one experiment; `experiments.py tables`
re-makes the tables from the runs on disk.

## Notes

- Every gold query asks for the largest page (500) with a deterministic order, and the oracle
  takes the same `LIMIT` in the same order — with `COLLATE "C"` on text columns, because YoDb
  orders ties by code point while a database collation such as `en_US` does not — so the two
  agree even when a question has more than 500 answers (the builder avoids such questions when
  it can; see above).
- The answer set is the distinct values of one row field (`id`, or `alias.id` for the traversed
  side); evidence is every record ID that appeared in any step's rows, by dataset.
- A second step carries at most 1,000 IDs (YoDb's `in` limit); the gold plans stay well under it.
