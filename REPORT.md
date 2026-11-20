# A Typed Abstract Query Layer for Heterogeneous Retrieval in AI Agent Workflows

*CSE 598 — final report draft. The method sections describe what is built; the result sections
are filled from `benchmark/results/results.md` once the experiments have run on the node.*

## Abstract

AI agents answer questions whose evidence is spread over relational tables, vector indexes and
graphs. The usual approach exposes each store as a tool and lets the model select tools, pass
intermediate identifiers and reconcile outputs, with weak guarantees about schema correctness,
dependencies and evidence lineage. We built AQL, a read-only layer over unmodified PostgreSQL
instances (relational tables, pgvector embeddings, Apache AGE graphs) whose contribution is a typed
intermediate representation: a plan is validated against a schema index before anything runs, each
operator is compiled to a native query (parameterized SQL, a pgvector ranking, Cypher inside
`ag_catalog.cypher()`), reads that do not depend on each other run concurrently, and every answer
carries the record IDs it rests on. We evaluate on a deterministic 10k-business / 100k-review sample
of the Yelp Open Dataset with 240 parameterized questions in four classes against a monolithic tool
agent (B1), framework routing (B2) and the oracle plan (B3). *[Results: one sentence per RQ.]*

## 1. Introduction

*[From the proposal's Background and Motivation; the open question; the three RQs and hypotheses.]*

## 2. Design

### 2.1 The catalog as the schema index (I1)

The catalog (`datasets.yaml`, `sources.yaml`, `relations.yaml`, `rules.yaml`) is the typed source
of truth: datasets with typed fields, descriptions and example values; the physical representation
of each field (table, column) and of each text field's embeddings (column, model, dimensions,
metric); relationships as column joins or as graph edges (a `graph` declaration names the AGE graph
and the vertex label standing for each dataset); and rules, named filters that encode business
vocabulary. `yodb validate` checks every claim against the live database — tables, columns, types,
identity keys, the AGE extension, the graph, its vertex labels and edge types — and refuses to
activate a catalog that does not match.

For the planner, the catalog is rendered in one of three representations (RQ3): the complete typed
catalog; descriptions only (names and prose, no types, keys, directions or edge information); or the
*Structured Semantic Schema Index*: every dataset, field, rule and relationship is an entry with a
short text and a BGE-small-en-v1.5 vector, a question retrieves entries by a hybrid of IDF-weighted
lexical overlap and cosine similarity, and the kept set is closed over datasets and relationships
into a typed fragment rendered exactly like the full catalog (`planner/sssi.py`, `planner/schema.py`).

### 2.2 The typed IR and capability-aware compilation (I2)

A query is JSON that names datasets, fields, relationships and rules and nothing physical:
`from`, `select`, `where` (typed predicates, `all`/`any`/`not`, rules, a `similar` vector search), one
`traverse` step over a declared relationship, `order_by`, `page`, `constraints`. A plan is one query or
two, the second taking the IDs the first reached (`{"$ids": …}`). Validation binds every name to the
catalog, checks operator/type compatibility and value types, the placement of extension terms, the
applicability and direction of a relationship, and the limits in force; the error carries a code and
a location (`[field_not_found] at where.all[1]: …`). Compilation is per source capability: PostgreSQL
declares what it can filter, order and limit, that it can rank by vector distance under a filter, and
that it can follow graph edges; the compiler emits parameterized SQL, `ORDER BY embedding <=> $q
LIMIT k` for a ranking, and a fixed Cypher `MATCH` whose only parameter is the list of IDs. The planner
pushes conjuncts to their source, keeps residuals at the coordinator, and chooses the read order with
a cost model over `pg_stats` statistics when they are available.

### 2.3 Dependency-aware execution and provenance (I3)

A join whose two sides each carry a filter is *independent*: both sides are read at the same time and
matched afterwards; otherwise the driving side runs first and the other side is probed with its keys
in batches of 1,000. A relationship stored as edges is an *edge join*: the driver's IDs go to the graph
as the Cypher parameter, the pairs it returns say which rows match, and the other side is read for the
IDs reached. Reads of a record assembly that need no learned IDs run together. Every read is bounded
(rows per read, keys per batch, joined rows, a statement timeout) and read-only (every session is
`default_transaction_read_only`). The executor returns, with the rows, the record IDs that appeared
in every step by dataset — the evidence lineage the provenance metric is computed over — and `explain`
shows the physical plan, which filters each source applied, the join strategy and the reasons.

### 2.4 The natural-language planner

Qwen2.5-Coder-7B-Instruct (temperature 0) receives the query-language guide YoDb gives agents, the
schema in one of the three representations, three examples, and the question, and must return one
JSON object: a plan with the field whose values answer the question and the source kinds used, or a
refusal with a code. Decoding is constrained by the JSON Schema of that object (vLLM `guided_json`).
Every step is bound and planned by YoDb before anything runs; the coded error is returned to the
model verbatim for at most two repairs. A plan that still fails is recorded with that error and no
native query is sent.

## 3. Implementation

*[Sizes: the engine, the Yelp data layer, the benchmark, the planner, the baselines; tests: 7xx unit
tests without a database, end-to-end suites against PostgreSQL + AGE + pgvector in Docker; the
sampler, loader and graph build; what ran on the node.]*

## 4. Evaluation

### 4.1 Data and workload

A deterministic sample of the Yelp Open Dataset (SHA-256 ordering of seed and record IDs): 10,000
businesses, 100,000 reviews, up to 50,000 users and 200,000 friendship edges (`examples/yelp`).
PostgreSQL holds the tables and a `vector(384)` column of BGE-small review embeddings with an HNSW
index; the AGE graph `yelp_graph` holds `Business`, `User` and `Category` vertices (only an `id`) and
`FRIENDS_WITH`, `REVIEWED` and `IN_CATEGORY` edges. 34 templates instantiate 240 questions — 60
single-source, 60 independent multi-source, 80 dependent chains (two or three sources), 40 stress
cases — with gold typed plans, source kinds and executable native-SQL oracles for the answer and
evidence sets; 40 tune, 200 held out (`benchmark/`).

### 4.2 Systems

AQL (the planner over the typed layer), B1 the monolithic tool agent (the same model, raw SQL /
vector / Cypher tools, up to 8 calls), B2 framework routing (sub-question decomposition, one engine
per store, aggregation; also as a LlamaIndex pipeline), B3 the oracle plan executed by the engine.
All systems share the model, the result limits, the embedding model and the database.

### 4.3 Metrics

Answer-set exact match and F1 (stress questions: a refusal or the expected error); source-selection
micro/macro-F1; plan validity; provenance precision/recall/F1 over gold evidence IDs; P50/P95
end-to-end and retrieval latency (medians of five runs after a warm-up); prompt tokens; repairs;
invalid native queries; transferred records; peak host and GPU memory; 95% bootstrap intervals over
1,000 resamples (`benchmark/score.py`).

### 4.4 E1 — accuracy by query class (RQ1)

*[Table from `results.md` § E1. H1: invalid native queries −20% relative with ≤ 2 points of answer F1 lost.
Discuss: where the typed layer refused what B1 ran incorrectly; failure modes per class; the stress class.]*

### 4.5 E2 — DAG against forced-sequential execution (RQ2)

*[Table from `results.md` § E2. H2: P95 retrieval latency −25% on independent multi-source questions.
Same gold plans, `--sequential` turns both the independent join and concurrent assembly reads off.]*

### 4.6 E3 — schema representations (RQ3)

*[Table from `results.md` § E3. H3: retrieved typed fragments use fewer prompt tokens than the full
schema and yield higher plan validity than descriptions alone. Report `schema_tokens` and validity.]*

### 4.7 Ablations and resources

*[No grammar; no repairs; peak memory from `resources.json`.]*

## 5. Error analysis

*[Per-stage failures: decomposition, binding, planning, execution, aggregation; examples.]*

## 6. Threats to validity

The sample is one city mix of one Yelp release; templates share their shapes with the few-shot
examples' *shapes* but not their slots or relationships; the oracle and the gold plans were
cross-checked by executing B3 (expected F1 ≈ 1.0) but remain our own; a 7B model at temperature 0
with grammar-constrained decoding is one point in the model space; the sequential ablation keeps the
same plans, so it measures scheduling, not planning.

## 7. Conclusion

*[What the typed structure bought, where its overhead was not justified (per RQ1's crossover), what
is left: multi-hop paths, aggregation, write paths, more backends.]*

## Reproducibility

`examples/yelp/README.md` (data), `benchmark/README.md` (questions, runs, scoring),
`planner/README.md` (the model), `baselines/README.md`, and `benchmark/experiments.py all` for the
whole protocol; every run is a file under `benchmark/results/runs`, and the tables are regenerated
from those files.
