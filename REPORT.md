# A Typed Abstract Query Layer for Heterogeneous Retrieval in AI Agent Workflows

*CSE 598, Group 9 — final report. All numbers are from `benchmark/results/results.md`, produced by
`scripts/run_all.sh --preset target --workers 32` on an ASU Sol compute node (16 cores, Apptainer) with
`qwen3-235b-a22b-instruct-2507` served by the ASU Research Computing vLLM endpoint, on 2026-11-20.*

## Abstract

AI agents answer questions whose evidence is spread over relational tables, vector indexes and
graphs. The usual approach exposes each store as a tool and lets the model select tools, pass
intermediate identifiers and reconcile outputs, with weak guarantees about schema correctness,
dependencies and evidence lineage. We built AQL (implemented as *YoDb*), a read-only layer over
unmodified PostgreSQL instances (relational tables, pgvector embeddings, Apache AGE graphs) whose
contribution is a typed intermediate representation: a plan is validated against a schema index before
anything runs, each operator is compiled to a native query (parameterized SQL, a pgvector ranking,
Cypher inside `ag_catalog.cypher()`), reads that do not depend on each other can run concurrently, and
every answer carries the record IDs it rests on. We evaluate on a deterministic 10k-business /
100k-review sample of the Yelp Open Dataset with 240 parameterized questions in four classes against a
monolithic tool agent (B1), framework routing (B2) and the oracle plan (B3), all driven by the same
235B-parameter model. On the 206 held-out questions the typed planner reaches answer-set F1 0.925
(95% CI 0.89–0.96) against 0.852 for the tool agent and 0.808 for the router, while sending no invalid
native query where the tool agent has 26% of its SQL and Cypher rejected by the stores (RQ1). The
repair loop driven by the layer's coded errors is what makes this work: without it F1 falls to 0.748,
and grammar-constrained decoding, the usual alternative, collapses to 0.291. The three schema
representations plan equally validly for this model, and the typed fragment uses 5% fewer prompt
tokens than the full catalog (RQ3). Dependency-aware execution does not reduce latency on a single
local PostgreSQL: once a cost rule stops it from reading whole join sides, it ties forced-sequential
execution within noise, and the one case where it still reads a side whole is slower (RQ2).

## 1. Introduction

Agent frameworks let a language model query data by calling tools. When the data lives in several
kinds of store, the model writes each store's native language itself, carries identifiers from one
call to the next by hand, and reconciles what comes back; nothing checks a query against the schema
before it runs, dependencies between calls are implicit in the conversation, and the provenance of an
answer is whatever the model chose to mention. This project asks whether putting a *typed* layer between
the model and the stores changes those outcomes, and at what cost. Three research questions structure
the evaluation. RQ1: does a typed intermediate representation, validated before execution, reduce
invalid native queries and improve answer quality compared with tool calling (H1: −20% invalid queries
at ≤ 2 points of answer F1 lost)? RQ2: does executing independent reads concurrently reduce retrieval
latency on multi-source questions (H2: −25% P95)? RQ3: which representation of the schema lets the
model plan best, and how much prompt does it cost (H3: a retrieved typed fragment is more valid than
descriptions alone and smaller than the full schema)?

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

A join whose two sides each carry a filter *may* be read as two independent reads at the same time
and matched afterwards; otherwise the driving side runs first and the other side is probed with its
keys in batches of 1,000. The choice is a cost decision (§ 4.5 explains why): the independent read is
taken only when statistics expect the other side to be at most four times the driver and at most
5,000 rows. A relationship stored as edges is an *edge join*: the driver's IDs go to the graph as the
Cypher parameter, the pairs it returns say which rows match, and the other side is read for the IDs
reached. Reads of a record assembly that need no learned IDs run together. Every read is bounded
(rows per read, keys per batch, joined rows, a statement timeout) and read-only (every session is
`default_transaction_read_only`). The executor returns, with the rows, the record IDs that appeared
in every step by dataset — the evidence lineage the provenance metric is computed over — and `explain`
shows the physical plan, which filters each source applied, the join strategy and the reasons.

### 2.4 The natural-language planner

The model (temperature 0) receives the query-language guide YoDb gives agents, the schema in one of
the three representations, three examples, and the question, and must return one JSON object: a plan
with the field whose values answer the question and the source kinds used, or a refusal with a code.
Every step is bound and planned by YoDb before anything runs; the coded error is returned to the
model verbatim for at most two repairs. A plan that still fails is recorded with that error and no
native query is sent. The JSON Schema of the output can also constrain decoding (vLLM structured
outputs); this is the *grammar* ablation, not the default, for the reason § 4.7 shows.

## 3. Implementation

The engine (`src/yodb`, ~13.6k lines) is the catalog, the IR with its parser, binder and validator, the
physical planner with pushdown and the cost model, the PostgreSQL compiler and executor (SQL,
pgvector, Cypher-in-SQL with `agtype` decoding), the join and edge-join operators, the semantic and
similarity extensions, the embedding provider (fastembed/ONNX, CPU or GPU), the CLI and an MCP server.
The Yelp data layer (`examples/yelp`) is the schema (tables, a `vector(384)` column with an HNSW
index, the AGE graph), a deterministic sampler, a resumable loader and the catalog. The benchmark
(`benchmark/`) is 34 templates, the builder with executable oracles, the harness (parallel workers,
resumable runs), the scorer with bootstrap intervals and the experiment runner. The planner
(`planner/`) and the two baselines (`baselines/`) share one OpenAI-compatible generator (streaming,
retried) so every system sees the same model under the same settings (4.4k lines for the data layer,
benchmark, planner and baselines together). 750 unit tests (15.6k lines) run without a
database; an end-to-end suite of 30 cases runs against PostgreSQL 16 + AGE 1.5 + pgvector in a
container (Docker locally, Apptainer on the node) and passed on the node before the data was loaded.
`scripts/run_all.sh` takes a clone to the result tables in one command.

## 4. Evaluation

### 4.1 Data and workload

A deterministic sample of the Yelp Open Dataset (SHA-256 ordering of seed and record IDs): 10,000
businesses, 100,000 reviews, 50,000 users, 200,000 friendship edges, 1,049 categories, 44,601
business–category pairs. PostgreSQL holds the tables and BGE-small embeddings of the review texts
(embedded on the node in 90 minutes on 16 cores); the AGE graph `yelp_graph` holds `Business`, `User`
and `Category` vertices (only an `id`) and `FRIENDS_WITH`, `REVIEWED` and `IN_CATEGORY` edges. 34
templates instantiate 240 questions — 60 single-source, 60 independent multi-source, 80 dependent
chains, 40 stress cases — with gold typed plans, source kinds and executable native-SQL oracles for
the answer and evidence sets; 34 tune, 206 held out. Every question's full result fits the 500-row page
(the builder prefers such instances), so the answer sets are complete, and the oracle plans executed
by the engine reproduce every gold answer exactly (B3, F1 1.000 on all 240).

### 4.2 Systems

AQL (the planner over the typed layer), B1 the monolithic tool agent (the same model, raw SQL /
vector / Cypher tools over the same database, up to 8 calls, the store's own error text returned on a
rejected query), B2 framework routing (sub-question decomposition, one engine per store, an aggregation
step), B3 the oracle plan executed by the engine. All systems share the model, the result limits, the
embedding model, the database and the answer contract (ids of one dataset; the baselines may name a
result column by reference instead of copying 500 ids).

### 4.3 Metrics

Answer-set exact match and F1 (stress questions: a refusal or the expected error counts as correct);
source-selection F1; plan validity; provenance F1 over gold evidence IDs; latency; prompt tokens;
repairs; invalid native queries; records transferred; 95% bootstrap intervals over 1,000 resamples.
The model-bound runs (E1, E3, ablations) answered 32 questions at a time against the shared endpoint,
so their latencies include queueing and are not reported as a result; E2 ran sequentially, five timed
repetitions per question after a warm-up, medians per question.

### 4.4 E1 — accuracy by query class (RQ1)

| system | class | n | EM | F1 (95% CI) | valid | src F1 | prov F1 | tokens | repairs | invalid native |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AQL planner | single | 50 | 0.980 | 0.996 (0.99–1.00) | 1.000 | 1.000 | 0.996 | 3110 | 0.00 | 0.00 |
| | independent | 51 | 0.784 | 0.840 (0.74–0.93) | 0.863 | 0.863 | 0.837 | 5851 | 0.80 | 0.00 |
| | dependent | 72 | 0.875 | 0.929 (0.87–0.98) | 0.972 | 0.958 | 0.912 | 3967 | 0.25 | 0.00 |
| | stress | 33 | 0.939 | 0.939 (0.85–1.00) | 0.939 | – | – | 3105 | 0.00 | 0.00 |
| | **all** | 206 | 0.888 | **0.925 (0.89–0.96)** | 0.947 | 0.942 | 0.914 | 4087 | 0.29 | **0.00** |
| B1 tool agent | single | 50 | 0.940 | 0.956 (0.90–1.00) | 0.960 | 0.960 | 0.956 | 6821 | 0.12 | 0.32 |
| | independent | 51 | 0.941 | 0.980 (0.94–1.00) | 0.980 | 1.000 | 0.930 | 12234 | 0.04 | 0.06 |
| | dependent | 72 | 0.694 | 0.746 (0.63–0.84) | 0.750 | 0.727 | 0.648 | 15790 | 0.12 | 0.90 |
| | stress | 33 | 0.727 | 0.727 (0.55–0.88) | 0.727 | – | – | 15543 | 0.58 | 0.24 |
| | **all** | 206 | 0.820 | **0.852 (0.80–0.90)** | 0.854 | 0.875 | 0.820 | 12693 | 0.17 | **0.45** |
| B2 router | single | 50 | 0.880 | 0.916 (0.84–0.98) | 0.940 | 0.940 | 0.916 | 10306 | 0.00 | 0.06 |
| | independent | 51 | 0.804 | 0.804 (0.69–0.90) | 0.961 | 1.000 | 0.777 | 14019 | 0.00 | 0.04 |
| | dependent | 72 | 0.778 | 0.786 (0.69–0.88) | 0.972 | 0.722 | 0.679 | 5017 | 0.00 | 0.04 |
| | stress | 33 | 0.697 | 0.697 (0.52–0.85) | 0.697 | – | – | 2595 | 0.00 | 0.00 |
| | **all** | 206 | 0.796 | **0.808 (0.75–0.86)** | 0.917 | 0.867 | 0.776 | 8141 | 0.00 | **0.04** |
| B3 oracle plan | all | 206 | 1.000 | 1.000 | 1.000 | 1.000 | 0.993 | 0 | 0.00 | 0.00 |

*tokens* = mean prompt tokens per question over all model calls; *invalid native* = native statements the
store rejected, per question.

**H1 is met and exceeded.** The planner sends no invalid native query by construction — every statement
it issues was bound and planned against the schema index first — where the tool agent had 92 of its
356 SQL, Cypher and vector statements (25.8%) rejected by PostgreSQL or AGE, and the router 8 of 204
(3.9%; one native query per sub-question). H1 allowed this to cost two points of answer F1; instead
the planner gains 7.3 points over the tool agent (0.925 vs 0.852, the intervals barely overlap) and
11.7 over the router, with better provenance (0.914 vs 0.820 / 0.776) and source selection (0.942 vs
0.875 / 0.867), on a third of the prompt tokens (4.1k vs 12.7k / 8.1k per question).

**Where each system wins is not where the others do.** By template (held-out questions):

| | planner | tool agent | router | | planner | tool agent | router |
|---|---:|---:|---:|---|---:|---:|---:|
| D01 friends of a user | 1.00 | 0.62 | 1.00 | I01 positive reviews, city, month | 1.00 | 1.00 | 0.27 |
| D06 open businesses in a category | 0.99 | 0.29 | 0.14 | I03 five-star reviews by active reviewers | 0.60 | 0.80 | 1.00 |
| D08 businesses sharing a category | 0.67 | 0.88 | 1.00 | I04 negative reviews by Elite users | 0.76 | 1.00 | 1.00 |
| D09 businesses reviewed by friends | 1.00 | 0.38 | 0.95 | I07 useful reviews by users with fans | 0.50 | 1.00 | 1.00 |
| D10 two-hop chain | 1.00 | 1.00 | 0.00 | S07 reviews similar to a phrase | 1.00 | 0.33 | 0.00 |
| D04 reviewers of a named business | 0.71 | 0.57 | 0.86 | X04 "how many" (expects a refusal) | 1.00 | 0.20 | 0.00 |

The tool agent's losses are the graph questions: on D01, D06 and D09 it wrote Cypher that AGE rejected
(its 92 rejected statements are Cypher AGE could not run and SQL against guessed column names), and
on 21 non-stress questions it gave up and refused. The router's losses are the two-hop chains
(D10, D06) it decomposed into sub-questions whose results it could not recombine, and the similarity
question (S07), whose vector engine it did not route to. Both baselines answered the count question
(X04) with a list instead of refusing. The planner is at or near 1.00 on all of these. Its own losses
are on the independent class and are analysed in § 5: in the main they are plans the model proposed
that the IR cannot express (two traversal steps), not wrong plans.

### 4.5 E2 — DAG against forced-sequential execution (RQ2)

Same gold plans, same database, `--sequential` turning the independent join read and the concurrent
assembly reads off; five timed runs after a warm-up; per-question medians.

| executor | class | p50 ms | p95 ms |
|---|---|---:|---:|
| dependency-aware (DAG) | single | 113 | 644 |
| | independent | 21 | 665 |
| | dependent | 68 | 510 |
| | all | 37 | 574 |
| forced-sequential | single | 84 | 483 |
| | independent | 20 | 441 |
| | dependent | 67 | 525 |
| | all | 33 | 511 |

**H2 is not supported on this deployment, and the first run of E2 taught us why.** In that run the
executor's rule was "read both sides at the same time whenever statistics say both fit their guards";
it did so for every independent question and was 10–100× *slower* than probing (I05: 1,773 ms against
11 ms). Reading the other side whole moved thousands of filtered rows to the coordinator — and, worse,
made the source scan the table, where a probe by 20 keys uses the index. The independent read saves one
round trip, under a millisecond on a local PostgreSQL, and risks a sequential scan. We replaced the
rule with the cost condition of § 2.3 (other side ≤ 4× the driver and ≤ 5,000 rows) and reran E2: the
two executors now tie on 16 of the 17 join templates (medians within a few milliseconds), and on the
one template where the estimate still called the sides balanced (I07) the whole read is 556 ms against
51 ms for the probe, which is the independent-class P95 gap in the table. The single-source P50
difference (113 vs 84 ms) is a cache effect: the DAG run went first after the data was rebuilt, and
the templates that scan reviews (S03, S04) warmed the cache for the sequential run. The conclusion we
draw is narrower than the hypothesis: dependency-aware scheduling pays when sources are round-trip
bound (remote stores, several databases); against one local PostgreSQL the right plan is almost always
to push the driver's keys to the source, and a cost model that knows the source's access path, not just
its row count, is what the executor is missing.

### 4.6 E3 — schema representations (RQ3)

| representation | n | EM | F1 (95% CI) | valid | src F1 | prov F1 | prompt tokens | repairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full typed catalog | 206 | 0.888 | 0.925 (0.89–0.96) | 0.947 | 0.942 | 0.914 | 4087 | 0.29 |
| descriptions only | 206 | 0.888 | 0.901 (0.86–0.94) | 0.947 | 0.963 | 0.912 | 3591 | 0.27 |
| retrieved typed fragment (SSSI) | 206 | 0.850 | 0.887 (0.85–0.93) | 0.942 | 0.960 | 0.905 | 3899 | 0.35 |

**H3 holds for size and not for validity.** The retrieved fragment is smaller than the full catalog
(−5% prompt tokens; the difference is modest because tokens count every repair round, and the fragment
needed more of them) but not smaller than descriptions, and plan validity is the same for all three
(0.942–0.947): with a 235B model and the repair loop, the representation does not decide whether a
plan validates. It decides what the model *attempts*. By class, descriptions beat the full catalog on
the independent questions (0.974 vs 0.840) and lose on the dependent ones (0.838 vs 0.929): shown the
relationships with their directions and the `optional` flag, the model writes more ambitious plans —
two traversal steps where one would do — and those are the plans the IR rejects (§ 5); shown prose
only, it writes the one-step plan and it works, but it fails more often to pick the right relationship
for a chain. The retrieved fragment sits between the two on both classes and does worst on the
dependent chains (0.805), where its top-12 entries sometimes omit the relationship the chain needs and
the model has to be repaired into it.

### 4.7 Ablations and resources

| configuration | F1 (95% CI) | valid | independent F1 | repairs / q | prompt tokens |
|---|---:|---:|---:|---:|---:|
| planner (full, repairs, no grammar) | 0.925 (0.89–0.96) | 0.947 | 0.840 | 0.29 | 4087 |
| no repairs | 0.748 (0.68–0.80) | 0.757 | 0.328 | 0.00 | 3115 |
| grammar-constrained decoding | 0.291 (0.24–0.35) | 0.447 | 0.004 | 1.15 | 7204 |

The repair loop is the mechanism behind RQ1. Without it the first attempt fails validation on a quarter
of the questions (28 `relationship_not_applicable`, 16 `query_feature_not_supported`), and on the
independent class on two thirds; with it, the coded, located error brings the second attempt home and
F1 rises by 17.7 points overall and 51 on that class. Constraining decoding with the output's JSON
Schema — the alternative to validation that structured-output APIs offer — is destructive on this
vLLM: the model emits structurally minimal plans (no `traverse`, no `page`, values of the wrong type:
65 `query_value_type_invalid`), validity falls to 0.447 and F1 to 0.291. "Validate and repair" beats
"constrain" by a wide margin here, and the two are not interchangeable.

Resources: every run fits in 330–620 MB of host memory on the node (the embedding model and the
schema index); no GPU was used for the evaluation (the model is remote). Wall time for the sweep with
32 workers: tool agent 52 min, router 7 min, planner runs 3–6 min each, E2 5 min.

## 5. Error analysis

The planner's 12 non-stress misses on the held-out split fall into four kinds.

*A shape the IR cannot express (7).* On I03, I04 and I07 the model's first plan traverses twice from
`review` — `review_of_business` (marked optional) and `user_wrote_review` — because the question
mentions both the business and the author; the layer answers "V0.1 supports one traversal step per
query", and after two repairs the model either stops producing JSON (5 `output_invalid`) or gives up
with a refusal (I07-005, D04-004). The tool agent, writing SQL, joins three tables and is at 1.00 on
I07. This is the IR's limit, not the model's, and the obvious fix (a second hop, or a residual join
at the coordinator) is the first item of future work.

*A wrong second step (3).* On D08 ("businesses in city A sharing a category with business B in city
B") the plan has the right shape — find B's categories, then businesses in A — but the second step
filters businesses by category *ids* through the wrong field, so the chain returns a subset (0.46) or
nothing. Validation cannot catch this: every name binds, every type matches; only the semantics are off.

*A wrong relationship (1).* D04-008 applies `user.id in <business ids>`; the layer rejects it as
`relationship_not_applicable` on every attempt.

*A dropped condition (stress).* On two of the five "price range" questions (X01) the planner produced a
valid plan that ignored the condition the catalog cannot express (once it searched the `categories`
text for "$$"), and answered confidently; it refused the other three. The tool agent and the router
answered all five. Validation checks what a plan says, not what it leaves out; a refusal needs the
model to notice the gap, and the typed catalog helps it do so more often (3/5 vs 0/5) but not always.

The baselines' errors are of a different nature. The tool agent's 92 invalid statements are AGE
rejecting Cypher it cannot parse (property-graph syntax that AGE 1.5 does not support, wrong label
names, `::text::int` casts), SQL against guessed column names, and a vector search called with the
wrong arguments; after a few rejections it refuses ("other") on 21 real questions. The router's
decomposition fails on the two-hop chains: it asks a sub-question per hop, but the second engine sees
the first's answer only as text, and the aggregation step then returns the wrong dataset's ids or none.
Both answered the count question with a list, where the planner refused because the IR has no
aggregate; all three refused the impossible date (the planner because the binder rejects the value).

## 6. Threats to validity

The sample is one mix of one Yelp release, and the workload is 34 templates whose shapes the few-shot
examples share (their slots and relationships differ); a paraphrase set written by people would make
the accuracy claims stronger. One model at temperature 0 is one point in the model space, and a strong
one; the results of RQ3 in particular (validity independent of the representation) may not hold for a
7B model. B2 is our re-implementation of the router/sub-question design, not the framework itself
(`LlamaIndexRouterSystem` exists in the code and was not run). The oracle and the gold plans were
cross-checked by executing B3 (F1 1.000) but remain our own. The model-bound runs answered 32 questions
at a time, so their latencies carry queueing and are not reported; E2 is sequential and unaffected. E2's
first run measured a planning rule we then changed; the reported numbers are from the rerun with the
rule of § 2.3, and the single-source P50 gap in it is a cache-order effect. Two templates (X01, D02) were
reworded after the tune-split runs because they were ambiguous or named a synonym of an existing field;
the held-out results reported here are from the final wording.

## 7. Conclusion

A typed, validated intermediate representation between an agent and heterogeneous stores bought what
the proposal hoped and more on RQ1: no invalid native queries instead of one rejected statement in
four, higher answer F1 rather than lower, better provenance and source selection, on a third of the
prompt. The mechanism is the coded error fed back to the model, not the schema shown to it: the repair
loop is worth 18 points, grammar-constrained decoding takes 63 away, and the three schema
representations plan equally validly. The overhead of the typed structure showed where the proposal
did not expect it: not in accuracy but in expressiveness — the planner's losses are plans the model
wanted to write and the IR could not take — and the concurrency of the executor, which is where the
proposal expected a gain, has nothing to buy on a single local PostgreSQL once it stops reading join
sides whole. What is left: a second hop and a coordinator-side residual join; aggregates; a cost model
that sees the source's access paths; a second, smaller model and a paraphrased question set; the
LlamaIndex pipeline as B2; and a second domain.

## Reproducibility

`scripts/run_all.sh --preset target --workers 32` goes from a clone to `benchmark/results/results.md`
(`RUNBOOK.md` has the laptop and ASU Sol instructions); `examples/yelp/README.md` (data),
`benchmark/README.md` (questions, runs, scoring), `planner/README.md` (the model),
`baselines/README.md`. Every run is a file under `benchmark/results/runs`, and the tables are
regenerated from those files with `benchmark/experiments.py tables`.
