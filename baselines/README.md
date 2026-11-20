# The baselines

Three systems the typed layer is compared with, all runnable through `benchmark/harness.py`
and scored by `benchmark/score.py` like every other system.

| System | Command | What it is |
| --- | --- | --- |
| **B1** monolithic tool agent | `harness.py tool-agent --endpoint ... --dsn "$YODB_CONN_YELP"` | the same model sees the raw schemas of the three stores (table DDL from `information_schema`, the vector column, the graph's labels and edge types) and calls three tools itself — `sql`, `vector_search`, `cypher` — one per turn, passing ids by hand, up to `--max-turns` (8) calls. No validation before a native query runs: a rejected query comes back as the store's error text and counts as an *invalid native query* |
| **B2** framework routing | `harness.py router --endpoint ... --dsn "$YODB_CONN_YELP"` | the router / sub-question design: the model decomposes the question into sub-questions and routes each to one engine (NL→SQL, vector search, NL→Cypher); each engine turns its sub-question into one native query; the model aggregates the sub-answers into the final id set. `baselines/router.py` also holds `LlamaIndexRouterSystem`, the same pipeline built from LlamaIndex's `RouterQueryEngine` + `SubQuestionQueryEngine` + `NLSQLTableQueryEngine` + `PGVectorStore`, for when those packages are installed |
| **B3** oracle pipeline | `harness.py gold` | each question's gold typed plan, executed by YoDb (the upper bound) |

The baselines read the stores directly as the read-only role (`--dsn`, default `$YODB_CONN_YELP`),
through `baselines/tools.py`: SELECT-only SQL, a vector search that embeds the phrase with the same
model as the data, and read-only Cypher wrapped in `ag_catalog.cypher()`. Reads are capped at 500
rows and the statement timeout applies.

What the scorer reads from them, beyond answers and evidence: `native_queries` and
`native_errors` (H1's "invalid native queries"), `tool_calls`, tokens over all turns, and
`sources` from the tools that succeeded. The typed systems report `native_queries` too (every
read YoDb issued) and never an invalid one: a plan that YoDb refuses is recorded as a rejected
plan and no native query is sent.
