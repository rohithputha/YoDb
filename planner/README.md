# The natural-language planner

A question becomes a typed YoDb plan: the model writes the plan in the query language, YoDb
validates it (binding, types, relationships, placement) before anything runs, the coded error
goes back to the model for at most two repairs, and the accepted plan is executed through the
same engine as every other system. It is a benchmark system (`benchmark/harness.py planner`).

```
question ──► prompt (query language + schema + 3 examples) ──► model ──► JSON ──► YoDb explain ──┬─► run ──► answer, evidence
                                                                 ▲                                │
                                                                 └──── error text (≤ 2 repairs) ◄─┘
```

## Pieces

| File | What |
| --- | --- |
| `schema.py` | the three schema representations (RQ3): `full` (typed, complete), `descriptions` (names and descriptions only), `retrieved` (SSSI fragments) |
| `sssi.py` | the Structured Semantic Schema Index: every dataset, field, rule and relationship as a typed entry with a BGE-small vector; hybrid lexical + vector retrieval, closed over datasets and relationships |
| `grammar.py` | the JSON Schema of the output (a plan of one or two queries + `answer_field` + `sources`, or a refusal), for grammar-constrained decoding |
| `prompt.py` | the system prompt (YoDb's own agent guide for the query language, the schema, the output contract), three few-shot examples, the repair message |
| `llm.py` | generators: an OpenAI-compatible endpoint (vLLM with `guided_json`, Ollama, an API), in-process transformers (4-bit, `outlines` grammar when installed), a scripted one for tests |
| `planner.py` | the loop, and the mapping to a benchmark `Prediction` (tokens summed over attempts, `repairs`, the last error when the budget is spent, `refused` for refusals) |

## Run

Serve Qwen2.5-Coder-7B-Instruct with vLLM on the node (grammar-constrained JSON comes for free):

```bash
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --quantization bitsandbytes --max-model-len 8192 --port 8000
```

then, with the Yelp data layer loaded and `benchmark/questions.jsonl` built:

```bash
export YODB_CONN_YELP="host=... dbname=yelp user=yodb_ro password=yodb_ro"
PYTHONPATH=src python benchmark/harness.py planner --endpoint http://localhost:8000/v1 --schema full        --split held_out
PYTHONPATH=src python benchmark/harness.py planner --endpoint http://localhost:8000/v1 --schema descriptions --split held_out
PYTHONPATH=src python benchmark/harness.py planner --endpoint http://localhost:8000/v1 --schema retrieved   --split held_out
PYTHONPATH=src python benchmark/score.py benchmark/runs/planner-*.jsonl benchmark/runs/gold.jsonl --split held_out
```

Without an endpoint the model runs in-process with transformers (`--model`, 4-bit when
`bitsandbytes` is installed; add `outlines` for the grammar). `--no-grammar` and `--repairs 0`
are the ablations of RQ1's typed-validation story; `--schema` is RQ3; `--top-k` sizes the
retrieved fragment.

## What the scorer sees

A plan the typed layer caught and the model could not fix is recorded with that error and no
answer (nothing ran: the invalid query never reached a database). A refusal is `refused` with
the model's own code. Tokens and repairs are summed over all attempts, `schema_tokens` (in the
planner outcome) is the cost of the representation alone.
