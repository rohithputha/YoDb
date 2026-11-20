"""The baselines of the evaluation plan, as benchmark systems.

    baselines/tools.py       raw, read-only access to the three stores (SQL, pgvector search, Cypher), shared by the baselines
    baselines/tool_agent.py  B1: a monolithic tool agent — the same LLM sees the raw schemas and calls the three tools itself
    baselines/router.py      B2: framework routing — a router picks one engine per (sub-)question and aggregates
                             (LlamaIndex's RouterQueryEngine/SubQuestionQueryEngine when installed, or the same
                             design in plain Python so the comparison never depends on a package)

B3, the oracle pipeline, is ``benchmark/harness.py gold``.
"""
