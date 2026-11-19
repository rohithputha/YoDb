"""The evaluation benchmark of the typed query layer over the Yelp data layer.

    benchmark/templates.py   the parameterized question templates (gold query, gold sources, oracle SQL)
    benchmark/build.py       instantiate them against the database into questions.jsonl, with gold answers
    benchmark/harness.py     run a *system* over the questions and write predictions.jsonl
    benchmark/score.py       score predictions against the gold: answer F1, source F1, validity, provenance, latency

See benchmark/README.md.
"""
