#!/usr/bin/env bash
# From a fresh clone to the result tables, in one go.  Every step is skipped when its result is already there,
# so the script can be rerun after an interruption and continues where it stopped.
#
#     git clone <repo> YoDb && cd YoDb
#     cp .env.example .env            # put OPENAI_API_KEY (the ASU endpoint key) and YELP_DATA (the dataset folder) in it
#     scripts/run_all.sh [--preset target|fallback|smoke] [--workers N] [--skip-tests] [--skip-e2e] [--only e1|e2|e3|ablations]
#
# What it does: Python environment -> unit tests -> PostgreSQL (AGE + pgvector) in a container, with Docker, Podman or
# Apptainer -> the engine's end-to-end check against the real extensions -> the Yelp load (sample, tables, embeddings,
# graph) -> the questions and the oracle check -> E2, E1, E3 and the ablations -> benchmark/results/results.md.
#
# The model is the endpoint in .env (YODB_LLM_ENDPOINT / YODB_LLM_MODEL / OPENAI_API_KEY); nothing runs on a GPU here.
# --workers is how many questions the model-bound runs answer at a time (8 is a good number for a vLLM endpoint).

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

PRESET=target
WORKERS=8
SKIP_TESTS=0
SKIP_E2E=0
ONLY=all
while [ $# -gt 0 ]; do
    case "$1" in
        --preset) PRESET="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --skip-tests) SKIP_TESTS=1; shift ;;
        --skip-e2e) SKIP_E2E=1; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "run_all.sh: unknown option $1" >&2; exit 2 ;;
    esac
done

STATE="$ROOT/.run_all"
mkdir -p "$STATE"
log() { printf '\n==== [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
done_step() { [ -f "$STATE/$1" ]; }
mark() { touch "$STATE/$1"; }

# --- 0. .env ----------------------------------------------------------------------------------------------------

if [ ! -f .env ]; then
    cp .env.example .env
    echo "run_all.sh: wrote .env from .env.example; put OPENAI_API_KEY and YELP_DATA in it and rerun." >&2
    exit 1
fi
set -a; . ./.env; set +a
if [ -z "${OPENAI_API_KEY:-}" ] || [ "$OPENAI_API_KEY" = "<your ASU key>" ]; then
    echo "run_all.sh: OPENAI_API_KEY is not set in .env (the model endpoint's key)." >&2
    exit 1
fi
export YODB_EMBED_THREADS="${YODB_EMBED_THREADS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

# --- 1. Python ----------------------------------------------------------------------------------------------------

log "Python environment"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_PREFIX:-}" ]; then
    if [ ! -d .venv ]; then
        PY=""
        for candidate in python3.12 python3.11 python3.13 python3; do
            if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then PY="$candidate"; break; fi
        done
        if [ -z "$PY" ]; then
            echo "run_all.sh: no python >= 3.11 on PATH. On ASU Sol: 'module load mamba/latest; mamba create -n yodb python=3.12 -y; source activate yodb' then rerun." >&2
            exit 1
        fi
        "$PY" -m venv .venv
    fi
    # shellcheck disable=SC1091
    . .venv/bin/activate
fi
python -c 'import yodb, fastembed, psycopg' 2>/dev/null || pip install -q -e ".[semantic,mcp]"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import sys, yodb; print("python", sys.version.split()[0], "yodb", getattr(yodb, "__version__", ""))'

# --- 2. unit tests -----------------------------------------------------------------------------------------------

if [ "$SKIP_TESTS" = 0 ] && ! done_step tests; then
    log "unit tests (no database)"
    python -m unittest discover -s tests -q
    mark tests
fi

# --- 3. PostgreSQL -----------------------------------------------------------------------------------------------

log "PostgreSQL with AGE and pgvector"
scripts/pg.sh up

# --- 4. the engine against the real extensions ----------------------------------------------------------------------

if [ "$SKIP_E2E" = 0 ] && ! done_step e2e; then
    log "end-to-end check of the graph and similarity paths (30 cases)"
    scripts/pg.sh psql -q -v ON_ERROR_STOP=1 -c "DROP DATABASE IF EXISTS yodb_age" -c "CREATE DATABASE yodb_age"
    YODB_PG_DB=yodb_age scripts/pg.sh psql -q -v ON_ERROR_STOP=1 <tests/e2e/age/seed.sql
    YODB_E2E_CONNINFO="host=localhost port=${YODB_PG_PORT:-55434} dbname=yodb_age user=yodb_ro password=yodb_ro" \
        python tests/e2e/age/run_age.py
    mark e2e
fi

# --- 5. the Yelp data ---------------------------------------------------------------------------------------------

loaded() {
    scripts/pg.sh psql -tA -c "SELECT count(*) FROM yelp.review WHERE text_embedding IS NOT NULL" 2>/dev/null | tr -d '[:space:]'
}
if ! done_step "load-$PRESET"; then
    log "Yelp load, preset $PRESET (sample, tables, embeddings on $YODB_EMBED_THREADS threads, graph)"
    if [ -z "${YELP_DATA:-}" ] || [ ! -f "$YELP_DATA/yelp_academic_dataset_business.json" ]; then
        if [ -z "${YELP_DATASET_URL:-}" ]; then
            echo "run_all.sh: YELP_DATA in .env must point at the folder with yelp_academic_dataset_*.json (or set YELP_DATASET_URL to the download link from https://business.yelp.com/data/resources/open-dataset/)." >&2
            exit 1
        fi
    fi
    python examples/yelp/load.py --schema --preset "$PRESET" --threads "$YODB_EMBED_THREADS"
    rm -f "$STATE"/load-* "$STATE"/questions "$STATE"/gold
    mark "load-$PRESET"
fi
echo "reviews with embeddings: $(loaded)"
yodb validate examples/yelp/catalog

# --- 6. questions and the oracle check -----------------------------------------------------------------------------

if ! done_step questions; then
    log "benchmark questions"
    python benchmark/build.py
    mark questions
fi
if ! done_step gold; then
    log "gold plans through the engine must match the oracle exactly"
    python benchmark/harness.py gold -q
    python benchmark/diff.py benchmark/runs/gold.jsonl
    mark gold
fi

# --- 7. experiments ------------------------------------------------------------------------------------------------

run_experiment() {
    log "experiment $1"
    python benchmark/experiments.py "$1" --workers "$WORKERS" --resume --repeat 5
}
case "$ONLY" in
    all) for e in e2 e1 e3 ablations; do run_experiment "$e"; done; python benchmark/experiments.py tables >/dev/null ;;
    *) run_experiment "$ONLY" ;;
esac

log "done"
echo "tables:  benchmark/results/results.md"
echo "runs:    benchmark/results/runs/*.jsonl   (scores: python benchmark/score.py <run>; diff: python benchmark/diff.py <run>)"
echo "memory:  benchmark/results/resources.json"
echo "server:  scripts/pg.sh status | down | destroy"
