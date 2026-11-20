#!/usr/bin/env bash
# PostgreSQL 16 + Apache AGE + pgvector as a container, with whatever runtime the machine has:
# Docker, Podman, or Apptainer/Singularity (an HPC node, no root).
#
#     scripts/pg.sh up        # build the image if needed, start the server, wait until it answers
#     scripts/pg.sh status    # is it answering?
#     scripts/pg.sh psql ...  # a psql on it (arguments are passed on)
#     scripts/pg.sh logs
#     scripts/pg.sh down      # stop it (the data directory stays)
#     scripts/pg.sh destroy   # stop it and delete the data
#
# Settings (environment, or .env): YODB_PG_PORT (55434), YODB_PG_PASSWORD (postgres), YODB_PG_DB (yelp),
# YODB_PG_HOME (where Apptainer keeps the data and the image: $SCRATCH/yodb-pg or ~/yodb-pg),
# YODB_PG_RUNTIME (docker|podman|apptainer, else detected), YODB_PG_IMAGE_TAR (a `docker save` archive
# to build the Apptainer image from when --fakeroot builds are not allowed).

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
[ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a

PORT="${YODB_PG_PORT:-55434}"
PASSWORD="${YODB_PG_PASSWORD:-postgres}"
DB="${YODB_PG_DB:-yelp}"
NAME="${YODB_PG_NAME:-yelp-pg}"
IMAGE_DIR="$ROOT/tests/e2e/age"
HOME_DIR="${YODB_PG_HOME:-${SCRATCH:-$HOME}/yodb-pg}"

runtime() {
    if [ -n "${YODB_PG_RUNTIME:-}" ]; then echo "$YODB_PG_RUNTIME"; return; fi
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then echo docker
    elif command -v podman >/dev/null 2>&1; then echo podman
    elif command -v apptainer >/dev/null 2>&1; then echo apptainer
    elif command -v singularity >/dev/null 2>&1; then echo singularity
    else echo "pg.sh: no container runtime found (docker, podman, apptainer or singularity)" >&2; exit 1
    fi
}
RT="$(runtime)"

ready() {
    if [ "$RT" = docker ] || [ "$RT" = podman ]; then
        "$RT" exec "$NAME" pg_isready -q -U postgres -d "$DB" >/dev/null 2>&1
    else
        pg_isready_in_container >/dev/null 2>&1
    fi
}

wait_ready() {
    local n=0
    until ready; do
        n=$((n + 1))
        if [ "$n" -gt 120 ]; then echo "pg.sh: the server did not come up in 2 minutes; see: $0 logs" >&2; exit 1; fi
        sleep 1
    done
    sleep 1
}

# --- Docker / Podman -----------------------------------------------------------------------------------

docker_up() {
    if ! "$RT" image inspect yodb-age >/dev/null 2>&1; then
        echo "pg.sh: building the image (PostgreSQL 16 + AGE + pgvector)..."
        "$RT" build -t yodb-age "$IMAGE_DIR"
    fi
    if "$RT" container inspect "$NAME" >/dev/null 2>&1; then
        "$RT" start "$NAME" >/dev/null
    else
        "$RT" run -d --name "$NAME" -e POSTGRES_PASSWORD="$PASSWORD" -e POSTGRES_DB="$DB" \
            -p "127.0.0.1:$PORT:5432" -v "$NAME-data:/var/lib/postgresql/data" yodb-age >/dev/null
    fi
    wait_ready
}

# --- Apptainer / Singularity (rootless): the official entrypoint runs as you, with the data directory bound --

SIF="$HOME_DIR/yodb-age.sif"
PGDATA_DIR="$HOME_DIR/data"
PGRUN_DIR="$HOME_DIR/run"
PIDFILE="$HOME_DIR/postgres.pid"
LOGFILE="$HOME_DIR/postgres.log"

apptainer_image() {
    [ -f "$SIF" ] && return
    mkdir -p "$HOME_DIR"
    if [ -n "${YODB_PG_IMAGE_TAR:-}" ] && [ -f "$YODB_PG_IMAGE_TAR" ]; then
        echo "pg.sh: building $SIF from $YODB_PG_IMAGE_TAR..."
        "$RT" build "$SIF" "docker-archive://$YODB_PG_IMAGE_TAR"
    else
        echo "pg.sh: building $SIF from $IMAGE_DIR/yodb-age.def (needs --fakeroot)..."
        if ! "$RT" build --fakeroot "$SIF" "$IMAGE_DIR/yodb-age.def"; then
            cat >&2 <<EOF
pg.sh: the --fakeroot build was refused on this machine.  Build the image where Docker runs and bring it here:
    docker build -t yodb-age tests/e2e/age && docker save yodb-age -o yodb-age.tar
    scp yodb-age.tar <you>@sol.asu.edu:$HOME_DIR/
then: YODB_PG_IMAGE_TAR=$HOME_DIR/yodb-age.tar $0 up
EOF
            exit 1
        fi
    fi
}

pg_isready_in_container() {
    "$RT" exec -B "$PGRUN_DIR:/var/run/postgresql" "$SIF" pg_isready -q -h 127.0.0.1 -p "$PORT" -U postgres -d "$DB"
}

apptainer_running() {
    [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

apptainer_up() {
    apptainer_image
    mkdir -p "$PGDATA_DIR" "$PGRUN_DIR"
    if apptainer_running; then wait_ready; return; fi
    echo "pg.sh: starting PostgreSQL (Apptainer, port $PORT, data in $PGDATA_DIR)..."
    # The entrypoint initialises an empty data directory, then execs postgres; not root, so it runs as you.
    nohup "$RT" exec --cleanenv --writable-tmpfs \
        -B "$PGDATA_DIR:/var/lib/postgresql/data" -B "$PGRUN_DIR:/var/run/postgresql" \
        --env POSTGRES_PASSWORD="$PASSWORD" --env POSTGRES_DB="$DB" --env PGDATA=/var/lib/postgresql/data \
        "$SIF" docker-entrypoint.sh postgres -p "$PORT" -c listen_addresses=127.0.0.1 >"$LOGFILE" 2>&1 &
    echo $! >"$PIDFILE"
    wait_ready
}

apptainer_down() {
    if apptainer_running; then
        kill -INT "$(cat "$PIDFILE")"
        local n=0
        while apptainer_running && [ "$n" -lt 30 ]; do n=$((n + 1)); sleep 1; done
    fi
    rm -f "$PIDFILE"
}

# --- commands ---------------------------------------------------------------------------------------------

case "${1:-}" in
    up)
        if [ "$RT" = docker ] || [ "$RT" = podman ]; then docker_up; else apptainer_up; fi
        echo "pg.sh: PostgreSQL is up ($RT, 127.0.0.1:$PORT, database $DB)"
        ;;
    status)
        if ready; then echo "up ($RT, 127.0.0.1:$PORT)"; else echo "down"; exit 1; fi
        ;;
    psql)
        shift
        if [ "$RT" = docker ] || [ "$RT" = podman ]; then
            "$RT" exec -i "$NAME" psql -U postgres -d "$DB" "$@"
        else
            PGPASSWORD="$PASSWORD" "$RT" exec -B "$PGRUN_DIR:/var/run/postgresql" "$SIF" psql -h 127.0.0.1 -p "$PORT" -U postgres -d "$DB" "$@"
        fi
        ;;
    logs)
        if [ "$RT" = docker ] || [ "$RT" = podman ]; then "$RT" logs --tail 100 "$NAME"; else tail -n 100 "$LOGFILE"; fi
        ;;
    down)
        if [ "$RT" = docker ] || [ "$RT" = podman ]; then "$RT" stop "$NAME" >/dev/null 2>&1 || true; else apptainer_down; fi
        echo "pg.sh: stopped"
        ;;
    destroy)
        if [ "$RT" = docker ] || [ "$RT" = podman ]; then
            "$RT" rm -f "$NAME" >/dev/null 2>&1 || true
            "$RT" volume rm "$NAME-data" >/dev/null 2>&1 || true
        else
            apptainer_down
            rm -rf "$PGDATA_DIR" "$PGRUN_DIR"
        fi
        echo "pg.sh: removed the server and its data"
        ;;
    *)
        sed -n '2,16p' "$0"
        exit 2
        ;;
esac
