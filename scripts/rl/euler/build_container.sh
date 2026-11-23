#!/bin/bash
# =============================================================================
# build_container.sh — build the RL container and (optionally) upload it to Euler.
# Run locally from anywhere in the repo (needs Docker; Apptainer for the .sif step).
#
#   scripts/rl/euler/build_container.sh                 # gr00t-rl image -> groot-rl.tar.gz
#   scripts/rl/euler/build_container.sh --base          # also rebuild the base gr00t image
#   scripts/rl/euler/build_container.sh --upload        # ... and rsync it to Euler
#   scripts/rl/euler/build_container.sh --upload-only   # just rsync an existing tarball
#
# Same recipe as submit_euler.sh's header (docker -> apptainer .sif -> tar.gz), for the
# image that adds the simulator venvs (docker/Dockerfile.rl). Rebuild only when
# dependencies change: code is bind-mounted at run time.
# Env: EULER_HOST (default euler), EULER_USER (default $USER), OUT_DIR (default repo root).
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EULER_USER="${EULER_USER:-$USER}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT}"
TARBALL="$OUT_DIR/groot-rl.tar.gz"
BASE=0; UPLOAD=0; BUILD=1
for arg in "$@"; do
    case "$arg" in
        --base) BASE=1 ;;
        --upload) UPLOAD=1 ;;
        --upload-only) UPLOAD=1; BUILD=0 ;;
        *) echo "unknown option $arg" >&2; exit 2 ;;
    esac
done

cd "$REPO_ROOT"
if [ "$BUILD" -eq 1 ]; then
    if [ "$BASE" -eq 1 ] || ! docker image inspect gr00t:latest >/dev/null 2>&1; then
        echo "[build] base image gr00t"
        DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile -t gr00t .
    fi
    echo "[build] gr00t-rl"
    DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rl -t gr00t-rl .
    echo "[build] apptainer .sif"
    rm -f "$OUT_DIR/groot-rl.sif"
    apptainer build "$OUT_DIR/groot-rl.sif" docker-daemon://gr00t-rl:latest
    echo "[build] compressing"
    tar -C "$OUT_DIR" -czf "$TARBALL" groot-rl.sif
    rm -f "$OUT_DIR/groot-rl.sif"
    ls -lh "$TARBALL"
fi

if [ "$UPLOAD" -eq 1 ]; then
    # shellcheck source=_ssh.sh
    source "$(dirname "${BASH_SOURCE[0]}")/_ssh.sh"
    euler_connect
    euler_ssh "mkdir -p /cluster/work/mrl/$EULER_USER/containers"
    euler_rsync -avP "$TARBALL" "$EULER_HOST:/cluster/work/mrl/$EULER_USER/containers/"
fi
