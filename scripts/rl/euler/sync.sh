#!/bin/bash
# =============================================================================
# sync.sh — rsync the code (and optionally MolmoSpaces assets) to Euler.
#
#   scripts/rl/euler/sync.sh             # code -> $EULER_HOST:$EULER_PROJECT_DIR
#   scripts/rl/euler/sync.sh --assets    # also local MolmoSpaces assets (saves the download)
#
# Code excludes mirror submit_euler.sh (no datasets, checkpoints, venvs, vendored deps:
# Euler $HOME is quota-limited). Assets go to /cluster/work/mrl/$EULER_USER/mlspaces and
# keep their symlinks (they point into /mlspaces/cache, the path both the local container
# and the Euler job bind them at).
# Env: EULER_HOST (euler), EULER_USER ($USER), EULER_PROJECT_DIR (~/groot-rl),
#      MLSPACES_LOCAL (~/mlspaces).
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EULER_HOST="${EULER_HOST:-euler}"
EULER_USER="${EULER_USER:-$USER}"
EULER_PROJECT_DIR="${EULER_PROJECT_DIR:-groot-rl}"
MLSPACES_LOCAL="${MLSPACES_LOCAL:-$HOME/mlspaces}"

cd "$REPO_ROOT"
rsync -avP --delete \
    --exclude='.git' --exclude='__pycache__' --exclude='*.egg-info' \
    --exclude='.venv' --exclude='venv' --exclude='*_uv/' \
    --exclude='checkpoints*' --exclude='g1_dataset*' --exclude='*_dataset*/' \
    --exclude='external_dependencies' --exclude='media' --exclude='demo_data' \
    --exclude='wandb' --exclude='scratchpad' --exclude='outputs' --exclude='vla_eval_results' \
    --exclude='*.gz' --exclude='*.sif' --exclude='*.tar' \
    --exclude='.env' --exclude='logs' \
    . "$EULER_HOST:$EULER_PROJECT_DIR/"
echo "[sync] code -> $EULER_HOST:$EULER_PROJECT_DIR (.env is NOT synced; create it there once)"

if [ "${1:-}" = "--assets" ]; then
    [ -d "$MLSPACES_LOCAL" ] || { echo "no assets at $MLSPACES_LOCAL" >&2; exit 1; }
    ssh "$EULER_HOST" "mkdir -p /cluster/work/mrl/$EULER_USER/mlspaces"
    rsync -aP "$MLSPACES_LOCAL/" "$EULER_HOST:/cluster/work/mrl/$EULER_USER/mlspaces/"
    echo "[sync] assets -> /cluster/work/mrl/$EULER_USER/mlspaces"
fi
