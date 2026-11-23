#!/bin/bash
# =============================================================================
# sync.sh — rsync the code (and optionally MolmoSpaces assets) to Euler.
#
#   scripts/rl/euler/sync.sh             # code -> $EULER_HOST:$EULER_PROJECT_DIR
#   scripts/rl/euler/sync.sh --assets    # also local MolmoSpaces assets (saves the download)
#   scripts/rl/euler/sync.sh --weights   # also the HF weights jobs need (N1.7, Cosmos-Reason2)
#   (options combine; Euler's proxy downloads at ~4 MB/s, rsync from here is ~20x faster)
#
# Code excludes mirror submit_euler.sh (no datasets, checkpoints, venvs, vendored deps:
# Euler $HOME is quota-limited). Assets go to /cluster/work/mrl/$EULER_USER/mlspaces and
# keep their symlinks (they point into /mlspaces/cache, the path both the local container
# and the Euler job bind them at).
# Env: EULER_HOST (euler), EULER_USER ($USER), EULER_PROJECT_DIR (~/groot-rl),
#      MLSPACES_LOCAL (~/mlspaces), HF_HUB_LOCAL (~/.cache/huggingface/hub).
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EULER_USER="${EULER_USER:-$USER}"
EULER_PROJECT_DIR="${EULER_PROJECT_DIR:-groot-rl}"
MLSPACES_LOCAL="${MLSPACES_LOCAL:-$HOME/mlspaces}"
HF_HUB_LOCAL="${HF_HUB_LOCAL:-$HOME/.cache/huggingface/hub}"
HF_MODELS=(models--nvidia--GR00T-N1.7-3B models--nvidia--Cosmos-Reason2-2B)
DO_ASSETS=0; DO_WEIGHTS=0
for arg in "$@"; do
    case "$arg" in
        --assets) DO_ASSETS=1 ;;
        --weights) DO_WEIGHTS=1 ;;
        *) echo "unknown option $arg" >&2; exit 2 ;;
    esac
done

# shellcheck source=_ssh.sh
source "$(dirname "${BASH_SOURCE[0]}")/_ssh.sh"
cd "$REPO_ROOT"
euler_connect
euler_rsync -avP --delete \
    --exclude='.git' --exclude='__pycache__' --exclude='*.egg-info' \
    --exclude='.venv' --exclude='venv' --exclude='*_uv/' \
    --exclude='checkpoints*' --exclude='g1_dataset*' --exclude='*_dataset*/' \
    --exclude='external_dependencies' --exclude='media' --exclude='demo_data' \
    --exclude='wandb' --exclude='scratchpad' --exclude='outputs' --exclude='vla_eval_results' \
    --exclude='*.gz' --exclude='*.sif' --exclude='*.tar' \
    --exclude='.env' --exclude='logs' \
    --exclude='.ruff_cache' --exclude='.pytest_cache' --exclude='.mypy_cache' \
    --exclude='scripts/deployment/*/wheels' \
    . "$EULER_HOST:$EULER_PROJECT_DIR/"
echo "[sync] code -> $EULER_HOST:$EULER_PROJECT_DIR (.env is NOT synced; create it there once)"

if [ "$DO_ASSETS" -eq 1 ]; then
    [ -d "$MLSPACES_LOCAL" ] || { echo "no assets at $MLSPACES_LOCAL" >&2; exit 1; }
    euler_ssh "mkdir -p /cluster/work/mrl/$EULER_USER/mlspaces"
    euler_rsync -aP "$MLSPACES_LOCAL/" "$EULER_HOST:/cluster/work/mrl/$EULER_USER/mlspaces/"
    echo "[sync] assets -> /cluster/work/mrl/$EULER_USER/mlspaces"
fi

if [ "$DO_WEIGHTS" -eq 1 ]; then
    # Jobs use HF_HOME=/groot/cache/huggingface (= /cluster/scratch/$USER/groot/cache/huggingface).
    REMOTE_HUB="/cluster/scratch/$EULER_USER/groot/cache/huggingface/hub"
    euler_ssh "mkdir -p $REMOTE_HUB"
    for m in "${HF_MODELS[@]}"; do
        [ -d "$HF_HUB_LOCAL/$m" ] || { echo "[sync] $m not in $HF_HUB_LOCAL, skipping"; continue; }
        # -a keeps the snapshot -> blob symlinks; .locks/incomplete files are skipped.
        euler_rsync -aP --exclude='*.incomplete' "$HF_HUB_LOCAL/$m" "$EULER_HOST:$REMOTE_HUB/"
    done
    echo "[sync] weights -> $REMOTE_HUB"
fi
