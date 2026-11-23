#!/bin/bash
#SBATCH --job-name=groot-rl
#SBATCH --output=logs/rl-%x-%j.out
#SBATCH --error=logs/rl-%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=a100_80gb:1
#SBATCH --tmp=150G
# =============================================================================
# submit_rl_euler.sh — SLURM job for GR00T RL (OGPO / Best-of-N / evaluation) on Euler.
#
# Usage (from the repo root on Euler):
#   scripts/rl/euler/submit.sh rl_experiments/smoke.sh          # recommended: applies the
#   scripts/rl/euler/submit.sh rl_experiments/ogpo_arm_c.sh     # preset's SBATCH_ARGS and
#                                                               # chains long runs
#   sbatch submit_rl_euler.sh rl_experiments/pilot_zero_shot.sh # plain single job
#
# An experiment preset (rl_experiments/*.sh) is a bash file defining:
#   MODE        smoke | assets | space_check | list_tasks | eval | train
#   RUN_NAME    stable name; the run dir is reused, so resubmitting RESUMES the run
#   ARGS=(...)  arguments forwarded to the tool (see scripts/rl/euler/entrypoint.sh)
#   SBATCH_ARGS optional, e.g. "--time=02:00:00" (used by submit.sh only)
#   CHAIN       optional, number of chained 24 h jobs (used by submit.sh only)
#
# One-time setup: see scripts/rl/euler/README.md (container build + upload, sync, .env).
# Storage:
#   container  /cluster/work/mrl/$USER/containers/groot-rl.tar.gz   (extracted to $TMPDIR)
#   caches     /cluster/scratch/$USER/groot/cache                    (HF weights, triton)
#   run dirs   /cluster/scratch/$USER/groot/rl_runs/$RUN_NAME        (buffer, checkpoints)
#   results    /cluster/project/mrl/$USER/results/groot-rl/$RUN_NAME (final policy, metrics)
#   assets     /cluster/work/mrl/$USER/mlspaces/{assets,cache}       (MolmoSpaces)
# The existing finetuning runs (submit_euler.sh) live in /cluster/scratch/$USER/groot/runs
# and appear in the container as /groot/runs/<run>: use that as --model-path.
# =============================================================================
set -euo pipefail

EXPERIMENT="${1:-${EXPERIMENT:-}}"
if [ -z "$EXPERIMENT" ]; then
    echo "usage: sbatch submit_rl_euler.sh rl_experiments/<preset>.sh" >&2
    exit 2
fi

# Internet access on compute nodes (HF weights, MolmoSpaces assets). Skipped off-cluster.
if command -v module >/dev/null 2>&1; then module load eth_proxy; fi

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"
[ -f "$EXPERIMENT" ] || { echo "[euler] experiment file $EXPERIMENT not found in $PROJECT_DIR" >&2; exit 2; }

MODE=""; RUN_NAME=""; ARGS=()
# shellcheck disable=SC1090
source "$EXPERIMENT"
: "${MODE:?$EXPERIMENT must set MODE}"
: "${RUN_NAME:?$EXPERIMENT must set RUN_NAME}"

CONTAINER_TAR="${CONTAINER_TAR:-/cluster/work/mrl/$USER/containers/groot-rl.tar.gz}"
GROOT_DIR="${GROOT_DIR:-/cluster/scratch/$USER/groot}"
RUN_DIR="$GROOT_DIR/rl_runs/$RUN_NAME"
RESULTS_ROOT="${RESULTS_ROOT:-/cluster/project/mrl/$USER/results/groot-rl}"
RESULTS_DIR="$RESULTS_ROOT/$RUN_NAME"
MLSPACES_DIR="${MLSPACES_DIR:-/cluster/work/mrl/$USER/mlspaces}"

echo "=== GR00T RL job ==="
echo "Job ID:     ${SLURM_JOB_ID:-local}   Host: $(hostname)   Date: $(date)"
echo "Experiment: $EXPERIMENT   MODE=$MODE   RUN_NAME=$RUN_NAME"
echo "GPU:        ${CUDA_VISIBLE_DEVICES:-?}   CPUs: ${SLURM_CPUS_PER_TASK:-?}"
echo "Run dir:    $RUN_DIR"
echo "Args:       ${ARGS[*]:-}"
echo "===================="

if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "[euler] ERROR: $PROJECT_DIR/.env missing (needs HF_TOKEN, optional WANDB_API_KEY)." >&2
    exit 1
fi
mkdir -p "$PROJECT_DIR/logs" "$GROOT_DIR"/cache/{triton,torch_kernels,huggingface} \
    "$GROOT_DIR/runs" "$RUN_DIR" "$RESULTS_DIR" "$MLSPACES_DIR"/{assets,cache}

# --- Container: extract to node-local scratch (fast) --------------------------------
TMPDIR="${TMPDIR:-/tmp}"
SIF="$TMPDIR/groot-rl.sif"
if [ ! -f "$SIF" ]; then
    echo "[euler] Extracting container to \$TMPDIR..."
    time tar -xzf "$CONTAINER_TAR" -C "$TMPDIR"
fi

# --- Launch --------------------------------------------------------------------------
# Code directories are bound over the copy baked into the image, so a code rsync is
# enough to change what runs; rebuild the container only when dependencies change.
# MolmoSpaces assets must be bound at /mlspaces/{assets,cache}: the asset tree contains
# absolute symlinks into /mlspaces/cache.
BINDS=(
    --bind "$GROOT_DIR:/groot"
    --bind "$RUN_DIR:/rl_run"
    --bind "$RESULTS_DIR:/results"
    --bind "$MLSPACES_DIR:/mlspaces"
    --bind "$PROJECT_DIR/.env:/workspace/.env"
    --bind "$PROJECT_DIR/gr00t:/workspace/gr00t"
    --bind "$PROJECT_DIR/scripts:/workspace/scripts"
    --bind "$PROJECT_DIR/tests:/workspace/tests"
    --bind "$PROJECT_DIR/rl_experiments:/workspace/rl_experiments"
)
# EXTRA_BINDS / EXTRA_ENV: optional extra apptainer flags, e.g. EXTRA_ENV="--env SMOKE_TRAINABLE_REGEX=action_decoder"
# shellcheck disable=SC2206
BINDS+=(${EXTRA_BINDS:-} ${EXTRA_ENV:-})
echo "[euler] Starting container..."
time apptainer exec --nv "${BINDS[@]}" \
    --env TRITON_CACHE_DIR=/groot/cache/triton \
    --env TORCH_EXTENSIONS_DIR=/groot/cache/torch_kernels \
    --env XDG_CACHE_HOME=/groot/cache \
    --env HF_HOME=/groot/cache/huggingface \
    --env MLSPACES_ASSETS_DIR=/mlspaces/assets \
    --env MLSPACES_CACHE_DIR=/mlspaces/cache \
    --env MLSPACES_AUTO_INSTALL=True \
    --env RL_RUN_DIR=/rl_run \
    --env RL_RUN_NAME="$RUN_NAME" \
    --env RL_RESULTS_DIR=/results \
    --env WANDB_DIR=/rl_run \
    "$SIF" \
    bash /workspace/scripts/rl/euler/entrypoint.sh "$MODE" "${ARGS[@]}"

echo "[euler] Job completed at $(date)"
