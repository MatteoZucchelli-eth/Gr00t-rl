#!/bin/bash
#SBATCH --job-name=groot-perception-action
#SBATCH --output=logs/groot-perception-action-%j.out
#SBATCH --error=logs/groot-perception-action-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=2
#SBATCH --tmp=200G
# =============================================================================
# submit_perception_action_euler.sh — SLURM job script for the fine-approach
# PERCEPTION-CONDITIONED ACTION finetune (VLM backbone + target heatmap head +
# perception-conditioned flow-matching action head) on ETH Euler.
#
# Usage:
#   cd ~/groot && sbatch submit_perception_action_euler.sh
#
#   RUN_NAME=perception_action_v1 sbatch submit_perception_action_euler.sh
#
# Retrieving the trained model afterwards (newest weights only; see submit_euler.sh):
#   scripts/fetch_euler_checkpoint.sh <run-name>
#
#   # Optional: reuse an existing perception head (needs its select_layer):
#   RUN_NAME=perception_action_reuse \
#       PERCEPTION_MODEL_PATH=/groot/runs/perceptionv1 SELECT_LAYER=12 \
#       COND_NOISE_UV_PX=<from calibrate_perception_conditioning.py> \
#       sbatch submit_perception_action_euler.sh
#
#   # Ablation (perception trained but NOT fed to the DiT):
#   RUN_NAME=perception_action_noc NO_PERCEPTION_CONDITIONING=1 \
#       sbatch submit_perception_action_euler.sh
#
# Prerequisites: same container image and one-time setup as submit_euler.sh (see
# its header). Additionally:
#
#   1. Sync the dataset — INCLUDING the heatmap_labels/ sidecar — to the work dir:
#        rsync -avP g1_dataset/ euler:/cluster/work/mrl/$USER/datasets/g1_dataset_perception/
#
#   2. Make sure the codebase sync has picked up the new files; they are
#      bind-mounted individually below:
#        start_finetune_perception_action.sh, setup_and_train_perception_action.sh,
#        register_perception_action_modality.py, gr00t/
#
#   3. PERCEPTION_MODEL_PATH (if used) must be reachable INSIDE the container.
#      Scratch is mounted as /groot, so runs live at /groot/runs/<run>/checkpoint-XXXX.
#      The GR00T base downloads via HF_TOKEN from .env.
#
#   4. COND_NOISE_UV_PX defaults to 3.0 px, MEASURED from the
#      1500_perception_planner eval of the 1500-step perception checkpoint. If you
#      point PERCEPTION_MODEL_PATH at a different checkpoint, re-derive it:
#        python scripts/calibrate_perception_conditioning.py \
#            --eval-dir <that checkpoint's eval dir> --dataset-path ./g1_dataset
#      Noise smaller than the real closed-loop error teaches the policy to trust
#      perfect perception.
# =============================================================================

set -euo pipefail

# --- Load modules ---
module load eth_proxy

# --- Paths (adjust to match your Euler setup) ---
PROJECT_DIR="$HOME/groot"
CONTAINER_TAR="/cluster/work/mrl/$USER/containers/groot-environment.tar.gz"
GROOT_DIR="/cluster/scratch/$USER/groot"
RESULTS_DIR="/cluster/project/mrl/$USER/results/groot/$SLURM_JOB_ID"
DATASET_HOST_DIR="${DATASET_HOST_DIR:-/cluster/work/mrl/$USER/datasets/g1_dataset_perception}"

# Weight init: base is always nvidia/GR00T-N1.7-3B. Optional perception overlay
# (path as seen INSIDE the container).
PERCEPTION_MODEL_PATH="${PERCEPTION_MODEL_PATH:-}"

# --- Job info ---
echo "=== groot Perception-Action Finetune Job ==="
echo "Job ID:                $SLURM_JOB_ID"
echo "Host:                  $(hostname)"
echo "Date:                  $(date)"
echo "GPU:                   $CUDA_VISIBLE_DEVICES"
echo "TMPDIR:                $TMPDIR"
echo "RUN_NAME:              ${RUN_NAME:-perception_action_v1}"
echo "BASE_MODEL_PATH:       nvidia/GR00T-N1.7-3B (fixed)"
echo "SELECT_LAYER:          ${SELECT_LAYER:-16}"
echo "PERCEPTION_MODEL_PATH: ${PERCEPTION_MODEL_PATH:-<none: fresh perception head>}"
echo "COND_NOISE_UV_PX:      ${COND_NOISE_UV_PX:-3.0 (measured, 1500 perception ckpt)}"
echo "COND_BIAS_DEPTH_M:     ${COND_BIAS_DEPTH_M:-0.0 (set 0.0065 only when reusing perceptionv1)}"
echo "DATASET:               $DATASET_HOST_DIR"
echo "==========================================="

# --- Create directories ---
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$GROOT_DIR/cache/checkpoints"
mkdir -p "$GROOT_DIR/cache/triton"
mkdir -p "$GROOT_DIR/cache/torch_kernels"
mkdir -p "$GROOT_DIR/data"
mkdir -p "$GROOT_DIR/runs"
mkdir -p "$RESULTS_DIR"

# --- Sanity: the dataset must ship the heatmap label sidecar ---
if [ ! -d "$DATASET_HOST_DIR/heatmap_labels" ]; then
    echo "[euler] ERROR: $DATASET_HOST_DIR/heatmap_labels not found."
    echo "[euler] This finetune trains on fine-phase frames from the heatmap_labels/"
    echo "[euler] sidecar and teacher-forces the conditioning token from it."
    echo "[euler] Rsync the dataset including it (see header)."
    exit 1
fi

# --- Sanity: a perception checkpoint path must actually exist inside /groot ---
# (HF ids are resolved at runtime, local paths are not — catch typos before the
# 20-minute container extraction rather than after.)
case "$PERCEPTION_MODEL_PATH" in
    /groot/*)
        HOST_PERCEPTION_PATH="${GROOT_DIR}${PERCEPTION_MODEL_PATH#/groot}"
        if [ ! -d "$HOST_PERCEPTION_PATH" ]; then
            echo "[euler] ERROR: PERCEPTION_MODEL_PATH=$PERCEPTION_MODEL_PATH"
            echo "[euler]        resolves to $HOST_PERCEPTION_PATH on the host, which does not exist."
            exit 1
        fi
        ;;
esac

# --- Extract container to local scratch (CRITICAL for performance) ---
echo "[euler] Extracting container to \$TMPDIR..."
time tar -xzf "$CONTAINER_TAR" -C "$TMPDIR"

# --- Copy dataset to local scratch (avoids slow NFS/Lustre reads) ---
# The container path, the bind target and the launcher's --dataset-path all derive
# from the host directory name, so pointing DATASET_HOST_DIR at another dataset is
# the only edit needed. NOTE: the mount point is now the host basename
# (g1_dataset_perception), not the hardcoded /workspace/g1_dataset it used to be.
DATASET_NAME="$(basename "$DATASET_HOST_DIR")"
DATASET_LOCAL_DIR="$TMPDIR/$DATASET_NAME"
echo "[euler] Rsyncing dataset to local scratch (${DATASET_LOCAL_DIR})..."
time rsync -a --info=progress2 "$DATASET_HOST_DIR/" "$DATASET_LOCAL_DIR/"
echo "[euler] Dataset copy done."

# --- Launch with Apptainer ---
echo "[euler] Starting container..."
time apptainer exec --nv \
    --bind "$GROOT_DIR:/groot" \
    --bind "$PROJECT_DIR/.env:/workspace/.env" \
    --bind "$RESULTS_DIR:/results" \
    --bind "$PROJECT_DIR/scripts:/workspace/scripts" \
    --bind "$PROJECT_DIR/gr00t:/workspace/gr00t" \
    --bind "$PROJECT_DIR/setup_and_train_perception_action.sh:/workspace/setup_and_train_perception_action.sh" \
    --bind "$PROJECT_DIR/start_finetune_perception_action.sh:/workspace/start_finetune_perception_action.sh" \
    --bind "$PROJECT_DIR/register_perception_action_modality.py:/workspace/register_perception_action_modality.py" \
    --bind "$DATASET_LOCAL_DIR:/workspace/$DATASET_NAME" \
    --env TRITON_CACHE_DIR=/groot/cache/triton \
    --env TORCH_EXTENSIONS_DIR=/groot/cache/torch_kernels \
    --env XDG_CACHE_HOME=/groot/cache \
    --env RUN_NAME="${RUN_NAME:-perception_action_v1}" \
    --env PERCEPTION_MODEL_PATH="$PERCEPTION_MODEL_PATH" \
    --env SELECT_LAYER="${SELECT_LAYER:-}" \
    --env HEATMAP_SIGMA_PX="${HEATMAP_SIGMA_PX:-}" \
    --env HEATMAP_LOSS_WEIGHT="${HEATMAP_LOSS_WEIGHT:-}" \
    --env OFFSET_LOSS_WEIGHT="${OFFSET_LOSS_WEIGHT:-}" \
    --env VISIBILITY_LOSS_WEIGHT="${VISIBILITY_LOSS_WEIGHT:-}" \
    --env DEPTH_LOSS_WEIGHT="${DEPTH_LOSS_WEIGHT:-}" \
    --env WALL_NORMAL_LOSS_WEIGHT="${WALL_NORMAL_LOSS_WEIGHT:-}" \
    --env PERCEPTION_LOSS_WEIGHT="${PERCEPTION_LOSS_WEIGHT:-}" \
    --env COND_NOISE_UV_PX="${COND_NOISE_UV_PX:-}" \
    --env COND_NOISE_DEPTH_M="${COND_NOISE_DEPTH_M:-}" \
    --env COND_NOISE_NORMAL="${COND_NOISE_NORMAL:-}" \
    --env COND_BIAS_DEPTH_M="${COND_BIAS_DEPTH_M:-}" \
    --env COND_VIS_DROPOUT_PROB="${COND_VIS_DROPOUT_PROB:-}" \
    --env NO_PERCEPTION_CONDITIONING="${NO_PERCEPTION_CONDITIONING:-0}" \
    --env ACTION_HORIZON="${ACTION_HORIZON:-}" \
    --env LEARNING_RATE="${LEARNING_RATE:-}" \
    --env DATASET_DIR="${DATASET_DIR:-./$DATASET_NAME/}" \
    --env IMAGE_TARGET_SIZE="${IMAGE_TARGET_SIZE:-}" \
    --env IMAGE_CROP_SIZE="${IMAGE_CROP_SIZE:-}" \
    --env GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-}" \
    --env GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-}" \
    --env NUM_EPOCHS="${NUM_EPOCHS:-}" \
    --env MAX_STEPS="${MAX_STEPS:-}" \
    --env SAVE_STEPS="${SAVE_STEPS:-}" \
    "$TMPDIR/groot-environment.sif" \
    bash /workspace/setup_and_train_perception_action.sh

echo "[euler] Job completed at $(date)"
