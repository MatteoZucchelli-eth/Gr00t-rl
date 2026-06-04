#!/bin/bash
#SBATCH --job-name=groot-finetune-red-green-no-recovery
#SBATCH --output=logs/groot-%j.out
#SBATCH --error=logs/groot-%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=a100_80gb:2
#SBATCH --tmp=200G
# =============================================================================
# submit_euler.sh — SLURM job script for groot finetuning on ETH Euler
#
# Usage:
#   cd ~/groot && sbatch submit_euler.sh
#
# Prerequisites:
#   1. Build the Docker image locally (from repo root):
#        bash docker/build.sh
#      This produces the image tagged "gr00t".
#
#   2. Convert to Apptainer .sif and compress (rootful Docker):
#        apptainer build groot-environment.sif docker-daemon://gr00t:latest
#        tar -czf groot-environment.tar.gz groot-environment.sif
#        rm groot-environment.sif
#
#   3. Transfer to Euler:
#        # Send the heavy container image to the work directory
#        rsync -avP groot-environment.tar.gz \
#            euler:/cluster/work/mrl/mzucchelli/containers/
#
#        # Sync the codebase to the home directory, excluding heavy/local folders
#        rsync -avP \
#            --exclude='.git' \
#            --exclude='__pycache__' \
#            --exclude='*.egg-info' \
#            --exclude='.venv' \
#            --exclude='venv' \
#            --exclude='g1_dataset' \
#            --exclude='groot-environment.tar.gz' \
#            . euler:~/groot/
#
#        # Sync the dataset directly to the high-capacity work directory
#        rsync -avP g1_dataset/ euler:/cluster/work/mrl/mzucchelli/datasets/g1_dataset/
#
#   4. One-time setup on Euler (run once after first transfer):
#        mkdir -p /cluster/work/mrl/$USER/containers
#        mkdir -p /cluster/project/mrl/$USER/results/groot
#        mkdir -p /cluster/scratch/$USER/groot/{cache/checkpoints,data,runs}
#        mkdir -p ~/groot/logs
#        cp ~/groot/.env.sample ~/groot/.env  # then edit with your tokens
# =============================================================================

set -euo pipefail

# --- Load modules ---
module load eth_proxy


# --- Paths (adjust to match your Euler setup) ---
PROJECT_DIR="$HOME/groot"
CONTAINER_TAR="/cluster/work/mrl/$USER/containers/groot-environment.tar.gz"
GROOT_DIR="/cluster/scratch/$USER/groot"
RESULTS_DIR="/cluster/project/mrl/$USER/results/groot/$SLURM_JOB_ID"
DATASET_HOST_DIR="/cluster/work/mrl/$USER/datasets/g1_dataset_red_green"

# --- Job info ---
echo "=== groot Finetune Job ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Host:      $(hostname)"
echo "Date:      $(date)"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "TMPDIR:    $TMPDIR"
echo "========================="

# --- Create directories ---
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$GROOT_DIR/cache/checkpoints"
mkdir -p "$GROOT_DIR/cache/triton"
mkdir -p "$GROOT_DIR/cache/torch_kernels"
mkdir -p "$GROOT_DIR/data"
mkdir -p "$GROOT_DIR/runs"
mkdir -p "$RESULTS_DIR"

# --- Extract container to local scratch (CRITICAL for performance) ---
echo "[euler] Extracting container to \$TMPDIR..."
time tar -xzf "$CONTAINER_TAR" -C "$TMPDIR"

# --- Copy dataset to local scratch (avoids slow NFS reads during stats generation) ---
# generate_stats/generate_rel_stats scans all parquet files on rank 0 before the NCCL
# barrier — on Lustre this takes ~4.5 s/file (~10 min for 131 files), exceeding the
# 600 s barrier timeout.  Reading from local NVMe reduces this to <10 s total.
DATASET_LOCAL_DIR="$TMPDIR/g1_dataset"
echo "[euler] Rsyncing dataset to local scratch (${DATASET_LOCAL_DIR})..."
time rsync -a --info=progress2 "$DATASET_HOST_DIR/" "$DATASET_LOCAL_DIR/"
echo "[euler] Dataset copy done."

# --- Launch with Apptainer ---
# --nv:                 enable NVIDIA GPU support
# --bind /groot:        scratch storage for checkpoints, data, caches (read-write)
# --bind .env:          mount env file with secrets (HF_TOKEN, WANDB, etc.)
# --bind /results:      persistent results on project storage
# --env:                redirect triton/torch caches to writable scratch
echo "[euler] Starting container..."
time apptainer exec --nv \
    --bind "$GROOT_DIR:/groot" \
    --bind "$PROJECT_DIR/.env:/workspace/.env" \
    --bind "$RESULTS_DIR:/results" \
    --bind "$PROJECT_DIR/scripts:/workspace/scripts" \
    --bind "$PROJECT_DIR/setup_and_train.sh:/workspace/setup_and_train.sh" \
    --bind "$PROJECT_DIR/start_finetune.sh:/workspace/start_finetune.sh" \
    --bind "$PROJECT_DIR/register_new_modality.py:/workspace/register_new_modality.py" \
    --bind "$DATASET_LOCAL_DIR:/workspace/g1_dataset" \
    --env TRITON_CACHE_DIR=/groot/cache/triton \
    --env TORCH_EXTENSIONS_DIR=/groot/cache/torch_kernels \
    --env XDG_CACHE_HOME=/groot/cache \
    "$TMPDIR/groot-environment.sif" \
    bash /workspace/setup_and_train.sh

echo "[euler] Job completed at $(date)"
