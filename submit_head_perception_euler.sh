#!/bin/bash
#SBATCH --job-name=groot-head-perception
#SBATCH --output=logs/groot-head-perception-%j.out
#SBATCH --error=logs/groot-head-perception-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=2
#SBATCH --tmp=200G
# Same submission/training pipeline as wrist perception, with head defaults.
# Usage: sbatch submit_head_perception_euler.sh
set -euo pipefail
export PERCEPTION_CAMERA=head
export HEATMAP_LABELS_DIR="${HEATMAP_LABELS_DIR:-heatmap_labels_head}"
export RUN_NAME="${RUN_NAME:-head_perception_v1}"
export DATASET_HOST_DIR="${DATASET_HOST_DIR:-/cluster/work/mrl/$USER/datasets/g1_dataset_autocollect_v2_rightfreq}"
# sbatch copies this script to its spool; companion scripts stay in the submit dir.
exec bash "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/submit_perception_euler.sh"
