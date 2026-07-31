#!/bin/bash
set -euo pipefail

source .env

# Fine-approach PERCEPTION model: GR00T VLM backbone + target heatmap head.
# Wrist camera only, fine-phase frames only (heatmap_labels/ sidecar).
# Keep RUN_NAME stable across resubmissions of the SAME experiment so training
# resumes; bump it to start fresh (same caveats as start_finetune.sh).
RUN_NAME="${RUN_NAME:-perception_v1}"

# Dataset root, as seen from the training cwd (/workspace in the container). Must
# contain the heatmap_labels/ sidecar. g1_dataset_small_general has 15 instruction
# phrasings over red/green/blue targets with all three squares in frame, so the
# instruction genuinely selects the target — which is what the head's text
# cross-attention (--use-text-conditioning, on by default) is there to exploit.
DATASET_DIR="${DATASET_DIR:-./g1_dataset_small_general/}"

# Backbone init. Three choices:
#   - A finetuned VLA run. Action-head weights are ignored, only the backbone is
#     inherited. NOT recommended as of 2026-07-31: every VLA run to date actually
#     trained at 256px (the resolution overrides were silently dropped until commit
#     a66a334), so such a checkpoint's vision encoder is adapted to 256px and is a
#     WORSE init than the foundation weights for a 736px perception run. Revisit
#     once you have a VLA checkpoint whose logs confirm a matching resolution.
#   - nvidia/GR00T-N1.7-3B (the GR00T foundation VLA) — the default.
#   - "scratch" / "none" / "" (empty) — do NOT overlay any VLA checkpoint. The
#     VLM backbone still starts from the published Cosmos-Reason2-2B foundation
#     weights (that 2B VLM is never randomly initialised); only the perception
#     head is fresh. This is the "finetune GR00T from scratch" option: nothing
#     from your earlier drilling runs is inherited.
BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}"

# Build the base-model CLI arg. When BASE_MODEL_PATH is empty/scratch/none we omit
# the flag entirely, so the config default (base_model_path=None) is used and no
# checkpoint is overlaid.
BASE_MODEL_ARG=(--base-model-path "${BASE_MODEL_PATH}")
case "${BASE_MODEL_PATH,,}" in
    ""|scratch|none|fresh)
        BASE_MODEL_ARG=()
        BASE_MODEL_PATH="<none: raw Cosmos-Reason2-2B backbone + fresh head>"
        ;;
esac

# Loss weights (see PerceptionFinetuneConfig for docs). All overridable at submit
# time, e.g.  HEATMAP_SIGMA_PX=5 VISIBILITY_LOSS_WEIGHT=1.0 RUN_NAME=sweep_a sbatch ...
HEATMAP_SIGMA_PX="${HEATMAP_SIGMA_PX:-8.0}"        # Gaussian target std, processed-image px
HEATMAP_LOSS_WEIGHT="${HEATMAP_LOSS_WEIGHT:-1.0}"
OFFSET_LOSS_WEIGHT="${OFFSET_LOSS_WEIGHT:-1.0}"
VISIBILITY_LOSS_WEIGHT="${VISIBILITY_LOSS_WEIGHT:-0.5}"  # BCE: "is the target in the image at all?"
DEPTH_LOSS_WEIGHT="${DEPTH_LOSS_WEIGHT:-0.1}"       # 0 disables the depth output
WALL_NORMAL_LOSS_WEIGHT="${WALL_NORMAL_LOSS_WEIGHT:-0.1}"  # 0 disables the wall-normal output
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

# --- Image resolution ---
# Wrist frames are natively 720x720. 736 rather than 720 because Qwen's smart-resize
# rounds each side to a multiple of patch_size*merge_size = 32: 720 rounds DOWN to
# 704 (44x44 patches -> 484 visual tokens) and throws away 16 px, while 736 is exact
# (46x46 -> 529 tokens). Costs 9% more tokens for a 1:1 mapping of 32 native px per
# token. perceptionv1 was trained at 720/648; use those to reproduce it.
#
# IMAGE_CROP_SIZE is NOT the output size: the pipeline crops that fraction and
# resizes back up to IMAGE_TARGET_SIZE, i.e. it is a zoom augmentation. 662/736 =
# 0.900, matching the 648/720 = 0.900 perceptionv1 trained with.
IMAGE_TARGET_SIZE="${IMAGE_TARGET_SIZE:-736}"
IMAGE_CROP_SIZE="${IMAGE_CROP_SIZE:-662}"

# --- Batch ---
# Effective batch = GLOBAL_BATCH_SIZE x GRAD_ACCUM_STEPS. GLOBAL_BATCH_SIZE is the
# total across GPUs, so with 2 GPUs the per-GPU micro-batch is half of it.
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"

# --- Training length ---
# Set NUM_EPOCHS to size the run by the data instead of by a step count; it wins
# over MAX_STEPS. One epoch here is counted over the LABELLED FINE-PHASE frames
# (the heatmap_labels/ sidecar), which is what this model actually trains on.
# SAVE_STEPS defaults to ~1/5 of the run so a killed job loses at most 20%.
NUM_EPOCHS="${NUM_EPOCHS:-}"
MAX_STEPS="${MAX_STEPS:-10000}"
LENGTH_ARGS=(--max-steps "${MAX_STEPS}")
if [ -n "${NUM_EPOCHS}" ]; then
    LENGTH_ARGS+=(--num-epochs "${NUM_EPOCHS}")
    # max_steps is resolved inside the launcher, so a step-derived default for
    # SAVE_STEPS is not available here; fall back to a fixed cadence.
    SAVE_STEPS="${SAVE_STEPS:-1000}"
else
    SAVE_STEPS="${SAVE_STEPS:-$(( MAX_STEPS / 5 < 200 ? 200 : MAX_STEPS / 5 ))}"
fi

MASTER_PORT=$(shuf -i 10000-65000 -n 1)
OUTPUT_DIR="${GROOT_HOME:-/groot}/runs/${RUN_NAME}"

echo "[train] RUN_NAME=${RUN_NAME}"
echo "[train] DATASET_DIR=${DATASET_DIR}"
echo "[train] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[train] BASE_MODEL_PATH=${BASE_MODEL_PATH}"
echo "[train] RESOLUTION target=${IMAGE_TARGET_SIZE} crop=${IMAGE_CROP_SIZE} (crop is a zoom aug; output is ${IMAGE_TARGET_SIZE}px)"
echo "[train] BATCH global=${GLOBAL_BATCH_SIZE} x accum=${GRAD_ACCUM_STEPS} -> effective $(( GLOBAL_BATCH_SIZE * GRAD_ACCUM_STEPS ))"
echo "[train] LENGTH ${LENGTH_ARGS[*]}  SAVE_STEPS=${SAVE_STEPS}"
echo "[train] MASTER_PORT=${MASTER_PORT}"
if [ -d "$OUTPUT_DIR" ]; then
    echo "[train] WARNING: output dir already exists. Existing checkpoints:"
    ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null || echo "[train]   (none)"
else
    echo "[train] Fresh run — no existing output dir."
fi

set +e
torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
    gr00t/experiment/launch_finetune_perception.py \
    "${BASE_MODEL_ARG[@]}" \
    --dataset-path "${DATASET_DIR}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path register_perception_modality.py \
    --output-dir "${OUTPUT_DIR}" \
    --save-total-limit 5 \
    --save-steps "${SAVE_STEPS}" \
    "${LENGTH_ARGS[@]}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
    --dataloader-num-workers 4 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.25 hue 0.02 \
    --image-target-size "${IMAGE_TARGET_SIZE}" \
    --image-crop-size "${IMAGE_CROP_SIZE}" \
    --learning-rate ${LEARNING_RATE} \
    --heatmap-target-sigma-px ${HEATMAP_SIGMA_PX} \
    --heatmap-loss-weight ${HEATMAP_LOSS_WEIGHT} \
    --offset-loss-weight ${OFFSET_LOSS_WEIGHT} \
    --visibility-loss-weight ${VISIBILITY_LOSS_WEIGHT} \
    --depth-loss-weight ${DEPTH_LOSS_WEIGHT} \
    --wall-normal-loss-weight ${WALL_NORMAL_LOSS_WEIGHT} \
    --num-gpus 2 \
    --use-wandb \
    --tune-visual 2>&1
TORCHRUN_EXIT=$?
set -e
echo "[train] torchrun exited with code: ${TORCHRUN_EXIT}"
exit "${TORCHRUN_EXIT}"
