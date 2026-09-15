#!/bin/bash
set -euo pipefail

source .env

# Fine-approach PERCEPTION-CONDITIONED ACTION model: GR00T VLM backbone +
# target heatmap head + flow-matching action head, where the perception output is
# cast into conditioning tokens the DiT attends to.
# Wrist camera only, every frame in the heatmap_labels/ sidecar.
# Keep RUN_NAME stable across resubmissions of the SAME experiment so training
# resumes; bump it to start fresh (same caveats as start_finetune.sh).
RUN_NAME="${RUN_NAME:-perception_action_v1}"

# Dataset root as seen from the training cwd (/workspace in the container). Must
# contain the heatmap_labels/ sidecar — this model trains on its rows only,
# so a dataset without it (e.g. g1_dataset_red_2k) yields zero trainable steps.
# The submit script mounts the host dataset at this path.
DATASET_DIR="${DATASET_DIR:-./g1_dataset/}"

# --- Weight initialization ---
# The base is ALWAYS nvidia/GR00T-N1.7-3B; there is no flag for it. The model config
# is pinned to that checkpoint's architecture (32 DiT blocks, 4 VL self-attention
# layers, 16 LLM layers), so any other base would load partially and silently.
#
# PERCEPTION_MODEL_PATH is OPTIONAL and overlays a trained perception head + the
# backbone it was trained against. It is NOT free: perceptionv1 was trained at
# select_layer=12, so using it needs SELECT_LAYER=12 and gives up the foundation's
# top 4 LLM layers plus the match between GR00T's pretrained action stack and the
# features it now sees. Unset = the perception head trains from scratch on a fully
# coherent GR00T init, which teacher forcing makes safe (the conditioning token comes
# from the LABEL during training; only inference quality depends on the head).
PERCEPTION_MODEL_PATH="${PERCEPTION_MODEL_PATH:-}"
SELECT_LAYER="${SELECT_LAYER:-16}"

PERCEPTION_MODEL_ARG=()
if [ -n "${PERCEPTION_MODEL_PATH}" ]; then
    PERCEPTION_MODEL_ARG=(--perception-model-path "${PERCEPTION_MODEL_PATH}")
    if [ "${SELECT_LAYER}" != "12" ]; then
        echo "[train] NOTE: overlaying a perception checkpoint with SELECT_LAYER=${SELECT_LAYER}."
        echo "[train]       perceptionv1 was trained at 12; if this run aborts on a backbone"
        echo "[train]       coverage error, re-submit with SELECT_LAYER=12."
    fi
else
    PERCEPTION_MODEL_PATH="<none: perception head trains from scratch>"
fi

# --- Perception losses (same meaning as start_finetune_perception.sh) ---
HEATMAP_SIGMA_PX="${HEATMAP_SIGMA_PX:-8.0}"
HEATMAP_LOSS_WEIGHT="${HEATMAP_LOSS_WEIGHT:-1.0}"
OFFSET_LOSS_WEIGHT="${OFFSET_LOSS_WEIGHT:-1.0}"
VISIBILITY_LOSS_WEIGHT="${VISIBILITY_LOSS_WEIGHT:-0.5}"
DEPTH_LOSS_WEIGHT="${DEPTH_LOSS_WEIGHT:-0.1}"
WALL_NORMAL_LOSS_WEIGHT="${WALL_NORMAL_LOSS_WEIGHT:-0.1}"
# Weight of the WHOLE perception loss against the action loss.
PERCEPTION_LOSS_WEIGHT="${PERCEPTION_LOSS_WEIGHT:-1.0}"

# --- Teacher-forcing error model on the conditioning token ---
# MEASURED from the 1500_perception_planner closed-loop eval (the 1500-step
# perception checkpoint) with:
#   python scripts/calibrate_perception_conditioning.py \
#       --eval-dir vla_eval_results/1500_perception_planner --dataset-path ./g1_dataset
# That eval runs TWO models — the coarse one drives the motion, perception only
# fires at the parked accumulation stops — so the script isolates the parked,
# single-frame detections. RE-RUN IT whenever you change PERCEPTION_MODEL_PATH:
# the numbers are a property of the perception checkpoint, not of the task.
# Noise SMALLER than the real closed-loop error teaches the policy to trust
# perfect perception and it will be brittle on the robot.
COND_NOISE_UV_PX="${COND_NOISE_UV_PX:-3.0}"        # 0.85 px single-frame x3 (motion/far range)
COND_NOISE_DEPTH_M="${COND_NOISE_DEPTH_M:-0.004}"  # random part only: 4.32 mm spread
COND_BIAS_DEPTH_M="${COND_BIAS_DEPTH_M:-0.0}"      # 0 for a fresh head; 0.0065 when reusing perceptionv1
COND_NOISE_NORMAL="${COND_NOISE_NORMAL:-0.04}"     # rotation spread 3.49 deg
# Fraction of steps the policy is told "target not visible" even when it is, so it
# learns to hold/search instead of collapsing when perception drops out.
COND_VIS_DROPOUT_PROB="${COND_VIS_DROPOUT_PROB:-0.1}"

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
ACTION_HORIZON="${ACTION_HORIZON:-30}"

# --- Image resolution ---
# Wrist frames are natively 720x720, so 720 is 1:1 with the sensor (each Qwen visual
# token covers 32 native px). IMAGE_CROP_SIZE is NOT the output size: the pipeline
# crops that fraction and resizes back up to IMAGE_TARGET_SIZE, i.e. it is a zoom
# augmentation. Keep the ratio near 0.9 when changing the target size.
IMAGE_TARGET_SIZE="${IMAGE_TARGET_SIZE:-720}"
IMAGE_CROP_SIZE="${IMAGE_CROP_SIZE:-648}"

# --- Batch ---
# Effective batch = GLOBAL_BATCH_SIZE x GRAD_ACCUM_STEPS. GLOBAL_BATCH_SIZE is the
# total across GPUs. Lower than the perception-only run because this model also
# carries the DiT and a 30-step action chunk.
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"

# --- Training length ---
# NUM_EPOCHS sizes the run by the data and wins over MAX_STEPS. One epoch is counted
# over every frame represented in the heatmap_labels/ sidecar, not over episodes
# without labels.
NUM_EPOCHS="${NUM_EPOCHS:-}"
MAX_STEPS="${MAX_STEPS:-10000}"
LENGTH_ARGS=(--max-steps "${MAX_STEPS}")
if [ -n "${NUM_EPOCHS}" ]; then
    LENGTH_ARGS+=(--num-epochs "${NUM_EPOCHS}")
    # max_steps is resolved inside the launcher, so derive nothing from it here.
    SAVE_STEPS="${SAVE_STEPS:-1000}"
else
    SAVE_STEPS="${SAVE_STEPS:-$(( MAX_STEPS / 5 < 200 ? 200 : MAX_STEPS / 5 ))}"
fi

# Ablation switch: NO_PERCEPTION_CONDITIONING=1 trains the identical model with the
# perception head as a pure auxiliary task (no tokens reach the DiT). That is the
# baseline that shows whether the conditioning is what actually helps.
COND_ARGS=(--use-perception-conditioning)
[ "${NO_PERCEPTION_CONDITIONING:-0}" = "1" ] && COND_ARGS=(--no-use-perception-conditioning)

MASTER_PORT=$(shuf -i 10000-65000 -n 1)
OUTPUT_DIR="${GROOT_HOME:-/groot}/runs/${RUN_NAME}"

echo "[train] RUN_NAME=${RUN_NAME}"
echo "[train] DATASET_DIR=${DATASET_DIR}"
echo "[train] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[train] RESOLUTION target=${IMAGE_TARGET_SIZE} crop=${IMAGE_CROP_SIZE} (crop is a zoom aug; output is ${IMAGE_TARGET_SIZE}px)"
echo "[train] BATCH global=${GLOBAL_BATCH_SIZE} x accum=${GRAD_ACCUM_STEPS} -> effective $(( GLOBAL_BATCH_SIZE * GRAD_ACCUM_STEPS ))"
echo "[train] LENGTH ${LENGTH_ARGS[*]}  SAVE_STEPS=${SAVE_STEPS}"
echo "[train] BASE_MODEL_PATH=nvidia/GR00T-N1.7-3B (fixed)  SELECT_LAYER=${SELECT_LAYER}"
echo "[train] PERCEPTION_MODEL_PATH=${PERCEPTION_MODEL_PATH}  (perception head + backbone)"
echo "[train] COND_NOISE_UV_PX=${COND_NOISE_UV_PX} px  COND_BIAS_DEPTH_M=${COND_BIAS_DEPTH_M} m  (re-derive per perception ckpt: scripts/calibrate_perception_conditioning.py)"
echo "[train] CONDITIONING=${COND_ARGS[*]}"
echo "[train] MASTER_PORT=${MASTER_PORT}"
if [ -d "$OUTPUT_DIR" ]; then
    echo "[train] WARNING: output dir already exists. Existing checkpoints:"
    ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null || echo "[train]   (none)"
    echo "[train] If a checkpoint already hit --max-steps, training resumes and exits"
    echo "[train] immediately. Bump RUN_NAME to start fresh."
else
    echo "[train] Fresh run — no existing output dir."
fi

set +e
torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
    gr00t/experiment/launch_finetune_perception_action.py \
    "${PERCEPTION_MODEL_ARG[@]}" \
    --dataset-path "${DATASET_DIR}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path register_perception_action_modality.py \
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
    --action-horizon ${ACTION_HORIZON} \
    --select-layer ${SELECT_LAYER} \
    --learning-rate ${LEARNING_RATE} \
    --heatmap-target-sigma-px ${HEATMAP_SIGMA_PX} \
    --heatmap-loss-weight ${HEATMAP_LOSS_WEIGHT} \
    --offset-loss-weight ${OFFSET_LOSS_WEIGHT} \
    --visibility-loss-weight ${VISIBILITY_LOSS_WEIGHT} \
    --depth-loss-weight ${DEPTH_LOSS_WEIGHT} \
    --wall-normal-loss-weight ${WALL_NORMAL_LOSS_WEIGHT} \
    --perception-loss-weight ${PERCEPTION_LOSS_WEIGHT} \
    --perception-cond-noise-uv-px ${COND_NOISE_UV_PX} \
    --perception-cond-noise-depth-m ${COND_NOISE_DEPTH_M} \
    --perception-cond-noise-normal ${COND_NOISE_NORMAL} \
    --perception-cond-bias-depth-m ${COND_BIAS_DEPTH_M} \
    --perception-cond-vis-dropout-prob ${COND_VIS_DROPOUT_PROB} \
    "${COND_ARGS[@]}" \
    --num-gpus 2 \
    --use-wandb \
    --tune-visual 2>&1
TORCHRUN_EXIT=$?
set -e
echo "[train] torchrun exited with code: ${TORCHRUN_EXIT}"
exit "${TORCHRUN_EXIT}"
