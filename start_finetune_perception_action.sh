#!/bin/bash
set -euo pipefail

source .env

# Fine-approach PERCEPTION-CONDITIONED ACTION model: GR00T VLM backbone +
# target heatmap head + flow-matching action head, where the perception output is
# cast into conditioning tokens the DiT attends to.
# Wrist camera only, fine-phase frames only (heatmap_labels/ sidecar).
# Keep RUN_NAME stable across resubmissions of the SAME experiment so training
# resumes; bump it to start fresh (same caveats as start_finetune.sh).
RUN_NAME="${RUN_NAME:-perception_action_v1}"

# --- Weight initialization: TWO overlays, they do different jobs ---
# BASE_MODEL_PATH       -> the ACTION HEAD (trained DiT). One of your VLA runs, or
#                          nvidia/GR00T-N1.7-3B. A fresh DiT costs thousands of steps.
# PERCEPTION_MODEL_PATH -> the trained PERCEPTION HEAD *and* the backbone it was
#                          trained against (this backbone wins over the base one —
#                          the head's features only mean anything on it).
# Set both for the best init. Omitting PERCEPTION_MODEL_PATH means a fresh
# perception head, i.e. relearning localization from scratch alongside the actions.
BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}"
PERCEPTION_MODEL_PATH="${PERCEPTION_MODEL_PATH:-}"

BASE_MODEL_ARG=(--base-model-path "${BASE_MODEL_PATH}")
case "${BASE_MODEL_PATH,,}" in
    ""|scratch|none|fresh)
        BASE_MODEL_ARG=()
        BASE_MODEL_PATH="<none: raw Cosmos-Reason2-2B backbone + fresh action head>"
        ;;
esac

PERCEPTION_MODEL_ARG=()
if [ -n "${PERCEPTION_MODEL_PATH}" ]; then
    PERCEPTION_MODEL_ARG=(--perception-model-path "${PERCEPTION_MODEL_PATH}")
else
    PERCEPTION_MODEL_PATH="<none: fresh perception head>"
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
COND_BIAS_DEPTH_M="${COND_BIAS_DEPTH_M:-0.0065}"   # SYSTEMATIC +6.49 mm: 84% of the depth error
COND_NOISE_NORMAL="${COND_NOISE_NORMAL:-0.04}"     # rotation spread 3.49 deg
# Fraction of steps the policy is told "target not visible" even when it is, so it
# learns to hold/search instead of collapsing when perception drops out.
COND_VIS_DROPOUT_PROB="${COND_VIS_DROPOUT_PROB:-0.1}"

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
ACTION_HORIZON="${ACTION_HORIZON:-30}"

# Ablation switch: NO_PERCEPTION_CONDITIONING=1 trains the identical model with the
# perception head as a pure auxiliary task (no tokens reach the DiT). That is the
# baseline that shows whether the conditioning is what actually helps.
COND_ARGS=(--use-perception-conditioning)
[ "${NO_PERCEPTION_CONDITIONING:-0}" = "1" ] && COND_ARGS=(--no-use-perception-conditioning)

MASTER_PORT=$(shuf -i 10000-65000 -n 1)
OUTPUT_DIR="${GROOT_HOME:-/groot}/runs/${RUN_NAME}"

echo "[train] RUN_NAME=${RUN_NAME}"
echo "[train] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[train] BASE_MODEL_PATH=${BASE_MODEL_PATH}            (action head)"
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
    "${BASE_MODEL_ARG[@]}" \
    "${PERCEPTION_MODEL_ARG[@]}" \
    --dataset-path ./g1_dataset/ \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path register_perception_action_modality.py \
    --output-dir "${OUTPUT_DIR}" \
    --save-total-limit 5 \
    --save-steps 2000 \
    --max-steps 10000 \
    --global-batch-size 8 \
    --gradient-accumulation-steps 4 \
    --dataloader-num-workers 4 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.25 hue 0.02 \
    --image-target-size 720 \
    --image-crop-size 648 \
    --action-horizon ${ACTION_HORIZON} \
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
