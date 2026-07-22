#!/bin/bash
set -euo pipefail

source .env

# Output run name. Keep this STABLE across resubmissions of the *same* experiment
# so training resumes from the last checkpoint (each Euler job is time-limited).
# Bump it (e.g. red_and_green_v2) to start a FRESH experiment — reusing a name
# whose checkpoint already hit --max-steps makes the trainer exit immediately.
# Override per run with:  RUN_NAME=red_and_green_v2 sbatch submit_euler.sh
RUN_NAME="${RUN_NAME:-red_and_green}"

# Set to true to train the auxiliary tip-target head: predicts the target pose
# in the drill-tip frame as xyz(3) + rot6d(6) from the `next.tip_target_9d`
# label and feeds it back as a conditioning token to the action head
# (conditioning is on by default via --use-tip-target-conditioning).
USE_TIP_TARGET_HEAD=true
TIP_TARGET_LOSS_WEIGHT=0.5       # overall tip-target loss multiplier vs the action loss
TIP_TARGET_POS_LOSS_WEIGHT=1.0   # distance-normalized xyz term (relative error)
TIP_TARGET_ROT_LOSS_WEIGHT=1.0   # rot6d MSE term
TIP_TARGET_DIST_FLOOR=0.02       # metres; clamp for the xyz loss normalization
TIP_TARGET_XYZ_SCALE=0.1         # metres; scales xyz inside the head's prediction space

TIP_TARGET_ARGS=""
[ "$USE_TIP_TARGET_HEAD" = true ] && TIP_TARGET_ARGS="--use-tip-target-head \
    --tip-target-loss-weight ${TIP_TARGET_LOSS_WEIGHT} \
    --tip-target-pos-loss-weight ${TIP_TARGET_POS_LOSS_WEIGHT} \
    --tip-target-rot-loss-weight ${TIP_TARGET_ROT_LOSS_WEIGHT} \
    --tip-target-dist-floor ${TIP_TARGET_DIST_FLOOR} \
    --tip-target-xyz-scale ${TIP_TARGET_XYZ_SCALE}"

# Image resolution: the dataset videos are native 720x720; feeding them at full
# resolution roughly halves the mm-per-pixel the tip-target head has to work
# with vs the old 384/344. Visual tokens scale with area (~4.4x vs 344), so the
# per-GPU batch is dropped to 4 (global 8 / 2 GPUs) with 4 accumulation steps to
# keep the effective batch at 32 (= old 32 x 1). If it still OOMs, fall back to
# --image-target-size 512 --image-crop-size 460 before touching the batch math.

# Generate a random port between 10000 and 65000 to avoid collisions
MASTER_PORT=$(shuf -i 10000-65000 -n 1)

# Each experiment gets its own output dir keyed on RUN_NAME so that bumping
# RUN_NAME actually starts a fresh run. Reusing a name whose checkpoint already
# reached --max-steps makes HF Trainer return immediately (resume_from_checkpoint
# sees global_step >= max_steps), which looks like an instant, error-free "success".
OUTPUT_DIR="${GROOT_HOME:-/groot}/runs/${RUN_NAME}"

# --- Debug: surface the resolved configuration (goes to stdout / the .out log) ---
echo "[train] RUN_NAME=${RUN_NAME}"
echo "[train] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[train] MASTER_PORT=${MASTER_PORT}"
echo "[train] TIP_TARGET_ARGS=${TIP_TARGET_ARGS:-<none>}"
if [ -d "$OUTPUT_DIR" ]; then
    echo "[train] WARNING: output dir already exists. Existing checkpoints:"
    ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null || echo "[train]   (none)"
    echo "[train] If a checkpoint already hit --max-steps, training will resume-and-exit"
    echo "[train] immediately. Bump RUN_NAME to start fresh."
else
    echo "[train] Fresh run — no existing output dir."
fi
# --- Debug: verify the environment can actually run the entry point ---
echo "[train] cwd=$(pwd)"
echo "[train] which torchrun: $(command -v torchrun || echo MISSING)"
echo "[train] which python:   $(command -v python || echo MISSING)"
echo "[train] entry exists:   $(ls -l gr00t/experiment/launch_finetune.py 2>&1)"
echo "[train] gr00t import:   $(python -c 'import gr00t, torch; print("gr00t ok, torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())' 2>&1)"
echo "[train] Launching torchrun..."

# Run torchrun with stderr merged into stdout so EVERYTHING lands in the .out log,
# and capture the exit code instead of letting `set -e` swallow it silently.
set +e
torchrun --nproc_per_node=2 --master_port=$MASTER_PORT gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path ./g1_dataset/ \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path register_new_modality.py \
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
    --num-gpus 2 \
    --use-wandb \
    --tune-visual \
    $TIP_TARGET_ARGS 2>&1
TORCHRUN_EXIT=$?
set -e
echo "[train] torchrun exited with code: ${TORCHRUN_EXIT}"
exit "${TORCHRUN_EXIT}"
