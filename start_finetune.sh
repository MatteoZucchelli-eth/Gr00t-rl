#!/bin/bash
set -euo pipefail
set -euo pipefail

source .env

# Output run name. Keep this STABLE across resubmissions of the *same* experiment
# so training resumes from the last checkpoint (each Euler job is time-limited).
# Bump it (e.g. red_and_green_v2) to start a FRESH experiment — reusing a name
# whose checkpoint already hit --max-steps makes the trainer exit immediately.
# Override per run with:  RUN_NAME=red_and_green_v2 sbatch submit_euler.sh
RUN_NAME="${RUN_NAME:-red_and_green}"

# Set to true to train the auxiliary task-progress head, false for plain policy.
USE_PROGRESS_HEAD=true

PROGRESS_ARGS=""
[ "$USE_PROGRESS_HEAD" = true ] && PROGRESS_ARGS="--use-progress-head"

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
echo "[train] PROGRESS_ARGS=${PROGRESS_ARGS:-<none>}"
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
# Each experiment gets its own output dir keyed on RUN_NAME so that bumping
# RUN_NAME actually starts a fresh run. Reusing a name whose checkpoint already
# reached --max-steps makes HF Trainer return immediately (resume_from_checkpoint
# sees global_step >= max_steps), which looks like an instant, error-free "success".
OUTPUT_DIR="${GROOT_HOME:-/groot}/runs/${RUN_NAME}"

# --- Debug: surface the resolved configuration (goes to stdout / the .out log) ---
echo "[train] RUN_NAME=${RUN_NAME}"
echo "[train] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[train] MASTER_PORT=${MASTER_PORT}"
echo "[train] PROGRESS_ARGS=${PROGRESS_ARGS:-<none>}"
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
    --output-dir "${OUTPUT_DIR}" \
    --save-total-limit 5 \
    --save-steps 2000 \
    --max-steps 10000 \
    --global-batch-size 32 \
    --dataloader-num-workers 4 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --image-target-size 384 \
    --image-crop-size 344 \
    --num-gpus 2 \
    --use-wandb \
    --tune-visual \
    $PROGRESS_ARGS 2>&1
TORCHRUN_EXIT=$?
set -e
echo "[train] torchrun exited with code: ${TORCHRUN_EXIT}"
exit "${TORCHRUN_EXIT}"
    $PROGRESS_ARGS 2>&1
TORCHRUN_EXIT=$?
set -e
echo "[train] torchrun exited with code: ${TORCHRUN_EXIT}"
exit "${TORCHRUN_EXIT}"
