#!/bin/bash
set -euo pipefail

source .env

# Output run name. Keep this STABLE across resubmissions of the *same* experiment
# so training resumes from the last checkpoint (each Euler job is time-limited).
# Bump it (e.g. red_and_green_v2) to start a FRESH experiment — reusing a name
# whose checkpoint already hit --max-steps makes the trainer exit immediately.
# Override per run with:  RUN_NAME=red_and_green_v2 sbatch submit_euler.sh
RUN_NAME="${RUN_NAME:-red_and_green}"

# Dataset root, as seen from the training cwd (/workspace in the container).
# g1_dataset_red_2k: 925 episodes / 837k frames carved out of g1_dataset_general_2k,
# keeping ONLY the red-target episodes (all 5 instruction phrasings). All three
# coloured squares are still physically in frame, so the instruction still has to
# select the target — but target-colour generalization is out of the picture, which
# isolates tip-target precision. Use this for the pooling ablation.
#   g1_dataset_general_2k/g1_dataset_general_2k  2763 eps, all 3 colours (superset)
#   g1_dataset_small_general                      747 eps, all 3 colours
#   g1_dataset                                   1515 eps, single phrasing, red only
# Override with DATASET_DIR=... to train on a different capture.
DATASET_DIR="${DATASET_DIR:-./g1_dataset_red_2k/}"

# Set to true to train the auxiliary tip-target head: predicts the target pose
# in the drill-tip frame as xyz(3) + rot6d(6) from the `next.tip_target_9d`
# label and feeds it back as a conditioning token to the action head
# (conditioning is on by default via --use-tip-target-conditioning).
USE_TIP_TARGET_HEAD=true
TIP_TARGET_LOSS_WEIGHT=0.5       # overall tip-target loss multiplier vs the action loss
TIP_TARGET_XYZ_SCALE=0.1         # metres; scales xyz inside the head's prediction space

# Objective: "mse" = one plain MSE over the scaled 9d (the original loss, and the
# only one measured to work). "rel_abs" = the distance-normalized position term
# plus an absolute anchor -- measured 5.7x worse in position error because
# dividing by distance makes far-range error nearly free and the model shrinks
# its predicted range toward the near-field mean. Do not use rel_abs without a
# reason; the four weights below are inert under "mse".
TIP_TARGET_LOSS_TYPE=mse
TIP_TARGET_POS_LOSS_WEIGHT=1.0   # rel_abs only: combined xyz term
TIP_TARGET_POS_ABS_WEIGHT=0.2    # rel_abs only: absolute MSE anchor
TIP_TARGET_ROT_LOSS_WEIGHT=1.0   # rel_abs only: rot6d MSE term
TIP_TARGET_DIST_FLOOR=0.05       # rel_abs only: metres, clamp for the normalization

# How the VL token sequence is pooled into the vector fed to the head's predictor
# MLP: "attn" (num_queries learned queries cross-attending over all vision +
# language tokens) or "mean" (masked mean). This is the ONLY thing that should
# differ between the two arms of the pooling ablation. The two are not
# weight-compatible, so give each arm its own RUN_NAME.
TIP_TARGET_POOL="${TIP_TARGET_POOL:-attn}"
TIP_TARGET_NUM_QUERIES=4         # attn only
TIP_TARGET_ATTN_HEADS=8          # attn only

TIP_TARGET_ARGS=""
[ "$USE_TIP_TARGET_HEAD" = true ] && TIP_TARGET_ARGS="--use-tip-target-head \
    --tip-target-loss-weight ${TIP_TARGET_LOSS_WEIGHT} \
    --tip-target-loss-type ${TIP_TARGET_LOSS_TYPE} \
    --tip-target-pool ${TIP_TARGET_POOL} \
    --tip-target-num-queries ${TIP_TARGET_NUM_QUERIES} \
    --tip-target-attn-heads ${TIP_TARGET_ATTN_HEADS} \
    --tip-target-pos-loss-weight ${TIP_TARGET_POS_LOSS_WEIGHT} \
    --tip-target-pos-abs-weight ${TIP_TARGET_POS_ABS_WEIGHT} \
    --tip-target-rot-loss-weight ${TIP_TARGET_ROT_LOSS_WEIGHT} \
    --tip-target-dist-floor ${TIP_TARGET_DIST_FLOOR} \
    --tip-target-xyz-scale ${TIP_TARGET_XYZ_SCALE}"

# Image resolution. These flags used to be INERT -- from_pretrained's override
# allowlist dropped image_target_size/image_crop_size/shortest_image_edge/
# crop_fraction, so every run before 2026-07-30 trained at the base model's 256px
# with crop_fraction 0.95 whatever these said. That is fixed; they now take effect,
# so they are PINNED to the values that reproduce the historical behaviour:
#   256 target with crop 243 -> crop fraction 243/256 = 0.949 ~= the old 0.95,
#   yielding 64 visual tokens per image and a 275-token sequence, as before.
# Raising this is a real experiment now, not a no-op. Reference points, measured:
#   target 384 crop 344 -> 144 tok/img, seq  595, ~54 native px per token
#   target 512 crop 460 -> 256 tok/img, seq 1043, ~40 native px per token
#   target 720 crop 648 -> 484 tok/img, seq 1955, ~30 native px per token
# The drill target is only ~25 px in the native 720px wrist frame, i.e. 0.30 of one
# token at 256 and still 0.86 at 720 -- so there is real headroom here, but change it
# on its own, not alongside another variable, and watch memory: visual tokens scale
# with area and the batch may need accumulation again.

# Batch. Effective batch = GLOBAL_BATCH_SIZE x GRAD_ACCUM_STEPS, and GLOBAL_BATCH_SIZE
# is split across the GPUs (32 over 2 GPUs = 16/GPU). Keep the product at 32 when
# trading one for the other, so the optimizer sees the same thing. Raising the image
# resolution multiplies visual tokens by (target/256)^2 -- 512px is 4x -- so this is
# the knob to drop first if a higher-resolution run OOMs:
#   GLOBAL_BATCH_SIZE=16 GRAD_ACCUM_STEPS=2   (same effective 32, half the activations)
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"

# Training length. Set NUM_EPOCHS to size the run by the dataset instead of by a step
# count: the launcher computes max_steps = ceil(epochs * trainable_steps / effective
# batch), where trainable_steps = sum(episode_length - action_horizon + 1). For
# g1_dataset_red_2k that is 810,321 steps, so at batch 32 one epoch = 25,323 steps.
#   NUM_EPOCHS=1 bash start_finetune.sh
# Leave NUM_EPOCHS empty to use MAX_STEPS verbatim.
# Note "epoch" is a sample-count target, not a true pass: batches are drawn from a
# shard buffer with replacement, so it means "each frame seen NUM_EPOCHS times in
# expectation". warmup_ratio scales with max_steps automatically; SAVE_STEPS does not,
# so it is derived below to give ~5 checkpoints either way.
NUM_EPOCHS="${NUM_EPOCHS:-}"
MAX_STEPS="${MAX_STEPS:-10000}"
if [ -n "$NUM_EPOCHS" ]; then
    TRAIN_LENGTH_ARGS="--num-epochs ${NUM_EPOCHS}"
    # ~810321 trainable steps for g1_dataset_red_2k; steps/epoch = that / effective batch.
    # Only a heuristic for checkpoint cadence -- the authoritative max_steps is computed
    # by launch_finetune.py from the dataset actually being used.
    SAVE_STEPS="${SAVE_STEPS:-$(python -c "import math;eff=${GLOBAL_BATCH_SIZE}*${GRAD_ACCUM_STEPS};print(max(200, math.ceil(${NUM_EPOCHS}*810321/eff/5/100)*100))")}"
else
    TRAIN_LENGTH_ARGS="--max-steps ${MAX_STEPS}"
    SAVE_STEPS="${SAVE_STEPS:-2000}"
fi
echo "[train] TRAIN_LENGTH_ARGS=${TRAIN_LENGTH_ARGS}  SAVE_STEPS=${SAVE_STEPS}"

IMAGE_TARGET_SIZE="${IMAGE_TARGET_SIZE:-256}"
IMAGE_CROP_SIZE="${IMAGE_CROP_SIZE:-243}"

# Generate a random port between 10000 and 65000 to avoid collisions
MASTER_PORT=$(shuf -i 10000-65000 -n 1)

# Each experiment gets its own output dir keyed on RUN_NAME so that bumping
# RUN_NAME actually starts a fresh run. Reusing a name whose checkpoint already
# reached --max-steps makes HF Trainer return immediately (resume_from_checkpoint
# sees global_step >= max_steps), which looks like an instant, error-free "success".
OUTPUT_DIR="${GROOT_HOME:-/groot}/runs/${RUN_NAME}"

# --- Debug: surface the resolved configuration (goes to stdout / the .out log) ---
echo "[train] RUN_NAME=${RUN_NAME}"
echo "[train] DATASET_DIR=${DATASET_DIR}"
echo "[train] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[train] MASTER_PORT=${MASTER_PORT}"
echo "[train] TIP_TARGET_ARGS=${TIP_TARGET_ARGS:-<none>}"
echo "[train] GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS} (effective $((GLOBAL_BATCH_SIZE*GRAD_ACCUM_STEPS)))"
echo "[train] IMAGE_TARGET_SIZE=${IMAGE_TARGET_SIZE} IMAGE_CROP_SIZE=${IMAGE_CROP_SIZE}"
echo "[train]   -> requested crop fraction $(python -c "print(f'{${IMAGE_CROP_SIZE}/${IMAGE_TARGET_SIZE}:.3f}')"), ~$(python -c "print((${IMAGE_TARGET_SIZE}//32)**2)") visual tokens/image"
echo "[train]   (grep the training log for '[resolution]' to see what the processor ACTUALLY used)"
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
    --dataset-path "${DATASET_DIR}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path register_new_modality.py \
    --output-dir "${OUTPUT_DIR}" \
    --save-total-limit 5 \
    --save-steps ${SAVE_STEPS} \
    ${TRAIN_LENGTH_ARGS} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --gradient-accumulation-steps ${GRAD_ACCUM_STEPS} \
    --dataloader-num-workers 4 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.03 \
    --image-target-size ${IMAGE_TARGET_SIZE} \
    --image-crop-size ${IMAGE_CROP_SIZE} \
    --num-gpus 2 \
    --use-wandb \
    --tune-visual \
    $TIP_TARGET_ARGS 2>&1
TORCHRUN_EXIT=$?
set -e
echo "[train] torchrun exited with code: ${TORCHRUN_EXIT}"
exit "${TORCHRUN_EXIT}"
