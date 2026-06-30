#!/bin/bash

source .env

# Set to true to train the auxiliary task-progress head, false for plain policy.
USE_PROGRESS_HEAD=true

PROGRESS_ARGS=""
[ "$USE_PROGRESS_HEAD" = true ] && PROGRESS_ARGS="--use-progress-head"

# Generate a random port between 10000 and 65000 to avoid collisions
MASTER_PORT=$(shuf -i 10000-65000 -n 1)

torchrun --nproc_per_node=2 --master_port=$MASTER_PORT gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path ./g1_dataset/ \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path register_new_modality.py \
    --output-dir "${GROOT_HOME:-/groot}/runs/red_and_green" \
    --save-total-limit 5 \
    --save-steps 2000 \
    --max-steps 10000 \
    --global-batch-size 32 \
    --dataloader-num-workers 4 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --num-gpus 2 \
    --use-wandb \
    --tune-visual \
    $PROGRESS_ARGS
