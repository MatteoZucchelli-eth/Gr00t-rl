# Arm C: warm-started checkpoint B + OGPO on the 5 training families.
# Set MODEL_PATH to B's finetuning run (submit_euler.sh writes it to
# /cluster/scratch/$USER/groot/runs/<run>, visible here as /groot/runs/<run>) and fill
# splits/train.txt / splits/heldout.txt after the pilot. Resubmitting resumes.
MODE=train
RUN_NAME=ogpo_arm_c_seed0
CHAIN=6
MODEL_PATH=/groot/runs/REPLACE_WITH_B_RUN
ARGS=(
    --model-path "$MODEL_PATH"
    --algo ogpo --seed 0
    --wandb-group arm_c
    --collect.tasks @rl_experiments/splits/train.txt
    --collect.eval-tasks @rl_experiments/splits/train.txt @rl_experiments/splits/heldout.txt
    --env.backend molmospaces --env.num-envs 24 --env.max-episode-steps 450
    --num-workers 6
    --kill-switch-after-rounds 5 --kill-switch-min-gain 0.10
)
