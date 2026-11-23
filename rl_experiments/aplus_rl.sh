# A+RL (conditional on the zero-shot gate): base N1.7 + OGPO, no warm-start SFT.
MODE=train
RUN_NAME=aplus_rl_seed0
CHAIN=6
ARGS=(
    --model-path nvidia/GR00T-N1.7-3B
    --algo ogpo --seed 0
    --collect.tasks @rl_experiments/splits/train.txt
    --collect.eval-tasks @rl_experiments/splits/train.txt @rl_experiments/splits/heldout.txt
    --env.backend molmospaces --env.num-envs 24 --env.max-episode-steps 450
    --num-workers 6
)
