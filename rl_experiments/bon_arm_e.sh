# Arm E: checkpoint B frozen, learned Q-critic (SARSA + V, as Korniak's BoN) with
# Best-of-32 during collection AND evaluation. Same splits as arm C.
MODE=train
RUN_NAME=bon_arm_e_seed0
CHAIN=3
MODEL_PATH=/groot/runs/REPLACE_WITH_B_RUN
ARGS=(
    --model-path "$MODEL_PATH"
    --algo bon --seed 0 --critic.bootstrap sarsa --collect.eval-best-of-n 32
    --wandb-group arm_e
    --steps-per-round 10000
    --collect.tasks @rl_experiments/splits/train.txt
    --collect.eval-tasks @rl_experiments/splits/train.txt @rl_experiments/splits/heldout.txt
    --env.backend molmospaces --env.num-envs 24 --env.max-episode-steps 450
    --num-workers 6
)
