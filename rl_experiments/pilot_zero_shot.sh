# PLAN.md milestone 2 (the zero-shot gate): base N1.7 (DROID tag) on all candidate
# tasks, 32 episodes each. Output eval.json -> per-task success, used to fix the family
# split and decide whether to run A+RL (mean >= 5% and successes on >= half the tasks).
MODE=eval
RUN_NAME=pilot_zero_shot
CHAIN=1
ARGS=(
    --model-path nvidia/GR00T-N1.7-3B
    --tasks @rl_experiments/splits/pilot_candidates.txt
    --episodes-per-task 32
    --env.backend molmospaces --env.num-envs 24 --env.max-episode-steps 450
)
