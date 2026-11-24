# PLAN.md milestone 2 (the zero-shot gate): base N1.7 (DROID tag) on the candidate tasks,
# 32 episodes each, one job chain per task family:
#   for f in pick pnp open close; do FAMILY=$f scripts/rl/euler/submit.sh rl_experiments/pilot_zero_shot.sh; done
# Output per family: eval.json (per-task success) in results/groot-rl/pilot_zero_shot_<family>.
# Gate for A+RL: mean success >= 5% and successes on >= half the training tasks.
# 4 h jobs start much sooner than 24 h ones; the evaluator saves eval_partial.json as it
# goes and the next job in the chain resumes from it (same seeds, identical results).
# GPU: N1.7 plus ~1.1 GB per MuJoCo rendering worker. RTX Pro 6000 (96 GB, 80 on Euler)
# fits 20 workers: ~8.6 episodes/min, ~2.5 h per family. A 24 GB RTX 4090 fits ~9 (the
# worker count is capped automatically). Turing GPUs (Quadro RTX 6000, 2080 Ti) cannot run
# the VLM (FlashAttention-2 / bf16).
FAMILY="${FAMILY:?set FAMILY=pick|pnp|open|close (see splits/pilot_candidates.txt)}"
MODE=eval
RUN_NAME=pilot_zero_shot_$FAMILY
CHAIN=2
SBATCH_ARGS="--time=04:00:00 --gpus=rtx_pro_6000:1 --cpus-per-task=24 --mem-per-cpu=8G --tmp=100G"
ARGS=(
    --model-path nvidia/GR00T-N1.7-3B
    --tasks "$(grep -E "^$FAMILY/" rl_experiments/splits/pilot_candidates.txt)"
    --episodes-per-task 32
    --env.backend molmospaces --env.num-envs 20 --env.max-episode-steps 450
)
