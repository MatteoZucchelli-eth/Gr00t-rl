# Smoke test on Euler: unit tests, GPU checks on N1.7 (full action head trainable),
# OGPO end-to-end with fake envs, MolmoSpaces space check + 2-episode zero-shot eval.
#   scripts/rl/euler/submit.sh rl_experiments/smoke.sh
MODE=smoke
RUN_NAME=smoke
SBATCH_ARGS="--time=02:00:00 --cpus-per-task=8"
ARGS=()
