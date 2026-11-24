# Smoke test on Euler: unit tests, GPU checks on N1.7, OGPO end-to-end with fake envs
# (logs to wandb project groot-rl-smoke), MolmoSpaces space check + 2-episode zero-shot eval.
#   scripts/rl/euler/submit.sh rl_experiments/smoke.sh
# Runs on an RTX 4090 (24 GB): training only the action decoder keeps the OGPO step at
# ~7 GB. The full action head (1.6 B params, ~26 GB with Adam) needs an A100 80 GB:
#   scripts/rl/euler/submit.sh rl_experiments/smoke.sh --gpus=a100_80gb:1   (and drop EXTRA_ENV)
MODE=smoke
RUN_NAME=smoke
SBATCH_ARGS="--time=01:30:00 --cpus-per-task=8 --gpus=rtx_4090:1 --tmp=150G"
EXTRA_ENV="--env SMOKE_TRAINABLE_REGEX=action_decoder"
ARGS=()
