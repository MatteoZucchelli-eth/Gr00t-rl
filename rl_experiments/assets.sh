# Install MolmoSpaces resources and prefetch the scenes of the pilot candidates, so
# later jobs do not download concurrently. (Skip if you ran sync.sh --assets and the
# tasks you need are already local.)
MODE=assets
RUN_NAME=assets
SBATCH_ARGS="--time=08:00:00 --cpus-per-task=4"
ARGS=(--tasks @rl_experiments/splits/pilot_candidates.txt)
