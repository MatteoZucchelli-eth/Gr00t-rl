# Episode counts + per-episode scene/object listing of the Franka benchmarks; writes
# task lists to results/groot-rl/list_tasks/task_lists/<benchmark>.txt.
MODE=list_tasks
RUN_NAME=list_tasks
SBATCH_ARGS="--time=01:00:00 --cpus-per-task=4"
ARGS=(--detail)
