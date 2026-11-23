#!/bin/bash
# =============================================================================
# submit.sh — submit an RL experiment preset to SLURM on Euler.
#
#   scripts/rl/euler/submit.sh rl_experiments/ogpo_arm_c.sh [--chain N] [--dry-run] [extra sbatch args]
#
# Reads the preset's RUN_NAME (job name), SBATCH_ARGS (e.g. a shorter --time) and CHAIN
# (number of chained 24 h jobs; --chain overrides). Chained jobs depend on the previous
# one with afterany, and each resumes the same run dir, so a run longer than the 24 h
# limit continues automatically; a job that finds the run finished exits immediately.
# Run from the repo root on Euler.
# =============================================================================
set -euo pipefail

PRESET="${1:?usage: submit.sh rl_experiments/<preset>.sh [--chain N] [--dry-run] [sbatch args...]}"
shift
[ -f "$PRESET" ] || { echo "preset $PRESET not found (run from the repo root)" >&2; exit 2; }

read -r RUN_NAME CHAIN_PRESET SBATCH_PRESET < <(
    bash -c 'set -e; MODE=""; RUN_NAME=""; CHAIN=1; SBATCH_ARGS=""; ARGS=(); source "$1";
             echo "$RUN_NAME ${CHAIN:-1} ${SBATCH_ARGS:-}"' _ "$PRESET"
)
CHAIN="$CHAIN_PRESET"
DRY=0
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --chain) CHAIN="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done
[ -n "$RUN_NAME" ] || { echo "$PRESET does not set RUN_NAME" >&2; exit 2; }
if grep -v '^\s*#' "$PRESET" | grep -q 'REPLACE_WITH'; then
    echo "$PRESET still contains a REPLACE_WITH_* placeholder (e.g. MODEL_PATH); edit it first." >&2
    exit 2
fi
for f in $(grep -o '@rl_experiments/[^ )"]*' "$PRESET" | sed 's/^@//' | sort -u); do
    if [ ! -f "$f" ] || [ -z "$(grep -v '^\s*#' "$f" | tr -d '[:space:]')" ]; then
        echo "$PRESET uses task list $f, which is missing or has no tasks yet." >&2
        exit 2
    fi
done

mkdir -p logs
prev=""
for i in $(seq 1 "$CHAIN"); do
    dep=()
    [ -n "$prev" ] && dep=(--dependency="afterany:$prev")
    # shellcheck disable=SC2206
    cmd=(sbatch --parsable --job-name="$RUN_NAME" ${SBATCH_PRESET} "${dep[@]}" "${EXTRA[@]}" submit_rl_euler.sh "$PRESET")
    if [ "$DRY" -eq 1 ]; then
        echo "${cmd[*]}"
        prev="DRYRUN$i"
    else
        prev="$("${cmd[@]}")"
        echo "submitted $RUN_NAME part $i/$CHAIN: job $prev"
    fi
done
echo "logs: logs/rl-$RUN_NAME-<jobid>.out   run dir: /cluster/scratch/\$USER/groot/rl_runs/$RUN_NAME"
