#!/bin/bash
# =============================================================================
# fetch_run.sh — pull the results of an RL run from Euler.
#
#   scripts/rl/euler/fetch_run.sh <run-name> [dest-dir]        # results only (small)
#   scripts/rl/euler/fetch_run.sh <run-name> [dest-dir] --policy  # + final_policy weights
#
# Results (metrics.jsonl, final_eval.json / eval.json, config.json, task_lists/) come
# from /cluster/project/mrl/$EULER_USER/results/groot-rl/<run>. The final policy is a
# normal GR00T checkpoint (loadable by Gr00tPolicy / launch_finetune --base-model-path).
# Env: EULER_HOST (euler), EULER_USER ($USER).
# =============================================================================
set -euo pipefail

RUN="${1:?usage: fetch_run.sh <run-name> [dest-dir] [--policy]}"
DEST="${2:-./rl_results/$RUN}"
POLICY=0
for a in "$@"; do [ "$a" = "--policy" ] && POLICY=1; done
[ "$DEST" = "--policy" ] && DEST="./rl_results/$RUN"
EULER_USER="${EULER_USER:-$USER}"
SRC="/cluster/project/mrl/$EULER_USER/results/groot-rl/$RUN"

mkdir -p "$DEST"
EXCLUDES=()
[ "$POLICY" -eq 1 ] || EXCLUDES=(--exclude='final_policy')
# shellcheck source=_ssh.sh
source "$(dirname "${BASH_SOURCE[0]}")/_ssh.sh"
euler_connect
euler_rsync -avP "${EXCLUDES[@]}" "$EULER_HOST:$SRC/" "$DEST/"
echo "[fetch] $RUN -> $DEST"
for f in final_eval.json eval.json; do
    if [ -f "$DEST/$f" ]; then
        python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print({k: v for k, v in d.items() if not isinstance(v, dict)})" "$DEST/$f"
    fi
done
