#!/bin/bash
# =============================================================================
# entrypoint.sh — runs INSIDE the gr00t-rl container (Apptainer on Euler, or Docker).
# Called by submit_rl_euler.sh as:  entrypoint.sh <MODE> [args forwarded to the tool]
#
# Modes:
#   smoke        unit tests + GPU smoke test on the real checkpoint + a 2-round OGPO run
#                with fake env workers + a MolmoSpaces space check (~15-30 min)
#   assets       install MolmoSpaces resources; with --tasks also prefetch their scenes
#   space_check  scripts/rl/check_droid_spaces.py   (args forwarded)
#   list_tasks   scripts/rl/list_molmospaces_tasks.py (args forwarded)
#   eval         python -m gr00t.rl.evaluate --output-dir $RL_RUN_DIR <args>
#   train        python -m gr00t.rl.train    --output-dir $RL_RUN_DIR <args>
#
# Expected environment (set by submit_rl_euler.sh): RL_RUN_DIR (writable run dir),
# RL_RESULTS_DIR (persistent results), HF_HOME, MLSPACES_ASSETS_DIR, MLSPACES_CACHE_DIR.
# Secrets come from /workspace/.env (HF_TOKEN, WANDB_API_KEY).
# =============================================================================
set -euo pipefail

# The whole body lives in main(), which bash parses completely before running it. Bash
# otherwise reads a script lazily from disk, and on Euler's NFS home a sync that replaces
# this file mid-job leaves the running bash with a stale handle ("error reading input
# file: Stale file handle"), killing the job after its current step.
main() {
MODE="${1:?usage: entrypoint.sh <smoke|assets|space_check|list_tasks|eval|train> [args...]}"
shift

cd /workspace
if [ -f /workspace/.env ]; then
    set -a
    # shellcheck disable=SC1091
    source /workspace/.env
    set +a
fi

export GROOT_PATCH_MISTRAL=1 GROOT_HF_LOCAL_FIRST=1 NO_ALBUMENTATIONS_UPDATE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
# apptainer --nv binds the HOST's GLVND dispatch libs (libEGL.so.1, libGLdispatch.so.0, ...)
# ahead of the image's; if the host glibc is newer than the image's they fail to load
# ("GLIBC_2.38 not found"). Use the image's dispatchers and only the host's NVIDIA vendor
# libraries (libEGL_nvidia.so.0, found via /.singularity.d/libs).
if [ -d /.singularity.d/libs ]; then
    GLVND_DIR="$(mktemp -d /tmp/glvnd.XXXXXX)"
    for lib in libEGL.so.1 libGLdispatch.so.0 libGL.so.1 libGLX.so.0 libOpenGL.so.0 libGLESv2.so.2; do
        [ -e "/usr/lib/x86_64-linux-gnu/$lib" ] && ln -s "/usr/lib/x86_64-linux-gnu/$lib" "$GLVND_DIR/$lib"
    done
    export LD_LIBRARY_PATH="$GLVND_DIR:${LD_LIBRARY_PATH:-}"
fi
export PYTHONUNBUFFERED=1
# Euler mounts $HOME read-only inside jobs; give every tool that defaults to ~ (tests'
# ~/.cache/g00t, wandb, triton, matplotlib, ...) a writable home on scratch.
# GROOT_HOME (from .env, = /groot, the scratch bind) is the writable root, as in submit_euler.sh.
GROOT_HOME="${GROOT_HOME:-/groot}"
if [ -d "$GROOT_HOME" ] && ! ( mkdir -p "$HOME/.cache" && touch "$HOME/.cache/.rw_test" ) 2>/dev/null; then
    export HOME="$GROOT_HOME/home"
    mkdir -p "$HOME"
    echo "[rl] \$HOME is read-only here; using HOME=$HOME"
fi
export TEST_CACHE_PATH="${TEST_CACHE_PATH:-${XDG_CACHE_HOME:-$HOME/.cache}/g00t_tests}"
# multiprocessing (DataLoader workers) puts Unix sockets under $TMPDIR; Linux caps socket
# paths at 107 bytes, so fall back to /tmp when the inherited host path is long.
_tmp="${TMPDIR:-}"; if [ "${#_tmp}" -gt 60 ]; then export TMPDIR=/tmp; fi
RL_RUN_DIR="${RL_RUN_DIR:-/tmp/rl_run}"
RL_RESULTS_DIR="${RL_RESULTS_DIR:-}"
mkdir -p "$RL_RUN_DIR"
PY=/workspace/.venv/bin/python
RL_RUN_NAME="${RL_RUN_NAME:-$(basename "$RL_RUN_DIR")}"
# wandb: on whenever .env provides WANDB_API_KEY (set WANDB_MODE=disabled to opt out).
# The run id is RL_RUN_NAME, so chained jobs of one run continue the same wandb run.
WANDB_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-}" != "disabled" ]; then
    WANDB_ARGS=(--use-wandb)
    echo "[rl] wandb enabled (run id: $RL_RUN_NAME)"
else
    echo "[rl] wandb disabled (no WANDB_API_KEY in .env, or WANDB_MODE=disabled)"
fi
SIM_PY="${RL_MOLMOSPACES_PYTHON:-gr00t/eval/sim/MolmoSpaces/molmospaces_uv/.venv/bin/python}"

echo "[rl] mode=$MODE run_dir=$RL_RUN_DIR results=${RL_RESULTS_DIR:-<none>} host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

base_model_dir() {
    # Local snapshot of the base checkpoint (downloads on first use).
    $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('nvidia/GR00T-N1.7-3B'))"
}

publish() {
    # Copy the small, final artifacts of a run to persistent storage.
    [ -n "$RL_RESULTS_DIR" ] || return 0
    mkdir -p "$RL_RESULTS_DIR"
    for f in config.json metrics.jsonl final_eval.json eval.json; do
        [ -f "$RL_RUN_DIR/$f" ] && cp "$RL_RUN_DIR/$f" "$RL_RESULTS_DIR/"
    done
    if [ -d "$RL_RUN_DIR/final_policy" ]; then
        rm -rf "$RL_RESULTS_DIR/final_policy"
        cp -r "$RL_RUN_DIR/final_policy" "$RL_RESULTS_DIR/final_policy"
    fi
    echo "[rl] published results to $RL_RESULTS_DIR"
}

case "$MODE" in
smoke)
    echo "[rl] 1/4 unit tests"
    $PY -m pytest tests/gr00t/rl -q -p no:cacheprovider
    M="$(base_model_dir)"
    echo "[rl] 2/4 GPU smoke test ($M)"
    $PY scripts/rl/smoke_test_policy.py --model-path "$M" --group 8 --trainable-regex "${SMOKE_TRAINABLE_REGEX:-}"
    echo "[rl] 3/4 OGPO end-to-end with fake env workers"
    $PY -m gr00t.rl.train --model-path "$M" --output-dir "$RL_RUN_DIR/fake_ogpo" --no-resume \
        --rounds 2 --steps-per-round 3 --buffer-capacity 5000 --num-workers 2 --encode-chunk 8 \
        --prefix-batch 8 --log-every 1 \
        --env.backend fake --env.num-envs 2 --env.max-episode-steps 24 --env.base-port 6300 \
        --env.backend-kwargs-json '{"close_steps": 2, "success_prob": 0.06, "image_size": [90, 160]}' \
        --collect.tasks a b --collect.episodes-per-task 2 --collect.eval-episodes-per-task 1 \
        --collect.best-of-n 4 --critic.horizon 8 --critic.batch-size 16 --critic.num-vr 2 \
        --actor.batch-states 4 --actor.group-size 4 --actor.micro-batch 8 --actor.bc-batch 4 \
        "${WANDB_ARGS[@]}" --wandb-project groot-rl-smoke --run-name "smoke_${SLURM_JOB_ID:-local}" \
        ${SMOKE_TRAINABLE_REGEX:+--actor.trainable-regex "$SMOKE_TRAINABLE_REGEX"}
    echo "[rl] 4/4 MolmoSpaces space check + 2-episode zero-shot eval"
    $SIM_PY -c "from molmo_spaces.molmo_spaces_constants import get_resource_manager; get_resource_manager()"
    $SIM_PY scripts/rl/check_droid_spaces.py --statistics "$M/statistics.json" \
        --tasks pick/0 pick/5 --hold-steps 2 --out-dir "$RL_RUN_DIR/space_check"
    $PY -m gr00t.rl.evaluate --model-path "$M" --tasks pick/5 pick/40 --episodes-per-task 2 \
        --env.backend molmospaces --env.num-envs 2 --env.max-episode-steps 150 --env.base-port 6400 \
        --output-dir "$RL_RUN_DIR/zero_shot"
    echo "[rl] SMOKE PASSED"
    ;;
assets)
    $SIM_PY scripts/rl/prefetch_molmospaces.py --done-file "$RL_RUN_DIR/prefetch_done.txt" "$@"
    ;;
space_check)
    M="$(base_model_dir)"
    $SIM_PY scripts/rl/check_droid_spaces.py --statistics "$M/statistics.json" \
        --out-dir "$RL_RUN_DIR/space_check" "$@"
    ;;
list_tasks)
    $SIM_PY scripts/rl/list_molmospaces_tasks.py --write "$RL_RUN_DIR/task_lists" "$@"
    publish
    [ -n "$RL_RESULTS_DIR" ] && cp -r "$RL_RUN_DIR/task_lists" "$RL_RESULTS_DIR/" || true
    ;;
eval)
    $PY -m gr00t.rl.evaluate --output-dir "$RL_RUN_DIR" --run-name "$RL_RUN_NAME" "${WANDB_ARGS[@]}" "$@"
    publish
    ;;
train)
    if [ -f "$RL_RUN_DIR/final_eval.json" ]; then
        echo "[rl] $RL_RUN_DIR/final_eval.json exists: run already finished, nothing to do."
        publish
        exit 0
    fi
    $PY -m gr00t.rl.train --output-dir "$RL_RUN_DIR" --run-name "$RL_RUN_NAME" "${WANDB_ARGS[@]}" "$@"
    publish
    ;;
*)
    echo "[rl] unknown mode '$MODE'" >&2
    exit 2
    ;;
esac
}

main "$@"
exit $?
