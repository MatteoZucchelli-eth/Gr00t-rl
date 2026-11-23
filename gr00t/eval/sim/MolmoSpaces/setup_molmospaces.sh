#!/usr/bin/env bash
# Create the MolmoSpaces simulator venv used by RL env workers (gr00t.rl.envs.server).
#
# MolmoSpaces pins numpy>=2.2 / jax / mujoco 3.4, which conflict with the model stack, so
# it lives in its own venv (like LIBERO's). The learner launches workers with
#   --env.python gr00t/eval/sim/MolmoSpaces/molmospaces_uv/.venv/bin/python
#
# We install the MolmoSpaces fork shipped with Korniak et al. 2026
# (github.com/lasgroup/generalist-improvement, MIT): it adds the initial-state
# randomisation options their (and our) env wrapper uses.
#
# Run from inside the gr00t container (so the venv's interpreter path is valid there):
#   bash gr00t/eval/sim/MolmoSpaces/setup_molmospaces.sh
# Env vars: MLSPACES_ASSETS_DIR / MLSPACES_CACHE_DIR (asset download locations),
#           PYTHON (interpreter for the venv, default /usr/bin/python3.10).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_REPO="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
VENV_DIR="$SCRIPT_DIR/molmospaces_uv/.venv"
REF_DIR="$PROJECT_REPO/external_dependencies/generalist-improvement"
REF_URL="https://github.com/lasgroup/generalist-improvement.git"
REF_SHA="b2886525ccab025bc5f3a5a06fa262e702b5b0f6"
PYTHON="${PYTHON:-/usr/bin/python3.10}"

if [ ! -d "$REF_DIR/.git" ]; then
    git clone --filter=blob:none "$REF_URL" "$REF_DIR"
fi
git -c safe.directory='*' -C "$REF_DIR" fetch --quiet origin "$REF_SHA" || true
git -c safe.directory='*' -C "$REF_DIR" checkout --quiet "$REF_SHA"

rm -rf "$VENV_DIR"
mkdir -p "$(dirname "$VENV_DIR")"
uv venv "$VENV_DIR" --python "$PYTHON"
export VIRTUAL_ENV="$VENV_DIR"
uv pip install --python "$VENV_DIR/bin/python" -e "$REF_DIR/molmospaces"
# molmo_spaces.evaluation.configs imports openpi_client at module level (baseline policies).
uv pip install --python "$VENV_DIR/bin/python" -e "$REF_DIR/openpi/packages/openpi-client"
uv pip install --python "$VENV_DIR/bin/python" pyzmq msgpack==1.1.0 msgpack-numpy==0.4.8 pillow
# Env-server modules only (gr00t/rl/envs imports numpy, zmq, msgpack, PIL).
uv pip install --python "$VENV_DIR/bin/python" -e "$PROJECT_REPO" --no-deps
# MolmoSpaces downloads these WordNet corpora on import; parallel env workers would race
# on the download. Fetch them once into <venv>/nltk_data, which NLTK searches by default.
"$VENV_DIR/bin/python" -m nltk.downloader -q -d "$VENV_DIR/nltk_data" wordnet wordnet2022

"$VENV_DIR/bin/python" - <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco, molmo_spaces
from gr00t.rl.envs.molmospaces import tcp_pose_to_eef_9d  # noqa: F401
print("mujoco", mujoco.__version__, "| molmo_spaces at", os.path.dirname(molmo_spaces.__file__))
PY
echo "MolmoSpaces venv ready: $VENV_DIR/bin/python"
echo "Assets download lazily on first use (MLSPACES_ASSETS_DIR=${MLSPACES_ASSETS_DIR:-<package>/assets})."
