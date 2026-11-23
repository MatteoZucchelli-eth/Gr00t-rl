# RL image: the gr00t image + simulator venvs for RL env workers (gr00t.rl.envs.server).
#
#   docker build -f docker/Dockerfile.rl -t gr00t-rl .        (after building `gr00t`)
#   or simply: bash scripts/rl/euler/build_container.sh
#
# Layout added on top of `gr00t`:
#   /opt/sim/molmospaces/.venv   MolmoSpaces (Korniak et al.'s fork) + WordNet data
#   /opt/sim/libero/.venv        LIBERO + robosuite (pipeline sanity check)
#   /opt/rl-deps/                the cloned simulator sources
# Both venvs install gr00t editable from /workspace, so code bind-mounted over
# /workspace/gr00t at run time (Euler) is what the workers import.
FROM gr00t:latest
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 libgl1 libglvnd0 libosmesa6 libglew-dev libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300

# --- MolmoSpaces -------------------------------------------------------------------
ARG MOLMOSPACES_REF_SHA=b2886525ccab025bc5f3a5a06fa262e702b5b0f6
RUN git clone --filter=blob:none https://github.com/lasgroup/generalist-improvement.git \
        /opt/rl-deps/generalist-improvement \
    && git -C /opt/rl-deps/generalist-improvement checkout --quiet ${MOLMOSPACES_REF_SHA}
COPY gr00t/eval/sim/MolmoSpaces/setup_molmospaces.sh /opt/rl-deps/setup_molmospaces.sh
RUN VENV_DIR=/opt/sim/molmospaces/.venv REF_DIR=/opt/rl-deps/generalist-improvement PROJECT_REPO=/workspace \
        bash /opt/rl-deps/setup_molmospaces.sh \
    && uv cache clean

# --- LIBERO (mirrors gr00t/eval/sim/LIBERO/setup_libero.sh) --------------------------
ARG LIBERO_SHA=8f1084e3132a39270c3a13ebe37270a43ece2a01
RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/rl-deps/LIBERO \
    && git -C /opt/rl-deps/LIBERO checkout --quiet ${LIBERO_SHA}
RUN <<EOF
set -euxo pipefail
V=/opt/sim/libero/.venv
uv venv $V --python /usr/bin/python3.10
P="--python $V/bin/python"
uv pip install $P "cmake<4"
PATH="$V/bin:$PATH" uv pip install $P --requirements /opt/rl-deps/LIBERO/requirements.txt
uv pip install $P -e /opt/rl-deps/LIBERO --config-settings editable_mode=compat
uv pip install $P torch==2.5.1 torchvision==0.20.1 pydantic av tianshou==0.5.1 tyro pandas dm_tree \
    einops==0.8.1 albumentations==1.4.18 zmq
uv pip install $P transformers==4.57.3 msgpack==1.1.0 msgpack-numpy==0.4.8 gymnasium==0.29.1
uv pip install $P numpy==1.26.4 mujoco==3.3.0
uv pip install $P -e /workspace --no-deps
uv cache clean
EOF
# LIBERO asks for a config interactively on first import; provide one.
ENV LIBERO_CONFIG_PATH=/opt/rl-deps/libero_config
RUN mkdir -p $LIBERO_CONFIG_PATH && L=/opt/rl-deps/LIBERO/libero/libero && cat > $LIBERO_CONFIG_PATH/config.yaml <<EOF
benchmark_root: $L
bddl_files: $L/bddl_files
init_states: $L/init_files
datasets: $L/../datasets
assets: $L/assets
EOF

ENV RL_MOLMOSPACES_PYTHON=/opt/sim/molmospaces/.venv/bin/python \
    RL_LIBERO_PYTHON=/opt/sim/libero/.venv/bin/python \
    NVIDIA_DRIVER_CAPABILITIES=all

# Refresh the source tree last (the base image may predate gr00t/rl), so code edits
# do not invalidate the simulator layers above.
COPY . /workspace/
RUN uv pip install --python /workspace/.venv/bin/python -e /workspace --no-deps

# Import check (no rendering: there is no GPU at build time).
RUN $RL_MOLMOSPACES_PYTHON -c "import molmo_spaces, gr00t.rl.envs.molmospaces; print('molmospaces ok')" \
    && $RL_LIBERO_PYTHON -c "import libero.libero, robosuite, gr00t.rl.envs.libero; print('libero ok')"
