# `gr00t.rl`: critic-guided RL (OGPO / Best-of-N) for GR00T N1.7

Code for the GR00T RL → demo-efficiency transfer experiment
(design: `~/experiments/groot-rl-transfer/PLAN.md`). It covers PLAN milestones
**M1** (MolmoSpaces ↔ GR00T-DROID env wrapper) and **M3** (SDE sampling + per-step
log-probs, BroNet critic, OGPO update, Best-of-N collection, batch-online loop).
Euler packaging (M5) and the actual runs are not done yet.

## Layout

| File | What |
|---|---|
| `sde.py` | Pure math: marginal-preserving SDE step (tapered / constant), Gaussian log-probs, group and conservative advantages, PPO clip, ensemble aggregation |
| `flow_policy.py` | `FlowPolicy`: frozen-VLM encoding + critic prefix, ODE/SDE sampling with trajectories, differentiable `traj_log_prob`, BC loss, action decode/re-encode, EMA "slow" policy, checkpoint save in `Gr00tPolicy`/`launch_finetune` layout |
| `critic.py` | BroNet Q/V ensembles (members stacked into batched matmuls), Polyak target net |
| `ogpo.py` | `OGPOLearner`: TD critic (policy bootstrap with Nvr variance reduction, or SARSA+V), OGPO(+CA) policy update, success-BC, KL-to-base diagnostic |
| `replay.py` | Episode → H-step chunk transitions (termination vs truncation), ring buffer with JPEG frames stored once per step, history-frame lookup, save/load |
| `rollout.py` | `Collector`: batched inference over remote envs, Best-of-N, open-loop chunks, recording; `fill_missing_prefixes` |
| `train.py` | Batch-online loop (collect → prefixes → train), DataLoader workers for batch prep, per-round checkpoints + resume, kill switch, final eval |
| `evaluate.py` | Evaluate a checkpoint without training (zero-shot pilot, held-out eval) |
| `config.py` | All knobs, defaults from Korniak 2026 Table 5 / Patil 2026 |
| `envs/` | ZeroMQ env server/client + adapters (`molmospaces`, `libero`, `fake`); run in the simulator's own venv, no torch imports |

`gr00t/model/gr00t_n1d7/gr00t_n1d7.py`: the denoising loop body of
`get_action_with_features` is factored into `build_denoising_condition()` and
`predict_velocity()` (behaviour-preserving, verified bit-exact against `main`), so RL
code calls the exact same network step with or without gradients.

## Setup

Everything runs in the `gr00t` container (`make build`). Extra flags: `NVIDIA_DRIVER_CAPABILITIES=all`
enables EGL rendering for MuJoCo inside the container; the `GROOT_*` flags avoid a transformers Hub
call when loading Cosmos-Reason2.

```bash
docker compose run --rm --no-deps \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/mlspaces:/mlspaces -e MLSPACES_ASSETS_DIR=/mlspaces/assets -e MLSPACES_CACHE_DIR=/mlspaces/cache \
  -e GROOT_PATCH_MISTRAL=1 -e GROOT_HF_LOCAL_FIRST=1 \
  gr00t bash
```

Simulator venvs (build once, inside the container):

```bash
bash gr00t/eval/sim/MolmoSpaces/setup_molmospaces.sh   # Korniak's MolmoSpaces fork @ b288652, + NLTK data
bash gr00t/eval/sim/LIBERO/setup_libero.sh             # fixed here: cmake<4 and mujoco==3.3.0 pins
```

MolmoSpaces assets: `python -c "from molmo_spaces.molmo_spaces_constants import get_resource_manager; get_resource_manager()"`
with `MLSPACES_AUTO_INSTALL=True` (robots, benchmarks; ~12 GB). Scenes and objects download on first use.

## Commands

```bash
# Tests (CPU) and GPU smoke test on the real checkpoint
python -m pytest tests/gr00t/rl -q
python scripts/rl/smoke_test_policy.py --model-path nvidia/GR00T-N1.7-3B

# M1: MolmoSpaces vs OXE_DROID observation-space check (in the MolmoSpaces venv)
gr00t/eval/sim/MolmoSpaces/molmospaces_uv/.venv/bin/python scripts/rl/check_droid_spaces.py \
    --statistics <GR00T-N1.7-3B snapshot>/statistics.json --tasks pick/0 pick/5 pick/40

# M2: zero-shot pilot (32 episodes/task); writes eval.json with per-task success
python -m gr00t.rl.evaluate --model-path nvidia/GR00T-N1.7-3B \
    --env.backend molmospaces --env.python gr00t/eval/sim/MolmoSpaces/molmospaces_uv/.venv/bin/python \
    --env.num-envs 16 --episodes-per-task 32 --tasks pick/0 pick/1 ... --output-dir outputs/pilot

# Arm C (OGPO) from the warm-started checkpoint B
python -m gr00t.rl.train --model-path <B checkpoint> --output-dir outputs/arm_c \
    --env.backend molmospaces --env.python gr00t/eval/sim/MolmoSpaces/molmospaces_uv/.venv/bin/python \
    --env.num-envs 16 --collect.tasks <train tasks> --collect.eval-tasks <train + held-out tasks>

# Arm E (critic only, Best-of-N at eval)
python -m gr00t.rl.train --algo bon --critic.bootstrap sarsa --collect.eval-best-of-n 32 ...
```

Task ids: MolmoSpaces `"<benchmark>/<episode>"` (benchmark names map to directories via
`--env.backend-kwargs-json '{"benchmarks": {"pick": "...", "open": "..."}}'`, default
`pick` = `FrankaPickDroidMiniBench`), LIBERO `"libero_10/3"`.

Outputs: `metrics.jsonl` (+ wandb with `--use-wandb`), `rounds/round_XXX/` (policy,
learner state, replay buffer of the newest round), `final_policy/`: a normal GR00T
checkpoint, the `--base-model-path` for the adaptation sweep. Re-running the same command
resumes after the last completed round (24 h SLURM limits).

## Protocol as implemented

* **Reward** −1 per step, 0 on success; terminate on success only; time limit = truncation
  (bootstrapped: discount γ^H unless the episode *terminated* inside the window).
* **Transitions**: one per env step; action = the executed chunk from that step on, stitched
  across policy queries, re-encoded into the model's normalized relative space w.r.t. the
  state at that step (BC target; the critic reads the first H = 16 steps). Windows that
  run past the episode end are dropped (Korniak).
* **Critic**: BroNet ×10 (width 1024, depth 2) on [mean-pooled VLM tokens (2048), normalized
  17-D state, flattened 16×17 chunk]; subsample-min aggregation; Nvr = 16 slow-policy
  samples in the TD target; second update on a success-only batch; target decay 0.95.
* **Actor**: VLM frozen, action head trained (fp32 master weights, bf16 autocast, all 144
  dropout layers removed); G = 32 SDE trajectories/state from the EMA policy (0.995); PPO clip
  0.01 on the product of K = 4 per-step ratios; conservative advantages; success-BC λ = 1.
* **Collection**: batch-online, M = 10 rounds × K = 20 episodes/task, policy frozen during
  collection, BoN N = 32 with SDE samples ranked by the target critic (round 0 plain);
  execute 8 of 40 actions per query at 15 Hz. Eval: ODE, N = 1, seeds disjoint from training.

## Where this deviates from the references, and why

1. **OGPO is not in Korniak's repo** (README: "on its way"). The critic, buffer, BoN and env
   wrapper are ported from it; the OGPO update follows Patil 2026 (§3-4, App. A/B/E) with
   Korniak's App. D.2 choices.
2. **SDE noise**: tapered σ_τ = σ_init √(1−τ) with the score correction (Patil's recommended
   form; their paper runs used a constant 0.01). Default σ_init = 0.1 is a guess; tune it on LIBERO.
3. **Chunking**: GR00T predicts 40 steps; critic H = 16, execute 8 (Korniak: π0.5, 15 / 5).
4. **Prefix embeddings** are computed for every stored observation after each round (Korniak
   reuses the query-time embedding for the following open-loop steps).
5. **EEF frame**: MolmoSpaces' `tcp_pose` sensor is relative to the robot *mount*; the
   adapter reports the grasp site in the arm base frame `fr3_link0`, like DROID (see below).
6. **Single GPU**. No DDP yet (Korniak used 4×GH200 per run).

## Verified so far (2026-11-23, RTX 5070 Ti)

* 34 CPU unit tests: SDE marginals preserved on a Gaussian toy, log-prob / advantage /
  PPO math, critic fits, buffer windows / discounts / eviction / history frames / save-load,
  OGPO on a toy flow policy (ratio starts at 1; moves the policy toward a fixed critic's optimum;
  zero update when ensemble members disagree), env server/client, Collector with BoN.
* Real N1.7 checkpoint: refactored `get_action` == pre-refactor (bit-exact); `FlowPolicy` ODE ==
  `get_action`; SDE log-probs recompute exactly (PPO ratio starts at exactly 1); decode→re-encode
  of joints/gripper exact to 4e-7 (`eef_9d` within 0.03: the model's rot6d is not orthonormal);
  BC backprop; OGPO step.
* End to end: `gr00t.rl.train` with real model + real MolmoSpaces workers (2 rounds, resume
  tested with the fake backend); `gr00t.rl.evaluate`: base N1.7 picked the spoon 2/2, the
  remote 0/2 (tiny sample, not a result).
* M1 space check on 4 pick episodes: joints, gripper, rot6d and EEF xyz all inside DROID's
  [q01, q99]; GR00T's `DROID_EEF_ROTATION_CORRECT` is the best of all 24 axis-aligned
  conventions (score 0.24 vs 2.31 runner-up); no joint drift when holding a pose.
* LIBERO adapter: reset with fixed init states + settle, truncation, two suites in parallel.

## Open items / knobs to watch

* **PPO ratio saturation.** In the short MolmoSpaces run `clip_frac` ≈ 1 from the second step:
  the likelihood sums 40×17 dims × 4 steps, so small parameter changes push π_θ/π_slow far past
  ε = 0.01. If this persists once advantages are non-zero, try `--actor.logprob-reduce mean`,
  `--actor.logprob-horizon 16`, a larger `--actor.sigma`, or a larger `--actor.clip-eps`. LIBERO
  validation (πRL: GR00T N1.5 52.5 → 89.9 %) is the place to settle this.
* **Memory**: the full action head is 1.62 B params (DiT 1.09 B, VL self-attention 0.20 B,
  32-embodiment projectors 0.33 B); fp32 weights + Adam ≈ 26 GB, fine on an A100 80 GB. On a
  16 GB card use `--actor.trainable-regex` (the smoke test trains `action_decoder` only).
* **TCP point**: the grasp site is used as EEF; if DROID's TCP sits elsewhere along the tool
  axis, set `tcp_offset_z` in the backend kwargs (only affects the state input, not control).
* **Buffer RAM**: ≈ 35 KB/step (two JPEG frames + fp16 prefix + chunk), i.e. ~40 GB at 1 M steps.
* **Cost per training step** with defaults: 4 VLM passes of 512 observations (s′ for both
  critic batches, s for the actor and for BC) + 2×512×16 bootstrap samples + 512×32 actor
  trajectories. `--critic.bootstrap sarsa` removes the s′ passes.
* Two existing tests (`test_coexists_with_progress_head`, `test_default_config`) fail on
  `main` too; unrelated to this code.
