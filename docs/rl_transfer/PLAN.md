# GR00T RL → demo-efficiency transfer: implementation brief

Self-contained brief for starting implementation in a fresh session. Status: **design frozen 2026-11-23; M1 + M3 code written 2026-11-23** (see §10).
Literature: [papers/INDEX.md](papers/INDEX.md) (28 papers, each with a summary and the decision it informs; `papers/download.sh` re-fetches the PDFs).

---

## 1. Question and hypothesis

Take GR00T N1.7, warm-start it with SFT on **5 task families** in MolmoSpaces, then RL-fine-tune it with **OGPO** on those families.
Then SFT it on **3 held-out families** with N demos and measure success vs N.

- **Positive:** the RL'd model reaches the SFT-only model's success with **≤ ½ the demos**. That means RL on some families makes a VLA cheaper to adapt to new ones.
- **Negative:** gains stay on the trained families. The value of environments is then "make task X work", not a general environment library.

Novelty, given prior work: Korniak et al. 2026 (ETH, Krause lab) showed that multi-task critic-based RL on π0.5 improves **zero-shot** success on unseen
MolmoSpaces tasks. **Nobody has measured the demo-efficiency of subsequent SFT on held-out families after RL.** That gap is this experiment, and it uses GR00T, not π0.5.
This is a **research experiment only**: no sales demo, no G1/humanoid.

## 2. Decisions

| Topic | Decision | Basis |
|---|---|---|
| Simulator | **MolmoSpaces**, Franka FR3 in DROID config. LIBERO only as a pipeline sanity check | Korniak 2026 App. C.1: no transfer on LIBERO-90 at ≤64 tasks (tasks too dissimilar) |
| Families | **Interaction type × object class** inside MolmoSpaces. Exact split TBD (§5) | MolmoSpaces has no insertion/rotation/peg tasks. Its base tasks are pick, pick-and-place(-next-to/-color), open, close (+ open-door on RB-Y1, navigate). Korniak §4.3: object/grasp diversity drives transfer, and held-out objects similar to trained ones transfer best |
| Base model | **GR00T N1.7 (3B)**, embodiment tag `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`, behind a swappable policy interface | GR00T 2 (Nov 2026 whitepaper) weights are not released yet (project page says December), it is not shown on Franka, and its rollouts also denoise video |
| Warm start | Short SFT on the 5 training families before RL (≈10 demos/task) | RL4VLA, Chu 2025, πRL |
| RL method | **OGPO** (Patil et al. 2026), as adapted by Korniak 2026 for π0.5 | The only method with evidence for multi-task → unseen-task transfer |
| Reward | **−1 per step, 0 on success.** Terminate on success. **Time limit = truncation**: bootstrap from V, don't treat it as a terminal failure | Korniak 2026 App. D.3 |
| Compute | ETH **Euler**, multiple **A100 80 GB**. Extend the existing Apptainer pipeline in the GR00T fork | Korniak: 4×GH200 per node, 2–5 h per single-task run, ~200 h for the largest multi-task split |
| Protocol | §4 | robomimic, Lin 2024, rliable |

## 3. Arms

All arms end with the same **adaptation step**: SFT on N demos/task of the 3 held-out families, for N ∈ {1, 5, 10, 25, 50}, then evaluate on the held-out families.
The arms differ only in the checkpoint that step starts from.

| Arm | Starting checkpoint | Role |
|---|---|---|
| **B** | N1.7 + SFT on the 5 training families | Control: same sim/robot/family exposure, no RL |
| **C** | B + OGPO on the 5 training families | Treatment. **C vs B is the headline** |
| E (cheap) | B + learned Q-critic doing Best-of-N over B's samples, no weight update | Is transfer in the critic or in the policy weights? (Korniak: critic alone transfers) |
| D (optional) | B with extra SFT, matched to C's compute | Rules out "C trained longer" |
| A (floor) | Base N1.7 | Reference only. C vs A is confounded by sim/embodiment adaptation |
| **A+RL (conditional)** | Base N1.7 + OGPO, no SFT | **Run only if the zero-shot gate passes** (below). Compare A vs A+RL, zero-shot and on the demo curve |

**Zero-shot gate.** Pilot first: evaluate base N1.7 (DROID tag) on all candidate MolmoSpaces tasks, 32 episodes each.
Run A+RL if mean success is ≥5% **and** successes appear on ≥ half the training tasks. With −1/0 reward, a task with zero successes gives the critic no signal.
Korniak selected tasks so the base averaged ≈30%. OGPO claims to work from poorly initialised policies, which makes a low-but-nonzero base viable.
If the gate fails, use warm-start SFT rounds (arm B path) only.

## 4. Evaluation protocol (fix before training)

- Demo counts {1, 5, 10, 25, 50} per held-out task × **3 seeds** × ≈50 eval episodes per task. Use evaluation initial states disjoint from training ones.
- **Fixed SFT budget per point** (same steps and LR for every arm). The same checkpoint-selection rule for every arm (by rollouts, not validation loss; robomimic).
- Also report: success on the training families (did RL work at all?), and zero-shot held-out success before adaptation (forgetting vs transfer, as in Korniak).
- Log the KL to the base policy during RL (RL's Razor).
- Stats: mean, IQM and **stratified bootstrap 95% CIs (rliable)**. Fit a power law to success vs N per arm.
- **Pre-registered win:** C reaches B's 50-demo success with ≤ 25 demos, with non-overlapping CIs.
- Report transfer **vs a pre-registered similarity score** between each held-out family and the trained ones (Korniak: transfer is strongest to similar objects).
  This turns a yes/no into a curve.
- **Kill switch:** if OGPO doesn't raise training-family success by ≥10 pts within ~200 GPU-h, stop. Transfer is then moot.

## 5. Family split (open, first design task)

Families = interaction type × object class, built from MolmoSpaces base tasks (pick, pick-and-place, pick-and-place-next-to, pick-and-place-color, open, close).
Illustrative only (to be replaced by an analysis of the available assets and base zero-shot rates):
- **Train (5):** pick thin tools (spatula, ladle, spoon); pick containers (box, cup, pot); pick-and-place into receptacles; open drawers; open hinged fixtures (cabinet, fridge).
- **Held out (3):** pick bottle-like objects (wine bottle, atomizer, shakers); pick-and-place-next-to; close fixtures.

Requirements: ~10 tasks per family (Korniak's transfer appeared from ~16 tasks upward). Families must differ in **manipulation strategy**, not just visuals.
Fix the split and the similarity score before any RL run. Korniak's splits (App. D.1, Tables 1–3) are a good starting point.

## 6. OGPO in brief, and how it differs from GRPO

Flow-matching action heads (GR00T, π0) generate an action chunk by integrating a **deterministic** ODE over K denoising steps, so there is no log π(a|s) for policy gradients.
The common fix (Flow-GRPO, πRL Flow-SDE, OGPO) is to **sample with an SDE** that has the same marginals. Each denoising step is then a Gaussian with a tractable likelihood.

**OGPO**, in two levels:
1. **Environment level, off-policy critic.** Store every transition, successes and failures, in a replay buffer. Train Q(s, a_chunk) and V(s) with **TD** targets
   (H-step chunk returns + γ^H·V). Use a BroNet residual MLP ensemble. Input: mean-pooled VLM prefix tokens + proprio + flattened action chunk. Take the min over a random 2 of 10 heads.
2. **Denoising level, PPO-style update.** For a replay state s, sample a **group of G action chunks** via SDE denoising and score each with the frozen target critic.
   Advantage Â_i = Q(s, a_i) − mean_j Q(s, a_j). Apply a **PPO-clipped** update to the product of denoising-step likelihood ratios vs a slow EMA policy.
   This pushes up denoising paths that land on high-Q actions.

Stabilizers:
- success-buffer BC loss
- two-sided conservative advantages (early critics are underfit)
- Q-variance reduction (average the TD target over N actor samples)
- EMA policy for rollouts
- Best-of-N (N=32) by Q during collection

**vs GRPO.** Both use a *group-relative* advantage with PPO clipping. The difference is what gets grouped and scored:
- **GRPO** (DeepSeek; used by GR00T 2's RL and SimpleVLA-RL) samples G **full episodes** from the same start, scores them by **real returns**, has **no critic**, and is **on-policy**.
  In robotics that means many expensive rollouts per update and resettable starts. Credit is spread over the whole episode, and all-fail groups give zero signal under sparse reward.
- **OGPO** samples G **action chunks at a single state**, scores them with a **learned Q-critic** (no extra environment steps), and is **off-policy**: the critic reuses all past data.
  The result is per-step credit assignment, much better sample efficiency, and a critic, which Korniak identifies as the ingredient that drives transfer to unseen tasks.

Reference code: Korniak's pipeline (π0.5, MolmoSpaces, OGPO/BoN/FSFT) at **github.com/lasgroup/generalist-improvement**. Port the critic and OGPO to the GR00T action head.

## 7. Existing assets on this machine

| Thing | Path | Notes |
|---|---|---|
| GR00T fork (N1.7) | `~/groot_ws/Isaac-GR00T` (origin `MatteoZucchelli-eth/Isaac-GR00T`) | Contains the user's perception/tip-target changes. Has its own `CLAUDE.md` |
| Flow action head | `gr00t/model/gr00t_n1d7/gr00t_n1d7.py:256` (`Gr00tN1d7ActionHead`); `get_action_with_features` ~L866 | Deterministic Euler integration over `num_inference_timesteps`. This is where SDE sampling plus per-step log-probs go |
| DROID embodiment | `gr00t/data/embodiment_tags.py` (`OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`), `examples/DROID/` | MolmoSpaces Franka uses the DROID camera/action layout. Verify the action-space match |
| SFT entry points | `gr00t/experiment/launch_finetune.py`, `examples/finetune.sh` | Reuse for the warm start and the adaptation sweep |
| Eval harness | `gr00t/eval/` (`rollout_policy.py`, `run_gr00t_server.py`, `sim/`) | LIBERO wrapper exists. A MolmoSpaces wrapper must be written |
| LIBERO (sanity check) | `examples/LIBERO/`, `gr00t/eval/sim/LIBERO/`, `external_dependencies/LIBERO` | Reproduce πRL-style few-shot SFT → RL gains to validate the RL code |
| Euler pipeline | `submit_euler.sh`, `setup_and_train.sh`, `scripts/fetch_euler_checkpoint.sh` | Apptainer `.sif` built from the `gr00t` Docker image. Container and dataset are copied to `$TMPDIR`. Paths: `/cluster/work/mrl/$USER/containers`, `/cluster/scratch/$USER/groot`, `/cluster/project/mrl/$USER/results/groot`. `~/groot/.env` is required |
| Local GPU | RTX 5070 Ti 16 GB | Smoke tests only. RL runs on Euler |

Repo-specific notes:
- **Never add Claude/Co-Authored-By attribution** to commits or PRs in the user's repos (standing preference).
- `experiment_cfg/final_processor_config.json` is the only reliable record of a run's image geometry. Don't trust `config.json` for that.
- Generate `meta/stats.json` for new datasets before training or serving.

## 8. Known pitfalls (from the papers)

- **Replace dropout with identity** in the action head during RL, because dropout corrupts the PPO likelihood ratio (πRL App. H, done on GR00T N1.5).
- **Critic warm-up:** success dips early while the critic is inaccurate (πRL). Use OGPO's conservative advantages.
- **Truncation vs termination** with the −1/0 reward (§2).
- **Which params to train:** Korniak found full-VLA FSFT gives +12 pp in-distribution but −10.6 pp on unseen objects, vs −2.2 pp for action-expert-only.
  For OGPO, start with the VLM frozen and train the action head (πRL did this for GR00T N1.5). Treat full fine-tuning as an ablation.
- **Distributional critics scale poorly multi-task.** Use TD + regression (Korniak §4.4.2).
- **Hold diversity fixed across arms** in the demo sweep (Lin 2024): same held-out tasks and scenes, only N varies.
- **Batch-online protocol** as in Korniak: M=10 rounds × K=20 episodes per task, policy frozen during collection.

## 9. Suggested milestones

1. **Environment:** install MolmoSpaces and write a GR00T-N1.7 ↔ MolmoSpaces (Franka-DROID) env wrapper. Check that observation and action spaces match `OXE_DROID`.
2. **Zero-shot pilot (the gate):** base N1.7 on candidate tasks, 32 episodes each. Use the per-task rates to fix the family split and similarity score (§5). Decide on A+RL.
3. **RL code:** SDE sampling + per-step log-probs in the action head; critic (BroNet ensemble on pooled VLM prefix); OGPO update; BoN collection.
   Validate on LIBERO (few-shot SFT → RL should give large gains, cf. πRL GR00T N1.5: 52.5 → 89.9%) and on 1 MolmoSpaces task.
4. **Data:** generate demos for the 8 families with MolmoBot-Engine (scripted planners; MolmoBot-Data may already cover pick/PnP/open). Use the warm-start set for the 5 and the adaptation pool (≥50/task) for the 3.
5. **Euler:** a new `groot-rl.sif` (gr00t image + MolmoSpaces + EGL), a chained SLURM job for OGPO with checkpoint/resume across 24 h limits, and a SLURM array for the adaptation sweep (arm × N × seed).
6. **Runs:** B, then C (watch the kill switch), then E, then the adaptation sweeps. Add A+RL if the gate passed, and D if budget allows.
7. **Analysis:** demo curves with CIs, power-law fits, transfer vs similarity, KL logs.

Optional: contact the Korniak/Krause group at ETH before step 3. They have the pipeline, and this is a natural follow-up to their paper.

## 10. Implementation status (2026-11-23)

Code: branch `rl-transfer` in `~/groot_ws/Isaac-GR00T` (remote `github.com/MatteoZucchelli-eth/Gr00t-rl`), package `gr00t/rl/`. Its `README.md` has the commands, the verified checks and the knobs.

- **M1 done.** `gr00t/rl/envs/molmospaces.py` (port of Korniak's wrapper → GR00T DROID layout), run by ZMQ env workers in their own venv (`gr00t/eval/sim/MolmoSpaces/setup_molmospaces.sh`). Space check (`scripts/rl/check_droid_spaces.py`): rot6d convention = GR00T's `DROID_EEF_ROTATION_CORRECT` (best of 24); joints/gripper/EEF in DROID's [q01, q99] **after** expressing the TCP in `fr3_link0` (MolmoSpaces' `tcp_pose` is relative to the robot mount, z ≈ 1.0 m, out of distribution). Base N1.7 runs zero-shot through the full stack.
- **M3 code done, not yet validated at scale.** SDE sampling + per-step log-probs, BroNet critic, OGPO(+CA, +success-BC), Best-of-N, batch-online loop with resume, `gr00t.rl.evaluate` for the M2 pilot. Korniak's repo has **no OGPO** (only FSFT/BoN); OGPO follows Patil 2026 + Korniak App. D.2.
- **Watch in LIBERO validation:** PPO ratio saturates (`clip_frac` ≈ 1) with ε = 0.01 on the 40×17×4-dim likelihood. Knobs: `logprob_reduce=mean`, `logprob_horizon`, `sigma`, `clip_eps`.
- **M5 packaging done** (2026-11-23): `docker/Dockerfile.rl` (gr00t + MolmoSpaces + LIBERO venvs) → `groot-rl.tar.gz`; `submit_rl_euler.sh` + presets in `rl_experiments/` (smoke, assets, list_tasks, pilot_zero_shot, ogpo_arm_c, bon_arm_e, aplus_rl, libero_ogpo); `scripts/rl/euler/{build_container,sync,submit,fetch_run}.sh`. How-to: `scripts/rl/euler/README.md`. Not done: DDP (single A100 per run).
- **Next:** smoke job on Euler, then the M2 pilot (`rl_experiments/pilot_zero_shot.sh`), fill `rl_experiments/splits/{train,heldout}.txt`, LIBERO few-shot SFT → `libero_ogpo.sh`.
