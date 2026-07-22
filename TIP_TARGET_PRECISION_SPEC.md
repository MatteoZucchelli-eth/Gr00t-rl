 # Tip-Target Precision — Implementation Spec

Handoff spec for improving target-prediction precision of the custom tip-target head in this
repo (Isaac GR00T N1.7, Unitree G1 drilling task). Written to be implemented by someone with
no prior context. Read section 0–2 before touching code.

---

## 0. Context

- The right end-effector is a **drill tip** that must be placed on a small target square.
- A custom auxiliary **tip-target head** predicts the target pose in the drill-tip frame
  (`xyz(3) + rot6d(6)`, the `next.tip_target_9d` label) from the pooled vision-language latent,
  and feeds that prediction back as **one conditioning token** to the action diffusion model (DiT).
  - Head + forward + loss: `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`
    (class `TipTargetHead` ~L107; `Gr00tN1d7ActionHead.forward` head/loss ~L429–521).
  - Config: `gr00t/configs/model/gr00t_n1d7.py` (~L154–160), `gr00t/configs/finetune_config.py` (~L73–92).
  - Enabled from `start_finetune.sh` via `--use-tip-target-head`.
  - Label pass-through: `gr00t/data/dataset/lerobot_episode_loader.py` (~L405),
    `gr00t/data/dataset/sharded_single_step_dataset.py` (~L86),
    `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py` (~L617).
  - Prediction surfaced for deployment: `gr00t/policy/gr00t_policy.py` (~L481).
- Training data: `g1_dataset_small_synthetic_1500` (synthetic). Eval runs in Isaac Sim.

## 1. Problem & root cause

- Goal: the drill tip should land on the target more accurately. The dominant error is the
  **head's target prediction**, not the low-level controller (tracking is already ~2–12 mm).
- Observed: when the target is **visible**, the head is ~1–2 cm off; when the target is **not
  visible** in the camera, the head outputs garbage (0.3–0.5 m). The garbage cases dominate the
  average and are the failures worth fixing.
- **Root cause:** the training set labels the target on ~100% of frames (0% occlusion) — the
  simulator knows the target pose even when it is off-camera — so the head learned to *always*
  emit a confident target, including on frames whose image shows nothing. A single-frame
  feed-forward head **cannot** localize an off-screen target. The fix must (a) let the model know
  when it can't see the target, and (b) remember where the target was.
- Secondary issues: eval `finalDist` is polluted by **contact-force spikes in Isaac** (use
  min / pre-contact distance instead); **action-horizon mismatch** (checkpoints trained at 40,
  `register_new_modality.py` now uses 30).

## 2. Prerequisites that are NOT code in this repo (owner: user / sim / deploy harness)

These gate how much the code changes below help. Note them; do not attempt them here.

- **P1 — sim data (enables Task A):** regenerate `g1_dataset_small_synthetic_1500` with occlusion
  handling. For each frame, project the target into each camera; on frames where it is outside
  **every** camera FOV, either write `next.tip_target_9d = NaN` or add a boolean column
  `next.tip_target_visible = False`. Without occluded frames, the visibility head has nothing to learn.
- **P2 — deploy/eval harness (external):** base-frame target memory. Transform each *visible*-frame
  head prediction into the world/base frame using the current tip pose (FK from proprioception),
  fuse over time (running median / Kalman), and hold it through occlusion, gated by the visibility
  signal the model exposes. This is likely what the existing "VLM planner" does (it reaches ~7 mm).
- **P3 — eval metric:** report min / pre-contact distance, not `finalDist`.

## 3. In-repo tasks (do these)

### Task A — Visibility head + conditioning gate  [highest priority]
Stop the policy from acting on a hallucinated target when the target is not visible.
- In `TipTargetHead`, add a **visibility logit** output (small linear head off the same pooled
  feature); return `(pred_scaled_9d, vis_logit)`.
- **Visibility target:** `visible = ~isnan(tip_target_9d).any(-1)` (this reuses the NaN mask already
  computed at `forward` ~L444; once P1 adds occluded/NaN frames it trains correctly). Optionally also
  consume an explicit `next.tip_target_visible` column if present (wire it through the loader +
  processor exactly like `tip_target_9d`).
- **Loss:** add `BCEWithLogits(vis_logit, visible)` over **all** rows (weight ≈ 0.2). Keep the 9d
  regression loss over **visible rows only** (already masked).
- **Conditioning gate:** where the conditioning token is built (~L451), when the frame is predicted
  invisible (`sigmoid(vis_logit) < τ` at inference; `visible == False` in training) feed a **learned
  `nn.Parameter` "unknown" token** instead of the pose token, so the DiT learns a distinct
  "target unknown" mode rather than trusting a garbage pose.
- Surface `vis_logit`/`sigmoid` next to `tip_target_pred` in `gr00t/policy/gr00t_policy.py` for P2.
- Add config flags in `gr00t/configs/model/gr00t_n1d7.py` + `finetune_config.py`
  (`use_tip_target_visibility`, `tip_target_visibility_loss_weight`, threshold).
- **Accept:** post-P1, visibility AUC > 0.9 on held-out episodes; on invisible frames the
  conditioning token is the "unknown" embedding, not a pose.

### Task B — Attention pooling over image tokens only  [high impact when visible]
Sharpen the visible-case prediction; the current pooling throws away spatial location.
- Replace `_masked_mean_pool(vl_embeds, attention_mask)` in `TipTargetHead.predict_scaled` (~L150)
  with a **learned-query cross-attention pool** (`nn.MultiheadAttention`, 1–4 learned query tokens)
  over the VLM tokens.
- Pool over **image tokens only**: pass `backbone_output.image_mask` (already available; used by the
  DiT at `forward` ~L459) into the head and exclude text/padding tokens. Currently the head pools
  *all* tokens including the language prompt.
- If feasible, pool **per camera** (split image tokens by view) so the head can use whichever view
  frames the target.
- Keep the 9d output shape and loss unchanged.
- **Accept:** visible-frame `‖xyz_pred − xyz_gt‖` drops vs the mean-pool baseline on a fixed val set.

### Task C — Decoupled position/rotation loss  [low risk]
- Split the tip-target loss (`F.mse_loss` at ~L517) into separate **xyz** and **rot6d** terms with
  independent weights. Position is the bottleneck; rotation is already accurate (~2.7–7°).
- Use **Huber/SmoothL1** on xyz so small near-target residuals keep gradient; keep MSE on rot6d
  (optionally Gram-Schmidt the predicted rot6d before the loss).
- Expose `tip_target_pos_loss_weight` / `tip_target_rot_loss_weight` in config.

### Task D — Config alignment & hygiene  [trivial]
- Make the action horizon consistent across `register_new_modality.py` (`G1_PALM_EEF_CHUNK_SIZE`),
  model config `action_horizon`, and the eval `--action-horizon` flag (checkpoints=40, config=30 now).
- Reduce color jitter in `start_finetune.sh` (task is red/green-semantic): `hue 0.08 → ~0.02`,
  `saturation 0.5 → ~0.25`; keep brightness/contrast.

### Task E — Exposure-bias mitigation  [optional, do with A/B]
- The conditioning token is teacher-forced on ground-truth + 5 mm noise (~L446), but at inference it
  receives the head's own (larger) error. Add **scheduled sampling** (ramp a probability of feeding
  the detached head prediction instead of GT during training) and/or raise the conditioning noise
  toward the real error scale. Combine with A/B, not instead of them.

## 4. Order & notes
- Implement **A → B → C → D**, then E. A and B need a retrain to show effect; C/D ride along.
- A's code should land **before** P1 so the data regeneration "just works."

## 5. Do NOT
- Do not add rotation/flip image augmentation — it corrupts the tip-frame `rot6d` label.
- Do not chase `finalDist` with a stopping/progress-head change — that tail is contact physics.
- Do not just deepen/widen the MLP head without fixing pooling (Task B) — the spatial signal is
  already gone before the MLP.
