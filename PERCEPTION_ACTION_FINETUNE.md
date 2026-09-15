# Perception-Conditioned Action Model (fine approach)

The perception model ([PERCEPTION_FINETUNE.md](PERCEPTION_FINETUNE.md)) localizes the wall
target in the wrist camera but emits no actions. This model keeps that head and adds the
standard GR00T flow-matching action head back on top, with the perception output **cast
into conditioning tokens** the DiT attends to — so the action chunk is generated
conditioned on where the target is.

It is the same mechanism the full VLA uses for `TipTargetHead`
([gr00t_n1d7.py:60](gr00t/model/gr00t_n1d7/gr00t_n1d7.py#L60)) — predict an auxiliary
quantity, embed it into a token, append it to the state tokens, teacher-force it during
training — but sourced from a **pixel-space heatmap** rather than a 3D pose regression,
which is the better-posed problem at short range.

```
wrist frame + prompt
   └─ VLM backbone ─┬─ perception_head ─► heatmap / uv / visibility / depth / normal
                    │        │
                    │        ├─ token A "where"  ─┐
                    │        └─ token B "what"   ─┤
                    └─────────────────────────────┴─► action_head (DiT)
                          state ⊕ A ⊕ B ⊕ 30 noisy action tokens
```

## The two tokens

| | Token A — "where" | Token B — "what" |
|---|---|---|
| Built from | Fourier-encoded (u,v), visibility, heatmap confidence, depth, wall normal | heatmap-weighted pool of the image tokens |
| Teacher-forced | **yes** (label + calibrated noise) | no — always the model's own heatmap |
| Gradient into the perception head | **no** (detached / label) | **yes** |
| Why it exists | the precise geometry the controller needs | appearance at the target + the only path for the action loss to sharpen perception |

Three details in token A matter more than they look:

- **Fourier encoding of (u, v).** At 720 px, 1 px is 0.0014 in normalized units. A plain
  `Linear(2 → 1536)` cannot resolve that above its own weight noise, so the sub-pixel
  precision the heatmap head was trained for would be destroyed *at the conditioning
  interface*. Eight octaves (1 … 128 cycles/image) make a 1 px shift move the top band by
  ~0.7 rad. `perception_num_fourier_bands` controls this.
- **A learned `unknown_token`.** Tokens are blended `vis · token + (1−vis) · unknown`, so
  "I don't see the target" is a distinct, learnable input instead of a meaningless
  coordinate the DiT cannot distinguish from a real detection.
- **Confidence channels** (top-1 heatmap probability, normalized entropy, always taken
  from the prediction even under teacher forcing) let the policy hedge — slow down or
  re-orient — instead of committing to an uncertain estimate.

## Teacher forcing and the calibrated noise

Training conditions token A on the **label plus noise**; inference conditions it on the
head's own prediction. The noise std is the single most important hyperparameter here,
and it is **measured, not guessed** — noise smaller than the real closed-loop error
teaches the policy to trust perfect perception, and it will be brittle on the robot.

The defaults come from the 1500-step perception checkpoint's closed-loop eval
(`vla_eval_results/1500_perception_planner`):

| Flag | Value | Derivation |
|---|---|---|
| `--perception-cond-noise-uv-px` | **3.0** | single-frame lateral error 0.85 processed px rms, ×3 margin (see below) |
| `--perception-cond-noise-depth-m` | **0.004** | random part of the depth error: 4.32 mm spread |
| `--perception-cond-bias-depth-m` | **0.0065** | systematic +6.49 mm — **84% of the depth error** |
| `--perception-cond-noise-normal` | **0.04** | rotation spread 3.49° → 0.043 per component |

Two findings shaped these more than the raw arithmetic did.

> **Re-derive these for every new perception checkpoint** — they are a property of the
> checkpoint, not of the task:
> ```bash
> python scripts/calibrate_perception_conditioning.py \
>     --eval-dir vla_eval_results/<your eval> --dataset-path ./g1_dataset
> ```

### How the derivation works

The eval CSVs record the planner's tip→target estimate against the truth, in **metres in
the tip frame**. The conditioning token needs **processed-image pixels** and
**camera-frame depth**, so the script recovers the geometry from the dataset rather than
assuming it:

- **Focal length** — the heatmap sidecar carries both `depth_z` (Z) and `ray_dist`
  (√(X²+Y²+Z²)) for the same labelled pixel, so `(ray/Z)² − 1 = r_px²/f²` solves for `f`.
  Result: **f = 388.09 px** at the raw 720 px camera (FOV 85.7°), fitted to a relative
  residual of 5·10⁻⁶ — the per-sample p5, p50 and p95 all land on 388.1. With the 648/720
  crop, 1 camera px = 1.111 processed px, so f_processed = 431.2 px.
- **Camera→tip extrinsics** — the sidecar's (u, v, depth_z) give the target in the camera
  frame and `next.tip_target_9d` gives the same point in the tip frame; Kabsch on 43 651
  paired points fits a **rigid** transform to 1.1 mm rms, confirming the camera and tip
  share a link. The optical axis in the tip frame is (0.904, −0.001, −0.428).

With both, the tip-frame error is projected back into the pixel and depth errors the
perception model itself made.

### Finding 1 — that eval contains two models, and pooling them is a trap

The coarse model drives the approach while the arm is moving; **the perception model only
fires when the arm parks to accumulate detections**. The parked stretches are recoverable
from the trace itself (`dist_target_actual` goes flat) with phase 0 excluded as the coarse
leg. The two are not remotely comparable:

| | n | uv rms (px) | depth rms (mm) | depth bias (mm) |
|---|---|---|---|---|
| coarse model, in motion | 80 | **7.69** | 9.01 | +0.75 |
| perception, all stop detections | 93 | **0.65** | 7.10 | +6.49 |
| perception, first detection of a stop | 16 | **0.85** | 7.32 | +6.01 |

Pooling everything gives ~14 px and would have over-noised training by an order of
magnitude. The perception model is *extremely* precise laterally — sub-pixel once
detections accumulate, 0.85 px from a single frame.

Note also that within a stop the estimate is averaged over successive detections
(0.85 px → 0.61 px). The new model gets **one frame per step** with no accumulation, so
the first-of-stop figure is the honest analogue.

### Why the uv figure carries a ×3 margin

The 0.85 px measurement is a best case in two ways the new model will not enjoy: the
camera was **stationary** (no motion blur) and perception only ever fired between 2 and
67 mm, while training spans 0–150 mm. There is no measurement at all of this checkpoint's
error while moving or at long range. The risk is asymmetric — under-noising produces a
policy that trusts perfect perception and breaks on the robot, over-noising just makes it
slightly conservative — so the default is 3 px, still only ~1 mm of target uncertainty
against a 5 mm tolerance.

If you have the perception run's own `uv_err_px_metric` from wandb, **prefer it**: it
covers every labelled frame including moving ones, which this eval does not.

### Finding 2 — the depth error is a bias, not noise

**+6.49 mm systematic, against a 2.88 mm spread — the bias is 84% of the mean-square
depth error, and accumulating detections does not reduce it** (7.26 → 7.05 mm). The depth
output is consistently *long*.

A zero-mean Gaussian cannot represent that, so `perception_cond_bias_depth_m` exists as a
separate term and is applied in eval mode too (a bias the model always exhibits is not a
training-time perturbation). Without it the policy would train on unbiased depth and meet
a 6.5 mm systematic offset at deployment — on a task with a 5 mm tolerance.

Worth chasing at the source: 6.5 mm on a ~125 mm working depth is a 5% scale error, which
smells like a depth-head calibration or extrinsics issue rather than noise. If a future
perception checkpoint fixes it, set the term to 0.

`--perception-cond-vis-dropout-prob` (default 0.1) additionally tells the policy "target
not visible" on a fraction of steps where it actually is, so it learns to hold or search
rather than collapse when perception drops out mid-approach.

## Weight initialization

**The base is always `nvidia/GR00T-N1.7-3B`.** There is no flag for it, by design: the
model config is pinned to that checkpoint's architecture (32 DiT blocks, 4 VL
self-attention layers, 16 LLM layers) so the whole pretrained action stack — **537/537
tensors** — lands. Any other base would load partially and silently.

That pinning is load-bearing. On the normal VLA path `Gr00tN1d7Pipeline` calls
`AutoModel.from_pretrained`, which takes the architecture from the *checkpoint's*
`config.json`; the dataclass defaults in `Gr00tN1d7Config` (16 DiT blocks, no VL
self-attention, `select_layer=12`) are dead code there and have never described a real
run. This pipeline builds from the dataclass instead — the checkpoint describes a
different model — so those defaults become live. Left inherited they would have quietly
produced a half-size denoiser with no VL self-attention.

### The optional perception overlay, and what it costs

`--perception-model-path` overlays a trained perception head **and the backbone it was
trained against**, which overrides the base backbone (the head's features are only
meaningful on it). It buys a head that already localizes to 0.85 px.

It is not free. `checkpoints-perceptionv1` was trained at `select_layer=12` — itself a
consequence of the same stale-default trap — so reusing it means:

| | pure GR00T (default) | with the perceptionv1 overlay |
|---|---|---|
| LLM layers | 16 (all of them) | 12 — top 4 discarded |
| action stack | pretrained on the features it now sees | pretrained on 16-layer features, fed 12-layer |
| perception head | trains from scratch | starts at 0.85 px |
| conditioning calibration | must be re-measured | the shipped defaults apply |

**Default recommendation: start clean, no overlay.** Teacher forcing is what makes this
safe — during training token A comes from the *label*, not the head, so a weak early
perception head does not hold back action learning. Only inference quality depends on the
head, and it gets the same losses on the same data as the perception run did, on a
strictly better backbone. Reusing perceptionv1 locks the whole model to a truncated trunk
to save a head that will retrain anyway.

If you do overlay it, pass `SELECT_LAYER=12`. The pipeline **hard-fails** rather than
assembling a mixed backbone, and also fails if the overlay contributes zero
`perception_head` tensors — so a silent partial load is not a failure mode you can hit.
Re-run `scripts/calibrate_perception_conditioning.py` afterwards either way: the shipped
noise and the +6.5 mm bias describe perceptionv1 specifically, and injecting that bias
into a head that does not have it would be actively harmful.

## Training

```bash
# Cluster (Euler) — the recommended run:
RUN_NAME=perception_action_v1 sbatch submit_perception_action_euler.sh

# Reusing an existing perception head instead (needs ITS select_layer):
RUN_NAME=perception_action_reuse \
  PERCEPTION_MODEL_PATH=/groot/runs/perceptionv1 SELECT_LAYER=12 \
  sbatch submit_perception_action_euler.sh

# Locally / inside the container:
RUN_NAME=perception_action_v1 bash start_finetune_perception_action.sh

# Or directly:
python gr00t/experiment/launch_finetune_perception_action.py \
    --dataset-path ./g1_dataset --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path register_perception_action_modality.py \
    --output-dir <dir> --tune-visual \
    --image-target-size 720 --image-crop-size 648 --action-horizon 30
```

**The ablation is one env var.** `NO_PERCEPTION_CONDITIONING=1` trains the identical model
with the perception head as a pure auxiliary task and no tokens reaching the DiT. That is
the baseline that shows whether the conditioning — rather than just the extra supervision —
is what helps. Run it.

### Data

Wrist camera only, single current frame (`register_perception_action_modality.py`), with a
30-step **relative** drill-tip action chunk and the same proprioceptive state keys as the
full VLA (so a VLA checkpoint's state encoder transfers). Training samples every
sidecar frame (`data.perception_labels_only=True`), matching the perception model. The
horizon-30 action chunk necessarily excludes the last 29 frames of each episode because
those frames cannot begin a complete target chunk.

### What to watch in wandb

| Metric | Meaning |
|---|---|
| `action_loss_scalar` | the flow-matching loss alone |
| `perception_loss` | the whole perception term (its own weighted sum) before `perception_loss_weight` |
| `uv_err_px_metric` | perception quality — **must not degrade** once actions are trained |
| `cond_uv_err_px_metric` | px error of what the policy is *actually* conditioned on (teacher noise included) — should sit near `perception_cond_noise_uv_px` |
| `cond_vis_mean_metric` | fraction of steps the policy is told the target is visible |
| `heatmap_loss`, `offset_loss`, `visibility_loss`, `depth_loss`, `normal_loss` | as in the perception-only run |

`loss == action_loss_scalar + perception_loss_weight * perception_loss`, so the lines
reconcile. If `uv_err_px_metric` degrades once the action loss is in play, the action
objective is corrupting the heatmap through token B — set
`--perception-appearance-stopgrad` to cut that path.

## Files

| File | Role |
|---|---|
| `gr00t/configs/model/gr00t_n1d7_perception_action.py` | `Gr00tN1d7PerceptionActionConfig` (conditioning knobs) |
| `gr00t/model/gr00t_n1d7/perception_action.py` | `Gr00tN1d7PerceptionAction` + `PerceptionConditioner` |
| `gr00t/model/gr00t_n1d7/processing_perception_action.py` | Processor registration (keypoint-aware, from the perception processor) |
| `gr00t/model/gr00t_n1d7/setup_perception_action.py` | Pipeline: the two-overlay weight init |
| `gr00t/model/gr00t_n1d7/perception.py` | `TargetPerceptionHead` + the shared `perception_losses` / `decode_uv` / `image_token_grid` helpers |
| `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` | `Gr00tN1d7ActionHead` — `extra_cond_tokens` plumbing |
| `register_perception_action_modality.py` | Wrist-only video + 30-step relative action chunk |
| `gr00t/experiment/launch_finetune_perception_action.py` | CLI entry point (`PerceptionActionFinetuneConfig`) |
| `start_finetune_perception_action.sh` / `setup_and_train_perception_action.sh` / `submit_perception_action_euler.sh` | Launch scripts |
| `tests/gr00t/model/test_perception_action.py` | CPU tests (mocked backbone): tokens, teacher forcing, gradient routing, whole-model forward |

## Caveats and known limits

- **Wrist-only, one frame.** The perception head reads the VLM image-token grid as a
  single spatial map, so exactly one image per sample is required. Adding `ego_view` or a
  previous frame means generalizing `image_token_grid` to slice one image's tokens out of
  several (the processor stacks time-major, so wrist-current is the last image) *and*
  applying keypoint-aware augmentation to only the wrist view. Not done here — this model
  is the close-range specialist; the coarse approach stays with the full VLA.
- Rotation augmentation stays off: the keypoint follows the transform, the wall-normal
  label does not.
- `get_action` never teacher-forces, even if labels happen to be present in the batch.
- The tip-target head and this conditioning path are independent; enabling both would put
  two competing target estimates in front of the DiT. Leave `use_tip_target_head` off.
