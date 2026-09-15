# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Finetune config used for single node post-training.
from dataclasses import dataclass


GR00T_BASE_MODEL = "nvidia/GR00T-N1.7-3B"
"""The one checkpoint the perception-action model ever starts from. Its architecture is
pinned in Gr00tN1d7PerceptionActionConfig, so this is not a user choice."""


@dataclass
class FinetuneConfig:
    """
    Configuration for fine-tuning a Vision-Language-Action (VLA) model.

    This dataclass defines all parameters needed to launch a fine-tuning job
    on a pretrained base model using a custom dataset and embodiment-specific
    modality configuration. It controls model tuning options, data augmentation,
    and training hyperparameters.
    """

    # --- Data and Model Paths ---
    base_model_path: str
    """Path to the pretrained base model checkpoint (e.g., Hugging Face model hub or local directory)."""

    dataset_path: str
    """Path to the dataset root directory containing trajectory data for fine-tuning."""

    embodiment_tag: str
    """Embodiment tag (name or value, case-insensitive). See EmbodimentTag for known tags."""

    modality_config_path: str | None = None
    """
    Path to a Python file defining the modality configuration for the given embodiment. 
    If None, use the pre-registered modality config in `gr00t/configs/data/embodiment_configs.py`. 
    """

    # --- Model Tuning Flags ---
    tune_llm: bool = False
    """If True, fine-tune the language model (LLM) backbone during training."""

    tune_visual: bool = False
    """If True, fine-tune the visual encoder (e.g., ViT or CNN backbone)."""

    tune_projector: bool = True
    """If True, fine-tune the multimodal projector layers that map vision/language features to a shared space."""

    tune_diffusion_model: bool = True
    """If True, fine-tune the diffusion-based action decoder (if present in the model)."""

    use_tip_target_head: bool = False
    """If True, add an auxiliary head that predicts the target pose in the end-effector (tip) frame
    as xyz(3) + rot6d(6) from the VLM latent (target = the `next.tip_target_9d` column in the
    parquet; NaN rows are masked out). The distance to target is derived as ||xyz||. Useful for
    precise end-effector placement: the action head is conditioned on where the target is relative
    to the tip."""

    tip_target_loss_weight: float = 0.5
    """Overall multiplier on the tip-target loss (pos + rot terms) when added to the action loss
    (only used if use_tip_target_head)."""

    tip_target_loss_type: str = "mse"
    """Tip-target objective. "mse" (default): the per-part MSEs of xyz and rot6d averaged
    50/50. Originally a single MSE over the whole scaled 9d vector, which gave rot6d 2x the
    weight of xyz simply by occupying 6 of the 9 averaged dims; the 50/50 split removes that.
    The comparison below was measured against that original 9d form. "rel_abs": position error
    normalized by the distance to target plus an absolute anchor. rel_abs is MEASURED WORSE
    (5.7x worse mean position error in-distribution, predicted-distance regression slope 0.82
    vs 1.02, -98 mm bias beyond 30 cm): dividing by distance makes far-range error nearly free
    so the model shrinks its range toward the near-field mean, and below tip_target_dist_floor
    the relative term is just absolute error times a constant, so it buys nothing up close
    either. Kept only so the comparison stays reproducible."""

    tip_target_pos_loss_weight: float = 1.0
    """Weight of the position term. Applies to tip_target_loss_type="rel_abs" ONLY; inert
    under "mse"."""

    tip_target_pos_abs_weight: float = 0.2
    """Weight of the absolute (scaled-space MSE) anchor inside the position term. Applies to
    tip_target_loss_type="rel_abs" ONLY; inert under "mse"."""

    tip_target_rot_loss_weight: float = 1.0
    """Weight of the rotation term (plain MSE on the raw rot6d part). Applies to
    tip_target_loss_type="rel_abs" ONLY; inert under "mse", which weights rot6d and xyz
    equally at 0.5 each."""

    tip_target_dist_floor: float = 0.05
    """Distance floor (metres) for the position-loss normalization. Applies to
    tip_target_loss_type="rel_abs" ONLY; inert under "mse"."""

    tip_target_pool: str = "attn"
    """How the variable-length vision-language token sequence is collapsed into the single
    vector fed to the head's predictor MLP. "attn" (default): `tip_target_num_queries` learned
    query tokens cross-attend over the whole sequence, so the head can focus on the patches
    containing the target and on the text tokens that name it. "mean": masked mean over all
    tokens. These give the predictor different input widths (num_queries*latent_dim vs
    latent_dim), so the two are NOT weight-compatible -- a checkpoint trained with one cannot
    be resumed with the other."""

    tip_target_num_queries: int = 4
    """Number of learned query tokens for tip_target_pool="attn" (inert under "mean")."""

    tip_target_attn_heads: int = 8
    """Attention heads of the pooling cross-attention for tip_target_pool="attn" (inert under
    "mean")."""

    use_tip_target_conditioning: bool = True
    """If True, feed the tip-target 9d back as a conditioning token to the action head.
    Teacher-forced: training conditions on the ground truth (+ small noise), inference on the
    head's prediction."""

    tip_target_xyz_scale: float = 0.1
    """Scale (metres) dividing the xyz part of the tip-target label inside the head so position and
    rot6d errors contribute comparable gradients. Stored in the model config so inference matches."""

    state_dropout_prob: float = 0.2
    """
    Dropout probability applied to state inputs for regularization during training.
    """

    # --- Image Resolution ---
    image_target_size: int = 256
    """Square edge length (px) the image is resized to before it is fed to the VLM
    (model config `image_target_size`). Larger = more detail preserved for small
    targets, at the cost of more visual tokens / GPU memory. The Cosmos-Reason2 (Qwen3-VL)
    backbone uses native dynamic resolution, so raising this is safe. Must be > image_crop_size."""

    image_crop_size: int = 230
    """Square edge length (px) of the crop taken during the resize pipeline (model config
    `image_crop_size`). Kept slightly below image_target_size (default ratio ~0.9) to allow
    the random/center crop augmentation. Set together with image_target_size."""

    # --- Data Augmentation ---
    random_rotation_angle: int | None = None
    """Maximum rotation angle (in degrees) for random rotation augmentation of input images."""

    color_jitter_params: dict[str, float] | None = None
    """
    Parameters for color jitter augmentation on images.

    Expected keys include:
      - "brightness": float
      - "contrast": float
      - "saturation": float
      - "hue": float
    Example: {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}

    If None, applying the default color jitter augmentation from the pretrained model.
    """
    extra_augmentation_config: str | None = None
    """
    JSON string for extra image augmentations (mask-based and others).

    Expected keys include:
      - "background_noise_transforms": list of dicts for noise on mask regions
          - "target_mask_values": list of int (e.g., [0])
          - "p": float (probability of applying)
      - "masked_region_transforms": list of dicts for color tint on mask regions
          - "target_mask_values": list of int (e.g., [4] or [5])
          - "p": float (probability of applying)
          - "alpha_range": [min, max] for random_tint intensity

    Example: {"background_noise_transforms": [{"target_mask_values": [0], "p": 0.9}],
              "masked_region_transforms": [{"target_mask_values": [4], "p": 1.0, "alpha_range": [0, 1]}]}

    If None, no extra augmentations are applied.
    """

    # --- Training Configuration ---
    global_batch_size: int = 64
    """Total effective batch size across all GPUs and accumulation steps."""

    dataloader_num_workers: int = 2
    """Number of parallel worker processes used for data loading."""

    learning_rate: float = 1e-4
    """Initial learning rate for optimizer."""

    gradient_accumulation_steps: int = 1
    """Number of forward passes to accumulate before performing a backward/update step."""

    output_dir: str = "./outputs"
    """Directory where model checkpoints, logs, and outputs are saved."""

    experiment_name: str | None = None
    """Optional experiment name used as the W&B run name. Defaults to the output directory basename."""

    wandb_project: str = "finetune-gr00t-n1d7"
    """W&B project name to log runs to."""

    save_steps: int = 1000
    """Frequency (in training steps) at which to save checkpoints."""

    save_total_limit: int = 5
    """Maximum number of checkpoints to keep before older ones are deleted."""

    num_gpus: int = 1
    """Number of GPUs available for distributed or single-node training."""

    use_wandb: bool = False
    """
    If True, log metrics and artifacts to Weights & Biases (wandb).
    The project is `finetune-gr00t-n1d7`.
    You need to login to wandb to view the logs.
    """

    max_steps: int = 10000
    """Total number of optimizer steps to run before stopping. Ignored when num_epochs is set."""

    num_epochs: float | None = None
    """If set, derive max_steps from the dataset size instead of specifying it directly:
    max_steps = ceil(num_epochs * trainable_steps / (global_batch_size * gradient_accumulation_steps)),
    where trainable_steps counts the timesteps that can start a full action chunk
    (sum over episodes of length - action_horizon + 1, read from meta/episodes.jsonl).
    Fractional values are allowed (0.5 = half a pass).

    CAVEAT: this is a sample-count target, not a true pass over the data. The training
    iterator draws batches from a shard buffer with replacement, so after "1 epoch" some
    frames have been seen several times and others not at all -- read it as "each frame
    seen num_epochs times in expectation". Note also that warmup_ratio is a fraction of
    max_steps and so scales automatically, but save_steps is absolute and does not."""

    weight_decay: float = 1e-5
    """Weight decay coefficient for optimizer (L2 regularization)."""

    warmup_ratio: float = 0.05
    """Proportion of total training steps used for learning rate warm-up."""

    shard_size: int = 2**10
    """Size of the shard to use for the dataset during preloading."""

    episode_sampling_rate: float = 0.1
    """Sampling rate for the episodes."""

    num_shards_per_epoch: int = int(1e5)
    """Number of shards to use for the dataset. reduce this number if vram is limited."""

    save_only_model: bool = False
    """If True, save only model weights (skip optimizer/scheduler/RNG states). Cannot resume training from these checkpoints."""

    skip_weight_loading: bool = False
    """If True, skip loading model weights from base_model_path (architecture only).
    The processor (tokenizer/config) is still loaded from base_model_path.
    Useful for CI/testing to skip the slow checkpoint shard loading."""


@dataclass
class _PerceptionCommonConfig:
    """Settings shared by the perception-only and perception-conditioned-action
    finetunes. Split out so the action variant can omit `base_model_path`: that model
    always starts from GR00T_BASE_MODEL and exposing a choice there would let a
    mismatched architecture load silently."""

    dataset_path: str = "./g1_dataset"
    """Path to the dataset root (must contain the heatmap_labels/ sidecar directory)."""

    embodiment_tag: str = "NEW_EMBODIMENT"
    """Embodiment tag (name or value, case-insensitive)."""

    modality_config_path: str | None = None
    """Path to a Python file registering the WRIST-ONLY modality config
    (see register_perception_modality.py)."""

    # --- Model Tuning Flags ---
    tune_llm: bool = False
    """If True, fine-tune the (truncated) language model of the backbone."""

    tune_visual: bool = True
    """If True, fine-tune the vision encoder. Recommended: the heatmap precision
    comes almost entirely from the visual features."""

    tune_top_llm_layers: int = 0
    """Number of top LLM layers to tune (0 = none). Cheaper alternative to tune_llm
    for adapting the fused vision-language features."""

    # --- Perception Head / Loss ---
    heatmap_hidden_dim: int = 512
    """Hidden width of the perception head MLPs."""

    heatmap_target_sigma_px: float = 8.0
    """Std (processed-image px) of the Gaussian target rendered around the label."""

    heatmap_loss_weight: float = 1.0
    """Weight of the heatmap cross-entropy (visible frames only)."""

    offset_loss_weight: float = 1.0
    """Weight of the subpixel offset loss at the ground-truth cell."""

    visibility_loss_weight: float = 0.5
    """Weight of the visibility BCE (all frames)."""

    visibility_neg_weight: float = 1.0
    """Per-sample weight on target_visible=0 rows in the visibility BCE. 1.0 is the
    unweighted loss. If class balancing is desired, derive n_pos/n_neg from the
    generated sidecar; the ratio is dataset-dependent now that every frame is used."""

    depth_loss_weight: float = 0.1
    """Weight of the target-depth loss (0 disables the depth output)."""

    wall_normal_loss_weight: float = 0.1
    """Weight of the wall-normal loss (0 disables the wall-normal output)."""

    # --- Language conditioning ---
    use_text_conditioning: bool = True
    """Let the perception head cross-attend over the instruction tokens.

    Mandatory on multi-target datasets (several candidate squares in frame, the
    instruction picks one): the causal backbone puts the whole image block before the
    instruction, so without this the head cannot see the instruction at all. Pass
    --no-use-text-conditioning to reproduce the old language-blind behaviour."""

    # --- Image Resolution ---
    image_target_size: int = 720
    """Square edge (px) the wrist image is resized to before the VLM. Native wrist
    resolution is 720; keep it for maximum px precision at short range."""

    image_crop_size: int = 648
    """Square edge (px) of the random/center crop (must be < image_target_size)."""

    # --- Data Augmentation ---
    color_jitter_params: dict[str, float] | None = None
    """Color jitter parameters ({"brightness","contrast","saturation","hue"}).
    Keep hue/saturation LOW: the target is red-semantic. NOTE: never enable
    rotation augmentation here without care — the keypoint follows the transform,
    but the wall-normal label does not."""

    # --- Training Configuration ---
    global_batch_size: int = 32
    dataloader_num_workers: int = 2
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 1
    output_dir: str = "./outputs-perception"
    experiment_name: str | None = None
    wandb_project: str = "finetune-gr00t-n1d7-perception"
    save_steps: int = 1000
    save_total_limit: int = 5
    num_gpus: int = 1
    use_wandb: bool = False
    max_steps: int = 10000
    """Total optimizer steps to run. Ignored when num_epochs is set."""

    num_epochs: float | None = None
    """If set, derive max_steps from the dataset size instead of specifying it directly.
    Same meaning and same caveat as FinetuneConfig.num_epochs, with one difference that
    matters: these finetunes run with perception_labels_only=True, so an "epoch" is
    counted over every row in the heatmap_labels/ sidecar, not over episodes without
    labels. Datasets without that sidecar cannot use num_epochs at all."""

    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    shard_size: int = 2**10
    episode_sampling_rate: float = 0.1
    num_shards_per_epoch: int = int(1e5)
    save_only_model: bool = False
    skip_weight_loading: bool = False
    """If True, skip loading weights from base_model_path (smoke tests)."""


@dataclass
class PerceptionFinetuneConfig(_PerceptionCommonConfig):
    """Fine-tune the perception-only model (VLM backbone + target heatmap head).

    The model sees ONLY the current wrist-camera frame + instruction and predicts the
    wall-target location as a heatmap in wrist-camera pixels, a visibility flag, and
    optionally target depth / wall normal. Training samples every frame represented in
    the heatmap_labels/ sidecar next to data/.
    """

    base_model_path: str | None = None
    """Checkpoint to initialize the VLM backbone from. Accepts a full VLA checkpoint
    (e.g. one of your finetuned GR00T runs — action-head weights are ignored) or a
    previous perception checkpoint. If None, starts from the raw Cosmos-Reason2-2B
    backbone weights."""


@dataclass
class PerceptionActionFinetuneConfig(_PerceptionCommonConfig):
    """Configuration for the perception-CONDITIONED action model (fine approach).

    Same wrist-only inputs and heatmap supervision as PerceptionFinetuneConfig, plus
    the flow-matching action head. The perception output is cast into conditioning
    tokens the DiT attends to, so the action chunk depends on where the target is
    (see Gr00tN1d7PerceptionActionConfig for the token layout).
    """

    # --- Weight initialization (see Gr00tN1d7PerceptionActionPipeline) ---
    # The base is ALWAYS nvidia/GR00T-N1.7-3B — there is no flag for it. The model
    # config is pinned to that checkpoint's architecture (32 DiT blocks, 4 VL
    # self-attention layers, 16 LLM layers), so any other base would load partially
    # and silently. GR00T_BASE_MODEL below is the single source of truth.

    perception_model_path: str | None = None
    """OPTIONAL perception checkpoint overlaid on top of the GR00T base. Supplies the
    trained perception head AND the backbone it was trained against, which overrides the
    base backbone — the head's features are only meaningful on that backbone.

    Not free: the existing perceptionv1 checkpoint was trained at select_layer=12, so
    using it forces --select-layer 12 and gives up the foundation's top 4 LLM layers plus
    the match between GR00T's pretrained action stack and the features it now sees. Left
    unset, the perception head trains from scratch on a fully coherent GR00T init — which
    teacher forcing makes safe, since the conditioning token comes from the LABEL during
    training and only inference quality depends on the head."""

    select_layer: int = 16
    """LLM layers kept in the backbone. 16 = the foundation's own value. Set to 12 only to
    match a perceptionv1-style overlay (the pipeline hard-fails on a mismatch)."""

    # --- Action head ---
    tune_projector: bool = True
    """If True, fine-tune the state/action encoders and decoder."""

    tune_diffusion_model: bool = True
    """If True, fine-tune the DiT."""

    state_dropout_prob: float = 0.2
    """Dropout applied to state inputs in the processor, for regularization."""

    # --- Perception loss vs action loss ---
    perception_loss_weight: float = 1.0
    """Multiplier on the whole perception loss when added to the action loss. Keep it at
    1.0 unless the perception metrics degrade once the action loss is in play; 0 trains
    the action head against a frozen-quality perception signal."""

    # --- Conditioning tokens ---
    use_perception_conditioning: bool = True
    """Master switch for the conditioning tokens. If False the perception head is still
    trained as an auxiliary task but the action head cannot see it (useful as the ablation
    baseline that shows the conditioning is what helps)."""

    use_perception_geometry_token: bool = True
    """Emit the "where" token (Fourier-encoded uv + visibility + confidence + depth/normal)."""

    use_perception_appearance_token: bool = True
    """Emit the "what" token (heatmap-weighted pooled image feature)."""

    perception_num_fourier_bands: int = 8
    """Octaves in the (u, v) positional encoding. 8 resolves ~1 px at 720; raising it
    costs nothing but can make the token noisier."""

    perception_appearance_temperature: float = 1.0
    """Softmax temperature of the heatmap used to pool the appearance token."""

    perception_appearance_stopgrad: bool = False
    """Cut the gradient from the action loss into the heatmap (through the appearance
    token). Turn on only if uv_err_px_metric degrades once actions are trained."""

    # --- Teacher forcing / exposure bias ---
    perception_cond_teacher_forcing: bool = True
    """Condition on the label (+ noise below) where one exists, on the head's own
    prediction elsewhere. Matches the tip-target scheme."""

    perception_cond_noise_uv_px: float = 3.0
    """Std (processed-image px) of the noise added to the teacher-forced (u, v).

    Measured via `scripts/calibrate_perception_conditioning.py` on the
    1500_perception_planner eval: 0.85 processed px rms for a single frame, inflated 3x
    because that eval only runs perception with a PARKED camera out to 67 mm while this
    model runs it every frame, moving, out to 150 mm. RE-RUN the script for every new
    perception checkpoint. Noise smaller than the real deployment error teaches the
    policy to trust perfect perception and it will be brittle on the robot. 1 processed
    px is 0.28-0.46 mm at the working depth (f = 388 px at the raw 720 camera, recovered
    from the dataset)."""

    perception_cond_noise_depth_m: float = 0.004
    """Std (metres) of the RANDOM part of the depth error (measured single-frame spread
    4.32 mm). The systematic part is perception_cond_bias_depth_m."""

    perception_cond_bias_depth_m: float = 0.0
    """Systematic depth offset (metres) added to the teacher-forced depth. 0 by default
    because the default run trains a fresh perception head whose bias is unknown, and a
    wrong bias is worse than none — it teaches the policy to compensate for an offset that
    is not there. Set to 0.0065 when overlaying checkpoints-perceptionv1 (its measured
    +6.49 mm, 84% of its mean-square depth error). Applied in eval mode too."""

    perception_cond_noise_normal: float = 0.04
    """Std of the noise on the teacher-forced wall normal. From the same eval's rotation
    spread (3.49 deg -> 0.043 per component); the least reliable of the four."""

    perception_cond_vis_dropout_prob: float = 0.1
    """Probability of telling the policy "target not visible" even when the label says it
    is, so it learns to hold/search instead of collapsing when perception drops out."""

    # --- Overrides of the perception defaults ---
    action_horizon: int = 30
    """Number of future steps predicted per chunk. MUST equal the length of the action
    delta_indices in the modality config (see register_perception_action_modality.py)."""

    output_dir: str = "./outputs-perception-action"
    wandb_project: str = "finetune-gr00t-n1d7-perception-action"
