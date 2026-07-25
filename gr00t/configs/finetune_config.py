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

    tip_target_pos_loss_weight: float = 1.0
    """Weight of the position term: squared xyz error normalized by the distance to the target
    (clamped to tip_target_dist_floor), plus an absolute anchor (see tip_target_pos_abs_weight).
    The relative part gives close-range frames as much gradient as far ones -- plain MSE would
    spend nearly all gradient on far, coarse frames."""

    tip_target_pos_abs_weight: float = 0.2
    """Weight of the absolute (scaled-space MSE) anchor inside the position term. Required: a
    purely relative loss caps far-range error cost at ~1 per frame, making 'always predict the
    close-range mean' a cheap constant minimizer (observed collapse). Raise to prioritize
    far-range fidelity, lower (not to 0) to prioritize close-range precision."""

    tip_target_rot_loss_weight: float = 1.0
    """Weight of the rotation term: plain MSE on the raw rot6d part of the prediction."""

    tip_target_dist_floor: float = 0.05
    """Distance floor (metres) for the position-loss normalization. Below this distance the
    divisor stops shrinking, so the loss does not blow up as the tip reaches the target. Also
    bounds how much extra weight close frames get vs far ones ((far/floor)^2)."""

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
    """Total number of training steps to run before stopping."""

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
class PerceptionFinetuneConfig:
    """Configuration for fine-tuning the perception-only model (VLM backbone +
    target heatmap head) used in the short-range fine approach.

    The model sees ONLY the current wrist-camera frame + instruction and predicts
    the wall-target location as a heatmap in wrist-camera pixels, a visibility
    flag, and optionally target depth / wall normal. Training samples only the
    labelled fine-phase frames (heatmap_labels/ sidecar next to data/).
    """

    # --- Data and Model Paths ---
    base_model_path: str | None = None
    """Checkpoint to initialize the VLM backbone from. Accepts a full VLA checkpoint
    (e.g. one of your finetuned GR00T runs — action-head weights are ignored) or a
    previous perception checkpoint. If None, starts from the raw Cosmos-Reason2-2B
    backbone weights."""

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

    depth_loss_weight: float = 0.1
    """Weight of the target-depth loss (0 disables the depth output)."""

    wall_normal_loss_weight: float = 0.1
    """Weight of the wall-normal loss (0 disables the wall-normal output)."""

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
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    shard_size: int = 2**10
    episode_sampling_rate: float = 0.1
    num_shards_per_epoch: int = int(1e5)
    save_only_model: bool = False
    skip_weight_loading: bool = False
    """If True, skip loading weights from base_model_path (smoke tests)."""
