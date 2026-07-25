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

from dataclasses import dataclass

from . import register_model_config
from .gr00t_n1d7 import Gr00tN1d7Config


@dataclass
class Gr00tN1d7PerceptionConfig(Gr00tN1d7Config):
    """Config for the perception-only GR00T model (VLM backbone + target heatmap head).

    Used by the short-range fine-approach stage: the model sees ONLY the wrist
    camera (single current frame) + the task instruction, and predicts
      - a spatial heatmap over the VLM image-token grid locating the wall
        target in wrist-camera pixel coordinates,
      - a subpixel (du, dv) offset at each cell (CenterNet-style refinement),
      - a visibility logit ("is the target in this image at all?"),
      - optionally the target depth (camera-frame z) and the wall normal.

    No action head / DiT is instantiated; state and action inputs are ignored.
    Labels come from the heatmap_labels/ sidecar parquets and training samples
    only the fine-phase frames (data.fine_phase_only=True).
    """

    # Model identification (overrides Gr00tN1d7)
    model_type: str = "Gr00tN1d7Perception"

    # Heatmap head
    heatmap_hidden_dim: int = 512
    """Hidden width of the per-token MLP producing heatmap/offset logits."""

    heatmap_target_sigma_px: float = 8.0
    """Std (in FINAL processed-image pixels) of the Gaussian rendered around the
    labelled target center to form the soft heatmap target."""

    heatmap_loss_weight: float = 1.0
    """Weight of the heatmap cross-entropy (visible frames only)."""

    offset_loss_weight: float = 1.0
    """Weight of the subpixel offset L1 loss at the ground-truth cell (visible frames only)."""

    visibility_loss_weight: float = 0.5
    """Weight of the visibility BCE (all frames)."""

    depth_loss_weight: float = 0.1
    """Weight of the target-depth SmoothL1 loss (visible frames only). 0 disables the head."""

    depth_scale: float = 0.3
    """Metres; depth is divided by this inside the head so its loss is O(1) at close range."""

    wall_normal_loss_weight: float = 0.1
    """Weight of the wall-normal MSE loss (visible frames only). 0 disables the head."""

    perception_pool_heads: int = 8
    """Attention heads of the learned-query pooling used by the visibility/depth/normal heads."""

    def __init__(self, **kwargs):
        # Defining __init__ here stops @dataclass from generating one; the parent
        # __init__ handles PretrainedConfig setup and fills dataclass-field
        # defaults (including this subclass's fields) via __dataclass_fields__.
        super().__init__(**kwargs)


register_model_config("Gr00tN1d7Perception", Gr00tN1d7PerceptionConfig)
