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

from dataclasses import dataclass, field

from . import register_model_config
from .gr00t_n1d7_perception import Gr00tN1d7PerceptionConfig


@dataclass
class Gr00tN1d7PerceptionActionConfig(Gr00tN1d7PerceptionConfig):
    """Config for the perception-CONDITIONED action model (fine approach).

    Same wrist-only VLM backbone + TargetPerceptionHead as the perception-only
    model, PLUS the standard GR00T flow-matching action head. The perception
    output is cast into conditioning tokens appended to the state tokens, so the
    DiT generates actions conditioned on where the target is:

      token A ("where"):  Fourier-encoded (u, v) in [0,1] + visibility +
                          heatmap confidence (top-1 prob, normalized entropy)
                          + optional depth and wall normal, each with a valid
                          flag -> MLP -> one token. Blended with a learned
                          `unknown_token` by the visibility weight, so "I don't
                          see the target" is a distinct, learnable input rather
                          than a meaningless coordinate.
      token B ("what"):   heatmap-weighted pool of the image tokens -> Linear ->
                          one token. Also the only differentiable path from the
                          action loss back into the heatmap (token A is blocked
                          by argmax decoding and by teacher forcing).

    Token A is TEACHER-FORCED on the labels + calibrated noise during training
    and fed the head's own prediction at inference, mirroring the tip-target
    scheme in Gr00tN1d7ActionHead.

    Why Fourier-encode (u, v): at 720 px, 1 px is 0.0014 in normalized units. A
    plain Linear on raw coordinates cannot resolve that above its own weight
    noise, which would throw away the sub-pixel precision the heatmap head was
    trained for at the conditioning interface. Multi-octave sin/cos makes 1-px
    differences linearly separable.
    """

    # Model identification (overrides Gr00tN1d7Perception)
    model_type: str = "Gr00tN1d7PerceptionAction"

    # --- Architecture: must match the checkpoints we initialize from ---
    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 32,
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 1024,
            "interleave_self_attention": True,
        }
    )
    """DiT geometry, pinned to nvidia/GR00T-N1.7-3B's own config (32 blocks).

    DO NOT fall back to the Gr00tN1d7Config default of 16 here. That default is dead
    code on the VLA path — Gr00tN1d7Pipeline calls AutoModel.from_pretrained, which
    takes the architecture from the CHECKPOINT's config.json (32 blocks), so no real
    run has ever used 16. This pipeline builds the model from this dataclass instead
    (the checkpoint describes a different model), so the value here is live: at 16 it
    would silently construct a half-size denoiser and inherit only blocks 0-15 of the
    pretrained DiT."""

    vl_self_attention_cfg: dict = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 4,
            "num_attention_heads": 32,
            "attention_head_dim": 64,
            "dropout": 0.2,
            "final_dropout": True,
        }
    )
    """The 4-layer self-attention stack the action head runs over the VL features before
    the DiT. Also pinned to the foundation's config — Gr00tN1d7Config does not declare
    this field at all, so leaving it out means nn.Identity() and 64 pretrained tensors
    silently dropped on the floor."""

    select_layer: int = 16
    """Number of LLM layers kept in the backbone. 16 = the foundation's own value, i.e.
    the whole Cosmos-Reason2-2B trunk that GR00T-N1.7-3B's action head was trained
    against.

    Lower it ONLY to match a perception checkpoint you are overlaying. The existing
    perceptionv1 checkpoint was trained at 12 (the stale dataclass default), so reusing
    it requires --select-layer 12 and costs you the top 4 LLM layers plus the alignment
    between the pretrained action stack and the features it sees.
    Gr00tN1d7PerceptionActionPipeline hard-fails rather than assembling a mixed backbone,
    so a mismatch is loud, not silent."""

    # --- Perception loss vs action loss ---
    perception_loss_weight: float = 1.0
    """Multiplier on the whole (already internally weighted) perception loss when it is
    added to the action loss. Set to 0 to freeze perception supervision and train the
    action head only against the conditioning tokens."""

    # --- Conditioning tokens ---
    use_perception_conditioning: bool = True
    """Master switch. If False the perception head is still trained (auxiliary task) but
    the action head receives no perception tokens."""

    use_perception_geometry_token: bool = True
    """Emit token A (Fourier-encoded uv + visibility + confidence + depth/normal)."""

    use_perception_appearance_token: bool = True
    """Emit token B (heatmap-weighted pooled image feature)."""

    perception_num_fourier_bands: int = 8
    """Octave count for the (u, v) positional encoding: frequencies 2^0..2^(n-1) cycles
    across the image. 8 -> highest period ~5.6 px at 720, so a 1 px shift moves the
    highest band by ~0.7 rad."""

    perception_appearance_temperature: float = 1.0
    """Softmax temperature for the heatmap used to pool image tokens for token B. <1
    sharpens toward the argmax cell, >1 blends more context."""

    perception_appearance_stopgrad: bool = False
    """If True, detach the heatmap before pooling token B, removing the gradient path
    from the action loss into the perception head. Turn on if the action loss is seen
    corrupting uv_err_px_metric."""

    # --- Teacher forcing / exposure bias ---
    perception_cond_teacher_forcing: bool = True
    """If True, condition on the LABEL (plus the noise below) whenever a label exists,
    and on the head's own detached prediction otherwise. If False, always condition on
    the prediction."""

    perception_cond_noise_uv_px: float = 3.0
    """Std (processed-image px) of the Gaussian noise added to the teacher-forced (u, v).

    Measured, not guessed: scripts/calibrate_perception_conditioning.py on the
    1500_perception_planner eval gives a single-frame lateral error of 0.85 processed px
    rms (n=16), inflated 3x here. The inflation is deliberate — that eval only exercises
    the perception model with a PARKED camera and only out to 67 mm, while this model
    runs it every frame, while moving, out to 150 mm.

    RE-DERIVE for every new perception checkpoint; the value is a property of the
    checkpoint. It must be >= the real error: noise smaller than deployment error
    teaches the policy to trust perfect perception and makes it brittle on the robot.
    For scale, 1 processed px is 0.28-0.46 mm at the 0.11-0.19 m working depth."""

    perception_cond_noise_depth_m: float = 0.004
    """Std (metres) of the RANDOM part of the depth error. Measured single-frame spread
    4.32 mm, with the systematic part split out into perception_cond_bias_depth_m."""

    perception_cond_bias_depth_m: float = 0.0
    """SYSTEMATIC depth offset (metres) added to the teacher-forced depth, so the policy
    trains on the depth it will actually be handed rather than an unbiased one.

    Defaults to 0 because the default run trains a FRESH perception head, whose depth
    bias is unknown. Unlike the noise terms, a bias is a directional claim: applying
    +6.5 mm to a head that does not have it would teach the policy to compensate for an
    offset that is not there, leaving it wrong by that much in the other direction. Noise
    is a robustness measure and is safe to keep on; a bias is not.

    Set it when you overlay a checkpoint whose bias you have measured. For
    checkpoints-perceptionv1 that value is 0.0065 (+6.49 mm on the
    1500_perception_planner eval, 84% of its mean-square depth error, and unaffected by
    averaging detections). Applied in eval mode too — a bias the model always exhibits is
    not a training-time perturbation. Re-measure with
    scripts/calibrate_perception_conditioning.py."""

    perception_cond_noise_normal: float = 0.04
    """Std (unit-vector units) of the noise added to the teacher-forced wall normal.
    From the same eval's rotation-error spread (3.49 deg -> 0.043 per component). The
    least reliable of the four: `rel_err_rot_deg` carries a ~164 deg frame-convention
    offset, so only its spread is meaningful."""

    perception_cond_vis_dropout_prob: float = 0.1
    """Probability of asserting "target not visible" during training even when the label
    says visible. Teaches the policy a search/hold behaviour instead of collapsing when
    perception drops out mid-approach."""

    def __init__(self, **kwargs):
        # Defining __init__ here stops @dataclass from generating one; the parent
        # __init__ handles PretrainedConfig setup and fills dataclass-field
        # defaults (including this subclass's fields) via __dataclass_fields__.
        super().__init__(**kwargs)


register_model_config("Gr00tN1d7PerceptionAction", Gr00tN1d7PerceptionActionConfig)
