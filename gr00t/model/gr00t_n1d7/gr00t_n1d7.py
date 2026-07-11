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

import logging
from typing import Any, Tuple

import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)


logger = logging.getLogger(__name__)


def _masked_mean_pool(
    vl_embeds: torch.Tensor, attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean-pool VLM token embeddings over valid (non-padding) tokens, shape (B, D)."""
    if attention_mask is None:
        return vl_embeds.mean(dim=1)
    mask = attention_mask[..., None].to(vl_embeds.dtype)
    return (vl_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)


def rot6d_orthonormalize(rot6d: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt a raw 6D rotation prediction into a valid one (Zhou et al., CVPR 2019).

    Input/output shape (..., 6): two 3-vectors; the first is normalized, the second
    is made orthonormal to it.
    """
    a = F.normalize(rot6d[..., :3], dim=-1)
    b = rot6d[..., 3:]
    b = F.normalize(b - (a * b).sum(dim=-1, keepdim=True) * a, dim=-1)
    return torch.cat((a, b), dim=-1)


class ProgressHead(nn.Module):
    """Auxiliary head predicting per-step task progress in [0, 1].

    The head mean-pools the (post-vlln) VLM latent over valid tokens and
    regresses task progress as a raw logit (apply sigmoid for the value). When
    conditioning is enabled it also embeds the estimate back into a single token
    that is appended to the state tokens, letting the action DiT switch regimes
    (e.g. coarse approach vs. fine alignment) based on its own progress estimate
    -- no ground-truth progress is required at inference time.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        cond_dim: int,
        use_conditioning: bool,
    ):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        if use_conditioning:
            self.cond_embed = nn.Sequential(
                nn.Linear(1, cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, cond_dim),
            )
        else:
            self.cond_embed = None

    def predict_logit(
        self, vl_embeds: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Mean-pool valid VLM tokens and return the progress logit, shape (B, 1)."""
        return self.predictor(_masked_mean_pool(vl_embeds, attention_mask))

    def conditioning_token(self, progress_value: torch.Tensor) -> torch.Tensor:
        """Embed a progress value in [0, 1] -> a single conditioning token (B, 1, cond_dim)."""
        assert self.cond_embed is not None
        return self.cond_embed(progress_value).unsqueeze(1)


class TipTargetHead(nn.Module):
    """Auxiliary head predicting the target pose in the end-effector (tip) frame.

    Regresses xyz(3) + rot6d(6) -- the `next.tip_target_9d` label -- from the
    mean-pooled (post-vlln) VLM latent. The head works in a scaled space where
    xyz is divided by `xyz_scale` so millimetre-level position errors and the
    unit-scale rot6d receive comparable MSE gradients. When conditioning is
    enabled, the (scaled) 9d value is embedded into a single token appended to
    the state tokens, so the action DiT can condition fine motion on where the
    target is relative to the tip -- teacher-forced on ground truth during
    training, fed its own prediction at inference.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        cond_dim: int,
        use_conditioning: bool,
        xyz_scale: float,
    ):
        super().__init__()
        self.xyz_scale = xyz_scale
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 9),
        )
        if use_conditioning:
            self.cond_embed = nn.Sequential(
                nn.Linear(9, cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, cond_dim),
            )
        else:
            self.cond_embed = None

    def predict_scaled(
        self, vl_embeds: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict the scaled 9d target (xyz / xyz_scale, raw rot6d), shape (B, 9)."""
        return self.predictor(_masked_mean_pool(vl_embeds, attention_mask))

    def scale(self, tip_target_9d: torch.Tensor) -> torch.Tensor:
        """Metric 9d label -> scaled space the head predicts in."""
        xyz = tip_target_9d[..., :3] / self.xyz_scale
        return torch.cat((xyz, tip_target_9d[..., 3:]), dim=-1)

    def unscale(self, scaled_9d: torch.Tensor, orthonormalize: bool = False) -> torch.Tensor:
        """Scaled prediction -> metric 9d; optionally Gram-Schmidt the rot6d part."""
        xyz = scaled_9d[..., :3] * self.xyz_scale
        rot = scaled_9d[..., 3:]
        if orthonormalize:
            rot = rot6d_orthonormalize(rot)
        return torch.cat((xyz, rot), dim=-1)

    def conditioning_token(self, scaled_9d: torch.Tensor) -> torch.Tensor:
        """Embed a scaled 9d value -> a single conditioning token (B, 1, cond_dim)."""
        assert self.cond_embed is not None
        return self.cond_embed(scaled_9d).unsqueeze(1)


class Gr00tN1d7ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d7Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            logger.info("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )
            logger.info("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        vl_self_attention_cfg = getattr(config, "vl_self_attention_cfg", None)
        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # Optional progress / time-to-completion head (see ProgressHead).
        self.use_progress_head = getattr(config, "use_progress_head", False)
        if self.use_progress_head:
            self.progress_loss_weight = config.progress_loss_weight
            self.use_progress_conditioning = config.use_progress_conditioning
            self.progress_conditioning_noise_std = config.progress_conditioning_noise_std
            self.progress_head = ProgressHead(
                latent_dim=config.backbone_embedding_dim,
                hidden_dim=config.progress_hidden_dim,
                cond_dim=self.input_embedding_dim,
                use_conditioning=config.use_progress_conditioning,
            )

        # Optional tip-target head (see TipTargetHead); independent of the
        # progress head -- both may be enabled, each appending its own
        # conditioning token (progress first, then tip target).
        self.use_tip_target_head = getattr(config, "use_tip_target_head", False)
        if self.use_tip_target_head:
            self.tip_target_loss_weight = config.tip_target_loss_weight
            self.use_tip_target_conditioning = config.use_tip_target_conditioning
            self.tip_target_head = TipTargetHead(
                latent_dim=config.backbone_embedding_dim,
                hidden_dim=config.tip_target_hidden_dim,
                cond_dim=self.input_embedding_dim,
                use_conditioning=config.use_tip_target_conditioning,
                xyz_scale=config.tip_target_xyz_scale,
            )
            # Teacher-forcing noise std in the head's scaled space: metric xyz
            # noise is divided by xyz_scale, rot6d noise is used as-is.
            noise = torch.tensor(
                [config.tip_target_conditioning_noise_xyz / config.tip_target_xyz_scale] * 3
                + [config.tip_target_conditioning_noise_rot] * 6
            )
            self.register_buffer("tip_target_cond_noise_std", noise, persistent=False)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
            self.vl_self_attention.requires_grad_(False)
        logger.debug(f"Tune action head projector: {self.tune_projector}")
        logger.debug(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        logger.debug(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, log a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    logger.debug(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if not self.tune_vlln:
                self.vlln.eval()
                self.vl_self_attention.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def _progress_cond_token(self, progress_value: torch.Tensor) -> torch.Tensor | None:
        """Build the progress conditioning token from a progress value in [0, 1], shape (B, 1)."""
        if not self.use_progress_conditioning:
            return None
        return self.progress_head.conditioning_token(progress_value)

    def _tip_target_cond_token(self, scaled_9d: torch.Tensor) -> torch.Tensor | None:
        """Build the tip-target conditioning token from a scaled 9d value, shape (B, 9)."""
        if not self.use_tip_target_conditioning:
            return None
        return self.tip_target_head.conditioning_token(scaled_9d)

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Handle state history
        assert action_input.state.shape[1] == self.config.state_history_length
        action_input.state = action_input.state.view(action_input.state.shape[0], 1, -1)

        # Embed state. State dropout (config.state_dropout_prob) is applied once,
        # in the processor, on the raw normalized states — not re-applied here on
        # the encoded features, which would compound the two dropout draws.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        vl_attn_mask = backbone_output.backbone_attention_mask

        # Optional progress head: predict task progress from the VLM latent (for
        # the auxiliary loss) and condition the action tokens on progress.
        # Teacher forcing: condition on the ground-truth progress (+ small noise)
        # so the action head cleanly learns the progress -> action-magnitude
        # relationship; the predicted progress is used for conditioning only at
        # inference time. Falls back to the (detached) prediction if no target.
        progress_logit = None
        progress_target = action_input.get("progress", None)
        if self.use_progress_head:
            progress_logit = self.progress_head.predict_logit(vl_embeds, vl_attn_mask)
            if progress_target is not None:
                cond_value = progress_target.to(vl_embeds.dtype).view(-1, 1)
                if self.training and self.progress_conditioning_noise_std > 0:
                    cond_value = (
                        cond_value
                        + self.progress_conditioning_noise_std * torch.randn_like(cond_value)
                    )
                    cond_value = cond_value.clamp(0.0, 1.0)
            else:
                cond_value = torch.sigmoid(progress_logit).detach()
            cond_token = self._progress_cond_token(cond_value)
            if cond_token is not None:
                state_features = torch.cat((state_features, cond_token), dim=1)

        # Optional tip-target head: same teacher-forcing scheme as the progress
        # head, in the head's scaled space. Rows whose label is NaN (unresolved
        # frames) are conditioned on the detached prediction instead of ground
        # truth and excluded from the loss below.
        tip_pred_scaled = None
        tip_target_scaled = None
        tip_valid = None
        tip_target = action_input.get("tip_target_9d", None)
        if self.use_tip_target_head:
            tip_pred_scaled = self.tip_target_head.predict_scaled(vl_embeds, vl_attn_mask)
            tip_cond_value = tip_pred_scaled.detach()
            if tip_target is not None:
                tip_target_scaled = self.tip_target_head.scale(
                    tip_target.to(tip_pred_scaled.dtype).view(-1, 9)
                )
                tip_valid = ~torch.isnan(tip_target_scaled).any(dim=-1, keepdim=True)
                teacher_value = torch.nan_to_num(tip_target_scaled)
                if self.training:
                    teacher_value = teacher_value + self.tip_target_cond_noise_std.to(
                        teacher_value.dtype
                    ) * torch.randn_like(teacher_value)
                tip_cond_value = torch.where(tip_valid, teacher_value, tip_cond_value)
            tip_cond_token = self._tip_target_cond_token(tip_cond_value)
            if tip_cond_token is not None:
                state_features = torch.cat((state_features, tip_cond_token), dim=1)

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        outputs = {
            "loss": loss,
            "action_loss": action_loss,
            # Scalar masked-mean action loss (== total loss before the progress
            # term is added), logged separately so the wandb lines reconcile:
            # loss == action_loss_scalar + progress_loss_weight * progress_loss.
            "action_loss_scalar": loss.detach(),
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

        # Progress head: regress task progress with a BCE-with-logits loss on the
        # soft [0, 1] target, and fold it into the total loss.
        if self.use_progress_head and progress_logit is not None:
            outputs["progress_pred"] = torch.sigmoid(progress_logit)
            if progress_target is not None:
                progress_target = progress_target.to(progress_logit.dtype).view(-1, 1)
                progress_loss = F.binary_cross_entropy_with_logits(progress_logit, progress_target)
                outputs["progress_loss"] = progress_loss
                outputs["loss"] = loss + self.progress_loss_weight * progress_loss

        # Tip-target head: MSE in the scaled space (xyz / xyz_scale + raw rot6d)
        # over rows with a resolved label, folded into the total loss.
        if self.use_tip_target_head and tip_pred_scaled is not None:
            outputs["tip_target_pred"] = self.tip_target_head.unscale(
                tip_pred_scaled, orthonormalize=True
            )
            if tip_target_scaled is not None and tip_valid is not None and tip_valid.any():
                valid_rows = tip_valid.squeeze(-1)
                tip_target_loss = F.mse_loss(
                    tip_pred_scaled[valid_rows], tip_target_scaled[valid_rows]
                )
                outputs["tip_target_loss"] = tip_target_loss
                outputs["loss"] = outputs["loss"] + self.tip_target_loss_weight * tip_target_loss

        return outputs

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_history_length, max_state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, 1, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Handle state history: if we have fewer timesteps than expected, repeat to fill
        state = action_input.state
        current_T = state.shape[1]
        assert current_T == self.config.state_history_length, "current_T != state_history_length"
        # Reshape state from [B, state_history_length, max_state_dim] to [B, 1, state_history_length * max_state_dim]
        state = state.view(state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps
        vel_strength = torch.ones_like(actions)

        if "action" in action_input:
            # If action in input when doing get action, it means we want to use RTC.
            # action_horizon is the action horizon of the input action.
            # rtc_overlap_steps is the number of steps to overlap with the previous action chunks.
            # rtc_frozen_steps is the number of steps to freeze the action, which is the latency of the policy inference.
            # rtc_ramp_rate is the rate of the ramp of denoising the actions.
            assert options is not None, "options is not None"
            assert "action_horizon" in options, "action_horizon is not in options"
            assert "rtc_overlap_steps" in options, "rtc_overlap_steps is not in options"
            assert "rtc_frozen_steps" in options, "rtc_frozen_steps is not in options"
            assert "rtc_ramp_rate" in options, "rtc_ramp_rate is not in options"

            action_horizon_before_padding = options["action_horizon"]

            # Use previous action instead of pure noise to do inpainting
            actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                :,
                action_horizon_before_padding
                - options["rtc_overlap_steps"] : action_horizon_before_padding,
                :,
            ]
            vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
            # NOTE: use an exponential ramp strength to set the remaining unfrozen rtc_steps
            intermediate_steps = options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
            # Create exponential ramp from 0 to 1 over intermediate steps
            t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
            ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t)
            ramp = ramp / ramp[-1].clamp_min(1e-8)  # normalize to [0,1]
            ramp = ramp[
                1:-1
            ]  # we will only take the middle part of the ramp, ignore the 0.0 and 1.0
            # Apply ramp to the intermediate steps [batch, intermediate_steps, action_dim]
            vel_strength[
                :,
                options["rtc_frozen_steps"] : options["rtc_overlap_steps"],
                :,
            ] = ramp[None, :, None].to(device)

        # Progress head: predict task progress once (independent of the action
        # noise) and pre-build the conditioned state tokens reused every step.
        progress_pred = None
        sa_state_features = state_features
        if self.use_progress_head:
            attn_mask = backbone_output.get("backbone_attention_mask", None)
            progress_logit = self.progress_head.predict_logit(vl_embeds, attn_mask)
            progress_pred = torch.sigmoid(progress_logit)
            # At inference we condition on the model's own predicted progress.
            cond_token = self._progress_cond_token(progress_pred)
            if cond_token is not None:
                sa_state_features = torch.cat((sa_state_features, cond_token), dim=1)

        # Tip-target head: predict once and condition on the (raw, scaled)
        # prediction, matching the teacher-forced training scheme. Token order
        # matches training: state, [progress], [tip target].
        tip_target_pred = None
        if self.use_tip_target_head:
            attn_mask = backbone_output.get("backbone_attention_mask", None)
            tip_pred_scaled = self.tip_target_head.predict_scaled(vl_embeds, attn_mask)
            tip_target_pred = self.tip_target_head.unscale(tip_pred_scaled, orthonormalize=True)
            tip_cond_token = self._tip_target_cond_token(tip_pred_scaled)
            if tip_cond_token is not None:
                sa_state_features = torch.cat((sa_state_features, tip_cond_token), dim=1)

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((sa_state_features, action_features), dim=1)

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity * vel_strength

        result = {
            "action_pred": actions,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }
        if progress_pred is not None:
            result["progress_pred"] = progress_pred
        if tip_target_pred is not None:
            result["tip_target_pred"] = tip_target_pred
        return BatchFeature(data=result)

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
            options=options,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d7Config):
    if "nvidia/Cosmos-Reason2" in config.model_name or "Qwen/Qwen3-VL" in config.model_name:
        # We import here as Qwen3Backbone depends on newer transformers versions than the rest of the code.
        from gr00t.model.modules.qwen3_backbone import Qwen3Backbone

        return Qwen3Backbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d7(PreTrainedModel):
    """Gr00tN1d7: VLA model with Cosmos-Reason2-2B (Qwen3-VL) backbone."""

    config_class = Gr00tN1d7Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d7Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d7 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d7ActionHead(config)
        from .processing_gr00t_n1d7 import Gr00tN1d7DataCollator

        self.collator = Gr00tN1d7DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def get_action(self, inputs: dict, options: dict[str, Any] | None = None) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)

        # Capture the raw VLM latent (last hidden state, full token sequence) before the
        # action head's process_backbone_output reassigns the "backbone_features" key
        # with the post-vlln / self-attention version.
        raw_backbone_features = backbone_outputs["backbone_features"]

        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs, options)

        # Surface the raw backbone features plus masks for downstream embedding extraction.
        action_outputs["backbone_features"] = raw_backbone_features
        if "backbone_attention_mask" in backbone_outputs:
            action_outputs["backbone_attention_mask"] = backbone_outputs["backbone_attention_mask"]
        if "image_mask" in backbone_outputs:
            action_outputs["image_mask"] = backbone_outputs["image_mask"]

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)
