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


def rot6d_orthonormalize(rot6d: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt a raw 6D rotation prediction into a valid one (Zhou et al., CVPR 2019).

    Input/output shape (..., 6): two 3-vectors; the first is normalized, the second
    is made orthonormal to it.
    """
    a = F.normalize(rot6d[..., :3], dim=-1)
    b = rot6d[..., 3:]
    b = F.normalize(b - (a * b).sum(dim=-1, keepdim=True) * a, dim=-1)
    return torch.cat((a, b), dim=-1)


def _masked_mean_pool(
    embeds: torch.Tensor, attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean-pool token embeddings over valid (non-padding) tokens, shape (B, D)."""
    if attention_mask is None:
        return embeds.mean(dim=1)
    mask = attention_mask.to(embeds.dtype).unsqueeze(-1)
    return (embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class TipTargetHead(nn.Module):
    """Auxiliary head predicting the target pose in the end-effector (tip) frame.

    Regresses xyz(3) + rot6d(6) -- the `next.tip_target_9d` label -- from the
    (post-vlln) VLM latent, pooled by a small cross-attention: `num_queries`
    learned query tokens attend over the full vision-language sequence. Unlike
    a mean pool, attention preserves *where* the target is (queries can focus
    on the patch tokens containing it), and it deliberately attends over text
    tokens too so language can select which target to localize (e.g. "drill
    the blue square"). The head works in a scaled space where xyz is divided
    by `xyz_scale`. When conditioning is enabled, the (scaled) 9d value is
    embedded into a single token appended to the state tokens, so the action
    DiT can condition fine motion on where the target is relative to the tip
    -- teacher-forced on ground truth during training, fed its own prediction
    at inference.

    `pool="mean"` selects the original masked-mean pooling (no learned queries,
    predictor takes a single `latent_dim` vector instead of
    `num_queries * latent_dim`). Both modes are supported and selectable with
    `--tip-target-pool`; which one is actually better here is untested -- the
    only checkpoint trained with "mean" differed from the "attn" one in the loss
    as well, so the comparison was confounded. Note the predictor input widths
    differ, so a checkpoint trained with one pool cannot be resumed with the
    other.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        cond_dim: int,
        use_conditioning: bool,
        xyz_scale: float,
        num_queries: int = 4,
        num_attn_heads: int = 8,
        pool: str = "attn",
    ):
        super().__init__()
        if pool not in ("attn", "mean"):
            raise ValueError(f"tip_target_pool must be 'attn' or 'mean', got {pool!r}")
        self.xyz_scale = xyz_scale
        self.pool = pool
        if pool == "attn":
            self.pool_queries = nn.Parameter(torch.randn(num_queries, latent_dim) * 0.02)
            self.pool_attn = nn.MultiheadAttention(latent_dim, num_attn_heads, batch_first=True)
            self.pool_norm = nn.LayerNorm(latent_dim)
            predictor_in_dim = latent_dim * num_queries
        else:
            self.pool_queries = None
            self.pool_attn = None
            self.pool_norm = None
            predictor_in_dim = latent_dim
        self.predictor = nn.Sequential(
            nn.Linear(predictor_in_dim, hidden_dim),
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

    def reset_parameters(self) -> None:
        """Re-initialize every parameter of the head to its intended fresh state.

        MUST be called after `from_pretrained` whenever the head's keys were
        missing from the checkpoint: transformers materializes the model on the
        meta device, so constructor init values are lost, and its missing-key
        fallback does not initialize bare `nn.Parameter`s like `pool_queries`
        (they come back as uninitialized/zero memory -- garbage queries NaN the
        attention softmax on the first forward).
        """
        if self.pool == "attn":
            with torch.no_grad():
                self.pool_queries.normal_(0.0, 0.02)
            self.pool_attn._reset_parameters()
            self.pool_norm.reset_parameters()
        for module in list(self.predictor) + (list(self.cond_embed) if self.cond_embed else []):
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def predict_scaled(
        self, vl_embeds: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict the scaled 9d target (xyz / xyz_scale, raw rot6d), shape (B, 9)."""
        if self.pool == "mean":
            return self.predictor(_masked_mean_pool(vl_embeds, attention_mask))
        queries = (
            self.pool_queries.to(vl_embeds.dtype).unsqueeze(0).expand(vl_embeds.shape[0], -1, -1)
        )
        # key_padding_mask: True = ignore that token (padding).
        key_padding_mask = None if attention_mask is None else ~attention_mask.bool()
        pooled, _ = self.pool_attn(
            queries, vl_embeds, vl_embeds, key_padding_mask=key_padding_mask, need_weights=False
        )
        return self.predictor(self.pool_norm(pooled).flatten(1))

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

        # Optional tip-target head (see TipTargetHead); when enabled it appends
        # its own conditioning token after the state tokens.
        self.use_tip_target_head = getattr(config, "use_tip_target_head", False)
        if self.use_tip_target_head:
            self.tip_target_loss_weight = config.tip_target_loss_weight
            self.tip_target_loss_type = getattr(config, "tip_target_loss_type", "mse")
            if self.tip_target_loss_type not in ("mse", "rel_abs"):
                raise ValueError(
                    "tip_target_loss_type must be 'mse' or 'rel_abs', "
                    f"got {self.tip_target_loss_type!r}"
                )
            self.tip_target_pos_loss_weight = getattr(config, "tip_target_pos_loss_weight", 1.0)
            self.tip_target_pos_abs_weight = getattr(config, "tip_target_pos_abs_weight", 0.2)
            self.tip_target_rot_loss_weight = getattr(config, "tip_target_rot_loss_weight", 1.0)
            self.tip_target_dist_floor = getattr(config, "tip_target_dist_floor", 0.05)
            self.use_tip_target_conditioning = config.use_tip_target_conditioning
            self.tip_target_head = TipTargetHead(
                latent_dim=config.backbone_embedding_dim,
                hidden_dim=config.tip_target_hidden_dim,
                cond_dim=self.input_embedding_dim,
                use_conditioning=config.use_tip_target_conditioning,
                xyz_scale=config.tip_target_xyz_scale,
                num_queries=getattr(config, "tip_target_num_queries", 4),
                num_attn_heads=getattr(config, "tip_target_attn_heads", 8),
                pool=getattr(config, "tip_target_pool", "attn"),
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

    def _tip_target_cond_token(self, scaled_9d: torch.Tensor) -> torch.Tensor | None:
        """Build the tip-target conditioning token from a scaled 9d value, shape (B, 9)."""
        if not self.use_tip_target_conditioning:
            return None
        return self.tip_target_head.conditioning_token(scaled_9d)

    def forward(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        extra_cond_tokens: torch.Tensor | None = None,
    ) -> BatchFeature:
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
            extra_cond_tokens: optional [B, K, input_embedding_dim] conditioning
                tokens appended after the state (and tip-target) tokens, so the
                DiT's self-attention stream sees them alongside state and action.
                Used by Gr00tN1d7PerceptionAction to inject the perception
                geometry/appearance tokens.

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

        # Optional tip-target head: teacher-forced conditioning in the head's
        # scaled space. Rows whose label is NaN (unresolved frames) are
        # conditioned on the detached prediction instead of ground truth and
        # excluded from the loss below.
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

        if extra_cond_tokens is not None:
            state_features = torch.cat(
                (state_features, extra_cond_tokens.to(state_features.dtype)), dim=1
            )

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
            # Scalar masked-mean action loss (== total loss before the tip-target
            # term is added), logged separately so the wandb lines reconcile:
            # loss == action_loss_scalar + tip_target_loss_weight * tip_target_loss.
            "action_loss_scalar": loss.detach(),
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

        # Tip-target head loss over rows with a resolved label. Two objectives,
        # selected by `tip_target_loss_type` (see the model config for the
        # measurements behind the default):
        #   "mse":     one plain MSE over the whole scaled 9d vector.
        #   "rel_abs": xyz error normalized by the (clamped) distance to target
        #              plus an absolute anchor, and a separate rot6d MSE.
        # The per-part MSEs are always logged as diagnostics so the two
        # objectives stay comparable on the same wandb panels.
        if self.use_tip_target_head and tip_pred_scaled is not None:
            outputs["tip_target_pred"] = self.tip_target_head.unscale(
                tip_pred_scaled, orthonormalize=True
            )
            if tip_target_scaled is not None and tip_valid is not None and tip_valid.any():
                valid_rows = tip_valid.squeeze(-1)
                pred_v = tip_pred_scaled[valid_rows]
                tgt_v = tip_target_scaled[valid_rows]
                tip_pos_abs_loss = F.mse_loss(pred_v[:, :3], tgt_v[:, :3])
                tip_rot_loss = F.mse_loss(pred_v[:, 3:], tgt_v[:, 3:])
                outputs["tip_target_pos_abs_loss"] = tip_pos_abs_loss
                outputs["tip_target_rot_loss"] = tip_rot_loss

                if self.tip_target_loss_type == "mse":
                    # Single MSE over all 9 dims. NOT the same as summing the two
                    # per-part MSEs above: averaging over 9 dims weights rot6d
                    # (6 dims) twice as much as xyz (3 dims).
                    tip_target_loss = F.mse_loss(pred_v, tgt_v)
                else:
                    xyz_scale = self.tip_target_head.xyz_scale
                    xyz_residual_m = (pred_v[:, :3] - tgt_v[:, :3]) * xyz_scale
                    dist_m = torch.linalg.vector_norm(
                        tgt_v[:, :3] * xyz_scale, dim=-1, keepdim=True
                    )
                    rel_residual = xyz_residual_m / dist_m.clamp_min(self.tip_target_dist_floor)
                    tip_pos_rel_loss = rel_residual.pow(2).mean()
                    tip_pos_loss = (
                        tip_pos_rel_loss + self.tip_target_pos_abs_weight * tip_pos_abs_loss
                    )
                    tip_target_loss = (
                        self.tip_target_pos_loss_weight * tip_pos_loss
                        + self.tip_target_rot_loss_weight * tip_rot_loss
                    )
                    outputs["tip_target_pos_loss"] = tip_pos_loss
                    outputs["tip_target_pos_rel_loss"] = tip_pos_rel_loss
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
        extra_cond_tokens: torch.Tensor | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
            extra_cond_tokens: optional [B, K, input_embedding_dim] conditioning
                tokens appended after the state (and tip-target) tokens. Token
                order must match training exactly.
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

        # Tip-target head: predict once and condition on the (raw, scaled)
        # prediction, matching the teacher-forced training scheme. Token order
        # matches training: state, [tip target].
        sa_state_features = state_features
        tip_target_pred = None
        if self.use_tip_target_head:
            attn_mask = backbone_output.get("backbone_attention_mask", None)
            tip_pred_scaled = self.tip_target_head.predict_scaled(vl_embeds, attn_mask)
            tip_target_pred = self.tip_target_head.unscale(tip_pred_scaled, orthonormalize=True)
            tip_cond_token = self._tip_target_cond_token(tip_pred_scaled)
            if tip_cond_token is not None:
                sa_state_features = torch.cat((sa_state_features, tip_cond_token), dim=1)

        if extra_cond_tokens is not None:
            sa_state_features = torch.cat(
                (sa_state_features, extra_cond_tokens.to(sa_state_features.dtype)), dim=1
            )

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
        if tip_target_pred is not None:
            result["tip_target_pred"] = tip_target_pred
        return BatchFeature(data=result)

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
        extra_cond_tokens: torch.Tensor | None = None,
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
            extra_cond_tokens: optional [B, K, input_embedding_dim] conditioning
                tokens (see get_action_with_features).

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
            extra_cond_tokens=extra_cond_tokens,
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
