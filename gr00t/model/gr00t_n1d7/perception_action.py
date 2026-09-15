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

"""Perception-CONDITIONED action model for the short-range fine approach.

The perception-only model (Gr00tN1d7Perception) localizes the wall target in the
wrist camera but emits no actions. This model keeps that perception head and adds
the standard GR00T flow-matching action head back on top, with the perception
output cast into conditioning tokens that are appended to the state tokens — so
the DiT generates the action chunk conditioned on where the target is, the same
mechanism the full VLA uses for its tip-target head (see TipTargetHead in
gr00t_n1d7.py), but sourced from a pixel-space heatmap instead of a 3D regression.

Two tokens are emitted (see PerceptionConditioner):
  A "where":  Fourier-encoded (u, v) + visibility + heatmap confidence
              (+ depth / wall normal), blended with a learned `unknown_token`
              by the visibility weight. Teacher-forced on the labels + calibrated
              noise during training, fed the head's own prediction at inference.
  B "what":   heatmap-weighted pool of the image tokens. Not teacher-forceable,
              and deliberately so: it is the only differentiable path from the
              action loss back into the heatmap.

Wrist camera only, single current frame — same input contract as the perception
model, so its checkpoints (and their keypoint-aware processor) carry over
unchanged. Trains on every sidecar frame (data.perception_labels_only=True).
"""

import logging
import math
from typing import Any, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

from gr00t.configs.model.gr00t_n1d7_perception_action import Gr00tN1d7PerceptionActionConfig

from .gr00t_n1d7 import Gr00tN1d7ActionHead, get_backbone_cls
from .perception import (
    TargetPerceptionHead,
    decode_uv,
    image_token_grid,
    perception_losses,
    processed_image_size,
)


logger = logging.getLogger(__name__)


class PerceptionConditioner(nn.Module):
    """Casts the perception head's output into conditioning tokens for the DiT.

    Token A ("where") is built from a compact geometry vector:
        [ sin/cos(2^k * 2pi * u), sin/cos(2^k * 2pi * v) ]  k = 0 .. bands-1
        [ u, v ]                                     raw normalized coords
        [ visibility ]                               teacher label or predicted prob
        [ top-1 heatmap prob, normalized entropy ]   confidence, always predicted
        [ depth_scaled, depth_valid ]                optional
        [ wall_normal(3), normal_valid ]             optional
    -> MLP -> (B, 1, cond_dim), then blended with a learned `unknown_token`:
        token = vis * geom_token + (1 - vis) * unknown_token
    so "target not visible" is a distinct, learnable input rather than a
    meaningless coordinate. Every input to token A is a label or a DETACHED
    prediction: the action loss must not be able to reshape the perception head
    through this path (and under teacher forcing it could not anyway).

    Token B ("what") pools the image tokens with the softmax heatmap and projects
    them. This one keeps its gradient, so the action objective can sharpen the
    heatmap; set `stopgrad=True` to cut it.
    """

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        num_fourier_bands: int,
        use_depth: bool,
        use_normal: bool,
        use_geometry_token: bool = True,
        use_appearance_token: bool = True,
        appearance_temperature: float = 1.0,
        appearance_stopgrad: bool = False,
        teacher_forcing: bool = True,
        noise_uv_px: float = 0.0,
        noise_depth_m: float = 0.0,
        noise_normal: float = 0.0,
        bias_depth_m: float = 0.0,
        vis_dropout_prob: float = 0.0,
        depth_scale: float = 0.3,
    ):
        super().__init__()
        if not (use_geometry_token or use_appearance_token):
            raise ValueError(
                "PerceptionConditioner needs at least one token enabled; disable "
                "use_perception_conditioning instead of both tokens."
            )
        self.use_depth = use_depth
        self.use_normal = use_normal
        self.use_geometry_token = use_geometry_token
        self.use_appearance_token = use_appearance_token
        self.appearance_temperature = appearance_temperature
        self.appearance_stopgrad = appearance_stopgrad
        self.teacher_forcing = teacher_forcing
        self.noise_uv_px = noise_uv_px
        self.noise_depth_m = noise_depth_m
        self.noise_normal = noise_normal
        self.bias_depth_m = bias_depth_m
        self.vis_dropout_prob = vis_dropout_prob
        self.depth_scale = depth_scale
        self.num_fourier_bands = num_fourier_bands

        if use_geometry_token:
            # Frequencies in cycles-per-image: 1, 2, 4, ... 2^(bands-1).
            freqs = 2.0 ** torch.arange(num_fourier_bands, dtype=torch.float32) * (2 * torch.pi)
            self.register_buffer("fourier_freqs", freqs, persistent=False)
            geom_dim = 4 * num_fourier_bands + 2 + 1 + 2
            if use_depth:
                geom_dim += 2
            if use_normal:
                geom_dim += 4
            self.geom_dim = geom_dim
            self.geom_mlp = nn.Sequential(
                nn.Linear(geom_dim, cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, cond_dim),
            )
            self.unknown_token = nn.Parameter(torch.randn(1, 1, cond_dim) * 0.02)
        else:
            self.geom_dim = 0

        if use_appearance_token:
            self.appearance_norm = nn.LayerNorm(latent_dim)
            self.appearance_proj = nn.Linear(latent_dim, cond_dim)
            self.unknown_appearance_token = nn.Parameter(torch.randn(1, 1, cond_dim) * 0.02)

    @property
    def num_tokens(self) -> int:
        return int(self.use_geometry_token) + int(self.use_appearance_token)

    def reset_parameters(self) -> None:
        """Re-initialize every parameter (same meta-device caveat as
        TipTargetHead.reset_parameters: transformers' missing-key fallback does
        not initialize bare nn.Parameters, which would leave `unknown_token` as
        uninitialized memory)."""
        with torch.no_grad():
            if self.use_geometry_token:
                self.unknown_token.normal_(0.0, 0.02)
            if self.use_appearance_token:
                self.unknown_appearance_token.normal_(0.0, 0.02)
        if self.use_geometry_token:
            for module in self.geom_mlp:
                if isinstance(module, nn.Linear):
                    module.reset_parameters()
        if self.use_appearance_token:
            self.appearance_norm.reset_parameters()
            self.appearance_proj.reset_parameters()

    def _fourier(self, uv_norm: torch.Tensor) -> torch.Tensor:
        """(B, 2) normalized coords -> (B, 4 * bands) sin/cos features."""
        angles = uv_norm.unsqueeze(-1) * self.fourier_freqs.view(1, 1, -1)  # (B, 2, bands)
        return torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(1)

    def resolve_conditioning(
        self,
        head_out: dict[str, torch.Tensor],
        pred_uv: torch.Tensor,
        labels: dict | None,
    ) -> dict[str, torch.Tensor]:
        """Decide WHAT the geometry token is conditioned on for this batch.

        Teacher forcing: use the label plus noise on rows that have one, and the
        head's own detached prediction elsewhere. `vis` is the label when
        available (0/1) and the predicted probability otherwise; visibility
        dropout randomly forces it to 0 during training.

        All returned tensors are detached — token A never carries gradient into
        the perception head.
        """
        B = pred_uv.shape[0]
        vis_prob = torch.sigmoid(head_out["vis_logit"].float()).detach().view(B)
        cond_uv = pred_uv.detach().float()
        vis = vis_prob

        depth = None
        depth_valid = None
        if self.use_depth:
            depth = head_out["depth_scaled"].float().detach().view(B)
            depth_valid = torch.ones_like(depth)
        normal = None
        normal_valid = None
        if self.use_normal:
            normal = F.normalize(head_out["wall_normal"].float().detach(), dim=-1).view(B, 3)
            normal_valid = torch.ones(B, device=pred_uv.device, dtype=torch.float32)

        if self.teacher_forcing and labels is not None and "target_uv" in labels:
            target_uv = labels["target_uv"].float().view(B, 2)
            target_visible = labels["target_visible"].float().view(B)
            uv_valid = (~torch.isnan(target_uv).any(dim=-1)) & (target_visible > 0.5)

            teacher_uv = torch.nan_to_num(target_uv)
            if self.training and self.noise_uv_px > 0:
                teacher_uv = teacher_uv + self.noise_uv_px * torch.randn_like(teacher_uv)
            cond_uv = torch.where(uv_valid.unsqueeze(-1), teacher_uv, cond_uv)
            # The label is definitive for every frame the sidecar covers.
            vis = target_visible

            if self.use_depth and "target_depth" in labels:
                target_depth = labels["target_depth"].float().view(B)
                d_valid = ~torch.isnan(target_depth)
                teacher_depth = torch.nan_to_num(target_depth)
                # The measured depth error is mostly a SYSTEMATIC offset, not noise
                # (see scripts/calibrate_perception_conditioning.py). Applied in both
                # train and eval mode: a bias the model will always exhibit at
                # deployment is not a training-time perturbation.
                if self.bias_depth_m != 0.0:
                    teacher_depth = teacher_depth + self.bias_depth_m
                if self.training and self.noise_depth_m > 0:
                    teacher_depth = teacher_depth + self.noise_depth_m * torch.randn_like(
                        teacher_depth
                    )
                teacher_depth = teacher_depth / self.depth_scale
                depth = torch.where(d_valid, teacher_depth, depth)
                depth_valid = d_valid.float()

            if self.use_normal and "target_wall_normal" in labels:
                target_normal = labels["target_wall_normal"].float().view(B, 3)
                n_valid = ~torch.isnan(target_normal).any(dim=-1)
                teacher_normal = torch.nan_to_num(target_normal)
                if self.training and self.noise_normal > 0:
                    teacher_normal = teacher_normal + self.noise_normal * torch.randn_like(
                        teacher_normal
                    )
                teacher_normal = F.normalize(teacher_normal, dim=-1)
                normal = torch.where(n_valid.unsqueeze(-1), teacher_normal, normal)
                normal_valid = n_valid.float()

        if self.training and self.vis_dropout_prob > 0:
            drop = torch.rand(B, device=vis.device) < self.vis_dropout_prob
            vis = torch.where(drop, torch.zeros_like(vis), vis)

        resolved = {"cond_uv": cond_uv, "vis": vis.clamp(0.0, 1.0)}
        if depth is not None:
            resolved["depth_scaled"] = depth
            resolved["depth_valid"] = depth_valid
        if normal is not None:
            resolved["wall_normal"] = normal
            resolved["normal_valid"] = normal_valid
        return resolved

    def forward(
        self,
        head_out: dict[str, torch.Tensor],
        image_tokens: torch.Tensor,
        gh: int,
        gw: int,
        proc_w: float,
        proc_h: float,
        labels: dict | None = None,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Build the conditioning tokens.

        Returns:
            tokens: (B, num_tokens, cond_dim)
            info:   pred_uv / cond_uv (processed-image px) and scalar metrics.
        """
        heat_logits = head_out["heat_logits"].float()
        offsets = head_out["offsets"].float()
        B = heat_logits.shape[0]

        pred_uv = decode_uv(heat_logits, offsets, gh, gw, proc_w, proc_h)
        resolved = self.resolve_conditioning(head_out, pred_uv, labels)
        vis = resolved["vis"]

        tokens = []
        if self.use_geometry_token:
            uv_norm = torch.stack(
                (
                    (resolved["cond_uv"][:, 0] / proc_w).clamp(0.0, 1.0),
                    (resolved["cond_uv"][:, 1] / proc_h).clamp(0.0, 1.0),
                ),
                dim=-1,
            )
            with torch.no_grad():
                probs = F.softmax(heat_logits, dim=-1)
                top1 = probs.max(dim=-1).values
                entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)
                entropy = entropy / math.log(probs.shape[-1])
            feats = [self._fourier(uv_norm), uv_norm, vis.unsqueeze(-1)]
            feats.append(torch.stack((top1, entropy), dim=-1))
            if self.use_depth:
                feats.append(
                    torch.stack((resolved["depth_scaled"], resolved["depth_valid"]), dim=-1)
                )
            if self.use_normal:
                feats.append(resolved["wall_normal"])
                feats.append(resolved["normal_valid"].unsqueeze(-1))
            geom_in = torch.cat(feats, dim=-1)
            assert geom_in.shape[-1] == self.geom_dim, (
                f"geometry feature dim {geom_in.shape[-1]} != configured {self.geom_dim}"
            )
            geom_token = self.geom_mlp(geom_in.to(self.geom_mlp[0].weight.dtype)).unsqueeze(1)
            gate = vis.to(geom_token.dtype).view(B, 1, 1)
            unknown = self.unknown_token.to(geom_token.dtype).expand(B, -1, -1)
            tokens.append(gate * geom_token + (1.0 - gate) * unknown)

        if self.use_appearance_token:
            logits = heat_logits.detach() if self.appearance_stopgrad else heat_logits
            weights = F.softmax(logits / self.appearance_temperature, dim=-1)
            pooled = torch.einsum(
                "bn,bnd->bd", weights.to(image_tokens.dtype), image_tokens
            )  # (B, D)
            app_token = self.appearance_proj(self.appearance_norm(pooled)).unsqueeze(1)
            gate = vis.to(app_token.dtype).view(B, 1, 1)
            unknown = self.unknown_appearance_token.to(app_token.dtype).expand(B, -1, -1)
            tokens.append(gate * app_token + (1.0 - gate) * unknown)

        info = {"pred_uv": pred_uv, "cond_uv": resolved["cond_uv"]}
        with torch.no_grad():
            info["cond_vis_mean_metric"] = vis.mean()
            # Px error of what the policy is ACTUALLY conditioned on (teacher noise
            # included) vs the label, over rows with a visible label.
            if labels is not None and "target_uv" in labels:
                target_uv = labels["target_uv"].float().view(B, 2)
                valid = (
                    (~torch.isnan(target_uv).any(dim=-1))
                    & (labels["target_visible"].float().view(B) > 0.5)
                ).float()
                err = torch.linalg.vector_norm(
                    resolved["cond_uv"] - torch.nan_to_num(target_uv), dim=-1
                )
                info["cond_uv_err_px_metric"] = (err * valid).sum() / valid.sum().clamp_min(1.0)
        return torch.cat(tokens, dim=1), info


class Gr00tN1d7PerceptionAction(PreTrainedModel):
    """Wrist-only VLM backbone + target perception head + perception-conditioned action head."""

    config_class = Gr00tN1d7PerceptionActionConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d7PerceptionActionConfig,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
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

        self.perception_head = TargetPerceptionHead(
            latent_dim=config.backbone_embedding_dim,
            hidden_dim=config.heatmap_hidden_dim,
            use_depth=config.depth_loss_weight > 0,
            use_normal=config.wall_normal_loss_weight > 0,
            num_attn_heads=config.perception_pool_heads,
        )

        self.action_head = Gr00tN1d7ActionHead(config)

        if config.use_perception_conditioning:
            self.conditioner = PerceptionConditioner(
                latent_dim=config.backbone_embedding_dim,
                cond_dim=config.input_embedding_dim,
                num_fourier_bands=config.perception_num_fourier_bands,
                use_depth=self.perception_head.use_depth,
                use_normal=self.perception_head.use_normal,
                use_geometry_token=config.use_perception_geometry_token,
                use_appearance_token=config.use_perception_appearance_token,
                appearance_temperature=config.perception_appearance_temperature,
                appearance_stopgrad=config.perception_appearance_stopgrad,
                teacher_forcing=config.perception_cond_teacher_forcing,
                noise_uv_px=config.perception_cond_noise_uv_px,
                noise_depth_m=config.perception_cond_noise_depth_m,
                noise_normal=config.perception_cond_noise_normal,
                bias_depth_m=config.perception_cond_bias_depth_m,
                vis_dropout_prob=config.perception_cond_vis_dropout_prob,
                depth_scale=config.depth_scale,
            )
        else:
            self.conditioner = None

        from .processing_gr00t_n1d7 import Gr00tN1d7DataCollator

        self.collator = Gr00tN1d7DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        vision_cfg = self.backbone.model.config.vision_config
        self._merge_size = getattr(vision_cfg, "spatial_merge_size", 2)

    # ------------------------------------------------------------------
    # Input preparation (mirrors Gr00tN1d7.prepare_input)
    # ------------------------------------------------------------------
    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        if "vlm_content" in inputs:
            vlm_content_list = inputs["vlm_content"]
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_dtype(x):
            if not torch.is_tensor(x):
                return x
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)
        return backbone_inputs, action_inputs

    # ------------------------------------------------------------------
    # Perception + conditioning
    # ------------------------------------------------------------------
    def _run_perception(
        self, backbone_output: BatchFeature, image_grid_thw: torch.Tensor
    ) -> Tuple[dict[str, torch.Tensor], torch.Tensor, int, int]:
        """Perception head on the RAW VLM latent.

        Must run BEFORE the action head, whose process_backbone_output reassigns
        "backbone_features" with the post-vlln / self-attention version.
        """
        image_tokens, gh, gw = image_token_grid(
            backbone_output["backbone_features"],
            backbone_output["image_mask"],
            image_grid_thw,
            self._merge_size,
        )
        return self.perception_head(image_tokens), image_tokens, gh, gw

    def _conditioning_tokens(
        self,
        head_out: dict[str, torch.Tensor],
        image_tokens: torch.Tensor,
        gh: int,
        gw: int,
        labels: dict | None,
    ) -> Tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        proc_w, proc_h = processed_image_size(self.config.image_target_size)
        if self.conditioner is None:
            heat_logits = head_out["heat_logits"].float()
            offsets = head_out["offsets"].float()
            pred_uv = decode_uv(heat_logits, offsets, gh, gw, proc_w, proc_h)
            return None, {"pred_uv": pred_uv}
        return self.conditioner(head_out, image_tokens, gh, gw, proc_w, proc_h, labels=labels)

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------
    def forward(self, inputs: dict) -> dict:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)

        head_out, image_tokens, gh, gw = self._run_perception(
            backbone_outputs, action_inputs["image_grid_thw"]
        )
        cond_tokens, cond_info = self._conditioning_tokens(
            head_out, image_tokens, gh, gw, labels=action_inputs
        )

        outputs = self.action_head(backbone_outputs, action_inputs, extra_cond_tokens=cond_tokens)

        perception_out = perception_losses(
            self.config,
            head_out,
            action_inputs,
            gh,
            gw,
            use_depth=self.perception_head.use_depth,
            use_normal=self.perception_head.use_normal,
            pred_uv=cond_info.get("pred_uv"),
        )
        perception_loss = perception_out.pop("perception_loss")
        outputs.update(perception_out)
        outputs["perception_loss"] = perception_loss
        outputs["loss"] = outputs["loss"] + self.config.perception_loss_weight * perception_loss

        for key, value in cond_info.items():
            if key.endswith("_metric"):
                outputs[key] = value
        return outputs

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_action(self, inputs: dict, options: dict[str, Any] | None = None) -> BatchFeature:
        """Generate an action chunk conditioned on the model's OWN perception output.

        Never teacher-forced (there are no labels at deployment): the geometry
        token is built from the decoded prediction, matching the token order used
        in training.
        """
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)

        # Raw VLM latent, before the action head's process_backbone_output
        # reassigns the key with the post-vlln version.
        raw_backbone_features = backbone_outputs["backbone_features"]

        head_out, image_tokens, gh, gw = self._run_perception(
            backbone_outputs, action_inputs["image_grid_thw"]
        )
        cond_tokens, cond_info = self._conditioning_tokens(
            head_out, image_tokens, gh, gw, labels=None
        )

        action_outputs = self.action_head.get_action(
            backbone_outputs, action_inputs, options, extra_cond_tokens=cond_tokens
        )

        action_outputs["backbone_features"] = raw_backbone_features
        if "backbone_attention_mask" in backbone_outputs:
            action_outputs["backbone_attention_mask"] = backbone_outputs["backbone_attention_mask"]
        if "image_mask" in backbone_outputs:
            action_outputs["image_mask"] = backbone_outputs["image_mask"]

        # Surface the perception outputs alongside the action, so a deployment
        # can log / gate on them without a second forward pass.
        heat_logits = head_out["heat_logits"].float()
        action_outputs["target_uv"] = cond_info["pred_uv"]
        action_outputs["visibility"] = torch.sigmoid(head_out["vis_logit"].float())
        if (options or {}).get("return_heatmap", False):
            action_outputs["heatmap"] = F.softmax(heat_logits, dim=-1).view(-1, gh, gw)
        if self.perception_head.use_depth:
            action_outputs["target_depth"] = (
                head_out["depth_scaled"].float() * self.config.depth_scale
            )
        if self.perception_head.use_normal:
            action_outputs["wall_normal"] = F.normalize(head_out["wall_normal"].float(), dim=-1)
        return action_outputs

    def uv_processed_to_camera(
        self, uv: torch.Tensor, camera_size: Tuple[int, int] = (720, 720)
    ) -> torch.Tensor:
        """Map uv from processed-image px back to RAW wrist-camera px.

        Same inverse transform as Gr00tN1d7Perception.uv_processed_to_camera.
        """
        cam_w, cam_h = float(camera_size[0]), float(camera_size[1])
        proc_w, proc_h = processed_image_size(self.config.image_target_size)
        crop = self.config.image_crop_size
        crop_h, crop_w = crop if isinstance(crop, (tuple, list)) else (crop, crop)
        x_min = (proc_w - crop_w) // 2
        y_min = (proc_h - crop_h) // 2
        u = (uv[..., 0] * crop_w / proc_w + x_min) * cam_w / proc_w
        v = (uv[..., 1] * crop_h / proc_h + y_min) * cam_h / proc_h
        return torch.stack((u, v), dim=-1)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


AutoConfig.register("Gr00tN1d7PerceptionAction", Gr00tN1d7PerceptionActionConfig)
AutoModel.register(Gr00tN1d7PerceptionActionConfig, Gr00tN1d7PerceptionAction)
