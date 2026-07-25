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

"""Perception-only GR00T model for the short-range fine approach.

VLM backbone (Cosmos-Reason2 / Qwen3-VL, same as Gr00tN1d7) + a target heatmap
head. Input: the CURRENT wrist-camera frame + task instruction. Output: where
the wall target is in wrist-camera pixel coordinates (heatmap over the VLM
image-token grid + subpixel offset), whether it is visible at all, and
optionally its depth and the wall normal in the camera frame.

There is no action head: this model does perception only. The coarse approach
keeps using the full VLA; this model takes over close to the wall, where the
heatmap_labels/ sidecar provides fine-phase supervision.
"""

import logging
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

from gr00t.configs.model.gr00t_n1d7_perception import Gr00tN1d7PerceptionConfig

from .gr00t_n1d7 import get_backbone_cls


logger = logging.getLogger(__name__)


class TargetPerceptionHead(nn.Module):
    """Heatmap + visibility (+ depth / wall-normal) head over the VLM image tokens.

    Per image token (spatial grid cell after the ViT merge): a small MLP produces
    a heatmap logit and a subpixel (du, dv) offset in cell units. Globally: one
    learned query cross-attends over the image tokens and predicts a visibility
    logit, the target depth (scaled), and the wall normal.

    All coordinates inside the head are NORMALIZED to [0, 1] over the processed
    image, so the head is independent of the VLM's internal smart-resize.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        use_depth: bool,
        use_normal: bool,
        num_attn_heads: int = 8,
    ):
        super().__init__()
        self.use_depth = use_depth
        self.use_normal = use_normal

        self.token_norm = nn.LayerNorm(latent_dim)
        # 3 outputs per token: heatmap logit, du, dv (cell units, centered).
        self.token_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

        self.pool_query = nn.Parameter(torch.randn(1, latent_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(latent_dim, num_attn_heads, batch_first=True)
        self.pool_norm = nn.LayerNorm(latent_dim)
        global_out = 1 + (1 if use_depth else 0) + (3 if use_normal else 0)
        self.global_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, global_out),
        )

    def reset_parameters(self) -> None:
        """Re-initialize all parameters (needed after meta-device materialization,
        same caveat as TipTargetHead.reset_parameters)."""
        with torch.no_grad():
            self.pool_query.normal_(0.0, 0.02)
        self.pool_attn._reset_parameters()
        self.token_norm.reset_parameters()
        self.pool_norm.reset_parameters()
        for module in list(self.token_mlp) + list(self.global_mlp):
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, image_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """image_tokens: (B, N, D) in raster order of the (gh, gw) token grid."""
        tokens = self.token_norm(image_tokens)
        per_token = self.token_mlp(tokens)  # (B, N, 3)
        heat_logits = per_token[..., 0]
        offsets = per_token[..., 1:3]

        query = self.pool_query.to(tokens.dtype).unsqueeze(0).expand(tokens.shape[0], -1, -1)
        pooled, _ = self.pool_attn(query, tokens, tokens, need_weights=False)
        pooled = self.pool_norm(pooled).squeeze(1)
        global_out = self.global_mlp(pooled)

        out = {"heat_logits": heat_logits, "offsets": offsets, "vis_logit": global_out[..., 0]}
        idx = 1
        if self.use_depth:
            out["depth_scaled"] = global_out[..., idx]
            idx += 1
        if self.use_normal:
            out["wall_normal"] = global_out[..., idx : idx + 3]
        return out


class Gr00tN1d7Perception(PreTrainedModel):
    """VLM-only perception model: Qwen3-VL backbone + TargetPerceptionHead."""

    config_class = Gr00tN1d7PerceptionConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d7PerceptionConfig,
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
    def prepare_input(self, inputs: dict) -> BatchFeature:
        if "vlm_content" in inputs:
            vlm_content_list = inputs["vlm_content"]
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        def to_device_with_dtype(x):
            if not torch.is_tensor(x):
                return x
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            return x.to(self.device)

        return tree.map_structure(to_device_with_dtype, dict(inputs))

    # ------------------------------------------------------------------
    # Token grid helpers
    # ------------------------------------------------------------------
    def _image_token_grid(
        self, backbone_output: BatchFeature, image_grid_thw: torch.Tensor
    ) -> Tuple[torch.Tensor, int, int]:
        """Gather image-token features into (B, gh*gw, D) plus the grid shape.

        Requires exactly one image per sample with identical grid shape across
        the batch (the perception setup: single current wrist frame at a fixed
        resolution)."""
        vl_embeds = backbone_output["backbone_features"]
        image_mask = backbone_output["image_mask"]
        B = vl_embeds.shape[0]
        assert image_grid_thw.shape[0] == B, (
            f"Perception model expects exactly 1 image per sample, got "
            f"{image_grid_thw.shape[0]} images for batch of {B}. Use a wrist-only "
            f"modality config with video delta_indices=[0]."
        )
        merge = self._merge_size
        t, h, w = (int(x) for x in image_grid_thw[0])
        gh, gw = (h // merge), (w // merge)
        n_tokens = t * gh * gw
        counts = image_mask.sum(dim=1)
        assert bool((counts == n_tokens).all()), (
            f"Image token count mismatch: expected {n_tokens}, got {counts.tolist()}"
        )
        image_tokens = vl_embeds[image_mask].view(B, n_tokens, -1)
        return image_tokens, gh, gw

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------
    def forward(self, inputs: dict) -> BatchFeature:
        inputs = self.prepare_input(inputs)
        backbone_output = self.backbone(BatchFeature(data=inputs))
        image_tokens, gh, gw = self._image_token_grid(backbone_output, inputs["image_grid_thw"])
        head_out = self.perception_head(image_tokens)

        heat_logits = head_out["heat_logits"].float()  # (B, N)
        offsets = head_out["offsets"].float()  # (B, N, 2)
        vis_logit = head_out["vis_logit"].float()  # (B,)
        B, N = heat_logits.shape
        device = heat_logits.device

        proc_w, proc_h = self._processed_image_size()

        # Labels (from the perception processor): uv in processed-image px.
        target_uv = inputs["target_uv"].float().view(B, 2)
        target_visible = inputs["target_visible"].float().view(B)
        heat_valid = target_visible * (~torch.isnan(target_uv).any(dim=-1)).float()

        # Normalized target coords in [0,1] (NaN-safe; invalid rows are masked out).
        uv = torch.nan_to_num(target_uv)
        u_norm = (uv[:, 0] / proc_w).clamp(0.0, 1.0 - 1e-6)
        v_norm = (uv[:, 1] / proc_h).clamp(0.0, 1.0 - 1e-6)
        gx = u_norm * gw  # continuous grid coords, cell units
        gy = v_norm * gh

        # --- Heatmap CE against a normalized Gaussian rendered on the cell grid ---
        sigma_x = self.config.heatmap_target_sigma_px / proc_w * gw
        sigma_y = self.config.heatmap_target_sigma_px / proc_h * gh
        xs = torch.arange(gw, device=device, dtype=torch.float32) + 0.5
        ys = torch.arange(gh, device=device, dtype=torch.float32) + 0.5
        dx2 = (xs.view(1, 1, gw) - gx.view(B, 1, 1)) ** 2 / (2 * sigma_x**2)
        dy2 = (ys.view(1, gh, 1) - gy.view(B, 1, 1)) ** 2 / (2 * sigma_y**2)
        target_heat = torch.exp(-(dx2 + dy2)).view(B, N)
        target_heat = target_heat / target_heat.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        log_probs = F.log_softmax(heat_logits, dim=-1)
        heat_ce = -(target_heat * log_probs).sum(dim=-1)  # (B,)
        heatmap_loss = (heat_ce * heat_valid).sum() / heat_valid.sum().clamp_min(1.0)

        # --- Subpixel offset L1 at the ground-truth cell ---
        cell_x = gx.long().clamp(max=gw - 1)
        cell_y = gy.long().clamp(max=gh - 1)
        cell_idx = cell_y * gw + cell_x  # (B,)
        pred_off = offsets[torch.arange(B, device=device), cell_idx]  # (B, 2)
        target_off = torch.stack((gx - cell_x.float() - 0.5, gy - cell_y.float() - 0.5), dim=-1)
        offset_l1 = F.smooth_l1_loss(pred_off, target_off, reduction="none").sum(dim=-1)
        offset_loss = (offset_l1 * heat_valid).sum() / heat_valid.sum().clamp_min(1.0)

        # --- Visibility BCE over all rows ---
        visibility_loss = F.binary_cross_entropy_with_logits(vis_logit, target_visible)

        loss = (
            self.config.heatmap_loss_weight * heatmap_loss
            + self.config.offset_loss_weight * offset_loss
            + self.config.visibility_loss_weight * visibility_loss
        )

        outputs = {
            "heatmap_loss": heatmap_loss,
            "offset_loss": offset_loss,
            "visibility_loss": visibility_loss,
        }

        # --- Optional depth ---
        if self.perception_head.use_depth and "target_depth" in inputs:
            depth_target = torch.nan_to_num(
                inputs["target_depth"].float().view(B) / self.config.depth_scale
            )
            depth_valid = heat_valid * (~torch.isnan(inputs["target_depth"].view(B))).float()
            depth_err = F.smooth_l1_loss(
                head_out["depth_scaled"].float(), depth_target, reduction="none"
            )
            depth_loss = (depth_err * depth_valid).sum() / depth_valid.sum().clamp_min(1.0)
            loss = loss + self.config.depth_loss_weight * depth_loss
            outputs["depth_loss"] = depth_loss

        # --- Optional wall normal ---
        if self.perception_head.use_normal and "target_wall_normal" in inputs:
            normal_target = inputs["target_wall_normal"].float().view(B, 3)
            normal_valid = heat_valid * (~torch.isnan(normal_target).any(dim=-1)).float()
            normal_err = F.mse_loss(
                head_out["wall_normal"].float(), torch.nan_to_num(normal_target), reduction="none"
            ).sum(dim=-1)
            normal_loss = (normal_err * normal_valid).sum() / normal_valid.sum().clamp_min(1.0)
            loss = loss + self.config.wall_normal_loss_weight * normal_loss
            outputs["normal_loss"] = normal_loss

        outputs["loss"] = loss

        # --- Monitoring metrics (no grad) ---
        with torch.no_grad():
            pred_uv = self._decode_uv(heat_logits, offsets, gh, gw)
            err_px = torch.linalg.vector_norm(pred_uv - uv, dim=-1)
            n_vis = heat_valid.sum().clamp_min(1.0)
            outputs["uv_err_px_metric"] = (err_px * heat_valid).sum() / n_vis
            vis_pred = (torch.sigmoid(vis_logit) > 0.5).float()
            outputs["visibility_acc_metric"] = (vis_pred == target_visible).float().mean()

        # Plain dict (not BatchFeature): HF Trainer's compute_loss requires a dict
        # subclass to index outputs["loss"].
        return outputs

    def _processed_image_size(self) -> Tuple[float, float]:
        """(W, H) of the processor's output image — the frame target_uv lives in."""
        size = self.config.image_target_size
        if isinstance(size, (tuple, list)):
            return float(size[1]), float(size[0])  # stored as (H, W)
        return float(size), float(size)

    def _decode_uv(
        self, heat_logits: torch.Tensor, offsets: torch.Tensor, gh: int, gw: int
    ) -> torch.Tensor:
        """Argmax cell + its subpixel offset -> uv in processed-image px, (B, 2)."""
        B = heat_logits.shape[0]
        proc_w, proc_h = self._processed_image_size()
        best = heat_logits.argmax(dim=-1)  # (B,)
        off = offsets[torch.arange(B, device=heat_logits.device), best].clamp(-0.5, 0.5)
        cx = (best % gw).float() + 0.5 + off[:, 0]
        cy = (best // gw).float() + 0.5 + off[:, 1]
        return torch.stack((cx / gw * proc_w, cy / gh * proc_h), dim=-1)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, inputs: dict) -> BatchFeature:
        """Run perception on processed inputs (from the perception processor).

        Returns:
            target_uv: (B, 2) predicted target center, processed-image px
                       (use uv_processed_to_camera to map back to raw wrist px)
            visibility: (B,) sigmoid probability the target is visible
            heatmap: (B, gh, gw) softmax heatmap over the token grid
            depth: (B,) target depth in metres (if the depth head is enabled)
            wall_normal: (B, 3) wall normal in camera frame (if enabled)
        """
        inputs = self.prepare_input(inputs)
        backbone_output = self.backbone(BatchFeature(data=inputs))
        image_tokens, gh, gw = self._image_token_grid(backbone_output, inputs["image_grid_thw"])
        head_out = self.perception_head(image_tokens)

        heat_logits = head_out["heat_logits"].float()
        offsets = head_out["offsets"].float()
        result = {
            "target_uv": self._decode_uv(heat_logits, offsets, gh, gw),
            "visibility": torch.sigmoid(head_out["vis_logit"].float()),
            "heatmap": F.softmax(heat_logits, dim=-1).view(-1, gh, gw),
        }
        if self.perception_head.use_depth:
            result["depth"] = head_out["depth_scaled"].float() * self.config.depth_scale
        if self.perception_head.use_normal:
            result["wall_normal"] = F.normalize(head_out["wall_normal"].float(), dim=-1)
        return BatchFeature(data=result)

    def uv_processed_to_camera(
        self, uv: torch.Tensor, camera_size: Tuple[int, int] = (720, 720)
    ) -> torch.Tensor:
        """Map uv from processed-image px back to RAW wrist-camera px.

        Inverts the deterministic eval transform (letterbox -> resize shortest
        edge to image_target_size -> center crop crop_fraction -> resize back).
        Assumes a square camera image (no letterbox padding).
        """
        cam_w, cam_h = float(camera_size[0]), float(camera_size[1])
        proc_w, proc_h = self._processed_image_size()
        crop = self.config.image_crop_size
        crop_h, crop_w = crop if isinstance(crop, (tuple, list)) else (crop, crop)
        # Forward: resize cam -> (proc_h, proc_w); center-crop to (crop_h, crop_w)
        # using integer offsets; resize back to (proc_h, proc_w).
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


AutoConfig.register("Gr00tN1d7Perception", Gr00tN1d7PerceptionConfig)
AutoModel.register(Gr00tN1d7PerceptionConfig, Gr00tN1d7Perception)
