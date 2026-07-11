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

"""
Test Gr00tN1d7ActionHead: flow matching forward, get_action, feature encoding.

These tests instantiate the action head directly (no backbone required)
and feed it synthetic backbone output tensors.
"""

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


def _small_config(**overrides) -> Gr00tN1d7Config:
    defaults = dict(
        backbone_embedding_dim=64,
        hidden_size=64,
        input_embedding_dim=64,
        max_state_dim=7,
        max_action_dim=7,
        action_horizon=4,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=4,
        add_pos_embed=True,
        use_vlln=True,
        max_seq_len=32,
        use_alternate_vl_dit=False,
        attend_text_every_n_blocks=2,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.0,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        attn_dropout=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 32,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 64,
            "interleave_self_attention": True,
        },
    )
    defaults.update(overrides)
    return Gr00tN1d7Config(**defaults)


@pytest.fixture
def action_head():
    config = _small_config()
    head = Gr00tN1d7ActionHead(config)
    head.eval()
    return head, config


def _make_backbone_output(config, batch_size=2, seq_len=8):
    return BatchFeature(
        data={
            "backbone_features": torch.randn(batch_size, seq_len, config.backbone_embedding_dim),
            "backbone_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "image_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        }
    )


def _make_action_input(config, batch_size=2):
    return BatchFeature(
        data={
            "state": torch.randn(batch_size, config.state_history_length, config.max_state_dim),
            "action": torch.randn(batch_size, config.action_horizon, config.max_action_dim),
            "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
            "action_mask": torch.ones(batch_size, config.action_horizon, config.max_action_dim),
        }
    )


class TestActionHeadForward:
    """Test training forward pass."""

    def test_forward_returns_loss(self, action_head):
        head, config = action_head
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert "loss" in out
        assert out["loss"].dim() == 0
        assert torch.isfinite(out["loss"])

    def test_forward_loss_shape(self, action_head):
        head, config = action_head
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert out["action_loss"].shape == (2, config.action_horizon, config.max_action_dim)

    def test_head_does_not_apply_state_dropout(self):
        # State dropout is owned by the processor (raw states); the head must not
        # re-apply it to the encoded features (double dropout). With prob=1.0 the
        # old head-side dropout would zero every state feature entering the DiT.
        config = _small_config(state_dropout_prob=1.0)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        action_input = _make_action_input(config)
        out = head.forward(_make_backbone_output(config), action_input)
        assert torch.isfinite(out["loss"])
        assert out["state_features"].abs().sum() > 0


class TestActionHeadGetAction:
    """Test inference (denoising loop)."""

    def test_get_action_output_shape(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        del action_input["action"]  # get_action doesn't need ground-truth action
        out = head.get_action(_make_backbone_output(config), action_input)
        assert "action_pred" in out
        assert out["action_pred"].shape == (2, config.action_horizon, config.max_action_dim)

    def test_get_action_no_grad(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        del action_input["action"]
        out = head.get_action(_make_backbone_output(config), action_input)
        assert not out["action_pred"].requires_grad

    def test_get_action_single_sample(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config, batch_size=1)
        del action_input["action"]
        out = head.get_action(
            _make_backbone_output(config, batch_size=1),
            action_input,
        )
        assert out["action_pred"].shape[0] == 1


class TestActionHeadEncodeFeatures:
    """Test feature encoding helper."""

    def test_encode_features_shapes(self, action_head):
        head, config = action_head
        result = head._encode_features(
            _make_backbone_output(config),
            _make_action_input(config),
        )
        assert result["backbone_features"].shape == (2, 8, config.backbone_embedding_dim)
        assert result["state_features"].shape == (2, 1, config.input_embedding_dim)


class TestActionHeadTrainableParams:
    """Test parameter freezing."""

    def test_all_trainable_by_default(self, action_head):
        head, _ = action_head
        head.set_trainable_parameters(True, True, True)
        assert all(p.requires_grad for p in head.parameters())

    def test_freeze_projector(self):
        config = _small_config()
        head = Gr00tN1d7ActionHead(config)
        head.set_trainable_parameters(False, True, True)
        for p in head.state_encoder.parameters():
            assert not p.requires_grad
        for p in head.action_encoder.parameters():
            assert not p.requires_grad

    def test_freeze_diffusion(self):
        config = _small_config()
        head = Gr00tN1d7ActionHead(config)
        head.set_trainable_parameters(True, False, True)
        for p in head.model.parameters():
            assert not p.requires_grad


class TestTipTargetHead:
    """Test the optional tip-target auxiliary head (xyz + rot6d regression)."""

    @staticmethod
    def _tip_config(**overrides):
        defaults = dict(use_tip_target_head=True, tip_target_loss_weight=0.5)
        defaults.update(overrides)
        return _small_config(**defaults)

    @staticmethod
    def _tip_target(batch_size=2):
        target = torch.randn(batch_size, 9) * 0.1
        target[:, 3:] = torch.nn.functional.normalize(target[:, 3:], dim=-1)
        return target

    def test_rot6d_orthonormalize(self):
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import rot6d_orthonormalize

        r = rot6d_orthonormalize(torch.randn(4, 6))
        a, b = r[..., :3], r[..., 3:]
        assert torch.allclose(a.norm(dim=-1), torch.ones(4), atol=1e-5)
        assert torch.allclose(b.norm(dim=-1), torch.ones(4), atol=1e-5)
        assert torch.allclose((a * b).sum(-1), torch.zeros(4), atol=1e-5)

    def test_forward_adds_tip_loss(self):
        config = self._tip_config()
        head = Gr00tN1d7ActionHead(config)
        head.train()
        action_input = _make_action_input(config)
        action_input["tip_target_9d"] = self._tip_target()
        out = head.forward(_make_backbone_output(config), action_input)
        assert "tip_target_loss" in out
        assert torch.isfinite(out["tip_target_loss"])
        assert torch.isfinite(out["loss"])
        # Total loss = action loss + weight * tip loss.
        expected = out["action_loss_scalar"] + 0.5 * out["tip_target_loss"]
        assert torch.allclose(out["loss"], expected, atol=1e-5)
        # Conditioning appends one token to the single state token.
        assert out["state_features"].shape[1] == 2

    def test_forward_masks_nan_rows(self):
        config = self._tip_config()
        head = Gr00tN1d7ActionHead(config)
        head.train()
        action_input = _make_action_input(config)
        target = self._tip_target()
        target[0] = float("nan")  # unresolved label on one row
        action_input["tip_target_9d"] = target
        out = head.forward(_make_backbone_output(config), action_input)
        assert torch.isfinite(out["tip_target_loss"])
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        for p in head.tip_target_head.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()

    def test_forward_all_nan_skips_tip_loss(self):
        config = self._tip_config()
        head = Gr00tN1d7ActionHead(config)
        head.train()
        action_input = _make_action_input(config)
        action_input["tip_target_9d"] = torch.full((2, 9), float("nan"))
        out = head.forward(_make_backbone_output(config), action_input)
        assert "tip_target_loss" not in out
        assert torch.isfinite(out["loss"])

    def test_forward_without_target_conditions_on_prediction(self):
        config = self._tip_config()
        head = Gr00tN1d7ActionHead(config)
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert "tip_target_loss" not in out
        assert "tip_target_pred" in out
        assert torch.isfinite(out["loss"])

    def test_get_action_returns_orthonormal_prediction(self):
        config = self._tip_config()
        head = Gr00tN1d7ActionHead(config)
        head.eval()
        action_input = _make_action_input(config)
        del action_input["action"]
        out = head.get_action(_make_backbone_output(config), action_input)
        assert out["action_pred"].shape == (2, config.action_horizon, config.max_action_dim)
        pred = out["tip_target_pred"]
        assert pred.shape == (2, 9)
        a, b = pred[..., 3:6], pred[..., 6:]
        assert torch.allclose(a.norm(dim=-1), torch.ones(2), atol=1e-4)
        assert torch.allclose((a * b).sum(-1), torch.zeros(2), atol=1e-4)

    def test_no_conditioning_keeps_single_state_token(self):
        config = self._tip_config(use_tip_target_conditioning=False)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        action_input = _make_action_input(config)
        action_input["tip_target_9d"] = self._tip_target()
        out = head.forward(_make_backbone_output(config), action_input)
        assert head.tip_target_head.cond_embed is None
        assert out["state_features"].shape[1] == 1
        assert "tip_target_loss" in out

    def test_coexists_with_progress_head(self):
        config = self._tip_config(use_progress_head=True)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        action_input = _make_action_input(config)
        action_input["tip_target_9d"] = self._tip_target()
        action_input["progress"] = torch.rand(2, 1)
        out = head.forward(_make_backbone_output(config), action_input)
        assert "progress_loss" in out and "tip_target_loss" in out
        # state token + progress token + tip token
        assert out["state_features"].shape[1] == 3
        expected = (
            out["action_loss_scalar"]
            + config.progress_loss_weight * out["progress_loss"]
            + 0.5 * out["tip_target_loss"]
        )
        assert torch.allclose(out["loss"], expected, atol=1e-5)

    def test_xyz_scaling_roundtrip(self):
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import TipTargetHead

        head = TipTargetHead(
            latent_dim=8, hidden_dim=8, cond_dim=8, use_conditioning=True, xyz_scale=0.1
        )
        raw = self._tip_target()
        assert torch.allclose(head.unscale(head.scale(raw)), raw, atol=1e-6)
        assert torch.allclose(head.scale(raw)[..., :3], raw[..., :3] / 0.1)
