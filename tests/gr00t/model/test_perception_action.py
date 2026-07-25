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

"""Test the perception -> action conditioning path.

Covers PerceptionConditioner (token shapes, teacher forcing, the unknown token,
Fourier resolution, gradient routing), the extra_cond_tokens plumbing through
Gr00tN1d7ActionHead, and a whole-model forward/get_action of
Gr00tN1d7PerceptionAction. The VLM backbone is mocked (same pattern as
test_model_forward.py), so everything here runs on CPU with no download.
"""

import math
from unittest.mock import MagicMock, patch

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.configs.model.gr00t_n1d7_perception_action import Gr00tN1d7PerceptionActionConfig
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from gr00t.model.gr00t_n1d7.perception import perception_losses
from gr00t.model.gr00t_n1d7.perception_action import PerceptionConditioner
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


GH, GW = 4, 5  # token grid
N_CELLS = GH * GW
PROC_W = PROC_H = 720.0
LATENT_DIM = 16
COND_DIM = 64


def _make_conditioner(**overrides) -> PerceptionConditioner:
    kwargs = dict(
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
        num_fourier_bands=8,
        use_depth=True,
        use_normal=True,
        teacher_forcing=True,
        noise_uv_px=0.0,
        noise_depth_m=0.0,
        noise_normal=0.0,
        vis_dropout_prob=0.0,
        depth_scale=0.3,
    )
    kwargs.update(overrides)
    conditioner = PerceptionConditioner(**kwargs)
    conditioner.eval()
    return conditioner


def _make_head_out(batch_size=2, peak_cell=7):
    """Synthetic perception head output with a sharp peak at `peak_cell`."""
    heat_logits = torch.zeros(batch_size, N_CELLS)
    heat_logits[:, peak_cell] = 10.0
    return {
        "heat_logits": heat_logits,
        "offsets": torch.zeros(batch_size, N_CELLS, 2),
        "vis_logit": torch.full((batch_size,), 3.0),
        "depth_scaled": torch.full((batch_size,), 0.5),
        "wall_normal": torch.tensor([[0.0, 0.0, 1.0]]).repeat(batch_size, 1),
    }


def _make_labels(batch_size=2, visible=(True, True), uv=(100.0, 200.0)):
    vis = torch.tensor([1.0 if v else 0.0 for v in visible])
    target_uv = torch.tensor([uv], dtype=torch.float32).repeat(batch_size, 1)
    target_uv[vis == 0] = float("nan")
    depth = torch.full((batch_size,), 0.15)
    depth[vis == 0] = float("nan")
    normal = torch.tensor([[0.0, 0.0, 1.0]]).repeat(batch_size, 1)
    normal[vis == 0] = float("nan")
    return {
        "target_uv": target_uv,
        "target_visible": vis,
        "target_depth": depth,
        "target_wall_normal": normal,
    }


def _tokens(conditioner, head_out, labels, image_tokens=None, batch_size=2):
    if image_tokens is None:
        image_tokens = torch.randn(batch_size, N_CELLS, LATENT_DIM)
    return conditioner(head_out, image_tokens, GH, GW, PROC_W, PROC_H, labels=labels)


class TestConditionerTokens:
    def test_emits_two_tokens(self):
        conditioner = _make_conditioner()
        assert conditioner.num_tokens == 2
        tokens, _ = _tokens(conditioner, _make_head_out(), _make_labels())
        assert tokens.shape == (2, 2, COND_DIM)
        assert torch.isfinite(tokens).all()

    def test_single_token_modes(self):
        for geom, app, expected in ((True, False, 1), (False, True, 1)):
            conditioner = _make_conditioner(use_geometry_token=geom, use_appearance_token=app)
            assert conditioner.num_tokens == expected
            tokens, _ = _tokens(conditioner, _make_head_out(), _make_labels())
            assert tokens.shape == (2, expected, COND_DIM)

    def test_rejects_both_tokens_disabled(self):
        with pytest.raises(ValueError, match="at least one token"):
            _make_conditioner(use_geometry_token=False, use_appearance_token=False)

    def test_geometry_feature_dim_matches_config(self):
        # 4*bands fourier + 2 raw uv + 1 vis + 2 confidence + 2 depth + 4 normal
        conditioner = _make_conditioner(num_fourier_bands=8)
        assert conditioner.geom_dim == 4 * 8 + 2 + 1 + 2 + 2 + 4
        no_extras = _make_conditioner(use_depth=False, use_normal=False)
        assert no_extras.geom_dim == 4 * 8 + 2 + 1 + 2


class TestTeacherForcing:
    def test_conditions_on_label_when_visible(self):
        conditioner = _make_conditioner()
        _, info = _tokens(conditioner, _make_head_out(), _make_labels(uv=(100.0, 200.0)))
        assert torch.allclose(info["cond_uv"], torch.tensor([[100.0, 200.0]]).repeat(2, 1))

    def test_falls_back_to_prediction_when_label_missing(self):
        conditioner = _make_conditioner()
        head_out = _make_head_out(peak_cell=7)
        _, info = _tokens(conditioner, head_out, _make_labels(visible=(True, False)))
        # Row 0 is teacher-forced onto the label, row 1 onto the decoded prediction.
        assert torch.allclose(info["cond_uv"][0], torch.tensor([100.0, 200.0]))
        assert torch.allclose(info["cond_uv"][1], info["pred_uv"][1])

    def test_no_labels_uses_prediction(self):
        conditioner = _make_conditioner()
        _, info = _tokens(conditioner, _make_head_out(), None)
        assert torch.allclose(info["cond_uv"], info["pred_uv"])

    def test_teacher_forcing_disabled_uses_prediction(self):
        conditioner = _make_conditioner(teacher_forcing=False)
        _, info = _tokens(conditioner, _make_head_out(), _make_labels())
        assert torch.allclose(info["cond_uv"], info["pred_uv"])

    def test_depth_bias_applies_in_both_modes(self):
        """The measured depth error is 84% systematic offset. Unlike the noise, the
        bias must be present at eval too — it is what the model always outputs."""
        bias = 0.0065
        conditioner = _make_conditioner(bias_depth_m=bias, depth_scale=0.3)
        labels = _make_labels()  # depth label = 0.15 m
        for mode in ("eval", "train"):
            getattr(conditioner, mode)()
            resolved = conditioner.resolve_conditioning(_make_head_out(), torch.zeros(2, 2), labels)
            got = resolved["depth_scaled"] * 0.3
            assert torch.allclose(got, torch.full((2,), 0.15 + bias), atol=1e-6), mode

    def test_zero_bias_leaves_depth_untouched(self):
        conditioner = _make_conditioner(bias_depth_m=0.0, depth_scale=0.3)
        resolved = conditioner.resolve_conditioning(
            _make_head_out(), torch.zeros(2, 2), _make_labels()
        )
        assert torch.allclose(resolved["depth_scaled"] * 0.3, torch.full((2,), 0.15), atol=1e-6)

    def test_noise_only_in_training_mode(self):
        conditioner = _make_conditioner(noise_uv_px=5.0)
        labels = _make_labels()
        _, eval_info = _tokens(conditioner, _make_head_out(), labels)
        assert torch.allclose(eval_info["cond_uv"], torch.tensor([[100.0, 200.0]]).repeat(2, 1))

        conditioner.train()
        torch.manual_seed(0)
        _, train_info = _tokens(conditioner, _make_head_out(), labels)
        assert not torch.allclose(train_info["cond_uv"], eval_info["cond_uv"])
        # Noise is calibrated, not wild: within a few sigma of the label.
        assert (train_info["cond_uv"] - eval_info["cond_uv"]).abs().max() < 5 * 5.0


class TestUnknownToken:
    def test_invisible_row_gets_exactly_the_unknown_token(self):
        conditioner = _make_conditioner()
        tokens, _ = _tokens(conditioner, _make_head_out(), _make_labels(visible=(True, False)))
        assert torch.allclose(tokens[1, 0], conditioner.unknown_token[0, 0], atol=1e-6)
        assert torch.allclose(tokens[1, 1], conditioner.unknown_appearance_token[0, 0], atol=1e-6)
        # The visible row must NOT be the unknown token.
        assert not torch.allclose(tokens[0, 0], conditioner.unknown_token[0, 0], atol=1e-4)

    def test_visibility_dropout_hides_a_visible_target(self):
        conditioner = _make_conditioner(vis_dropout_prob=1.0)
        conditioner.train()
        tokens, info = _tokens(conditioner, _make_head_out(), _make_labels())
        assert info["cond_vis_mean_metric"].item() == 0.0
        for row in range(tokens.shape[0]):
            assert torch.allclose(tokens[row, 0], conditioner.unknown_token[0, 0], atol=1e-6)


class TestFourierResolution:
    """The reason (u, v) is Fourier-encoded: a plain linear encoding cannot
    resolve a 1 px difference at 720 px above its own weight noise."""

    def test_one_pixel_shift_is_resolvable(self):
        conditioner = _make_conditioner(num_fourier_bands=8)
        uv_a = torch.tensor([[360.0 / PROC_W, 360.0 / PROC_H]])
        uv_b = torch.tensor([[361.0 / PROC_W, 360.0 / PROC_H]])
        feat_a = conditioner._fourier(uv_a)
        feat_b = conditioner._fourier(uv_b)
        raw_delta = (uv_a - uv_b).norm().item()
        fourier_delta = (feat_a - feat_b).norm().item()
        assert raw_delta < 2e-3  # what a raw-coordinate encoding would have to see
        assert fourier_delta > 50 * raw_delta

    def test_highest_band_period_covers_a_few_pixels(self):
        conditioner = _make_conditioner(num_fourier_bands=8)
        highest_cycles = conditioner.fourier_freqs[-1].item() / (2 * math.pi)
        period_px = PROC_W / highest_cycles
        assert period_px < 8.0  # a 1 px shift moves this band by > 0.75 rad


class TestGradientRouting:
    """Token A must not carry gradient into the perception head (it is a label or
    a detached prediction); token B is the deliberate differentiable path."""

    def test_appearance_token_backprops_into_the_heatmap(self):
        conditioner = _make_conditioner(use_geometry_token=False, use_appearance_token=True)
        head_out = _make_head_out()
        head_out["heat_logits"] = head_out["heat_logits"].clone().requires_grad_(True)
        tokens, _ = _tokens(conditioner, head_out, _make_labels())
        tokens.sum().backward()
        assert head_out["heat_logits"].grad is not None
        assert head_out["heat_logits"].grad.abs().sum() > 0

    def test_geometry_token_does_not_backprop_into_the_heatmap(self):
        conditioner = _make_conditioner(use_geometry_token=True, use_appearance_token=False)
        head_out = _make_head_out()
        head_out["heat_logits"] = head_out["heat_logits"].clone().requires_grad_(True)
        head_out["vis_logit"] = head_out["vis_logit"].clone().requires_grad_(True)
        tokens, _ = _tokens(conditioner, head_out, _make_labels())
        tokens.sum().backward()
        assert head_out["heat_logits"].grad is None
        assert head_out["vis_logit"].grad is None

    def test_stopgrad_cuts_the_appearance_path(self):
        conditioner = _make_conditioner(
            use_geometry_token=False, use_appearance_token=True, appearance_stopgrad=True
        )
        head_out = _make_head_out()
        head_out["heat_logits"] = head_out["heat_logits"].clone().requires_grad_(True)
        tokens, _ = _tokens(conditioner, head_out, _make_labels())
        tokens.sum().backward()
        assert head_out["heat_logits"].grad is None


class TestConditionerParameterInit:
    def test_reset_parameters_reinitializes_unknown_tokens(self):
        conditioner = _make_conditioner()
        with torch.no_grad():
            conditioner.unknown_token.zero_()
            conditioner.unknown_appearance_token.zero_()
        conditioner.reset_parameters()
        assert conditioner.unknown_token.abs().sum() > 0
        assert conditioner.unknown_appearance_token.abs().sum() > 0


def _small_action_config(**overrides) -> Gr00tN1d7Config:
    defaults = dict(
        backbone_embedding_dim=LATENT_DIM,
        hidden_size=32,
        input_embedding_dim=COND_DIM,
        max_state_dim=7,
        max_action_dim=7,
        action_horizon=4,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=4,
        max_seq_len=32,
        use_alternate_vl_dit=False,
        state_dropout_prob=0.0,
        attn_dropout=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            # inner_dim (heads * head_dim) must equal input_embedding_dim,
            # and output_dim must equal hidden_size.
            "num_attention_heads": 2,
            "attention_head_dim": COND_DIM // 2,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 32,
            "interleave_self_attention": True,
        },
    )
    defaults.update(overrides)
    return Gr00tN1d7Config(**defaults)


def _backbone_output(config, batch_size=2, seq_len=8):
    return BatchFeature(
        data={
            "backbone_features": torch.randn(batch_size, seq_len, config.backbone_embedding_dim),
            "backbone_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "image_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        }
    )


def _action_input(config, batch_size=2):
    return BatchFeature(
        data={
            "state": torch.randn(batch_size, config.state_history_length, config.max_state_dim),
            "action": torch.randn(batch_size, config.action_horizon, config.max_action_dim),
            "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
            "action_mask": torch.ones(batch_size, config.action_horizon, config.max_action_dim),
        }
    )


class TestActionHeadExtraCondTokens:
    """The conditioning tokens must reach the DiT's self-attention stream without
    disturbing the action slice."""

    def test_forward_accepts_extra_tokens(self):
        config = _small_action_config()
        head = Gr00tN1d7ActionHead(config)
        head.train()
        extra = torch.randn(2, 2, config.input_embedding_dim)
        out = head.forward(_backbone_output(config), _action_input(config), extra_cond_tokens=extra)
        assert torch.isfinite(out["loss"])
        assert out["action_loss"].shape == (2, config.action_horizon, config.max_action_dim)
        # state token + 2 conditioning tokens
        assert out["state_features"].shape[1] == 3

    def test_get_action_accepts_extra_tokens(self):
        config = _small_action_config()
        head = Gr00tN1d7ActionHead(config)
        head.eval()
        action_input = _action_input(config)
        del action_input["action"]
        extra = torch.randn(2, 2, config.input_embedding_dim)
        out = head.get_action(_backbone_output(config), action_input, extra_cond_tokens=extra)
        assert out["action_pred"].shape == (2, config.action_horizon, config.max_action_dim)
        assert torch.isfinite(out["action_pred"]).all()

    def test_extra_tokens_change_the_prediction(self):
        # If the tokens did not reach the DiT this would silently pass as a no-op.
        torch.manual_seed(0)
        config = _small_action_config()
        head = Gr00tN1d7ActionHead(config)
        head.eval()
        backbone_output = _backbone_output(config)
        action_input = _action_input(config)
        del action_input["action"]
        extra = torch.randn(2, 2, config.input_embedding_dim) * 5.0

        torch.manual_seed(1)
        without = head.get_action(backbone_output, action_input)["action_pred"]
        torch.manual_seed(1)
        with_tokens = head.get_action(backbone_output, action_input, extra_cond_tokens=extra)[
            "action_pred"
        ]
        assert not torch.allclose(without, with_tokens, atol=1e-4)


class TestPerceptionLossHelper:
    """perception_losses is shared by the perception-only and perception-action
    models; it must expose the weighted sum separately from the individual terms."""

    def test_returns_weighted_sum_and_terms(self):
        config = Gr00tN1d7PerceptionActionConfig(
            image_target_size=(int(PROC_H), int(PROC_W)),
            depth_loss_weight=0.1,
            wall_normal_loss_weight=0.1,
        )
        head_out = _make_head_out()
        out = perception_losses(
            config, head_out, _make_labels(), GH, GW, use_depth=True, use_normal=True
        )
        for key in (
            "heatmap_loss",
            "offset_loss",
            "visibility_loss",
            "depth_loss",
            "normal_loss",
            "perception_loss",
            "uv_err_px_metric",
            "visibility_acc_metric",
        ):
            assert key in out, key
            assert torch.isfinite(out[key]), key

        expected = (
            config.heatmap_loss_weight * out["heatmap_loss"]
            + config.offset_loss_weight * out["offset_loss"]
            + config.visibility_loss_weight * out["visibility_loss"]
            + config.depth_loss_weight * out["depth_loss"]
            + config.wall_normal_loss_weight * out["normal_loss"]
        )
        assert torch.allclose(out["perception_loss"], expected)

    def test_invisible_rows_do_not_poison_the_heatmap_loss(self):
        config = Gr00tN1d7PerceptionActionConfig(
            image_target_size=(int(PROC_H), int(PROC_W)),
            depth_loss_weight=0.0,
            wall_normal_loss_weight=0.0,
        )
        out = perception_losses(
            config,
            _make_head_out(),
            _make_labels(visible=(True, False)),
            GH,
            GW,
            use_depth=False,
            use_normal=False,
        )
        assert torch.isfinite(out["heatmap_loss"])
        assert torch.isfinite(out["offset_loss"])


# ----------------------------------------------------------------------------
# Whole-model forward with a mocked backbone (mirrors test_model_forward.py)
# ----------------------------------------------------------------------------

MERGE = 2
IMG_SIZE = 64  # processed-image edge, px
TEXT_TOKENS = 4
SEQ_LEN = N_CELLS + TEXT_TOKENS


def _perception_action_config(**overrides) -> Gr00tN1d7PerceptionActionConfig:
    defaults = dict(
        model_name="nvidia/Cosmos-Reason2-2B",
        backbone_model_type="qwen",
        backbone_embedding_dim=LATENT_DIM,
        hidden_size=32,
        input_embedding_dim=COND_DIM,
        max_state_dim=7,
        max_action_dim=7,
        action_horizon=4,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=4,
        max_seq_len=32,
        use_alternate_vl_dit=False,
        select_layer=1,
        use_flash_attention=False,
        # The real config pins a 4-layer VL self-attention sized for the 2048-d
        # backbone; at these toy dimensions it must be off.
        vl_self_attention_cfg={"num_layers": 0},
        image_target_size=(IMG_SIZE, IMG_SIZE),
        image_crop_size=(IMG_SIZE - 8, IMG_SIZE - 8),
        heatmap_hidden_dim=16,
        perception_pool_heads=2,
        state_dropout_prob=0.0,
        attn_dropout=0.0,
        perception_cond_noise_uv_px=0.0,
        perception_cond_noise_depth_m=0.0,
        perception_cond_noise_normal=0.0,
        perception_cond_vis_dropout_prob=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": COND_DIM // 2,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 32,
            "interleave_self_attention": True,
        },
    )
    defaults.update(overrides)
    return Gr00tN1d7PerceptionActionConfig(**defaults)


def _mock_backbone(config):
    """Backbone stub emitting a VL sequence whose first N_CELLS tokens are image tokens."""
    backbone = MagicMock()

    def fake_forward(vl_input):
        batch = vl_input["state"].shape[0]
        image_mask = torch.zeros(batch, SEQ_LEN, dtype=torch.bool)
        image_mask[:, :N_CELLS] = True
        return BatchFeature(
            data={
                "backbone_features": torch.randn(batch, SEQ_LEN, config.backbone_embedding_dim),
                "backbone_attention_mask": torch.ones(batch, SEQ_LEN, dtype=torch.long),
                "image_mask": image_mask,
            }
        )

    backbone.side_effect = fake_forward
    backbone.prepare_input = lambda x: BatchFeature(data=x)
    backbone.model.config.vision_config.spatial_merge_size = MERGE
    backbone.parameters = lambda: iter([torch.zeros(1)])
    return backbone


@pytest.fixture
def perception_action_model():
    config = _perception_action_config()
    with patch("gr00t.model.gr00t_n1d7.perception_action.get_backbone_cls") as mock_get_cls:
        mock_get_cls.return_value = lambda **kwargs: _mock_backbone(config)
        with patch("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor"):
            from gr00t.model.gr00t_n1d7.perception_action import Gr00tN1d7PerceptionAction

            model = Gr00tN1d7PerceptionAction(config)
    model.eval()
    return model, config


def _model_inputs(config, batch_size=2, visible=(True, True)):
    labels = _make_labels(batch_size=batch_size, visible=visible, uv=(30.0, 20.0))
    return {
        "state": torch.randn(batch_size, config.state_history_length, config.max_state_dim),
        "action": torch.randn(batch_size, config.action_horizon, config.max_action_dim),
        "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
        "action_mask": torch.ones(batch_size, config.action_horizon, config.max_action_dim),
        "image_grid_thw": torch.tensor([[1, GH * MERGE, GW * MERGE]]).repeat(batch_size, 1),
        **labels,
    }


class TestPerceptionActionModel:
    def test_forward_combines_action_and_perception_loss(self, perception_action_model):
        model, config = perception_action_model
        model.train()
        out = model(_model_inputs(config))
        assert torch.isfinite(out["loss"]) and out["loss"].dim() == 0
        for key in ("action_loss_scalar", "heatmap_loss", "offset_loss", "visibility_loss"):
            assert key in out, key
        expected = (
            out["action_loss_scalar"] + config.perception_loss_weight * out["perception_loss"]
        )
        assert torch.allclose(out["loss"], expected, atol=1e-5)

    def test_forward_loss_backprops_to_both_heads(self, perception_action_model):
        model, config = perception_action_model
        model.train()
        out = model(_model_inputs(config))
        out["loss"].backward()
        assert model.perception_head.token_mlp[0].weight.grad.abs().sum() > 0
        action_grad = sum(
            p.grad.abs().sum() for p in model.action_head.parameters() if p.grad is not None
        )
        assert action_grad > 0
        assert model.conditioner.geom_mlp[0].weight.grad.abs().sum() > 0
        assert model.conditioner.appearance_proj.weight.grad.abs().sum() > 0

    def test_conditioning_tokens_reach_the_dit(self, perception_action_model):
        model, config = perception_action_model
        model.train()
        out = model(_model_inputs(config))
        # 1 state token + geometry + appearance
        assert out["state_features"].shape[1] == 1 + model.conditioner.num_tokens

    def test_metrics_are_logged(self, perception_action_model):
        model, config = perception_action_model
        model.train()
        out = model(_model_inputs(config))
        for key in ("uv_err_px_metric", "visibility_acc_metric", "cond_uv_err_px_metric"):
            assert key in out and torch.isfinite(out[key]), key
        # Teacher forcing with zero noise: the policy sees the exact label.
        assert out["cond_uv_err_px_metric"].item() == pytest.approx(0.0, abs=1e-4)

    def test_get_action_returns_actions_and_perception(self, perception_action_model):
        model, config = perception_action_model
        inputs = _model_inputs(config)
        del inputs["action"]
        out = model.get_action(inputs, options={"return_heatmap": True})
        assert out["action_pred"].shape == (2, config.action_horizon, config.max_action_dim)
        assert torch.isfinite(out["action_pred"]).all()
        assert out["target_uv"].shape == (2, 2)
        assert out["visibility"].shape == (2,)
        assert out["heatmap"].shape == (2, GH, GW)
        assert out["target_depth"].shape == (2,)
        assert out["wall_normal"].shape == (2, 3)

    def test_get_action_never_teacher_forces(self, perception_action_model):
        """At deployment there are no labels; passing them must change nothing."""
        model, config = perception_action_model
        torch.manual_seed(0)
        inputs = _model_inputs(config)
        del inputs["action"]
        torch.manual_seed(1)
        with_labels = model.get_action(dict(inputs))["target_uv"]
        stripped = {
            k: v
            for k, v in inputs.items()
            if k not in ("target_uv", "target_visible", "target_depth", "target_wall_normal")
        }
        torch.manual_seed(1)
        without_labels = model.get_action(stripped)["target_uv"]
        assert torch.allclose(with_labels, without_labels)

    def test_conditioning_can_be_disabled(self):
        config = _perception_action_config(use_perception_conditioning=False)
        with patch("gr00t.model.gr00t_n1d7.perception_action.get_backbone_cls") as mock_get_cls:
            mock_get_cls.return_value = lambda **kwargs: _mock_backbone(config)
            with patch("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor"):
                from gr00t.model.gr00t_n1d7.perception_action import Gr00tN1d7PerceptionAction

                model = Gr00tN1d7PerceptionAction(config)
        model.train()
        out = model(_model_inputs(config))
        assert model.conditioner is None
        assert torch.isfinite(out["loss"])
        # State token only: nothing extra reached the DiT.
        assert out["state_features"].shape[1] == 1

    def test_invisible_rows_still_produce_a_finite_loss(self, perception_action_model):
        model, config = perception_action_model
        model.train()
        out = model(_model_inputs(config, visible=(False, False)))
        assert torch.isfinite(out["loss"])
