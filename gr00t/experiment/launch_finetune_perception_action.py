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

# Launch finetuning of the PERCEPTION-CONDITIONED ACTION model for the short-range
# fine approach: wrist-only VLM backbone + target heatmap head + flow-matching
# action head, where the perception output is cast into conditioning tokens the
# DiT attends to. Trains on fine-phase frames only (heatmap_labels/ sidecar).

import os

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import GR00T_BASE_MODEL, PerceptionActionFinetuneConfig
from gr00t.configs.model.gr00t_n1d7_perception_action import Gr00tN1d7PerceptionActionConfig
from gr00t.experiment.experiment import run
from gr00t.experiment.launch_finetune import load_modality_config


if __name__ == "__main__":
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    ft_config = tyro.cli(PerceptionActionFinetuneConfig, description=__doc__)
    from gr00t.data.embodiment_tags import EmbodimentTag

    embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag).value

    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [ft_config.dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    # Swap the model config for the perception-action one.
    model = Gr00tN1d7PerceptionActionConfig()
    config.model = model

    model.tune_llm = ft_config.tune_llm
    model.tune_visual = ft_config.tune_visual
    model.tune_top_llm_layers = ft_config.tune_top_llm_layers
    model.select_layer = ft_config.select_layer
    model.tune_projector = ft_config.tune_projector
    model.tune_diffusion_model = ft_config.tune_diffusion_model
    model.tune_vlln = True

    model.load_bf16 = False
    model.reproject_vision = False
    model.model_name = "nvidia/Cosmos-Reason2-2B"
    model.backbone_trainable_params_fp32 = True
    # Relative actions (deltas w.r.t. the current pose, key-matched against the
    # state) — matches the ActionRepresentation.RELATIVE in the modality config and
    # what the full VLA is trained with.
    model.use_relative_action = True
    model.state_dropout_prob = ft_config.state_dropout_prob
    model.action_horizon = ft_config.action_horizon

    # Image resolution fed to the VLM.
    assert ft_config.image_crop_size < ft_config.image_target_size, (
        "image_crop_size must be smaller than image_target_size"
    )
    model.image_target_size = (ft_config.image_target_size, ft_config.image_target_size)
    model.image_crop_size = (ft_config.image_crop_size, ft_config.image_crop_size)
    model.shortest_image_edge = None
    model.crop_fraction = None
    model.color_jitter_params = ft_config.color_jitter_params
    model.random_rotation_angle = None  # would desync the wall-normal label
    model.extra_augmentation_config = None

    # Perception head / loss (unchanged from the perception-only finetune).
    model.heatmap_hidden_dim = ft_config.heatmap_hidden_dim
    model.heatmap_target_sigma_px = ft_config.heatmap_target_sigma_px
    model.heatmap_loss_weight = ft_config.heatmap_loss_weight
    model.offset_loss_weight = ft_config.offset_loss_weight
    model.visibility_loss_weight = ft_config.visibility_loss_weight
    model.depth_loss_weight = ft_config.depth_loss_weight
    model.wall_normal_loss_weight = ft_config.wall_normal_loss_weight
    model.perception_loss_weight = ft_config.perception_loss_weight

    # Perception -> action conditioning tokens.
    model.use_perception_conditioning = ft_config.use_perception_conditioning
    model.use_perception_geometry_token = ft_config.use_perception_geometry_token
    model.use_perception_appearance_token = ft_config.use_perception_appearance_token
    model.perception_num_fourier_bands = ft_config.perception_num_fourier_bands
    model.perception_appearance_temperature = ft_config.perception_appearance_temperature
    model.perception_appearance_stopgrad = ft_config.perception_appearance_stopgrad
    model.perception_cond_teacher_forcing = ft_config.perception_cond_teacher_forcing
    model.perception_cond_noise_uv_px = ft_config.perception_cond_noise_uv_px
    model.perception_cond_noise_depth_m = ft_config.perception_cond_noise_depth_m
    model.perception_cond_noise_normal = ft_config.perception_cond_noise_normal
    model.perception_cond_bias_depth_m = ft_config.perception_cond_bias_depth_m
    model.perception_cond_vis_dropout_prob = ft_config.perception_cond_vis_dropout_prob

    config.training.experiment_name = ft_config.experiment_name
    # The base is always the GR00T foundation checkpoint; see GR00T_BASE_MODEL.
    config.training.start_from_checkpoint = GR00T_BASE_MODEL
    config.training.perception_start_from_checkpoint = ft_config.perception_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project
    config.training.save_only_model = ft_config.save_only_model
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
    # Train only on the labelled fine-phase (close-range) frames, matching the
    # perception model this one is initialized from.
    config.data.fine_phase_only = True

    run(config)
