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

"""Training pipeline for the perception-conditioned action model.

Weight initialization is a two-stage overlay, because the good init for this model
lives in two different checkpoints:

  1. ``start_from_checkpoint`` (base) — a full VLA run or nvidia/GR00T-N1.7-3B.
     Supplies ``action_head.*`` (a trained DiT is worth a lot; a fresh one costs
     thousands of steps) and, failing anything better, ``backbone.*``.
  2. ``perception_start_from_checkpoint`` (optional overlay) — one of your
     perception runs. Supplies ``perception_head.*`` AND ``backbone.*``, and it
     deliberately WINS on the backbone: the perception head's features are only
     meaningful on the backbone it was trained against.

Anything absent from both keeps its constructor initialization (the conditioner is
normally fresh). Unlike the base pipeline this never calls
``AutoModel.from_pretrained`` on the checkpoint — the checkpoint's config
describes a different model — so parameters are real tensors throughout and the
meta-device caveat that forces ``reset_parameters()`` elsewhere does not apply.
"""

import json
import logging
from pathlib import Path

from gr00t.configs.model.gr00t_n1d7_perception_action import Gr00tN1d7PerceptionActionConfig
from gr00t.model.gr00t_n1d7.perception_action import Gr00tN1d7PerceptionAction
from gr00t.model.gr00t_n1d7.processing_perception_action import Gr00tN1d7PerceptionActionProcessor
from gr00t.model.gr00t_n1d7.setup_perception import (
    Gr00tN1d7PerceptionPipeline,
    _resolve_checkpoint_dir,
)
from gr00t.model.registry import register_model


def _checkpoint_files(checkpoint: str) -> list[Path]:
    ckpt_dir = _resolve_checkpoint_dir(checkpoint)
    index_file = ckpt_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            weight_map = json.load(f)["weight_map"]
        return sorted({ckpt_dir / fname for fname in weight_map.values()})
    if (ckpt_dir / "model.safetensors").exists():
        return [ckpt_dir / "model.safetensors"]
    raise FileNotFoundError(f"No safetensors weights found in {ckpt_dir}")


def load_partial_weights(model, checkpoint: str, prefixes: tuple[str, ...], label: str) -> set[str]:
    """Load only the keys under `prefixes` whose name and shape match the model.

    Returns the set of loaded keys so a later overlay can report what it replaced.
    Shape-mismatched or unknown keys are skipped (counted, not fatal): a VLA
    checkpoint legitimately carries e.g. tip_target_head weights this model has no
    slot for.
    """
    from safetensors import safe_open

    own_state = model.state_dict()
    to_load = {}
    skipped = 0
    for file in _checkpoint_files(checkpoint):
        with safe_open(file, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                if not key.startswith(prefixes):
                    continue
                if key not in own_state:
                    skipped += 1
                    continue
                tensor = sf.get_tensor(key)
                if own_state[key].shape != tensor.shape:
                    skipped += 1
                    continue
                to_load[key] = tensor

    model.load_state_dict(to_load, strict=False)
    by_group: dict[str, int] = {}
    for key in to_load:
        by_group[key.split(".")[0]] = by_group.get(key.split(".")[0], 0) + 1
    logging.info(
        f"[{label}] loaded {len(to_load)} tensors from {checkpoint} "
        f"({by_group or 'nothing matched!'}; {skipped} skipped)"
    )
    return set(to_load)


class Gr00tN1d7PerceptionActionPipeline(Gr00tN1d7PerceptionPipeline):
    model_class = Gr00tN1d7PerceptionAction
    processor_class = Gr00tN1d7PerceptionActionProcessor

    def _create_model(self):
        model = self.model_class(
            self.config.model,
            transformers_loading_kwargs=self.transformers_loading_kwargs,
        )

        skip_weight_loading = getattr(self.config.training, "skip_weight_loading", False)
        base_ckpt = self.config.training.start_from_checkpoint
        perception_ckpt = getattr(self.config.training, "perception_start_from_checkpoint", None)

        if skip_weight_loading:
            logging.info("skip_weight_loading=True — model keeps its fresh initialization")
        else:
            base_keys: set[str] = set()
            overlay_keys: set[str] = set()
            if base_ckpt is not None:
                base_keys = load_partial_weights(
                    model,
                    base_ckpt,
                    ("backbone.", "action_head.", "perception_head.", "conditioner."),
                    label="base",
                )
            if perception_ckpt is not None:
                overlay_keys = load_partial_weights(
                    model,
                    perception_ckpt,
                    ("backbone.", "perception_head.", "conditioner."),
                    label="perception-overlay",
                )
                replaced = len(base_keys & overlay_keys)
                logging.info(
                    f"[perception-overlay] {replaced} tensors from the base checkpoint were "
                    f"replaced (expected: the whole backbone — the perception head only "
                    f"works on the backbone it was trained against)"
                )

            loaded_any = bool(base_ckpt or perception_ckpt)
            if not loaded_any:
                logging.info(
                    "No checkpoint given — backbone starts from the published "
                    f"{self.config.model.model_name} weights, all heads are fresh"
                )
            else:
                own_state = model.state_dict()
                all_loaded = base_keys | overlay_keys
                missing_backbone = {k for k in own_state if k.startswith("backbone.")} - all_loaded
                if missing_backbone and base_ckpt is not None:
                    raise RuntimeError(
                        f"{len(missing_backbone)} backbone keys were not found in any given "
                        f"checkpoint (e.g. {sorted(missing_backbone)[:5]}). Wrong checkpoint?"
                    )
                for group in ("action_head", "perception_head", "conditioner"):
                    if not any(k.startswith(f"{group}.") for k in all_loaded):
                        logging.info(f"{group} not in any checkpoint — fresh initialization")

        # Re-apply trainable flags after weight loading.
        model.backbone.set_trainable_parameters(
            tune_llm=self.config.model.tune_llm,
            tune_visual=self.config.model.tune_visual,
            tune_top_llm_layers=self.config.model.tune_top_llm_layers,
        )
        model.action_head.set_trainable_parameters(
            tune_projector=self.config.model.tune_projector,
            tune_diffusion_model=self.config.model.tune_diffusion_model,
            tune_vlln=self.config.model.tune_vlln,
        )

        from gr00t.experiment.dist_utils import get_rank

        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_model_config.json", "w") as f:
                f.write(model.config.to_filtered_json())
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(
            f"Trainable parameters: {trainable_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )
        return model


register_model(Gr00tN1d7PerceptionActionConfig, Gr00tN1d7PerceptionActionPipeline)
