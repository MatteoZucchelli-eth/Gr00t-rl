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

"""Processor for the perception-conditioned action model.

Identical behaviour to Gr00tN1d7PerceptionProcessor — keypoint-aware augmentation
so the (u, v) target label follows the same crop/resize as the wrist image — but
registered under the Gr00tN1d7PerceptionAction model type. The state/action
normalization and action-chunk assembly come from the base Gr00tN1d7Processor, so
this model gets a full action chunk while keeping the aligned heatmap labels.
"""

from transformers import AutoProcessor

from .processing_perception import Gr00tN1d7PerceptionProcessor


class Gr00tN1d7PerceptionActionProcessor(Gr00tN1d7PerceptionProcessor):
    pass


AutoProcessor.register("Gr00tN1d7PerceptionAction", Gr00tN1d7PerceptionActionProcessor)
