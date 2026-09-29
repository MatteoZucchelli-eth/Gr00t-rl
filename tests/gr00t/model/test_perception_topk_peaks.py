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


"""decode_uv_topk: the runner-up square, not the argmax's own neighbours.

The head is decoded by argmax, so with a distractor in frame the detection is
whichever square peaked higher on that inference. The other square is a second
mode of the same heatmap, which is what makes a caller with an outside opinion
about where the target is able to pick between them -- but only if the
candidates it is handed are distinct modes rather than five cells of one blob.
"""

from gr00t.model.gr00t_n1d7.perception import decode_uv, decode_uv_topk
import torch


GH = GW = 26
PROC = 736.0
N = GH * GW


def _blob(logits, row, col, peak, spread=1.0):
    """A quadratic bump centred on one cell, as a trained heatmap produces."""
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            r, c = row + dr, col + dc
            if 0 <= r < GH and 0 <= c < GW:
                logits[r * GW + c] = peak - spread * (dr * dr + dc * dc)


def _bimodal(spread=1.0):
    """One frame, two squares: the distractor peaks higher than the target.

    ``spread`` is how fast a mode falls off per cell. A target spanning several
    cells of the grid makes a broad one, which is the case that decides whether
    NMS is needed: the flatter the mode, the more its own neighbours outrank
    the other square.
    """
    logits = torch.full((1, N), -20.0)
    _blob(logits[0], 12, 19, 8.0, spread)  # distractor -> wins the argmax
    _blob(logits[0], 12, 6, 7.2, spread)  # the true target
    offsets = torch.zeros(1, N, 2)
    offsets[0, 12 * GW + 6] = torch.tensor([0.25, -0.25])
    return logits, offsets


def test_first_candidate_is_the_argmax():
    """Keeping candidate 0 has to be exactly today's behaviour, offset and all."""
    logits, offsets = _bimodal()
    uv, prob = decode_uv_topk(logits, offsets, GH, GW, PROC, PROC, k=3)
    assert torch.allclose(uv[:, 0], decode_uv(logits, offsets, GH, GW, PROC, PROC))
    assert prob[0, 0] > prob[0, 1] > 0.0


def test_second_candidate_is_the_other_square():
    """The runner-up mode lands on the true target, with its own subpixel offset."""
    logits, offsets = _bimodal()
    uv, _ = decode_uv_topk(logits, offsets, GH, GW, PROC, PROC, k=2)
    expected = torch.tensor(
        [(6 + 0.5 + 0.25) / GW * PROC, (12 + 0.5 - 0.25) / GH * PROC]
    )
    assert torch.allclose(uv[0, 1], expected, atol=1e-4)


def test_nms_is_what_separates_the_modes():
    """On a broad mode, the raw top-k is one square reported k times over."""
    logits, offsets = _bimodal(spread=0.1)
    cell_px = PROC / GW

    raw, _ = decode_uv_topk(
        logits, offsets, GH, GW, PROC, PROC, k=3, min_separation_cells=0
    )
    # Candidates 1 and 2 are the argmax's own neighbours, so the true square is
    # not among them however many are asked for.
    assert float((raw[0, 1] - raw[0, 0]).abs().max()) <= 2.0 * cell_px
    assert float((raw[0, 2] - raw[0, 0]).abs().max()) <= 2.0 * cell_px

    # Suppressed, the same heatmap gives up the other square as candidate 1.
    kept, _ = decode_uv_topk(
        logits, offsets, GH, GW, PROC, PROC, k=2, min_separation_cells=2
    )
    expected_u = (6 + 0.5 + 0.25) / GW * PROC
    assert abs(float(kept[0, 1, 0]) - expected_u) < 1e-3


def test_exhausted_grid_pads_at_zero_probability():
    """Fewer modes than asked for keeps the shape and says so in the probability."""
    logits = torch.full((1, 9), -20.0)
    logits[0, 4] = 5.0
    offsets = torch.zeros(1, 9, 2)
    uv, prob = decode_uv_topk(
        logits, offsets, 3, 3, PROC, PROC, k=4, min_separation_cells=2
    )
    assert uv.shape == (1, 4, 2) and prob.shape == (1, 4)
    assert prob[0, 0] > 0.0
    assert float(prob[0, 1:].max()) == 0.0
    # The padded slots repeat the last real candidate rather than an invented one.
    assert torch.allclose(uv[0, 1], uv[0, 0]) and torch.allclose(uv[0, 3], uv[0, 0])


def test_batch_rows_are_independent():
    """A unimodal frame beside a bimodal one is decoded on its own evidence."""
    bimodal, off_bi = _bimodal()
    unimodal = torch.full((1, N), -20.0)
    _blob(unimodal[0], 5, 5, 9.0)
    logits = torch.cat([bimodal, unimodal])
    offsets = torch.cat([off_bi, torch.zeros(1, N, 2)])
    uv, prob = decode_uv_topk(logits, offsets, GH, GW, PROC, PROC, k=2)
    assert torch.allclose(uv[0], decode_uv_topk(bimodal, off_bi, GH, GW, PROC, PROC, k=2)[0][0])
    # The unimodal row's second candidate is a far-away scrap of background,
    # which the probability is what exposes.
    assert prob[1, 1] < 0.01 * prob[1, 0]
