"""BroNet Q-ensemble used by OGPO / Best-of-N (port of Korniak et al. 2026's critic).

Input: ``[prefix_embedding, state, flattened action chunk]`` where ``prefix_embedding``
is the mean-pooled VLM backbone output (the critic never sees pixels). Each ensemble
member is a BroNet (Nauman et al. 2024): Linear -> LN -> ReLU, then ``depth`` residual
blocks (Linear -> LN -> ReLU -> Linear -> LN, + skip), then a scalar head. Members are
evaluated in one batched matmul per layer.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
import torch.nn.functional as F


class EnsembleLinear(nn.Module):
    """``num_members`` independent Linear layers applied to [M, B, in] -> [M, B, out]."""

    def __init__(self, num_members: int, in_features: int, out_features: int, gain: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_members, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(num_members, 1, out_features))
        for m in range(num_members):
            # Orthogonal init as in the reference implementation.
            w = torch.empty(out_features, in_features)
            nn.init.orthogonal_(w, gain=gain)
            self.weight.data[m].copy_(w.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.baddbmm(self.bias, x, self.weight)


class EnsembleLayerNorm(nn.Module):
    """Per-member LayerNorm over the last dim of [M, B, D]."""

    def __init__(self, num_members: int, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_members, 1, dim))
        self.bias = nn.Parameter(torch.zeros(num_members, 1, dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(x, x.shape[-1:], eps=self.eps)
        return x * self.weight + self.bias


class BroNetEnsemble(nn.Module):
    """Ensemble of BroNets mapping [B, in] (shared) or [M, B, in] to [M, B] scalars."""

    def __init__(self, input_dim: int, hidden_dim: int, depth: int, num_members: int):
        super().__init__()
        gain = math.sqrt(2.0)
        self.num_members = num_members
        self.proj = EnsembleLinear(num_members, input_dim, hidden_dim, gain)
        self.proj_norm = EnsembleLayerNorm(num_members, hidden_dim)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "linear1": EnsembleLinear(num_members, hidden_dim, hidden_dim, gain),
                        "norm1": EnsembleLayerNorm(num_members, hidden_dim),
                        "linear2": EnsembleLinear(num_members, hidden_dim, hidden_dim, gain),
                        "norm2": EnsembleLayerNorm(num_members, hidden_dim),
                    }
                )
            )
        self.head = EnsembleLinear(num_members, hidden_dim, 1, gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(self.num_members, -1, -1)
        x = F.relu(self.proj_norm(self.proj(x)))
        for block in self.blocks:
            res = F.relu(block["norm1"](block["linear1"](x)))
            res = block["norm2"](block["linear2"](res))
            x = x + res
        return self.head(x).squeeze(-1)


class QEnsemble(nn.Module):
    """Q(s, a) ensemble over critic observations ``(prefix, state)`` and flat action chunks."""

    def __init__(
        self,
        prefix_dim: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 1024,
        depth: int = 2,
        num_members: int = 10,
    ):
        super().__init__()
        self.prefix_dim = prefix_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = BroNetEnsemble(
            prefix_dim + state_dim + action_dim, hidden_dim, depth, num_members
        )

    @property
    def num_members(self) -> int:
        return self.net.num_members

    def forward(
        self, prefix: torch.Tensor, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Returns [M, B] Q-values. ``action`` may be [B, H, D] or already flat [B, H*D]."""
        action = action.reshape(action.shape[0], -1)
        x = torch.cat([prefix.float(), state.float(), action.float()], dim=-1)
        return self.net(x)


class VEnsemble(nn.Module):
    """V(s) ensemble, used by the SARSA-style Best-of-N critic (arm E)."""

    def __init__(
        self,
        prefix_dim: int,
        state_dim: int,
        hidden_dim: int = 1024,
        depth: int = 2,
        num_members: int = 2,
    ):
        super().__init__()
        self.net = BroNetEnsemble(prefix_dim + state_dim, hidden_dim, depth, num_members)

    def forward(self, prefix: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([prefix.float(), state.float()], dim=-1))


class TargetNetwork:
    """Polyak-averaged copy of a module: ``target <- decay * target + (1 - decay) * online``."""

    def __init__(self, online: nn.Module, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(online)
        self.module.requires_grad_(False)
        self.module.eval()

    @torch.no_grad()
    def update(self, online: nn.Module) -> None:
        for p_t, p in zip(self.module.parameters(), online.parameters()):
            p_t.lerp_(p.detach(), 1.0 - self.decay)
        for b_t, b in zip(self.module.buffers(), online.buffers()):
            b_t.copy_(b)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state):
        self.module.load_state_dict(state)


def td_loss(q_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean-squared TD error averaged over ensemble members; q_pred [M, B], target [B]."""
    return F.mse_loss(q_pred, target.detach().unsqueeze(0).expand_as(q_pred))
