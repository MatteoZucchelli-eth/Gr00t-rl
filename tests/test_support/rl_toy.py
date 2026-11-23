"""Small stand-ins for the GR00T model used by the RL unit tests (CPU only).

``ToyFlowPolicy`` implements the subset of ``gr00t.rl.flow_policy.FlowPolicy`` used by
``OGPOLearner`` and ``Collector`` with an MLP velocity field, so the RL logic can be
tested without the 3B checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

from gr00t.rl.sde import gaussian_log_prob, sde_step_params
import numpy as np
import torch
from torch import nn


@dataclass
class ToyFeatures:
    prefix: torch.Tensor  # [B, P]

    @property
    def batch_size(self) -> int:
        return int(self.prefix.shape[0])

    def index(self, idx) -> "ToyFeatures":
        return ToyFeatures(self.prefix[idx])

    def repeat_interleave(self, n: int) -> "ToyFeatures":
        return ToyFeatures(self.prefix.repeat_interleave(n, 0))


@dataclass
class ToySample:
    actions: torch.Tensor
    trajectory: torch.Tensor | None
    log_probs: torch.Tensor | None


class ToyFlowPolicy:
    """Flow policy over chunks [H, D] conditioned on a prefix vector."""

    def __init__(
        self,
        prefix_dim: int = 4,
        horizon: int = 4,
        action_dim: int = 2,
        num_steps: int = 4,
        sigma: float = 0.3,
        noise_schedule: str = "tapered",
        hidden: int = 64,
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        self.prefix_dim = prefix_dim
        self.state_dim = 3
        self.model_horizon = horizon
        self.valid_horizon = horizon
        self.max_action_dim = action_dim
        self.action_dim = action_dim
        self.num_steps = num_steps
        self.sigma = sigma
        self.noise_schedule = noise_schedule
        flat = horizon * action_dim
        self.net = nn.Sequential(
            nn.Linear(prefix_dim + flat + 1, hidden), nn.Tanh(), nn.Linear(hidden, flat)
        )
        self.mask = torch.ones(1, horizon, action_dim)
        # Collector-facing attributes (DROID-like layout of the fake env).
        self.camera_keys = ["exterior_image_1_left", "wrist_image_left"]
        self.video_delta_indices = [-2, 0]
        self.state_keys = ["eef_9d", "gripper_position", "joint_position"]
        self.state_dims = [9, 1, 7]
        self.action_keys = ["gripper_position", "joint_position"]
        self.language_key = "annotation.language.language_instruction"
        self.device = torch.device("cpu")

    # -- interface used by OGPOLearner --------------------------------------------------

    def trainable_parameters(self):
        return list(self.net.parameters())

    def named_trainable_parameters(self):
        return list(self.net.named_parameters())

    def _velocity(self, prefix: torch.Tensor, x: torch.Tensor, tau: float) -> torch.Tensor:
        B = x.shape[0]
        inp = torch.cat([prefix, x.reshape(B, -1), torch.full((B, 1), tau)], dim=-1)
        return self.net(inp).view_as(x)

    def sample(self, feats, mode="ode", return_trajectory=False, repeats=1, generator=None):
        prefix = feats.prefix.repeat_interleave(repeats, 0)
        B = prefix.shape[0]
        K = self.num_steps
        dt = 1.0 / K
        with torch.no_grad():
            x = torch.randn(B, self.model_horizon, self.max_action_dim, generator=generator)
            traj, lps = [x], []
            for k in range(K):
                tau = k / K
                v = self._velocity(prefix, x, tau)
                if mode == "ode":
                    x = x + v * dt
                else:
                    mean, std = sde_step_params(x, v, tau, dt, self.sigma, self.noise_schedule)
                    x_next = mean + std * torch.randn(x.shape, generator=generator)
                    lps.append(gaussian_log_prob(x_next, mean, std, self.mask))
                    x = x_next
                traj.append(x)
        return ToySample(
            x,
            torch.stack(traj) if (return_trajectory or mode == "sde") else None,
            torch.stack(lps) if lps else None,
        )

    def traj_log_prob(self, feats, trajectory, repeats=1):
        prefix = feats.prefix.repeat_interleave(repeats, 0)
        K = self.num_steps
        dt = 1.0 / K
        out = []
        for k in range(K):
            tau = k / K
            v = self._velocity(prefix, trajectory[k], tau)
            mean, std = sde_step_params(trajectory[k], v, tau, dt, self.sigma, self.noise_schedule)
            out.append(gaussian_log_prob(trajectory[k + 1], mean, std, self.mask))
        return torch.stack(out)

    def bc_loss(self, feats, actions):
        B = actions.shape[0]
        noise = torch.randn_like(actions)
        t = torch.rand(B, 1, 1)
        x = t * actions + (1 - t) * noise
        inp = torch.cat([feats.prefix, x.reshape(B, -1), t.reshape(B, 1)], dim=-1)
        pred = self.net(inp).view_as(actions)
        return ((pred - (actions - noise)) ** 2).mean()

    def valid_actions(self, actions):
        return actions[:, : self.valid_horizon, : self.action_dim]

    # -- interface used by Collector -------------------------------------------------

    def encode(self, observations):
        prefix = torch.stack(
            [
                torch.as_tensor(
                    np.asarray(o["state"]["joint_position"]).reshape(-1)[: self.prefix_dim],
                    dtype=torch.float32,
                )
                for o in observations
            ]
        )
        if prefix.shape[1] < self.prefix_dim:
            prefix = torch.nn.functional.pad(prefix, (0, self.prefix_dim - prefix.shape[1]))
        return ToyFeatures(prefix)

    def normalize_state(self, state):
        return np.concatenate([np.asarray(state[k]).reshape(-1) for k in self.state_keys])[
            : self.state_dim
        ].astype(np.float32)

    def decode_actions(self, actions, states):
        """Action dim 0 -> gripper command, dim 1 -> all 7 joint targets (toy mapping)."""
        a = actions.detach().cpu().numpy()
        out = []
        for i in range(a.shape[0]):
            out.append(
                {
                    "gripper_position": a[i, :, :1].astype(np.float32),
                    "joint_position": np.repeat(a[i, :, 1:2], 7, axis=1).astype(np.float32),
                }
            )
        return out

    def encode_action_chunk(self, state, chunk):
        return np.concatenate(
            [chunk["gripper_position"], chunk["joint_position"][:, :1]], axis=-1
        ).astype(np.float32)


class FixedQ(nn.Module):
    """Q(s, a) = -||a - target||^2 for every ensemble member (plus per-member offsets)."""

    def __init__(self, target: torch.Tensor, num_members: int = 4, member_noise: float = 0.0):
        super().__init__()
        self.register_buffer("target", target.reshape(-1))
        self.register_buffer("offsets", torch.randn(num_members) * member_noise)

    @property
    def num_members(self) -> int:
        return int(self.offsets.shape[0])

    def forward(self, prefix, state, action):
        a = action.reshape(action.shape[0], -1)
        q = -((a - self.target) ** 2).sum(-1)
        return q.unsqueeze(0) + self.offsets.unsqueeze(1)
