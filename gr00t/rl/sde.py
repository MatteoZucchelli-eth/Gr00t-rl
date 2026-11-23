"""Pure-tensor math for OGPO on flow-matching policies.

Conventions follow GR00T's action head (and Patil et al. 2026, App. E.2): flow time
``tau`` runs from 0 (pure noise) to 1 (data), the interpolant is
``x_tau = tau * a + (1 - tau) * z`` and the network predicts the velocity ``v = a - z``.
Deterministic inference is Euler integration ``x <- x + v * dt`` over K steps at
``tau_k = k / K``.

SDE sampling (marginal-preserving, Albergo et al. 2023) adds a score-correction drift
and Gaussian noise. With the tapered schedule ``sigma_tau = sigma_init * sqrt(1 - tau)``
the score correction simplifies to ``c = sigma_init**2 * (tau * v - x) / 2`` (OGPO
Eq. A.3 / E.16), so each denoising step is the Gaussian

    x_{k+1} ~ N(x_k + (v + c) * dt,  (sigma_tau * sqrt(dt))**2 I).

The ``constant`` schedule (the setting used for the OGPO paper's runs) adds isotropic
noise with a fixed per-step std and no drift correction (ReinFlow-style).
"""

from __future__ import annotations

import math

import torch


NOISE_SCHEDULES = ("tapered", "constant")


def sde_step_params(
    x: torch.Tensor,
    velocity: torch.Tensor,
    tau: float,
    dt: float,
    sigma: float,
    schedule: str = "tapered",
) -> tuple[torch.Tensor, float]:
    """Mean and (scalar) std of one SDE denoising step from ``x`` at flow time ``tau``.

    Args:
        x: current noisy actions [..., H, D].
        velocity: predicted flow velocity at (x, tau), same shape as ``x``.
        tau: flow time of the current step, in [0, 1).
        dt: step size (1 / K).
        sigma: ``sigma_init`` for the tapered schedule, the per-step std for ``constant``.
        schedule: ``"tapered"`` or ``"constant"``.

    Returns:
        ``(mean, std)``; ``std`` is a Python float shared by all elements.
    """
    if schedule == "tapered":
        correction = 0.5 * sigma**2 * (tau * velocity - x)
        mean = x + (velocity + correction) * dt
        std = sigma * math.sqrt(max(1.0 - tau, 0.0)) * math.sqrt(dt)
    elif schedule == "constant":
        mean = x + velocity * dt
        std = sigma
    else:
        raise ValueError(f"Unknown noise schedule {schedule!r}; expected one of {NOISE_SCHEDULES}")
    return mean, std


def gaussian_log_prob(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: float,
    mask: torch.Tensor | None = None,
    reduce: str = "sum",
) -> torch.Tensor:
    """Log-density of an isotropic Gaussian, reduced over the trailing (H, D) dims.

    Args:
        x, mean: [B, H, D].
        std: scalar std (> 0).
        mask: optional [B, H, D] or broadcastable 0/1 mask of the dims that are sampled
            stochastically; masked-out dims contribute nothing.
        reduce: ``"sum"`` (exact joint log-density) or ``"mean"`` (per-dimension average,
            a common stabiliser for high-dimensional chunks).

    Returns:
        [B] log-probabilities, computed in float32.
    """
    if std <= 0:
        raise ValueError("gaussian_log_prob needs std > 0")
    x = x.float()
    mean = mean.float()
    elem = -0.5 * ((x - mean) / std) ** 2 - math.log(std) - 0.5 * math.log(2 * math.pi)
    if mask is None:
        mask = torch.ones_like(elem)
    else:
        mask = mask.to(elem.dtype).expand_as(elem)
    total = (elem * mask).flatten(1).sum(-1)
    if reduce == "sum":
        return total
    if reduce == "mean":
        return total / mask.flatten(1).sum(-1).clamp_min(1.0)
    raise ValueError(f"Unknown reduce {reduce!r}")


def group_advantages(q: torch.Tensor) -> torch.Tensor:
    """Group-relative advantages ``A[..., j] = Q[..., j] - mean_j Q[..., j]`` over the last dim."""
    return q - q.mean(dim=-1, keepdim=True)


def conservative_advantages(q_heads: torch.Tensor) -> torch.Tensor:
    """OGPO+CA conservative advantages (Patil et al. 2026, Eq. 4.5).

    Args:
        q_heads: [M, B, G] Q-values of G sampled actions per state, for each of M
            ensemble members.

    Returns:
        [B, G]: per member compute the group-relative advantage; if all members agree on
        a positive sign take their minimum, if all agree on a negative sign take their
        maximum (smallest magnitude), else 0.
    """
    adv = group_advantages(q_heads)  # [M, B, G]
    adv_min = adv.min(dim=0).values
    adv_max = adv.max(dim=0).values
    zeros = torch.zeros_like(adv_min)
    return torch.where(adv_min > 0, adv_min, torch.where(adv_max < 0, adv_max, zeros))


def ppo_clip_objective(
    log_ratio: torch.Tensor, advantages: torch.Tensor, clip_eps: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PPO clipped surrogate to *maximise*, averaged over all elements.

    Args:
        log_ratio: log of the importance ratio (any shape).
        advantages: same shape, treated as constants.
        clip_eps: clip range epsilon.

    Returns:
        ``(objective, stats)`` where ``stats`` holds diagnostics (detached).
    """
    advantages = advantages.detach().float()
    log_ratio = log_ratio.float()
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    objective = torch.minimum(unclipped, clipped).mean()
    with torch.no_grad():
        stats = {
            "ratio_mean": ratio.mean(),
            "ratio_max": ratio.max(),
            "ratio_min": ratio.min(),
            "clip_frac": ((ratio - 1.0).abs() > clip_eps).float().mean(),
            "approx_kl": ((ratio - 1.0) - log_ratio).mean(),
        }
    return objective, stats


def aggregate_ensemble(
    q_heads: torch.Tensor, mode: str, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Reduce [M, ...] ensemble predictions to [...] (OGPO Eq. A.2).

    ``subsample``: min over 2 members drawn uniformly at random, independently per
    element; ``min``: min over all members; ``mean``: mean over members.
    """
    if mode == "mean":
        return q_heads.mean(dim=0)
    if mode == "min":
        return q_heads.min(dim=0).values
    if mode == "subsample":
        num_heads = q_heads.shape[0]
        if num_heads < 2:
            return q_heads[0]
        flat = q_heads.reshape(num_heads, -1)
        n = flat.shape[1]
        # Two distinct heads per element.
        idx1 = torch.randint(num_heads, (n,), device=q_heads.device, generator=generator)
        offset = torch.randint(1, num_heads, (n,), device=q_heads.device, generator=generator)
        idx2 = (idx1 + offset) % num_heads
        cols = torch.arange(n, device=q_heads.device)
        picked = torch.minimum(flat[idx1, cols], flat[idx2, cols])
        return picked.reshape(q_heads.shape[1:])
    raise ValueError(f"Unknown ensemble aggregation {mode!r}")
