"""Tests for the OGPO math in gr00t.rl.sde (CPU)."""

import math

from gr00t.rl.sde import (
    aggregate_ensemble,
    conservative_advantages,
    gaussian_log_prob,
    group_advantages,
    ppo_clip_objective,
    sde_step_params,
)
import pytest
import torch


def exact_velocity(x: torch.Tensor, tau: float, m: float, s: float) -> torch.Tensor:
    """E[a - z | x_tau = x] for a ~ N(m, s^2), z ~ N(0, 1), x_tau = tau a + (1 - tau) z."""
    var_x = tau**2 * s**2 + (1 - tau) ** 2
    cov = tau * s**2 - (1 - tau)
    return m + cov / var_x * (x - tau * m)


def integrate(
    schedule: str, sigma: float, steps: int, n: int, m: float, s: float, seed: int = 0
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, generator=g, dtype=torch.float64)
    dt = 1.0 / steps
    for k in range(steps):
        tau = k / steps
        v = exact_velocity(x, tau, m, s)
        if sigma == 0:
            x = x + v * dt
        else:
            mean, std = sde_step_params(x, v, tau, dt, sigma, schedule)
            x = mean + std * torch.randn(n, generator=g, dtype=torch.float64)
    return x


def test_ode_recovers_data_distribution():
    x = integrate("tapered", 0.0, steps=200, n=200_000, m=0.7, s=0.3)
    assert abs(x.mean().item() - 0.7) < 0.01
    assert abs(x.std().item() - 0.3) < 0.01


def test_tapered_sde_preserves_marginals():
    # The score correction keeps the SDE's marginals equal to the ODE's (Albergo et al.).
    x = integrate("tapered", 0.8, steps=400, n=200_000, m=0.7, s=0.3)
    assert abs(x.mean().item() - 0.7) < 0.01
    assert abs(x.std().item() - 0.3) < 0.015


def test_uncorrected_noise_inflates_variance():
    # Sanity check that the test above is sensitive: plain added noise biases the spread.
    x = integrate("constant", 0.8 * math.sqrt(1 / 400), steps=400, n=200_000, m=0.7, s=0.3)
    corrected = integrate("tapered", 0.8, steps=400, n=200_000, m=0.7, s=0.3)
    assert abs(x.std().item() - 0.3) > abs(corrected.std().item() - 0.3)


def test_tapered_std_vanishes_at_data_end():
    x = torch.zeros(2, 3, 4)
    _, std0 = sde_step_params(x, x, 0.0, 0.25, 0.5, "tapered")
    _, std_last = sde_step_params(x, x, 0.75, 0.25, 0.5, "tapered")
    assert std0 == pytest.approx(0.5 * math.sqrt(0.25))
    assert std_last == pytest.approx(0.5 * math.sqrt(0.25) * math.sqrt(0.25))


def test_gaussian_log_prob_matches_torch_normal():
    torch.manual_seed(0)
    x, mean = torch.randn(5, 3, 4), torch.randn(5, 3, 4)
    std = 0.37
    ref = torch.distributions.Normal(mean, std).log_prob(x).flatten(1).sum(-1)
    assert torch.allclose(gaussian_log_prob(x, mean, std), ref, atol=1e-5)


def test_gaussian_log_prob_mask_and_mean():
    torch.manual_seed(0)
    x, mean = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    mask = torch.zeros(1, 3, 4)
    mask[:, :2, :3] = 1
    ref = torch.distributions.Normal(mean[:, :2, :3], 0.5).log_prob(x[:, :2, :3]).flatten(1)
    assert torch.allclose(gaussian_log_prob(x, mean, 0.5, mask), ref.sum(-1), atol=1e-5)
    assert torch.allclose(
        gaussian_log_prob(x, mean, 0.5, mask, reduce="mean"), ref.mean(-1), atol=1e-5
    )


def test_group_advantages_zero_mean():
    q = torch.randn(3, 5, 8)
    adv = group_advantages(q)
    assert torch.allclose(adv.mean(-1), torch.zeros(3, 5), atol=1e-6)


def test_conservative_advantages_sign_agreement():
    # Two members, one state, three actions.
    q = torch.tensor(
        [
            [[3.0, 0.0, -3.0]],  # member 0: adv [3, 0, -3]
            [[1.0, 2.0, -3.0]],  # member 1: adv [1, 2, -3]
        ]
    )
    adv = conservative_advantages(q)
    # action 0: both positive -> min = 1; action 1: disagree (0 vs 2) -> 0; action 2: both -3 -> -3
    assert torch.allclose(adv, torch.tensor([[1.0, 0.0, -3.0]]))


def test_conservative_advantages_shift_invariant():
    torch.manual_seed(0)
    q = torch.randn(4, 2, 6)
    shifted = q + torch.randn(4, 1, 1) * 100  # per-member constant offsets don't matter
    assert torch.allclose(conservative_advantages(q), conservative_advantages(shifted), atol=1e-4)


def test_aggregate_modes():
    torch.manual_seed(0)
    q = torch.randn(10, 1000)
    assert torch.allclose(aggregate_ensemble(q, "mean"), q.mean(0))
    assert torch.allclose(aggregate_ensemble(q, "min"), q.min(0).values)
    sub = aggregate_ensemble(q, "subsample")
    assert sub.shape == (1000,)
    assert (sub >= q.min(0).values - 1e-6).all() and (sub <= q.max(0).values + 1e-6).all()
    # The min over a random pair is strictly below the max almost surely.
    assert (sub < q.max(0).values).float().mean() > 0.95


def test_ppo_objective_at_unit_ratio_and_clipping():
    adv = torch.tensor([1.0, -2.0, 0.5])
    obj, stats = ppo_clip_objective(torch.zeros(3), adv, 0.1)
    assert obj.item() == pytest.approx(adv.mean().item())
    assert stats["clip_frac"].item() == 0.0
    # Large positive log-ratio with positive advantage is clipped at 1 + eps.
    obj, _ = ppo_clip_objective(torch.tensor([2.0]), torch.tensor([1.0]), 0.1)
    assert obj.item() == pytest.approx(1.1)
    # The gradient pushes the log-ratio along the advantage sign.
    lr = torch.zeros(2, requires_grad=True)
    obj, _ = ppo_clip_objective(lr, torch.tensor([1.0, -1.0]), 0.2)
    obj.backward()
    assert lr.grad[0] > 0 and lr.grad[1] < 0
