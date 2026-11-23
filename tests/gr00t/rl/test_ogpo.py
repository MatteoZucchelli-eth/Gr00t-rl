"""OGPO learner tests on a toy flow policy (CPU)."""

import copy

from gr00t.rl.config import TrainConfig
from gr00t.rl.ogpo import CriticBatch, OGPOLearner
from test_support.rl_toy import FixedQ, ToyFeatures, ToyFlowPolicy
import torch


H, D, P = 4, 2, 4


def make_learner(algo="ogpo", bootstrap="policy", **actor_overrides):
    cfg = TrainConfig(algo=algo)
    cfg.critic.horizon = H
    cfg.critic.hidden_dim = 32
    cfg.critic.depth = 1
    cfg.critic.num_qs = 4
    cfg.critic.num_vr = 3
    cfg.critic.bootstrap = bootstrap
    cfg.critic.lr = 3e-3
    cfg.critic.target_decay = 0.5
    cfg.actor.group_size = 8
    cfg.actor.micro_batch = 32
    cfg.actor.warmup_steps = 0
    cfg.actor.bc_coeff = 0.0
    for k, v in actor_overrides.items():
        setattr(cfg.actor, k, v)
    policy = ToyFlowPolicy(prefix_dim=P, horizon=H, action_dim=D, sigma=0.5)
    learner = OGPOLearner(policy, cfg, P, policy.state_dim, D, "cpu")
    return learner, policy


def feats(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return ToyFeatures(torch.randn(n, P, generator=g))


def test_first_update_has_unit_ratio():
    learner, _ = make_learner(clip_eps=0.01, lr=1e-3)
    learner.q_target.module = FixedQ(torch.full((H * D,), 0.5), num_members=4)
    f = feats()
    info = learner.update_policy(f, f.prefix, torch.zeros(16, 3))
    assert abs(info["actor/ratio_mean"].item() - 1.0) < 1e-5
    assert abs(info["actor/approx_kl"].item()) < 1e-6
    assert info["actor/adv_nonzero_frac"].item() > 0.5


def test_ogpo_moves_policy_towards_critic_optimum():
    torch.manual_seed(0)
    learner, policy = make_learner(clip_eps=0.2, lr=3e-3, ema_decay=0.8)
    target = torch.full((H * D,), 0.8)
    learner.q_target.module = FixedQ(target, num_members=4, member_noise=0.1)
    f = feats(32)

    def dist():
        a = policy.sample(f, mode="ode", generator=torch.Generator().manual_seed(1)).actions
        return ((a.reshape(32, -1) - target) ** 2).sum(-1).mean().item()

    before = dist()
    for _ in range(150):
        learner.update_policy(f, f.prefix, torch.zeros(32, 3))
    after = dist()
    assert after < 0.5 * before, (before, after)


def test_bc_loss_participates():
    learner, policy = make_learner(bc_coeff=1.0, lr=3e-3)
    learner.q_target.module = FixedQ(torch.zeros(H * D), num_members=4)
    f = feats(32)
    actions = torch.full((32, H, D), 0.3)
    l0 = policy.bc_loss(f, actions).item()
    for _ in range(100):
        info = learner.update_policy(f, f.prefix, torch.zeros(32, 3), f, actions)
    assert "actor/bc_loss" in info
    torch.manual_seed(0)
    assert policy.bc_loss(f, actions).item() < l0


def critic_batch(n=64, reward=-1.0, discount=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return CriticBatch(
        prefix=torch.randn(n, P, generator=g),
        state=torch.randn(n, 3, generator=g),
        action=torch.randn(n, H, D, generator=g),
        reward=torch.full((n,), reward),
        discount=torch.full((n,), discount),
        next_prefix=torch.randn(n, P, generator=g),
        next_state=torch.randn(n, 3, generator=g),
        mc_return=torch.randn(n, generator=g),
    )


def test_critic_regresses_terminal_reward_with_policy_bootstrap():
    learner, _ = make_learner()
    b = critic_batch(reward=-2.0, discount=0.0)
    nf = ToyFeatures(b.next_prefix)
    for _ in range(300):
        info = learner.update_critic(b, nf)
        learner.update_targets()
    assert abs(info["critic/q_mean"].item() + 2.0) < 0.1


def test_td_target_uses_bootstrap():
    learner, _ = make_learner()
    learner.q_target.module = FixedQ(torch.zeros(H * D), num_members=4)
    b = critic_batch(reward=-1.0, discount=0.5)
    y = learner.td_target(b, ToyFeatures(b.next_prefix))
    # bootstrap is <= 0 (negative squared norm), so y <= reward
    assert (y <= -1.0 + 1e-6).all() and (y < -1.0).any()


def test_sarsa_bootstrap_and_bon_algo():
    learner, _ = make_learner(algo="bon", bootstrap="sarsa")
    assert learner.ema is None and not learner.train_policy
    b = critic_batch(reward=-1.0, discount=0.9)
    info = learner.update_critic(b)  # no next-state features needed
    assert "critic/v_loss" in info
    assert learner.update_policy(feats(), torch.zeros(16, P), torch.zeros(16, 3)) == {}


def test_conservative_advantage_zero_when_members_disagree():
    learner, _ = make_learner(lr=1e-2)

    class Disagree(torch.nn.Module):
        def forward(self, prefix, state, action):
            q = action.reshape(action.shape[0], -1).sum(-1)
            return torch.stack([q, -q])  # opposite rankings

    learner.q_target.module = Disagree()
    policy = learner.policy
    before = copy.deepcopy(policy.net.state_dict())
    info = learner.update_policy(feats(), torch.zeros(16, P), torch.zeros(16, 3))
    assert info["actor/adv_nonzero_frac"].item() == 0.0
    for k, v in policy.net.state_dict().items():
        assert torch.allclose(v, before[k])  # zero advantages -> zero gradient (no BC)


def test_state_dict_roundtrip():
    learner, policy = make_learner(lr=1e-2)
    f = feats()
    learner.update_critic(critic_batch(), ToyFeatures(torch.randn(64, P)))
    learner.update_policy(f, f.prefix, torch.zeros(16, 3))
    sd = learner.state_dict()
    other, policy2 = make_learner()
    other.load_state_dict(sd)
    for (n1, p1), (_, p2) in zip(
        policy.named_trainable_parameters(), policy2.named_trainable_parameters()
    ):
        assert torch.allclose(p1, p2), n1
    for s1, s2 in zip(learner.ema.shadow, other.ema.shadow):
        assert torch.allclose(s1, s2)
    assert other.policy_steps == 1 and other.critic_steps == 1
