"""OGPO learner (Patil et al. 2026) as adapted by Korniak et al. 2026 for VLAs.

Two levels:

1. **Critic (environment MDP, off-policy).** A BroNet Q-ensemble on
   ``(prefix embedding, normalized state, flattened H-step action chunk)`` regresses
   H-step TD targets ``r_{t:t+H} + gamma^H * Q_targ(s', a')``, with the bootstrap averaged
   over ``num_vr`` actions sampled from the slow policy at ``s'`` (Q-variance reduction)
   and the ensemble reduced by ``aggregation``. Optionally a second update per step on
   a batch from successful episodes only.
2. **Actor (denoising MDP, PPO-style).** For replay states, sample G SDE denoising
   trajectories from the slow (EMA) policy, score the final chunks with the target
   critic, form group-relative (conservative) advantages, and apply a PPO-clipped update
   to the product of per-denoising-step likelihood ratios ``pi_theta / pi_slow``.
   Plus a flow-matching BC loss on transitions from successful episodes (OGPO+).

``algo="bon"`` (arm E) trains the critic only (SARSA-style by default) and leaves the
policy frozen; action selection then happens by Best-of-N in the collector.

The learner only sees tensors and a policy object with the ``FlowPolicy`` interface
(``sample``, ``traj_log_prob``, ``bc_loss``, ``valid_actions``,
``trainable_parameters``), so it can be unit-tested with a toy flow policy.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from gr00t.rl.config import TrainConfig
from gr00t.rl.critic import QEnsemble, TargetNetwork, VEnsemble, td_loss
from gr00t.rl.flow_policy import EMAPolicyParams
from gr00t.rl.sde import (
    aggregate_ensemble,
    conservative_advantages,
    group_advantages,
    ppo_clip_objective,
)


@dataclass
class CriticBatch:
    prefix: torch.Tensor  # [B, P]
    state: torch.Tensor  # [B, S]
    action: torch.Tensor  # [B, H, Da] normalized chunk (critic horizon)
    reward: torch.Tensor  # [B]
    discount: torch.Tensor  # [B]
    next_prefix: torch.Tensor  # [B, P]
    next_state: torch.Tensor  # [B, S]
    mc_return: torch.Tensor | None = None  # [B] diagnostics

    @classmethod
    def from_numpy(cls, arrays: dict, device) -> "CriticBatch":
        def t(k):
            return torch.as_tensor(arrays[k], device=device, dtype=torch.float32)

        return cls(
            prefix=t("prefix"),
            state=t("state"),
            action=t("action"),
            reward=t("reward"),
            discount=t("discount"),
            next_prefix=t("next_prefix"),
            next_state=t("next_state"),
            mc_return=t("mc_return") if "mc_return" in arrays else None,
        )


def _pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.float() - a.float().mean()
    b = b.float() - b.float().mean()
    return (a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-8)


class OGPOLearner:
    def __init__(
        self,
        policy,
        cfg: TrainConfig,
        prefix_dim: int,
        state_dim: int,
        action_dim: int,
        device: torch.device | str,
    ):
        self.policy = policy
        self.cfg = cfg
        self.device = torch.device(device)
        c, a = cfg.critic, cfg.actor
        self.H = c.horizon
        self.action_dim = action_dim
        self.train_policy = cfg.algo == "ogpo"

        self.q = QEnsemble(
            prefix_dim, state_dim, self.H * action_dim, c.hidden_dim, c.depth, c.num_qs
        ).to(self.device)
        self.q_target = TargetNetwork(self.q, c.target_decay)
        self.q_opt = torch.optim.AdamW(self.q.parameters(), lr=c.lr, weight_decay=c.weight_decay)
        self.use_v = c.bootstrap == "sarsa"
        if self.use_v:
            self.v = VEnsemble(prefix_dim, state_dim, c.hidden_dim, c.depth, c.num_vs).to(
                self.device
            )
            self.v_target = TargetNetwork(self.v, c.target_decay)
            self.v_opt = torch.optim.AdamW(
                self.v.parameters(), lr=c.lr, weight_decay=c.weight_decay
            )

        self.ema: EMAPolicyParams | None = None
        self.base: EMAPolicyParams | None = None
        if self.train_policy:
            params = policy.trainable_parameters()
            if not params:
                raise ValueError("OGPO needs trainable policy parameters")
            self.pi_opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
            self.ema = EMAPolicyParams(params, a.ema_decay)
            # Frozen snapshot of the starting policy, for the KL-to-base diagnostic.
            self.base = EMAPolicyParams(params, 1.0)
        self.critic_steps = 0
        self.policy_steps = 0

    # ------------------------------------------------------------------ helpers

    def slow_policy(self):
        """Context in which the policy evaluates with the slow (EMA) parameters."""
        return self.ema.swapped() if self.ema is not None else nullcontext()

    def critic_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Padded normalized model actions [B, H_model, D_max] -> critic input [B, H, Da]."""
        return self.policy.valid_actions(actions)[:, : self.H]

    @torch.no_grad()
    def score(
        self, prefix: torch.Tensor, state: torch.Tensor, actions: torch.Tensor, target: bool = True
    ) -> torch.Tensor:
        """Aggregated Q of candidate chunks ``actions`` (padded model space) -> [B]."""
        net = self.q_target if target else self.q
        q = net(prefix, state, self.critic_actions(actions))
        return aggregate_ensemble(q, self.cfg.critic.aggregation)

    # ------------------------------------------------------------------ critic

    @torch.no_grad()
    def td_target(self, batch: CriticBatch, next_feats=None) -> torch.Tensor:
        c = self.cfg.critic
        if c.bootstrap == "policy":
            if next_feats is None:
                raise ValueError("bootstrap='policy' needs features of the next states")
            n = c.num_vr
            with self.slow_policy():
                out = self.policy.sample(next_feats, mode="sde", repeats=n)
            q = self.q_target(
                batch.next_prefix.repeat_interleave(n, 0),
                batch.next_state.repeat_interleave(n, 0),
                self.critic_actions(out.actions),
            )
            bootstrap = aggregate_ensemble(q, c.aggregation).view(-1, n).mean(1)
        else:
            bootstrap = self.v_target(batch.next_prefix, batch.next_state).min(0).values
        return batch.reward + batch.discount * bootstrap

    def update_critic(self, batch: CriticBatch, next_feats=None, tag: str = "critic") -> dict:
        c = self.cfg.critic
        target = self.td_target(batch, next_feats)
        q_pred = self.q(batch.prefix, batch.state, batch.action)
        loss = td_loss(q_pred, target)
        self.q_opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q.parameters(), c.grad_clip)
        self.q_opt.step()
        info = {
            f"{tag}/td_loss": loss.detach(),
            f"{tag}/q_mean": q_pred.detach().mean(),
            f"{tag}/q_ens_std": q_pred.detach().std(0).mean(),
            f"{tag}/target_mean": target.mean(),
            f"{tag}/grad_norm": grad_norm.detach(),
        }
        if batch.mc_return is not None:
            info[f"{tag}/q_mc_corr"] = _pearson(q_pred.detach().mean(0), batch.mc_return)
        if self.use_v:
            with torch.no_grad():
                v_tgt = self.q_target(batch.prefix, batch.state, batch.action).min(0).values
            v_pred = self.v(batch.prefix, batch.state)
            v_loss = td_loss(v_pred, v_tgt)
            self.v_opt.zero_grad(set_to_none=True)
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.v.parameters(), c.grad_clip)
            self.v_opt.step()
            info[f"{tag}/v_loss"] = v_loss.detach()
            info[f"{tag}/v_mean"] = v_pred.detach().mean()
        self.critic_steps += 1
        return info

    def update_targets(self) -> None:
        self.q_target.update(self.q)
        if self.use_v:
            self.v_target.update(self.v)

    # ------------------------------------------------------------------ actor

    def _lr_scale(self) -> float:
        warm = self.cfg.actor.warmup_steps
        return min(1.0, (self.policy_steps + 1) / warm) if warm > 0 else 1.0

    def update_policy(
        self,
        feats,
        prefix: torch.Tensor,
        state: torch.Tensor,
        bc_feats=None,
        bc_actions: torch.Tensor | None = None,
    ) -> dict:
        """One OGPO(+CA) policy step on replay states ``feats`` (+ success-BC batch)."""
        if not self.train_policy:
            return {}
        a = self.cfg.actor
        G = a.group_size
        B = feats.batch_size
        micro_states = max(1, a.micro_batch // G)
        params = self.policy.trainable_parameters()
        for g in self.pi_opt.param_groups:
            g["lr"] = a.lr * self._lr_scale()
        self.pi_opt.zero_grad(set_to_none=True)

        stats_acc: dict[str, list[torch.Tensor]] = {}

        def acc(d):
            for k, v in d.items():
                stats_acc.setdefault(k, []).append(v.detach().float())

        for start in range(0, B, micro_states):
            sl = slice(start, min(start + micro_states, B))
            f = feats.index(sl)
            m = f.batch_size
            with torch.no_grad(), self.slow_policy():
                out = self.policy.sample(f, mode="sde", repeats=G, return_trajectory=True)
                # Recompute the reference log-probs in exactly the layout used for the
                # current policy below, so the ratio is 1 when the parameters coincide.
                old_lp = self.policy.traj_log_prob(f, out.trajectory, repeats=G).sum(0)
            with torch.no_grad():
                q_heads = self.q_target(
                    prefix[sl].repeat_interleave(G, 0),
                    state[sl].repeat_interleave(G, 0),
                    self.critic_actions(out.actions),
                ).view(-1, m, G)
                if a.advantage == "conservative":
                    adv = conservative_advantages(q_heads)
                else:
                    adv = group_advantages(
                        aggregate_ensemble(q_heads, self.cfg.critic.aggregation).view(m, G)
                    )
            new_lp = self.policy.traj_log_prob(f, out.trajectory, repeats=G).sum(0)
            objective, st = ppo_clip_objective(new_lp - old_lp, adv.reshape(-1), a.clip_eps)
            (-objective * (m / B)).backward()
            acc(st)
            acc(
                {
                    "adv_abs_mean": adv.abs().mean(),
                    "adv_nonzero_frac": (adv != 0).float().mean(),
                    "q_group_std": q_heads.mean(0).std(-1).mean(),
                    "ppo_objective": objective,
                }
            )

        if a.bc_coeff > 0 and bc_feats is not None and bc_actions is not None:
            Bb = bc_feats.batch_size
            for start in range(0, Bb, a.micro_batch):
                sl = slice(start, min(start + a.micro_batch, Bb))
                fb = bc_feats.index(sl)
                bc = self.policy.bc_loss(fb, bc_actions[sl])
                (a.bc_coeff * bc * (fb.batch_size / Bb)).backward()
                acc({"bc_loss": bc})

        grad_norm = torch.nn.utils.clip_grad_norm_(params, a.grad_clip)
        self.pi_opt.step()
        self.ema.update()
        self.policy_steps += 1
        info = {f"actor/{k}": torch.stack(v).mean() for k, v in stats_acc.items()}
        info["actor/grad_norm"] = grad_norm.detach()
        info["actor/lr"] = torch.tensor(a.lr * self._lr_scale())
        return info

    @torch.no_grad()
    def kl_to_base(self, feats, num_samples: int = 4) -> dict:
        """Path-space KL(pi_theta || pi_base) on SDE trajectories (upper-bounds the
        marginal KL of the action distributions); RL's Razor diagnostic."""
        if self.base is None:
            return {}
        out = self.policy.sample(feats, mode="sde", repeats=num_samples, return_trajectory=True)
        lp = self.policy.traj_log_prob(feats, out.trajectory, repeats=num_samples).sum(0)
        with self.base.swapped():
            lp_base = self.policy.traj_log_prob(feats, out.trajectory, repeats=num_samples).sum(0)
        kl = (lp - lp_base).mean()
        return {"actor/kl_to_base_path": kl}

    # ------------------------------------------------------------------ state

    def state_dict(self) -> dict:
        sd = {
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "q_opt": self.q_opt.state_dict(),
            "critic_steps": self.critic_steps,
            "policy_steps": self.policy_steps,
        }
        if self.use_v:
            sd.update(
                v=self.v.state_dict(),
                v_target=self.v_target.state_dict(),
                v_opt=self.v_opt.state_dict(),
            )
        if self.train_policy:
            sd.update(
                pi_opt=self.pi_opt.state_dict(),
                ema=self.ema.state_dict(),
                base=self.base.state_dict(),
                policy_params={
                    n: p.detach().cpu() for n, p in self.policy.named_trainable_parameters()
                },
            )
        return sd

    def load_state_dict(self, sd: dict) -> None:
        self.q.load_state_dict(sd["q"])
        self.q_target.load_state_dict(sd["q_target"])
        self.q_opt.load_state_dict(sd["q_opt"])
        self.critic_steps = sd["critic_steps"]
        self.policy_steps = sd["policy_steps"]
        if self.use_v:
            self.v.load_state_dict(sd["v"])
            self.v_target.load_state_dict(sd["v_target"])
            self.v_opt.load_state_dict(sd["v_opt"])
        if self.train_policy:
            named = dict(self.policy.named_trainable_parameters())
            with torch.no_grad():
                for n, t in sd["policy_params"].items():
                    named[n].copy_(t.to(named[n].device, named[n].dtype))
            self.pi_opt.load_state_dict(sd["pi_opt"])
            self.ema.load_state_dict(sd["ema"])
            self.base.load_state_dict(sd["base"])
