"""Batch-online OGPO / Best-of-N training of GR00T N1.7 in remote simulators.

Protocol (Korniak et al. 2026): ``rounds`` x [collect ``episodes_per_task`` episodes per
training task with the frozen (slow) policy -> ``steps_per_round`` critic/actor updates
on the whole replay buffer] -> final evaluation. Checkpoints are written after every
round, so a time-limited SLURM job can be requeued and resumes where it stopped.

Example (MolmoSpaces, 1 GPU)::

    python -m gr00t.rl.train \
        --model-path /path/to/warm_started_checkpoint \
        --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
        --output-dir outputs/ogpo_c \
        --env.backend molmospaces \
        --env.python gr00t/eval/sim/MolmoSpaces/molmospaces_uv/.venv/bin/python \
        --env.num-envs 16 \
        --collect.tasks pick/267 pick/366 --collect.eval-tasks pick/267 pick/366

Outputs in ``output_dir``: ``metrics.jsonl``, ``rounds/round_XXX/`` (policy checkpoint,
learner state, replay buffer), and ``final_policy/`` (input to the adaptation sweep).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import dataclasses
from functools import partial
import json
import logging
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import tyro

from gr00t.rl.config import TrainConfig, expand_task_specs
from gr00t.rl.envs.client import RemoteVectorEnv
from gr00t.rl.flow_policy import FlowPolicy, prepare_inputs
from gr00t.rl.ogpo import CriticBatch, OGPOLearner
from gr00t.rl.replay import ReplayBuffer
from gr00t.rl.rollout import Collector, RolloutStats, cap_num_envs, fill_missing_prefixes


logger = logging.getLogger("gr00t.rl.train")


# --------------------------------------------------------------------------------------
# Training batches (built in DataLoader workers; the buffer is static while training)
# --------------------------------------------------------------------------------------


class StepBatches(torch.utils.data.Dataset):
    """Item ``i`` = everything training step ``i`` needs, prepared on CPU.

    For each sampled batch (``u``: uniform, ``s``: successful episodes) it returns the
    critic arrays plus processor-ready inputs for the observations whose backbone
    features the step needs (current states for the actor / BC, next states for the
    policy-bootstrapped TD target), in chunks of ``encode_chunk``.
    """

    def __init__(
        self,
        buffer: ReplayBuffer,
        prep,
        plan: list[dict],
        needs: dict[str, tuple[str, ...]],
        encode_chunk: int,
        language_key: str,
    ):
        self.buffer = buffer
        self.prep = prep
        self.plan = plan
        self.needs = needs
        self.encode_chunk = encode_chunk
        self.language_key = language_key

    def __len__(self) -> int:
        return len(self.plan)

    def _inputs(self, obs_idx: np.ndarray) -> list[dict]:
        chunks = []
        for i in range(0, len(obs_idx), self.encode_chunk):
            obs = self.buffer.policy_observations(
                obs_idx[i : i + self.encode_chunk], self.language_key
            )
            chunks.append(self.prep(obs))
        return chunks

    def __getitem__(self, i: int) -> dict:
        out = {}
        for name, idx in self.plan[i].items():
            cb = self.buffer.critic_batch(idx)
            item = {"critic": cb}
            for what in self.needs.get(name, ()):
                key, limit = what.split(":") if ":" in what else (what, None)
                obs_idx = cb["obs_idx"] if key == "obs" else cb["next_obs_idx"]
                if limit is not None:
                    obs_idx = obs_idx[: int(limit)]
                item[key] = self._inputs(obs_idx)
            out[name] = item
        return out


def _identity_collate(x):
    return x


# --------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------


class Trainer:
    def __init__(self, cfg: TrainConfig):
        cfg.collect.tasks = expand_task_specs(cfg.collect.tasks)
        cfg.collect.eval_tasks = expand_task_specs(cfg.collect.eval_tasks)
        self.cfg = cfg
        self.out = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(cfg.device)
        _seed_everything(cfg.seed)

        a = cfg.actor
        self.policy = FlowPolicy.from_pretrained(
            cfg.model_path,
            cfg.embodiment_tag,
            device=self.device,
            sigma=a.sigma,
            noise_schedule=a.noise_schedule,
            logprob_reduce=a.logprob_reduce,
            logprob_horizon=a.logprob_horizon,
            train_vlln=a.train_vlln,
            train_projector=a.train_projector,
            train_diffusion_model=a.train_diffusion_model,
            trainable_regex=a.trainable_regex,
        )
        p = self.policy
        n_train = sum(t.numel() for t in p.trainable_parameters())
        logger.info(
            "Policy: horizon %d (valid %d), action dim %d, state dim %d, K=%d, %d dropout layers removed, %.1fM trainable params",
            p.model_horizon,
            p.valid_horizon,
            p.action_dim,
            p.state_dim,
            p.num_steps,
            p.num_dropout_removed,
            n_train / 1e6,
        )
        if cfg.critic.horizon > p.valid_horizon:
            raise ValueError("critic.horizon must not exceed the policy's action horizon")
        if cfg.collect.execute_steps > p.valid_horizon:
            raise ValueError("collect.execute_steps must not exceed the policy's action horizon")

        self.learner = OGPOLearner(p, cfg, p.prefix_dim, p.state_dim, p.action_dim, self.device)
        self.buffer = ReplayBuffer(
            capacity=cfg.buffer_capacity,
            camera_keys=p.camera_keys,
            state_keys=p.state_keys,
            state_dims=p.state_dims,
            state_dim=p.state_dim,
            prefix_dim=p.prefix_dim,
            action_shape=(p.valid_horizon, p.action_dim),
            critic_horizon=cfg.critic.horizon,
            video_delta_indices=p.video_delta_indices,
            frame_codec=cfg.frame_codec,
            frame_quality=cfg.frame_quality,
            seed=cfg.seed,
        )
        self.round = 0
        self.history: list[dict] = []
        self._maybe_resume()
        self._wandb = None
        if cfg.use_wandb:
            import wandb

            name = cfg.run_name or self.out.name
            self._wandb = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                group=cfg.wandb_group,
                name=name,
                config=dataclasses.asdict(cfg),
                resume="allow",
                id=name.replace("/", "_"),
                dir=str(self.out),
            )
        (self.out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
        self.env: RemoteVectorEnv | None = None
        self.collector: Collector | None = None

    # ------------------------------------------------------------------ io

    def log(self, metrics: dict, step: int | None = None) -> None:
        clean = {
            k: (v.item() if isinstance(v, torch.Tensor) else float(v)) for k, v in metrics.items()
        }
        clean["round"] = self.round
        if step is not None:
            clean["step"] = step
        with open(self.out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(clean) + "\n")
        if self._wandb is not None:
            self._wandb.log(clean, step=step)

    def _round_dir(self, r: int) -> Path:
        return self.out / "rounds" / f"round_{r:03d}"

    def _maybe_resume(self) -> None:
        if not self.cfg.resume:
            return
        rounds = (
            sorted((self.out / "rounds").glob("round_*/COMPLETE"))
            if (self.out / "rounds").exists()
            else []
        )
        if not rounds:
            return
        rd = rounds[-1].parent
        meta = json.loads((rd / "meta.json").read_text())
        state = torch.load(rd / "learner.pt", map_location=self.device, weights_only=False)
        self.learner.load_state_dict(state)
        if (rd / "buffer.pkl").exists():
            self.buffer.load(rd / "buffer.pkl")
        else:
            logger.warning("Resuming from %s without a replay buffer", rd)
        self.round = meta["round"] + 1
        self.history = meta.get("history", [])
        logger.info("Resumed after round %d from %s", meta["round"], rd)

    def save_round(self, r: int) -> None:
        rd = self._round_dir(r)
        if rd.exists():
            shutil.rmtree(rd)
        rd.mkdir(parents=True)
        torch.save(self.learner.state_dict(), rd / "learner.pt")
        if self.cfg.algo == "ogpo":
            self.policy.save_pretrained(rd / "policy")
        if self.cfg.save_buffer:
            self.buffer.save(rd / "buffer.pkl")
        (rd / "meta.json").write_text(json.dumps({"round": r, "history": self.history}, indent=2))
        (rd / "COMPLETE").touch()
        # Keep only the newest buffer copy (they are large).
        for old in sorted((self.out / "rounds").glob("round_*")):
            if old != rd and (old / "buffer.pkl").exists():
                (old / "buffer.pkl").unlink()

    # ------------------------------------------------------------------ envs

    def _ensure_env(self) -> Collector:
        if self.collector is None:
            e = self.cfg.env
            num_envs = (
                cap_num_envs(e.num_envs, e.gpu_mem_per_env_gb, self.device, e.render_devices)
                if e.cap_envs_by_gpu_mem
                else e.num_envs
            )
            self.env = RemoteVectorEnv(
                num_envs,
                e.backend,
                host=e.host,
                base_port=e.base_port,
                spawn=e.spawn,
                python=e.python,
                render_devices=e.render_devices,
                backend_kwargs=json.loads(e.backend_kwargs_json),
                max_episode_steps=e.max_episode_steps,
                timeout_s=e.step_timeout_s,
                log_dir=str(self.out / "env_logs"),
            )
            self.collector = Collector(
                self.env,
                self.policy,
                self.learner,
                execute_steps=self.cfg.collect.execute_steps,
                critic_horizon=self.cfg.critic.horizon,
                gamma=self.cfg.gamma,
            )
        return self.collector

    # ------------------------------------------------------------------ phases

    def collect(self, r: int) -> RolloutStats:
        c = self.cfg.collect
        bon = 1 if (r == 0 and c.plain_first_round) else c.best_of_n
        stats = self._ensure_env().run(
            c.tasks,
            c.episodes_per_task,
            mode=c.sample_mode,
            best_of_n=bon,
            buffer=self.buffer,
            seed=self.cfg.seed * 1_000_003 + c.seed_offset + r * 100_003,
        )
        return stats

    def evaluate(self, tag: str) -> RolloutStats:
        c = self.cfg.collect
        tasks = c.eval_tasks or c.tasks
        return self._ensure_env().run(
            tasks,
            c.eval_episodes_per_task,
            mode=c.eval_mode,
            best_of_n=c.eval_best_of_n,
            buffer=None,
            seed=10_000_019
            + self.cfg.seed * 1_000_003
            + c.seed_offset,  # disjoint from collection seeds
        )

    def _plan(self, steps: int) -> list[dict]:
        cfgc = self.cfg.critic
        plan = []
        has_success = self.buffer.num_success_transitions > 0
        for _ in range(steps):
            item = {"u": self.buffer.sample(cfgc.batch_size)}
            if has_success and (cfgc.success_batch or self.cfg.actor.bc_coeff > 0):
                item["s"] = self.buffer.sample(cfgc.batch_size, success_only=True)
            plan.append(item)
        return plan

    def _needs(self) -> dict[str, tuple[str, ...]]:
        cfg = self.cfg
        needs: dict[str, list[str]] = {"u": [], "s": []}
        if cfg.critic.bootstrap == "policy":
            needs["u"].append("next")
            if cfg.critic.success_batch:
                needs["s"].append("next")
        if cfg.algo == "ogpo":
            needs["u"].append(f"obs:{cfg.actor.batch_states}")
            if cfg.actor.bc_coeff > 0:
                needs["s"].append(f"obs:{cfg.actor.bc_batch}")
        return {k: tuple(v) for k, v in needs.items()}

    def train_phase(self, steps: int) -> None:
        cfg = self.cfg
        if self.buffer.size == 0:
            logger.warning("Empty replay buffer; skipping training")
            return
        p = self.policy
        prep = partial(prepare_inputs, p.processor, p.embodiment_tag, p.language_key)
        dataset = StepBatches(
            self.buffer, prep, self._plan(steps), self._needs(), cfg.encode_chunk, p.language_key
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=_identity_collate,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
            multiprocessing_context="fork" if cfg.num_workers > 0 else None,
            # A crashed worker must fail the job instead of hanging it until the time limit.
            timeout=cfg.loader_timeout_s if cfg.num_workers > 0 else 0,
        )
        t0 = time.time()
        for i, item in enumerate(loader):
            info = self.train_step(item)
            if (i + 1) % cfg.log_every == 0 or i == steps - 1:
                info["time/steps_per_s"] = (i + 1) / (time.time() - t0)
                info["buffer/size"] = self.buffer.size
                self.log(info, step=self.learner.critic_steps)
                logger.info(
                    "round %d step %d/%d  td %.4f  q %.2f  %s",
                    self.round,
                    i + 1,
                    steps,
                    float(info.get("critic/td_loss", float("nan"))),
                    float(info.get("critic/q_mean", float("nan"))),
                    " ".join(
                        f"{k.split('/')[-1]} {float(v):.4g}"
                        for k, v in info.items()
                        if k.startswith("actor/")
                        and k.split("/")[-1]
                        in (
                            "ratio_max",
                            "clip_frac",
                            "adv_nonzero_frac",
                            "bc_loss",
                            "kl_to_base_path",
                        )
                    ),
                )

    def train_step(self, item: dict) -> dict:
        cfg = self.cfg
        p, L = self.policy, self.learner
        info: dict = {}

        def to_batch(name: str) -> CriticBatch:
            return CriticBatch.from_numpy(item[name]["critic"], self.device)

        u = to_batch("u")
        next_u = p.encode_prepared(item["u"]["next"]) if "next" in item["u"] else None
        info.update(L.update_critic(u, next_u, tag="critic"))
        if "s" in item and cfg.critic.success_batch:
            s = to_batch("s")
            next_s = p.encode_prepared(item["s"]["next"]) if "next" in item["s"] else None
            info.update(L.update_critic(s, next_s, tag="critic_succ"))
        L.update_targets()

        if cfg.algo == "ogpo" and L.critic_steps % cfg.actor.update_every == 0:
            feats = p.encode_prepared(item["u"]["obs"])
            n = feats.batch_size
            bc_feats = bc_actions = None
            if "s" in item and "obs" in item["s"]:
                bc_feats = p.encode_prepared(item["s"]["obs"])
                bc_actions = torch.as_tensor(
                    item["s"]["critic"]["bc_action"][: bc_feats.batch_size], device=self.device
                )
            info.update(L.update_policy(feats, u.prefix[:n], u.state[:n], bc_feats, bc_actions))
            if cfg.actor.kl_every and L.policy_steps % cfg.actor.kl_every == 0:
                info.update(L.kl_to_base(feats.index(slice(0, cfg.actor.kl_states))))
        return info

    # ------------------------------------------------------------------ main loop

    def run(self) -> None:
        cfg = self.cfg
        if not cfg.collect.tasks:
            raise ValueError("collect.tasks is empty")
        pool = ThreadPoolExecutor(max_workers=max(1, min(4, cfg.num_workers)))
        try:
            for r in range(self.round, cfg.rounds):
                self.round = r
                stats = self.collect(r)
                metrics = stats.metrics("train")
                metrics.update(
                    {
                        "buffer/size": self.buffer.size,
                        "buffer/success_transitions": self.buffer.num_success_transitions,
                        "buffer/episodes": self.buffer.num_episodes,
                    }
                )
                self.log(metrics, step=self.learner.critic_steps)
                logger.info(
                    "round %d collection: success %.3f over %d episodes",
                    r,
                    stats.success_rate,
                    sum(stats.episodes.values()),
                )
                n = fill_missing_prefixes(
                    self.buffer, self.policy, cfg.prefix_batch, prepare_pool=pool
                )
                logger.info("computed %d prefix embeddings", n)
                self.history.append({"round": r, "train_success": stats.success_rate})
                if self._kill_switch():
                    logger.warning(
                        "Kill switch: training-task success did not improve enough; stopping."
                    )
                    self.save_round(r)
                    break
                self.train_phase(cfg.steps_per_round)
                if cfg.collect.eval_every_rounds and (r + 1) % cfg.collect.eval_every_rounds == 0:
                    self.log(self.evaluate("eval").metrics("eval"), step=self.learner.critic_steps)
                self.save_round(r)
            self.round = cfg.rounds
            final = self.evaluate("final")
            self.log(final.metrics("final_eval"), step=self.learner.critic_steps)
            logger.info("final evaluation: success %.3f", final.success_rate)
            if cfg.algo == "ogpo":
                self.policy.save_pretrained(self.out / "final_policy")
            (self.out / "final_eval.json").write_text(
                json.dumps(final.metrics("final_eval"), indent=2)
            )
            if self._wandb is not None:
                self._wandb.summary.update(final.metrics("final_eval"))
        finally:
            pool.shutdown(wait=False)
            if self.env is not None:
                self.env.close()
            if self._wandb is not None:
                self._wandb.finish()

    def _kill_switch(self) -> bool:
        cfg = self.cfg
        if cfg.kill_switch_after_rounds <= 0 or len(self.history) <= cfg.kill_switch_after_rounds:
            return False
        base = self.history[0]["train_success"]
        best = max(h["train_success"] for h in self.history[1:])
        return best - base < cfg.kill_switch_min_gain


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(cfg: TrainConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    Trainer(cfg).run()


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
