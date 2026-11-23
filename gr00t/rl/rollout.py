"""Batched rollouts of a GR00T flow policy in remote simulators.

Every query: one VLM pass per active env, then either a single action chunk per env or
``best_of_n`` candidates ranked by the critic; the first ``execute_steps`` actions run
open-loop in the simulator (one round-trip per env). Observations after every env step
are recorded, so the replay buffer has one transition per env step (Korniak et al.).
"""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
import logging
import time

import numpy as np
import torch

from gr00t.rl.envs.client import EnvWorkerError, RemoteVectorEnv
from gr00t.rl.replay import EpisodeRecorder, ReplayBuffer, finalize_episode


logger = logging.getLogger(__name__)


@dataclass
class _EnvSlot:
    task: str | None = None
    seed: int | None = None
    recorder: EpisodeRecorder | None = None
    history: deque = field(default_factory=deque)
    state: dict | None = None
    language: str = ""
    steps: int = 0


@dataclass
class RolloutStats:
    episodes: dict[str, int] = field(default_factory=dict)
    successes: dict[str, int] = field(default_factory=dict)
    success_lengths: list[int] = field(default_factory=list)
    env_steps: int = 0
    worker_failures: int = 0
    wall_time_s: float = 0.0

    def add(self, task: str, success: bool, length: int) -> None:
        self.episodes[task] = self.episodes.get(task, 0) + 1
        self.successes[task] = self.successes.get(task, 0) + int(success)
        if success:
            self.success_lengths.append(length)

    @property
    def success_rate(self) -> float:
        n = sum(self.episodes.values())
        return sum(self.successes.values()) / n if n else 0.0

    def per_task(self) -> dict[str, float]:
        return {t: self.successes.get(t, 0) / n for t, n in self.episodes.items() if n}

    def metrics(self, prefix: str) -> dict[str, float]:
        out = {
            f"{prefix}/success_rate": self.success_rate,
            f"{prefix}/episodes": float(sum(self.episodes.values())),
            f"{prefix}/env_steps": float(self.env_steps),
            f"{prefix}/worker_failures": float(self.worker_failures),
            f"{prefix}/wall_time_s": self.wall_time_s,
        }
        if self.success_lengths:
            out[f"{prefix}/mean_success_length"] = float(np.mean(self.success_lengths))
        for t, sr in self.per_task().items():
            out[f"{prefix}/success_rate/{t}"] = sr
        return out


class Collector:
    def __init__(
        self,
        env: RemoteVectorEnv,
        policy,
        learner=None,
        *,
        execute_steps: int = 8,
        critic_horizon: int = 16,
        gamma: float = 0.995,
    ):
        self.env = env
        self.policy = policy
        self.learner = learner
        self.execute_steps = execute_steps
        self.critic_horizon = critic_horizon
        self.gamma = gamma
        self.max_history = max(0, -min(policy.video_delta_indices + [0]))

    # ------------------------------------------------------------------ helpers

    def _stack_video(self, history: deque) -> dict[str, np.ndarray]:
        n = len(history)
        video = {}
        for cam in self.policy.camera_keys:
            frames = [history[max(0, n - 1 + d)][cam] for d in self.policy.video_delta_indices]
            video[cam] = np.stack(frames)
        return video

    def _observation(self, slot: _EnvSlot) -> dict:
        return {
            "video": self._stack_video(slot.history),
            "state": {
                k: np.asarray(slot.state[k], dtype=np.float32)[None, :]
                for k in self.policy.state_keys
            },
            "language": {self.policy.language_key: [slot.language]},
        }

    def _start(self, slot: _EnvSlot, obs: dict, record: bool) -> None:
        slot.history = deque(maxlen=self.max_history + 1)
        slot.history.append(obs["video"])
        slot.state = obs["state"]
        slot.language = obs["language"]
        slot.steps = 0
        slot.recorder = EpisodeRecorder(task=obs["language"]) if record else None
        if slot.recorder is not None:
            slot.recorder.add_observation(obs["video"], obs["state"])

    @torch.no_grad()
    def _select_actions(self, observations: list[dict], mode: str, best_of_n: int) -> list[dict]:
        policy = self.policy
        feats = policy.encode(observations)
        slow = self.learner.slow_policy() if self.learner is not None else nullcontext()
        with slow:
            if best_of_n > 1:
                if self.learner is None:
                    raise ValueError("Best-of-N needs a learner (critic)")
                out = policy.sample(feats, mode=mode, repeats=best_of_n)
                states = torch.as_tensor(
                    np.stack([policy.normalize_state(o["state"]) for o in observations]),
                    device=feats.prefix.device,
                )
                scores = self.learner.score(
                    feats.prefix.repeat_interleave(best_of_n, 0),
                    states.repeat_interleave(best_of_n, 0),
                    out.actions,
                ).view(-1, best_of_n)
                best = scores.argmax(dim=1)
                rows = torch.arange(len(observations), device=best.device) * best_of_n + best
                actions = out.actions[rows]
            else:
                actions = policy.sample(feats, mode=mode).actions
        return policy.decode_actions(actions, [o["state"] for o in observations])

    # ------------------------------------------------------------------ main loop

    def run(
        self,
        tasks: list[str],
        episodes_per_task: int,
        *,
        mode: str = "sde",
        best_of_n: int = 1,
        buffer: ReplayBuffer | None = None,
        seed: int = 0,
    ) -> RolloutStats:
        """Run ``episodes_per_task`` episodes of every task; insert them into ``buffer``."""
        t0 = time.time()
        record = buffer is not None
        queue = deque(
            (t, seed + i * len(tasks) + j)
            for i in range(episodes_per_task)
            for j, t in enumerate(tasks)
        )
        slots = [_EnvSlot() for _ in range(self.env.num_envs)]
        stats = RolloutStats()

        def assign(indices: list[int]) -> None:
            requests = {}
            for i in indices:
                if queue:
                    slots[i].task, slots[i].seed = queue.popleft()
                    requests[i] = (slots[i].task, slots[i].seed)
                else:
                    slots[i].task = None
            while requests:
                try:
                    replies = self.env.reset(requests)
                except EnvWorkerError as e:
                    replies = e.replies
                    for i in e.failed:
                        logger.warning(
                            "Reset failed on worker %d, restarting: %s", i, e.failed[i][:500]
                        )
                        stats.worker_failures += 1
                        self.env.restart(i)
                for i, r in replies.items():
                    self._start(slots[i], r["obs"], record)
                    requests.pop(i)

        assign(list(range(self.env.num_envs)))
        while any(s.task is not None for s in slots):
            active = [i for i, s in enumerate(slots) if s.task is not None]
            observations = [self._observation(slots[i]) for i in active]
            decoded = self._select_actions(observations, mode, best_of_n)
            chunks = {}
            for i, act in zip(active, decoded):
                n = min(self.execute_steps, next(iter(act.values())).shape[0])
                chunks[i] = [{k: v[j] for k, v in act.items()} for j in range(n)]
            try:
                results = self.env.step_chunk(chunks)
                failed = {}
            except EnvWorkerError as e:
                results, failed = e.replies, e.failed
            for i in failed:
                logger.warning(
                    "Worker %d failed mid-episode (%s); discarding episode", i, failed[i][:500]
                )
                stats.worker_failures += 1
                queue.appendleft((slots[i].task, slots[i].seed))
                slots[i].task = None
                self.env.restart(i)
            finished = list(failed)
            for i, res in results.items():
                slot = slots[i]
                done = False
                for j, obs in enumerate(res["obs"]):
                    term, trunc = bool(res["terminated"][j]), bool(res["truncated"][j])
                    if slot.recorder is not None:
                        slot.recorder.add_step(chunks[i][j], res["reward"][j], term, trunc)
                        slot.recorder.add_observation(obs["video"], obs["state"])
                    slot.history.append(obs["video"])
                    slot.state = obs["state"]
                    slot.steps += 1
                    stats.env_steps += 1
                    if term or trunc:
                        done = True
                        stats.add(slot.task, term, slot.steps)
                        if buffer is not None:
                            self._insert(buffer, slot.recorder)
                        break
                if done:
                    finished.append(i)
            if finished:
                assign(finished)
        stats.wall_time_s = time.time() - t0
        return stats

    def _insert(self, buffer: ReplayBuffer, recorder: EpisodeRecorder) -> None:
        ep = finalize_episode(
            recorder,
            critic_horizon=self.critic_horizon,
            bc_horizon=self.policy.valid_horizon,
            gamma=self.gamma,
            encode_chunk=self.policy.encode_action_chunk,
            normalize_state=self.policy.normalize_state,
            state_keys=self.policy.state_keys,
        )
        if ep is not None:
            buffer.insert_episode(ep)


@torch.no_grad()
def fill_missing_prefixes(
    buffer: ReplayBuffer, policy, batch_size: int = 64, prepare_pool=None
) -> int:
    """Compute critic prefix embeddings for buffer observations that lack one.

    ``prepare_pool``: optional ``concurrent.futures`` executor; processor work for the
    next batch then overlaps with the GPU pass of the current one.
    """
    missing = buffer.missing_prefix_obs()
    batches = [missing[i : i + batch_size] for i in range(0, len(missing), batch_size)]

    def prep(idx):
        return policy.prepare_inputs(buffer.policy_observations(idx, policy.language_key))

    if prepare_pool is None:
        for idx in batches:
            feats = policy.encode_prepared(prep(idx))
            buffer.set_prefixes(idx, feats.prefix.float().cpu().numpy())
        return len(missing)
    futures = [prepare_pool.submit(prep, idx) for idx in batches[:2]]
    for b, idx in enumerate(batches):
        collated = futures[b].result()
        if b + 2 < len(batches):
            futures.append(prepare_pool.submit(prep, batches[b + 2]))
        feats = policy.encode_prepared(collated)
        buffer.set_prefixes(idx, feats.prefix.float().cpu().numpy())
    return len(missing)
