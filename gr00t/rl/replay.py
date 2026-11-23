"""Replay buffer for batch-online RL on GR00T (port of Korniak et al. 2026's buffer).

Layout
------
* An **observation ring**: one entry per environment step. It holds the camera frames
  (encoded once; history frames such as DROID's ``t-15`` are looked up by index), the
  raw proprio state, the normalized state the critic sees, the mean-pooled VLM prefix
  embedding the critic sees, and the task (language) id.
* A **transition ring**: one entry per environment step ``t`` whose chunk window
  ``[t, t + H)`` fits inside the episode. It points to observation ``t`` and ``t + H``
  and stores the H-step discounted reward, the bootstrap discount (0 iff the episode
  *terminated* inside the window; time-limit truncation bootstraps), the Monte-Carlo
  return (diagnostics), the success flag of the episode, and the normalized action
  chunk executed from ``t`` onward, stitched across policy queries, re-encoded relative
  to the state at ``t`` (the BC target; the critic reads its first H steps).

Episodes are inserted whole when they end, so each episode's observations are
contiguous in the ring and history-frame lookups never cross episodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
from pathlib import Path
import pickle
from typing import Callable, Sequence

import numpy as np


# --------------------------------------------------------------------------------------
# Episode recording and finalisation
# --------------------------------------------------------------------------------------


@dataclass
class EpisodeRecorder:
    """Accumulates one episode for one environment.

    ``frames[i]`` / ``states[i]`` describe observation ``i``; ``actions[i]`` is the
    absolute (decoded) action executed from observation ``i``; ``rewards[i]``,
    ``terminated[i]`` and ``truncated[i]`` are the outcome of that step. After the last
    step, :meth:`add_final_observation` appends observation ``n``.
    """

    task: str
    frames: list[dict[str, np.ndarray]] = field(default_factory=list)
    states: list[dict[str, np.ndarray]] = field(default_factory=list)
    actions: list[dict[str, np.ndarray]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)

    def add_observation(self, frames: dict[str, np.ndarray], state: dict[str, np.ndarray]):
        self.frames.append(frames)
        self.states.append(
            {k: np.asarray(v, dtype=np.float32).reshape(-1) for k, v in state.items()}
        )

    def add_step(
        self, action: dict[str, np.ndarray], reward: float, terminated: bool, truncated: bool
    ):
        self.actions.append(
            {k: np.asarray(v, dtype=np.float32).reshape(-1) for k, v in action.items()}
        )
        self.rewards.append(float(reward))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))

    def add_final_observation(self, frames: dict[str, np.ndarray], state: dict[str, np.ndarray]):
        self.add_observation(frames, state)

    @property
    def num_steps(self) -> int:
        return len(self.actions)

    @property
    def success(self) -> bool:
        return any(self.terminated)


@dataclass
class EpisodeData:
    """A finalised episode, ready to insert."""

    task: str
    frames: list[dict[str, np.ndarray]]  # n + 1 observations
    states_raw: np.ndarray  # [n + 1, S_raw]
    states_norm: np.ndarray  # [n + 1, S]
    # Transitions (one per valid window start t = 0 .. n_windows - 1).
    next_offset: np.ndarray  # [W] observation offset of the bootstrap state (t + H)
    reward: np.ndarray  # [W]
    discount: np.ndarray  # [W]
    mc_return: np.ndarray  # [W]
    action: np.ndarray  # [W, H_bc, Da] normalized chunk from t
    success: bool


def finalize_episode(
    episode: EpisodeRecorder,
    critic_horizon: int,
    bc_horizon: int,
    gamma: float,
    encode_chunk: Callable[[dict[str, np.ndarray], dict[str, np.ndarray]], np.ndarray],
    normalize_state: Callable[[dict[str, np.ndarray]], np.ndarray],
    state_keys: Sequence[str],
) -> EpisodeData | None:
    """Turn a recorded episode into H-step chunk transitions.

    Args:
        episode: the recorded episode (must have its final observation).
        critic_horizon: H, the chunk length the critic scores and the TD step.
        bc_horizon: model action horizon; length of the stored normalized chunk.
        gamma: per-env-step discount.
        encode_chunk: ``(state_dict, abs_action_chunk_dict) -> [bc_horizon, Da]``;
            converts an absolute chunk into the model's normalized (relative) space
            using the state at the chunk start as reference.
        normalize_state: ``state_dict -> [S]`` normalized state for the critic.
        state_keys: order of raw state groups when flattening.

    Returns:
        ``None`` if the episode is shorter than ``critic_horizon``.
    """
    n = episode.num_steps
    if len(episode.frames) != n + 1:
        raise ValueError(
            f"Episode has {len(episode.frames)} observations for {n} steps; "
            "call add_final_observation() before finalising."
        )
    H = critic_horizon
    n_windows = n - H + 1
    if n_windows <= 0:
        return None

    rewards = np.asarray(episode.rewards, dtype=np.float64)
    terminated = np.asarray(episode.terminated, dtype=bool)
    gammas = gamma ** np.arange(n, dtype=np.float64)

    reward_w = np.array([(rewards[t : t + H] * gammas[:H]).sum() for t in range(n_windows)])
    discount_w = np.array(
        [0.0 if terminated[t : t + H].any() else gamma**H for t in range(n_windows)]
    )
    # Discounted return-to-go, normalised so that entry t starts discounting at t.
    mc = (gammas * rewards)[::-1].cumsum()[::-1] / gammas
    if np.all(rewards == rewards[0]):
        # Never-succeeding episode with constant reward: treat as an infinite horizon.
        mc = np.full_like(mc, rewards[0] / (1.0 - gamma))
    mc_w = mc[:n_windows]

    action_keys = list(episode.actions[0].keys())
    stacked_actions = {k: np.stack([a[k] for a in episode.actions]) for k in action_keys}  # [n, Dk]
    chunks = []
    for t in range(n_windows):
        idx = np.minimum(np.arange(t, t + bc_horizon), n - 1)  # pad by repeating the last action
        chunk = {k: v[idx] for k, v in stacked_actions.items()}
        state = {k: episode.states[t][k][None, :] for k in state_keys}
        chunks.append(np.asarray(encode_chunk(state, chunk), dtype=np.float32))
    action_w = np.stack(chunks)

    states_raw = np.stack([np.concatenate([s[k] for k in state_keys]) for s in episode.states])
    states_norm = np.stack(
        [
            np.asarray(
                normalize_state({k: s[k][None, :] for k in state_keys}), dtype=np.float32
            ).reshape(-1)
            for s in episode.states
        ]
    )

    return EpisodeData(
        task=episode.task,
        frames=episode.frames,
        states_raw=states_raw.astype(np.float32),
        states_norm=states_norm.astype(np.float32),
        next_offset=np.arange(n_windows) + H,
        reward=reward_w.astype(np.float32),
        discount=discount_w.astype(np.float32),
        mc_return=mc_w.astype(np.float32),
        action=action_w,
        success=episode.success,
    )


# --------------------------------------------------------------------------------------
# Frame codec
# --------------------------------------------------------------------------------------


def encode_frame(frame: np.ndarray, codec: str, quality: int) -> bytes:
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    if codec == "raw":
        buf = io.BytesIO()
        np.save(buf, frame, allow_pickle=False)
        return buf.getvalue()
    from PIL import Image

    buf = io.BytesIO()
    if codec == "jpeg":
        Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    elif codec == "png":
        Image.fromarray(frame).save(buf, format="PNG")
    else:
        raise ValueError(f"Unknown frame codec {codec!r}")
    return buf.getvalue()


def decode_frame(data: bytes, codec: str) -> np.ndarray:
    if codec == "raw":
        return np.load(io.BytesIO(data), allow_pickle=False)
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


# --------------------------------------------------------------------------------------
# Replay buffer
# --------------------------------------------------------------------------------------


class ReplayBuffer:
    """Ring buffer of chunk transitions with linked, history-aware observations."""

    def __init__(
        self,
        capacity: int,
        camera_keys: Sequence[str],
        state_keys: Sequence[str],
        state_dims: Sequence[int],
        state_dim: int,
        prefix_dim: int,
        action_shape: tuple[int, int],
        critic_horizon: int,
        video_delta_indices: Sequence[int],
        frame_codec: str = "jpeg",
        frame_quality: int = 95,
        obs_capacity: int | None = None,
        seed: int = 0,
    ):
        """
        Args:
            capacity: max number of transitions.
            camera_keys: video modality keys, in model order.
            state_keys: raw state groups, in model order.
            state_dims: width of each raw state group (same order as ``state_keys``).
            state_dim: width of the normalized state vector the critic sees.
            prefix_dim: width of the critic's prefix embedding.
            action_shape: (bc_horizon, action_dim) of stored normalized chunks.
            critic_horizon: H (the critic reads ``action[:, :H]``).
            video_delta_indices: e.g. ``[-15, 0]``; frames before the episode start
                are clamped to the first frame.
            frame_codec: ``jpeg`` | ``png`` | ``raw``.
            obs_capacity: observation-ring size; defaults to ``capacity`` plus slack
                for the extra per-episode observations.
        """
        self.capacity = int(capacity)
        self.obs_capacity = int(obs_capacity or int(capacity * 1.1) + 4096)
        self.camera_keys = list(camera_keys)
        self.state_keys = list(state_keys)
        self.state_dims = [int(d) for d in state_dims]
        raw_state_dim = sum(self.state_dims)
        self.critic_horizon = int(critic_horizon)
        self.video_delta_indices = [int(d) for d in video_delta_indices]
        self.max_history = max(0, -min(self.video_delta_indices + [0]))
        self.frame_codec = frame_codec
        self.frame_quality = frame_quality
        self._rng = np.random.default_rng(seed)

        oc, tc = self.obs_capacity, self.capacity
        # Observation ring.
        self.frames = {k: np.empty(oc, dtype=object) for k in self.camera_keys}
        self.obs_state_raw = np.zeros((oc, raw_state_dim), dtype=np.float32)
        self.obs_state = np.zeros((oc, state_dim), dtype=np.float32)
        self.obs_prefix = np.zeros((oc, prefix_dim), dtype=np.float16)
        self.obs_has_prefix = np.zeros(oc, dtype=bool)
        self.obs_task = np.zeros(oc, dtype=np.int32)
        self.obs_episode_start = np.zeros(oc, dtype=np.int64)  # absolute index
        self.obs_total = 0

        # Transition ring.
        self.tr_obs = np.zeros(tc, dtype=np.int64)  # absolute obs index
        self.tr_next_obs = np.zeros(tc, dtype=np.int64)
        self.tr_reward = np.zeros(tc, dtype=np.float32)
        self.tr_discount = np.zeros(tc, dtype=np.float32)
        self.tr_mc_return = np.zeros(tc, dtype=np.float32)
        self.tr_success = np.zeros(tc, dtype=bool)
        self.tr_action = np.zeros((tc, *action_shape), dtype=np.float16)
        self.total_inserted = 0
        self.valid_start = 0  # absolute transition ordinal of the oldest valid transition

        self.tasks: list[str] = []
        self._task_index: dict[str, int] = {}
        self._success_cache: np.ndarray | None = None
        self.num_episodes = 0
        self.num_success_episodes = 0

    # ---------------------------------------------------------------- bookkeeping

    @property
    def size(self) -> int:
        return int(self.total_inserted - self.valid_start)

    def task_id(self, task: str) -> int:
        if task not in self._task_index:
            self._task_index[task] = len(self.tasks)
            self.tasks.append(task)
        return self._task_index[task]

    def _oldest_resident_obs(self) -> int:
        return self.obs_total - self.obs_capacity

    def _advance_valid_start(self) -> None:
        self.valid_start = max(self.valid_start, self.total_inserted - self.capacity)
        oldest = self._oldest_resident_obs()
        while self.valid_start < self.total_inserted:
            ring = self.valid_start % self.capacity
            obs = int(self.tr_obs[ring])
            # The observation itself must be resident before its ring slot (and hence its
            # episode start) can be trusted; then every history frame must be resident.
            if obs >= oldest:
                earliest = max(
                    obs - self.max_history, int(self.obs_episode_start[obs % self.obs_capacity])
                )
                if earliest >= oldest:
                    break
            self.valid_start += 1

    # ---------------------------------------------------------------- insertion

    def insert_episode(self, ep: EpisodeData) -> None:
        num_obs = len(ep.frames)
        if num_obs > self.obs_capacity:
            raise ValueError(f"Episode with {num_obs} observations exceeds obs_capacity")
        base = self.obs_total
        obs_idx = np.arange(base, base + num_obs)
        ring = obs_idx % self.obs_capacity
        task = self.task_id(ep.task)
        for i, r in enumerate(ring):
            for k in self.camera_keys:
                self.frames[k][r] = encode_frame(
                    ep.frames[i][k], self.frame_codec, self.frame_quality
                )
        self.obs_state_raw[ring] = ep.states_raw
        self.obs_state[ring] = ep.states_norm
        self.obs_has_prefix[ring] = False
        self.obs_task[ring] = task
        self.obs_episode_start[ring] = base
        self.obs_total += num_obs

        n_tr = len(ep.reward)
        tr_ring = np.arange(self.total_inserted, self.total_inserted + n_tr) % self.capacity
        self.tr_obs[tr_ring] = base + np.arange(n_tr)
        self.tr_next_obs[tr_ring] = base + ep.next_offset
        self.tr_reward[tr_ring] = ep.reward
        self.tr_discount[tr_ring] = ep.discount
        self.tr_mc_return[tr_ring] = ep.mc_return
        self.tr_success[tr_ring] = ep.success
        self.tr_action[tr_ring] = ep.action.astype(np.float16)
        self.total_inserted += n_tr
        self._advance_valid_start()
        self._success_cache = None
        self.num_episodes += 1
        self.num_success_episodes += int(ep.success)

    # ---------------------------------------------------------------- prefixes

    def missing_prefix_obs(self) -> np.ndarray:
        """Absolute indices of resident observations without a prefix embedding."""
        lo = max(0, self._oldest_resident_obs())
        idx = np.arange(lo, self.obs_total)
        return idx[~self.obs_has_prefix[idx % self.obs_capacity]]

    def set_prefixes(self, obs_idx: np.ndarray, prefixes: np.ndarray) -> None:
        ring = np.asarray(obs_idx) % self.obs_capacity
        self.obs_prefix[ring] = prefixes.astype(np.float16)
        self.obs_has_prefix[ring] = True

    # ---------------------------------------------------------------- sampling

    def _valid_ordinals(self) -> np.ndarray:
        return np.arange(self.valid_start, self.total_inserted)

    def sample(self, batch_size: int, success_only: bool = False) -> np.ndarray:
        """Uniformly sample transition ring indices (optionally from successful episodes)."""
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer")
        if not success_only:
            ordinals = self._rng.integers(self.valid_start, self.total_inserted, size=batch_size)
            return ordinals % self.capacity
        if self._success_cache is None:
            ords = self._valid_ordinals()
            self._success_cache = ords[self.tr_success[ords % self.capacity]] % self.capacity
        if len(self._success_cache) == 0:
            raise ValueError("No successful transitions in the buffer")
        return self._success_cache[self._rng.integers(0, len(self._success_cache), size=batch_size)]

    @property
    def num_success_transitions(self) -> int:
        ords = self._valid_ordinals()
        return int(self.tr_success[ords % self.capacity].sum())

    def critic_batch(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        """Arrays the critic needs for transitions ``idx`` (ring indices)."""
        obs = self.tr_obs[idx] % self.obs_capacity
        nxt = self.tr_next_obs[idx] % self.obs_capacity
        if not (self.obs_has_prefix[obs].all() and self.obs_has_prefix[nxt].all()):
            raise RuntimeError(
                "Sampled observations without prefix embeddings; call fill_prefixes first"
            )
        return {
            "prefix": self.obs_prefix[obs].astype(np.float32),
            "state": self.obs_state[obs],
            "action": self.tr_action[idx, : self.critic_horizon].astype(np.float32),
            "next_prefix": self.obs_prefix[nxt].astype(np.float32),
            "next_state": self.obs_state[nxt],
            "reward": self.tr_reward[idx],
            "discount": self.tr_discount[idx],
            "mc_return": self.tr_mc_return[idx],
            "is_success": self.tr_success[idx].astype(np.float32),
            "bc_action": self.tr_action[idx].astype(np.float32),
            "obs_idx": self.tr_obs[idx],
            "next_obs_idx": self.tr_next_obs[idx],
        }

    def policy_observations(self, obs_idx: Sequence[int], language_key: str) -> list[dict]:
        """Rebuild GR00T policy observations (single-sample, unbatched) for absolute obs indices.

        Each element is ``{"video": {cam: [T,H,W,C]}, "state": {key: [1, D]},
        "language": {language_key: [str]}}`` matching ``Gr00tPolicy`` inputs.
        """
        out = []
        bounds = np.cumsum([0] + self.state_dims)
        for a in obs_idx:
            a = int(a)
            r = a % self.obs_capacity
            start = int(self.obs_episode_start[r])
            video = {}
            for k in self.camera_keys:
                frames = []
                for d in self.video_delta_indices:
                    src = max(a + d, start) % self.obs_capacity
                    frames.append(decode_frame(self.frames[k][src], self.frame_codec))
                video[k] = np.stack(frames)
            flat = self.obs_state_raw[r]
            state = {
                k: flat[bounds[i] : bounds[i + 1]][None, :].copy()
                for i, k in enumerate(self.state_keys)
            }
            out.append(
                {
                    "video": video,
                    "state": state,
                    "language": {language_key: [self.tasks[int(self.obs_task[r])]]},
                }
            )
        return out

    # ---------------------------------------------------------------- persistence

    _ARRAYS = (
        "obs_state_raw",
        "obs_state",
        "obs_prefix",
        "obs_has_prefix",
        "obs_task",
        "obs_episode_start",
        "tr_obs",
        "tr_next_obs",
        "tr_reward",
        "tr_discount",
        "tr_mc_return",
        "tr_success",
        "tr_action",
    )
    _SCALARS = (
        "obs_total",
        "total_inserted",
        "valid_start",
        "num_episodes",
        "num_success_episodes",
    )

    def save(self, path: str | Path) -> None:
        """Write the buffer to ``path`` atomically (pickle of numpy arrays + frame bytes)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "arrays": {k: getattr(self, k) for k in self._ARRAYS},
            "scalars": {k: getattr(self, k) for k in self._SCALARS},
            "frames": self.frames,
            "tasks": self.tasks,
            "rng": self._rng.bit_generator.state,
            "meta": {
                "capacity": self.capacity,
                "obs_capacity": self.obs_capacity,
                "camera_keys": self.camera_keys,
                "frame_codec": self.frame_codec,
            },
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        meta = payload["meta"]
        if meta["capacity"] != self.capacity or meta["obs_capacity"] != self.obs_capacity:
            raise ValueError("Saved buffer capacity does not match this buffer")
        for k, v in payload["arrays"].items():
            setattr(self, k, v)
        for k, v in payload["scalars"].items():
            setattr(self, k, v)
        self.frames = payload["frames"]
        self.tasks = payload["tasks"]
        self._task_index = {t: i for i, t in enumerate(self.tasks)}
        self._rng.bit_generator.state = payload["rng"]
        self._success_cache = None
