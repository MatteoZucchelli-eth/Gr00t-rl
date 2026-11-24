"""Env server/client over ZeroMQ with the fake backend, and the Collector (CPU)."""

import sys

from gr00t.rl.config import TrainConfig
from gr00t.rl.envs.client import EnvWorkerError, RemoteVectorEnv
from gr00t.rl.envs.fake import FakeAdapter
from gr00t.rl.envs.server import EnvRunner
from gr00t.rl.ogpo import OGPOLearner
from gr00t.rl.replay import ReplayBuffer
from gr00t.rl.rollout import Collector, RolloutStats
import numpy as np
import pytest
from test_support.rl_toy import ToyFlowPolicy
import torch


def test_env_runner_reward_termination_truncation():
    runner = EnvRunner(FakeAdapter(close_steps=2), max_episode_steps=5)
    runner.reset("0", seed=1)
    out = runner.step_chunk([{"gripper_position": [1.0]}] * 4)
    # success on the 2nd closed step: rewards -1, 0 and the chunk stops early
    assert out["reward"] == [-1.0, 0.0]
    assert out["terminated"] == [False, True]
    assert out["truncated"] == [False, False]
    runner.reset("0", seed=1)
    out = runner.step_chunk([{"gripper_position": [0.0]}] * 8)
    assert len(out["reward"]) == 5 and out["truncated"][-1] and not any(out["terminated"])
    with pytest.raises(RuntimeError):
        runner.step_chunk([{"gripper_position": [0.0]}])


@pytest.fixture
def fake_envs(tmp_path):
    env = RemoteVectorEnv(
        2,
        "fake",
        base_port=17300 + (abs(hash(str(tmp_path))) % 500) * 2,
        spawn=True,
        python=sys.executable,
        backend_kwargs={"close_steps": 2, "image_size": [8, 12]},
        max_episode_steps=12,
        timeout_s=60,
        log_dir=str(tmp_path / "logs"),
    )
    yield env
    env.close()


def test_remote_vector_env_roundtrip(fake_envs):
    assert fake_envs.spec["camera_keys"] == ["exterior_image_1_left", "wrist_image_left"]
    r = fake_envs.reset({0: ("a", 1), 1: ("b", 2)})
    assert r[0]["obs"]["video"]["wrist_image_left"].shape == (8, 12, 3)
    assert r[1]["obs"]["language"] == "fake task b"
    out = fake_envs.step_chunk(
        {0: [{"gripper_position": np.ones(1)}] * 3, 1: [{"gripper_position": np.zeros(1)}] * 3}
    )
    assert out[0]["terminated"][-1] and len(out[0]["obs"]) == 2
    assert len(out[1]["obs"]) == 3 and not any(out[1]["terminated"])
    # Errors are reported per worker, and the good replies of the batch are kept.
    with pytest.raises(EnvWorkerError) as e:
        fake_envs.call_many({0: {"cmd": "bogus"}, 1: {"cmd": "ping"}})
    assert 0 in e.value.failed and 1 in e.value.replies
    fake_envs.restart(0)
    assert fake_envs.call(0, {"cmd": "ping"})["ok"]


class GripperPolicy(ToyFlowPolicy):
    """Closes the gripper (succeeds) on tasks whose id ends in 'win', else keeps it open."""

    def decode_actions(self, actions, states):
        out = super().decode_actions(actions, states)
        return out

    def encode(self, observations):
        feats = super().encode(observations)
        self._wins = ["win" in o["language"][self.language_key][0] for o in observations]
        return feats

    def sample(self, feats, mode="ode", return_trajectory=False, repeats=1, generator=None):
        out = super().sample(feats, mode, return_trajectory, repeats, generator)
        wins = torch.tensor(self._wins).repeat_interleave(repeats)
        out.actions[:, :, 0] = wins.float()[:, None]
        return out


def make_buffer(policy, H):
    return ReplayBuffer(
        capacity=10_000,
        camera_keys=policy.camera_keys,
        state_keys=policy.state_keys,
        state_dims=policy.state_dims,
        state_dim=policy.state_dim,
        prefix_dim=policy.prefix_dim,
        action_shape=(policy.valid_horizon, policy.action_dim),
        critic_horizon=H,
        video_delta_indices=policy.video_delta_indices,
        frame_codec="raw",
    )


@pytest.mark.parametrize("best_of_n", [1, 4])
def test_collector_records_transitions(fake_envs, best_of_n):
    policy = GripperPolicy(prefix_dim=4, horizon=4, action_dim=2)
    cfg = TrainConfig()
    cfg.critic.horizon = 2
    cfg.critic.hidden_dim = 16
    cfg.critic.depth = 1
    cfg.critic.num_qs = 2
    learner = OGPOLearner(
        policy, cfg, policy.prefix_dim, policy.state_dim, policy.action_dim, "cpu"
    )
    buf = make_buffer(policy, H=2)
    col = Collector(fake_envs, policy, learner, execute_steps=3, critic_horizon=2, gamma=0.9)
    seen = []
    stats = col.run(
        ["task_win", "task_lose"],
        episodes_per_task=3,
        mode="sde",
        best_of_n=best_of_n,
        buffer=buf,
        seed=5,
        on_episode=lambda st: seen.append(sum(st.episodes.values())),
    )
    assert seen == list(range(1, 7))  # called once per finished episode
    assert stats.episodes == {"task_win": 3, "task_lose": 3}
    assert stats.per_task() == {"task_win": 1.0, "task_lose": 0.0}
    assert stats.success_lengths == [2, 2, 2]
    assert buf.num_episodes == 6 and buf.num_success_episodes == 3
    # win episodes: 2 steps -> 1 transition each; lose: 12 steps -> 11 transitions each
    assert buf.size == 3 * 1 + 3 * 11
    assert buf.num_success_transitions == 3
    # Successful transitions end in termination; failing ones bootstrap at truncation.
    succ = buf.sample(16, success_only=True)
    assert (buf.tr_discount[succ] == 0).all()
    disc, succ_flag = buf.tr_discount[: buf.size], buf.tr_success[: buf.size]
    assert np.isclose(disc[~succ_flag], 0.9**2).all()
    # Stored language is the env's instruction, used to rebuild policy inputs.
    obs = buf.policy_observations([0], policy.language_key)[0]
    assert obs["video"]["wrist_image_left"].shape == (2, 8, 12, 3)
    assert obs["language"][policy.language_key][0].startswith("fake task")


def test_collector_skip_and_stats_merge(fake_envs):
    """Resume support: skipped episodes are not rerun, and merged stats add up."""
    from gr00t.rl.rollout import RolloutStats

    policy = GripperPolicy(prefix_dim=4, horizon=4, action_dim=2)
    col = Collector(fake_envs, policy, None, execute_steps=3, critic_horizon=2, gamma=0.9)
    tasks = ["task_win", "task_lose"]
    part1 = col.run(tasks, episodes_per_task=3, mode="ode", skip={"task_win": 3, "task_lose": 1})
    assert part1.episodes == {"task_lose": 2}
    part0 = col.run(tasks, episodes_per_task=3, mode="ode", skip={"task_lose": 2})
    assert part0.episodes == {"task_win": 3, "task_lose": 1}
    total = RolloutStats.from_dict(part0.to_dict()).merged(part1)
    assert total.episodes == {"task_win": 3, "task_lose": 3}
    assert total.per_task() == {"task_win": 1.0, "task_lose": 0.0}
    assert total.env_steps == part0.env_steps + part1.env_steps


def test_killed_worker_fails_fast(fake_envs):
    """A worker killed mid-request (e.g. by the OOM killer) fails at once, not after the
    step timeout, and the other workers' replies in the batch are kept."""
    import time

    fake_envs.reset({0: ("a", 1), 1: ("b", 2)})
    fake_envs.procs[0].kill()
    fake_envs.procs[0].wait()
    t0 = time.time()
    with pytest.raises(EnvWorkerError) as e:
        fake_envs.step_chunk({i: [{"gripper_position": np.zeros(1)}] for i in range(2)})
    assert time.time() - t0 < 10  # timeout_s is 60
    assert "SIGKILL" in e.value.failed[0] and 1 in e.value.replies
    fake_envs.restart(0)
    assert fake_envs.call(0, {"cmd": "ping"})["ok"]


def test_collector_recycles_workers_over_memory_share(tmp_path):
    env = RemoteVectorEnv(
        2,
        "fake",
        base_port=18400 + (abs(hash(str(tmp_path))) % 500) * 2,
        spawn=True,
        python=sys.executable,
        backend_kwargs={"close_steps": 2, "image_size": [8, 12], "leak_mb_per_reset": 64},
        max_episode_steps=12,
        timeout_s=60,
        log_dir=str(tmp_path / "logs"),
    )
    try:
        # Pretend the job may use just enough for the learner plus ~150 MB per worker:
        # each worker is recycled after its 2nd-3rd reset instead of growing without bound.
        from gr00t.rl.envs.memory import process_mem_gb

        env.job_mem_gb = (process_mem_gb() + 2 * 0.15) / env.mem_fraction
        pids = {p.pid for p in env.procs}
        policy = GripperPolicy(prefix_dim=4, horizon=4, action_dim=2)
        col = Collector(env, policy, None, execute_steps=3)
        stats = col.run(["task_win"], episodes_per_task=10, mode="ode", seed=0)
        assert stats.episodes == {"task_win": 10} and stats.worker_failures == 0
        assert stats.worker_recycles >= 2
        assert pids.isdisjoint(p.pid for p in env.procs)
        assert max(env.worker_mem_gb(i) for i in range(2)) < 0.5
        assert RolloutStats.from_dict(stats.to_dict()).worker_recycles == stats.worker_recycles
    finally:
        env.close()


def test_job_memory_limit_reads_slurm(monkeypatch):
    from gr00t.rl.envs import memory

    monkeypatch.setattr(memory, "_cgroup_limit_bytes", lambda: None)
    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "8192")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    assert memory.job_memory_limit_gb() == pytest.approx(16.0)
