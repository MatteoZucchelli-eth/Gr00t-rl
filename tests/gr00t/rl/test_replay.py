"""Tests for episode finalisation and the replay buffer (CPU)."""

from gr00t.rl.replay import EpisodeRecorder, ReplayBuffer, finalize_episode
import numpy as np
import pytest


STATE_KEYS = ["a", "b"]
STATE_DIMS = [2, 1]


def make_episode(
    n: int, success_at: int | None, task: str = "t", value_offset: float = 0.0
) -> EpisodeRecorder:
    """Episode with n steps; frames encode (episode offset + obs index) in their pixels."""
    rec = EpisodeRecorder(task=task)

    def obs(i):
        v = np.uint8((int(value_offset) + i) % 256)
        frames = {"cam": np.full((4, 6, 3), v, dtype=np.uint8)}
        state = {
            "a": np.array([i, -i], dtype=np.float32),
            "b": np.array([value_offset], dtype=np.float32),
        }
        return frames, state

    rec.add_observation(*obs(0))
    for t in range(n):
        success = success_at is not None and t == success_at
        rec.add_step(
            {"act": np.array([float(t)], dtype=np.float32)},
            0.0 if success else -1.0,
            success,
            (t == n - 1) and not success,
        )
        rec.add_observation(*obs(t + 1))
        if success:
            break
    return rec


def encode_chunk(state, chunk):
    # "Relative" encoding: subtract the state's first entry from the absolute action index.
    return (chunk["act"] - state["a"][-1, 0]).astype(np.float32)


def normalize_state(state):
    return np.concatenate([state[k].reshape(-1) for k in STATE_KEYS]) / 10.0


def finalize(rec, H=3, bc_h=5, gamma=0.9):
    return finalize_episode(rec, H, bc_h, gamma, encode_chunk, normalize_state, STATE_KEYS)


def test_truncated_episode_bootstraps():
    ep = finalize(make_episode(6, None))
    assert len(ep.reward) == 6 - 3 + 1
    np.testing.assert_allclose(ep.reward, -(1 + 0.9 + 0.81), rtol=1e-6)
    np.testing.assert_allclose(ep.discount, 0.9**3, rtol=1e-6)  # truncation is not terminal
    np.testing.assert_allclose(ep.mc_return, -1 / (1 - 0.9), rtol=1e-6)
    np.testing.assert_array_equal(ep.next_offset, np.arange(4) + 3)
    assert not ep.success


def test_terminated_episode_zero_discount_and_returns():
    ep = finalize(make_episode(10, success_at=4))  # 5 steps, success on the 5th
    assert len(ep.reward) == 5 - 3 + 1
    # window 0: steps 0,1,2 -> all -1; window 2: steps 2,3,4 -> -1, -1, 0
    np.testing.assert_allclose(ep.reward[0], -(1 + 0.9 + 0.81), rtol=1e-6)
    np.testing.assert_allclose(ep.reward[2], -(1 + 0.9), rtol=1e-6)
    np.testing.assert_allclose(ep.discount, [0.9**3, 0.9**3, 0.0], rtol=1e-6)
    # MC return-to-go from t=0: -1 - .9 - .81 - .729
    np.testing.assert_allclose(ep.mc_return[0], -(1 + 0.9 + 0.81 + 0.729), rtol=1e-6)
    assert ep.success


def test_bc_chunks_are_stitched_padded_and_relative():
    ep = finalize(make_episode(6, None), H=3, bc_h=5)
    assert ep.action.shape == (4, 5, 1)
    # window t: absolute actions t..t+4 clamped to the last action (5), minus state a[0] = t.
    for t in range(4):
        expect = np.minimum(np.arange(t, t + 5), 5) - t
        np.testing.assert_allclose(ep.action[t, :, 0], expect)


def test_too_short_episode_is_dropped():
    assert finalize(make_episode(2, success_at=1)) is None


def make_buffer(capacity=100, obs_capacity=None, codec="raw"):
    return ReplayBuffer(
        capacity=capacity,
        camera_keys=["cam"],
        state_keys=STATE_KEYS,
        state_dims=STATE_DIMS,
        state_dim=3,
        prefix_dim=4,
        action_shape=(5, 1),
        critic_horizon=3,
        video_delta_indices=[-2, 0],
        frame_codec=codec,
        obs_capacity=obs_capacity,
        seed=0,
    )


def test_buffer_insert_sample_and_history():
    buf = make_buffer()
    buf.insert_episode(finalize(make_episode(6, None, task="fail", value_offset=0)))
    buf.insert_episode(finalize(make_episode(10, success_at=4, task="win", value_offset=100)))
    assert buf.size == 4 + 3
    assert buf.num_success_transitions == 3
    idx = buf.sample(64, success_only=True)
    assert buf.tr_success[idx].all()

    # Prefixes must be filled before critic batches.
    with pytest.raises(RuntimeError):
        buf.critic_batch(buf.sample(4))
    missing = buf.missing_prefix_obs()
    assert len(missing) == 7 + 6
    buf.set_prefixes(missing, np.tile(missing[:, None].astype(np.float32), (1, 4)))
    cb = buf.critic_batch(np.arange(7))
    assert cb["action"].shape == (7, 3, 1) and cb["bc_action"].shape == (7, 5, 1)
    np.testing.assert_array_equal(cb["next_obs_idx"] - cb["obs_idx"], 3)
    np.testing.assert_allclose(cb["prefix"][:, 0], cb["obs_idx"])

    # History frames: delta -2, clamped to the episode start.
    obs = buf.policy_observations([7, 8, 10], language_key="lang")  # obs 0, 1, 3 of episode 2
    pix = [o["video"]["cam"][:, 0, 0, 0].tolist() for o in obs]
    assert pix == [[100, 100], [100, 101], [101, 103]]
    assert obs[0]["language"]["lang"] == ["win"]
    np.testing.assert_allclose(obs[2]["state"]["a"], [[3, -3]])
    np.testing.assert_allclose(obs[2]["state"]["b"], [[100]])


def test_buffer_eviction_keeps_only_resolvable_transitions():
    buf = make_buffer(capacity=10, obs_capacity=14)
    for e in range(5):
        buf.insert_episode(finalize(make_episode(6, None, value_offset=10 * e)))
    # Every valid transition must reference resident observations (incl. history).
    oldest = buf.obs_total - buf.obs_capacity
    for o in range(buf.valid_start, buf.total_inserted):
        r = o % buf.capacity
        obs = int(buf.tr_obs[r])
        assert obs >= oldest
        earliest = max(obs - 2, int(buf.obs_episode_start[obs % buf.obs_capacity]))
        assert earliest >= oldest
        assert buf.tr_next_obs[r] < buf.obs_total
    assert 0 < buf.size <= 10
    # Sampling never returns an evicted transition.
    idx = buf.sample(200)
    assert (buf.tr_obs[idx] >= oldest).all()


def test_jpeg_roundtrip_and_save_load(tmp_path):
    buf = make_buffer(codec="jpeg")
    buf.insert_episode(finalize(make_episode(6, None, value_offset=50)))
    path = tmp_path / "buf.pkl"
    buf.save(path)
    other = make_buffer(codec="jpeg")
    other.load(path)
    assert other.size == buf.size and other.tasks == buf.tasks
    a = buf.policy_observations([3], "l")[0]["video"]["cam"]
    b = other.policy_observations([3], "l")[0]["video"]["cam"]
    np.testing.assert_array_equal(a, b)
    assert abs(int(a[1, 0, 0, 0]) - 53) <= 2  # lossy but close
