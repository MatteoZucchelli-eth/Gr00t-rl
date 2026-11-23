"""Dependency-free stand-in environment for pipeline tests.

Observations mimic the DROID layout by default. The episode succeeds once the commanded
gripper has been closed (> 0.5) for ``close_steps`` consecutive steps, so a policy can
learn it and tests can force either outcome; ``success_prob`` adds random successes.
"""

from __future__ import annotations

import numpy as np


class FakeAdapter:
    def __init__(
        self,
        camera_keys: tuple[str, ...] = ("exterior_image_1_left", "wrist_image_left"),
        image_size: tuple[int, int] = (32, 48),
        state_dims: dict[str, int] | None = None,
        gripper_action_key: str = "gripper_position",
        close_steps: int = 3,
        success_prob: float = 0.0,
    ):
        self.camera_keys = list(camera_keys)
        self.image_size = tuple(image_size)
        self.state_dims = state_dims or {"eef_9d": 9, "gripper_position": 1, "joint_position": 7}
        self.gripper_action_key = gripper_action_key
        self.close_steps = close_steps
        self.success_prob = success_prob
        self._rng = np.random.default_rng(0)
        self._task = ""
        self._closed_for = 0
        self._t = 0

    def spec(self) -> dict:
        return {
            "camera_keys": self.camera_keys,
            "state_keys": list(self.state_dims),
            "image_size": list(self.image_size),
        }

    def _obs(self) -> dict:
        h, w = self.image_size
        video = {
            k: self._rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8) for k in self.camera_keys
        }
        state = {
            k: (self._rng.normal(size=d) * 0.01).astype(np.float32)
            for k, d in self.state_dims.items()
        }
        if "eef_9d" in state:  # keep a valid rot6d (identity rows)
            state["eef_9d"][3:9] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
        return {"video": video, "state": state, "language": self._task}

    def reset(self, task_id: str, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._task = f"fake task {task_id}"
        self._closed_for = 0
        self._t = 0
        return self._obs(), {"success": False}

    def step(self, action: dict):
        self._t += 1
        g = float(np.asarray(action.get(self.gripper_action_key, [0.0])).reshape(-1)[0])
        self._closed_for = self._closed_for + 1 if g > 0.5 else 0
        lucky = self.success_prob > 0 and self._rng.random() < self.success_prob
        return self._obs(), {"success": self._closed_for >= self.close_steps or lucky}

    def close(self) -> None:
        pass
