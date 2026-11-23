"""LIBERO adapter (pipeline sanity check for the RL code), for the ``LIBERO_PANDA`` tag.

Wraps ``gr00t.eval.sim.LIBERO.libero_env.LiberoEnv``. Task ids are
``"<suite>/<task_index>"``, e.g. ``"libero_10/3"`` or ``"libero_90/59"``. The seed picks
one of the suite's fixed initial states; after reset the robot waits ``settle_steps``
no-op steps so objects come to rest (as in openpi / Korniak et al.).
"""

from __future__ import annotations

import os

import numpy as np


class LiberoAdapter:
    def __init__(self, settle_steps: int = 10, image_size: int = 256):
        os.environ.setdefault("MUJOCO_GL", "egl")
        from libero.libero import benchmark, get_libero_path

        self._benchmark = benchmark
        self._bddl_root = get_libero_path("bddl_files")
        self.settle_steps = settle_steps
        self.image_size = image_size
        self._env = None
        self._loaded = None
        self._init_states = None
        self._language = ""

    def spec(self) -> dict:
        return {
            "camera_keys": ["image", "wrist_image"],
            "state_keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
            "image_size": [self.image_size, self.image_size],
        }

    def _load(self, suite_name: str, task_index: int) -> None:
        from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv

        suite = self._benchmark.get_benchmark_dict()[suite_name]()
        task = suite.get_task(task_index)
        bddl = os.path.join(self._bddl_root, task.problem_folder, task.bddl_file)
        if self._env is not None:
            self._env.close()
        self._env = LiberoEnv(task_bddl_file=bddl, task_description=task.language)
        self._init_states = suite.get_task_init_states(task_index)
        self._language = task.language
        self._loaded = (suite_name, task_index)

    def reset(self, task_id: str, seed: int | None = None):
        suite_name, idx = str(task_id).rsplit("/", 1)
        if self._loaded != (suite_name, int(idx)):
            self._load(suite_name, int(idx))
        self._env.reset(seed=seed)
        init = self._init_states[(seed or 0) % len(self._init_states)]
        raw = self._env._env.set_init_state(init)
        # Settle with the raw robosuite no-op (zero motion, gripper open = -1), as openpi does.
        raw_noop = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        for _ in range(self.settle_steps):
            raw, _, _, _ = self._env._env.step(raw_noop)
        obs = self._env._process_observation(raw)
        return self._convert(obs), {"success": bool(self._env._env.check_success())}

    def step(self, action: dict):
        env_action = {
            f"action.{k}": np.asarray(v, dtype=np.float32).reshape(-1) for k, v in action.items()
        }
        obs, _reward, _done, _trunc, info = self._env.step(env_action)
        return self._convert(obs), {"success": bool(info.get("success", False))}

    def close(self) -> None:
        if self._env is not None:
            self._env.close()

    def _convert(self, obs: dict) -> dict:
        video = {
            k[len("video.") :]: np.ascontiguousarray(v)
            for k, v in obs.items()
            if k.startswith("video.")
        }
        state = {
            k[len("state.") :]: np.asarray(v, dtype=np.float32).reshape(-1)
            for k, v in obs.items()
            if k.startswith("state.")
        }
        return {"video": video, "state": state, "language": self._language}
