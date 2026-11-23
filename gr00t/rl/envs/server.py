"""ZeroMQ server around one simulator environment.

Applies the RL conventions generically, so adapters only report ``success``:

* reward: -1 per step, 0 on the step that succeeds (time-to-success);
* termination: on success only;
* truncation: when ``max_episode_steps`` is reached (bootstrapped by the learner).

Run inside the simulator venv::

    python -m gr00t.rl.envs.server --backend molmospaces --port 6100 \
        --render-device 0 --max-episode-steps 450 --kwargs-json '{}'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any

import numpy as np

from gr00t.rl.envs.protocol import pack, unpack


def make_adapter(backend: str, render_device: int, kwargs: dict):
    if backend == "molmospaces":
        from gr00t.rl.envs.molmospaces import MolmoSpacesAdapter

        return MolmoSpacesAdapter(render_device=render_device, **kwargs)
    if backend == "libero":
        from gr00t.rl.envs.libero import LiberoAdapter

        return LiberoAdapter(**kwargs)
    if backend == "fake":
        from gr00t.rl.envs.fake import FakeAdapter

        return FakeAdapter(**kwargs)
    raise ValueError(f"Unknown backend {backend!r}")


class EnvRunner:
    """Episode bookkeeping on top of an adapter (reward, termination, truncation)."""

    def __init__(self, adapter, max_episode_steps: int):
        self.adapter = adapter
        self.max_episode_steps = int(max_episode_steps)
        self.steps = 0
        self.done = True

    def reset(self, task_id: str, seed: int | None) -> dict:
        obs, info = self.adapter.reset(task_id=task_id, seed=seed)
        self.steps = 0
        self.done = False
        return {"obs": obs, "info": _clean_info(info)}

    def step_chunk(self, actions: list[dict]) -> dict:
        """Execute actions open-loop until the chunk ends or the episode is done.

        Returns per executed step: observation after the step, reward, terminated,
        truncated, info.
        """
        if self.done:
            raise RuntimeError("step_chunk called on a finished episode; call reset first")
        out = {"obs": [], "reward": [], "terminated": [], "truncated": [], "info": []}
        for action in actions:
            obs, info = self.adapter.step(
                {k: np.asarray(v, dtype=np.float32) for k, v in action.items()}
            )
            self.steps += 1
            success = bool(info.get("success", False))
            truncated = (not success) and self.steps >= self.max_episode_steps
            out["obs"].append(obs)
            out["reward"].append(0.0 if success else -1.0)
            out["terminated"].append(success)
            out["truncated"].append(truncated)
            out["info"].append(_clean_info(info))
            if success or truncated:
                self.done = True
                break
        return out


def _clean_info(info: dict) -> dict:
    """Keep msgpack-friendly scalars only."""
    clean = {}
    for k, v in (info or {}).items():
        if isinstance(v, (bool, int, float, str)):
            clean[k] = v
        elif isinstance(v, np.generic):
            clean[k] = v.item()
    return clean


def serve(
    backend: str, host: str, port: int, render_device: int, max_episode_steps: int, kwargs: dict
):
    import zmq

    adapter = make_adapter(backend, render_device, kwargs)
    runner = EnvRunner(adapter, max_episode_steps)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(
        f"[env-server] {backend} listening on {host}:{port} (render device {render_device})",
        flush=True,
    )
    running = True
    while running:
        msg = unpack(sock.recv())
        cmd = msg.get("cmd")
        try:
            if cmd == "ping":
                reply: Any = {"ok": True, "spec": adapter.spec()}
            elif cmd == "reset":
                reply = {"ok": True, **runner.reset(msg["task_id"], msg.get("seed"))}
            elif cmd == "step_chunk":
                reply = {"ok": True, **runner.step_chunk(msg["actions"])}
            elif cmd == "close":
                reply = {"ok": True}
                running = False
            else:
                reply = {"ok": False, "error": f"unknown command {cmd!r}"}
        except Exception as e:  # report to the client instead of dying silently
            reply = {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
        sock.send(pack(reply))
    adapter.close()
    sock.close(linger=0)
    ctx.term()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--backend", required=True, choices=["molmospaces", "libero", "fake"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--render-device", type=int, default=0)
    p.add_argument("--max-episode-steps", type=int, default=450)
    p.add_argument("--kwargs-json", default="{}")
    args = p.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(args.render_device))
    serve(
        args.backend,
        args.host,
        args.port,
        args.render_device,
        args.max_episode_steps,
        json.loads(args.kwargs_json),
    )


if __name__ == "__main__":
    sys.exit(main())
