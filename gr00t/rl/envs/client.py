"""Client for a pool of env servers (one ZeroMQ REQ socket per worker).

Requests to different workers are issued together and gathered with a poller, so all
simulators step in parallel while the learner process only does model inference.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Sequence

from gr00t.rl.envs.protocol import pack, unpack


class EnvWorkerError(RuntimeError):
    """One or more workers failed. ``failed`` maps index -> message; ``replies`` holds the
    successful replies of the same request batch."""

    def __init__(self, failed: dict[int, str], replies: dict[int, dict] | None = None):
        msg = "; ".join(f"env worker {i}: {m}" for i, m in sorted(failed.items()))
        super().__init__(msg)
        self.failed = failed
        self.replies = replies or {}

    @property
    def index(self) -> int:
        return min(self.failed)


class RemoteVectorEnv:
    def __init__(
        self,
        num_envs: int,
        backend: str,
        *,
        host: str = "127.0.0.1",
        base_port: int = 6100,
        spawn: bool = True,
        python: str = "python",
        render_devices: Sequence[int] = (0,),
        backend_kwargs: dict | None = None,
        max_episode_steps: int = 450,
        timeout_s: float = 600.0,
        log_dir: str | None = None,
        startup_timeout_s: float = 900.0,
    ):
        import zmq

        self._zmq = zmq
        self.num_envs = num_envs
        self.backend = backend
        self.host = host
        self.base_port = base_port
        self.spawn = spawn
        self.python = python
        self.render_devices = list(render_devices)
        self.backend_kwargs = backend_kwargs or {}
        self.max_episode_steps = max_episode_steps
        self.timeout_ms = int(timeout_s * 1000)
        self.log_dir = Path(log_dir) if log_dir else None
        self.ctx = zmq.Context()
        self.procs: list[subprocess.Popen | None] = [None] * num_envs
        self.socks: list[Any] = [None] * num_envs
        for i in range(num_envs):
            if spawn:
                self._launch(i)
            self._connect(i)
        self.spec = None
        deadline = time.time() + startup_timeout_s
        for i in range(num_envs):
            while True:
                try:
                    reply = self.call(i, {"cmd": "ping"}, timeout_ms=30_000)
                    self.spec = reply["spec"]
                    break
                except EnvWorkerError:
                    if time.time() > deadline:
                        raise
                    self._check_alive(i)
                    self._reconnect(i)

    # ------------------------------------------------------------ process management

    def _port(self, i: int) -> int:
        return self.base_port + i

    def _launch(self, i: int) -> None:
        device = self.render_devices[i % len(self.render_devices)]
        cmd = [
            self.python,
            "-m",
            "gr00t.rl.envs.server",
            "--backend",
            self.backend,
            "--host",
            self.host,
            "--port",
            str(self._port(i)),
            "--render-device",
            str(device),
            "--max-episode-steps",
            str(self.max_episode_steps),
            "--kwargs-json",
            json.dumps(self.backend_kwargs),
        ]
        env = dict(os.environ)
        env.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID=str(device))
        # Workers must not grab the learner's GPU memory via CUDA.
        env.setdefault("CUDA_VISIBLE_DEVICES", str(device))
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            out = open(self.log_dir / f"env_worker_{i}.log", "a")
        else:
            out = subprocess.DEVNULL
        self.procs[i] = subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT)

    def _check_alive(self, i: int) -> None:
        proc = self.procs[i]
        if proc is not None and proc.poll() is not None:
            raise EnvWorkerError({i: f"process exited with code {proc.returncode} during startup"})

    def _connect(self, i: int) -> None:
        sock = self.ctx.socket(self._zmq.REQ)
        sock.setsockopt(self._zmq.LINGER, 0)
        sock.connect(f"tcp://{self.host}:{self._port(i)}")
        self.socks[i] = sock

    def _reconnect(self, i: int) -> None:
        if self.socks[i] is not None:
            self.socks[i].close(linger=0)
        self._connect(i)

    def restart(self, i: int) -> None:
        """Kill and relaunch worker ``i`` (e.g. after a simulator crash or hang)."""
        if self.spawn and self.procs[i] is not None:
            self.procs[i].kill()
            self.procs[i].wait()
            self._launch(i)
        self._reconnect(i)
        deadline = time.time() + 900
        while True:
            try:
                self.call(i, {"cmd": "ping"}, timeout_ms=30_000)
                return
            except EnvWorkerError:
                if time.time() > deadline:
                    raise
                self._reconnect(i)

    # ------------------------------------------------------------ requests

    def call(self, i: int, msg: dict, timeout_ms: int | None = None) -> dict:
        return self.call_many({i: msg}, timeout_ms)[i]

    def call_many(self, msgs: dict[int, dict], timeout_ms: int | None = None) -> dict[int, dict]:
        """Send one request to each listed worker and wait for all replies."""
        timeout_ms = self.timeout_ms if timeout_ms is None else timeout_ms
        poller = self._zmq.Poller()
        for i, msg in msgs.items():
            self.socks[i].send(pack(msg))
            poller.register(self.socks[i], self._zmq.POLLIN)
        replies: dict[int, dict] = {}
        failed: dict[int, str] = {}
        pending = set(msgs)
        deadline = time.time() + timeout_ms / 1000
        sock_to_idx = {id(self.socks[i]): i for i in msgs}
        while pending:
            remaining = int((deadline - time.time()) * 1000)
            if remaining <= 0:
                for i in pending:
                    failed[i] = f"timed out after {timeout_ms} ms"
                    self._reconnect(i)  # a REQ socket without a reply is unusable
                break
            events = dict(poller.poll(remaining))
            for sock, _ in events.items():
                i = sock_to_idx[id(sock)]
                reply = unpack(sock.recv())
                poller.unregister(sock)
                pending.discard(i)
                if reply.get("ok", False):
                    replies[i] = reply
                else:
                    failed[i] = (
                        reply.get("error", "unknown error") + "\n" + reply.get("traceback", "")
                    )
        if failed:
            raise EnvWorkerError(failed, replies)
        return replies

    def reset(self, requests: dict[int, tuple[str, int | None]]) -> dict[int, dict]:
        """Reset envs: ``{env_index: (task_id, seed)}`` -> ``{env_index: {"obs", "info"}}``."""
        return self.call_many(
            {i: {"cmd": "reset", "task_id": t, "seed": s} for i, (t, s) in requests.items()}
        )

    def step_chunk(self, actions: dict[int, list[dict]]) -> dict[int, dict]:
        """Execute open-loop action lists: ``{env_index: [action_dict, ...]}``."""
        return self.call_many({i: {"cmd": "step_chunk", "actions": a} for i, a in actions.items()})

    def close(self) -> None:
        for i in range(self.num_envs):
            try:
                self.call(i, {"cmd": "close"}, timeout_ms=10_000)
            except Exception:
                pass
            if self.socks[i] is not None:
                self.socks[i].close(linger=0)
            proc = self.procs[i]
            if proc is not None:
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self.ctx.term()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
