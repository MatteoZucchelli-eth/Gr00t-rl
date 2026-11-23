"""Simulator workers for RL collection.

Each worker is a separate process, usually started with the simulator's own virtualenv
(MolmoSpaces and LIBERO pin dependencies that conflict with the model stack), serving
one environment over ZeroMQ: ``python -m gr00t.rl.envs.server --backend molmospaces``.
Modules here must not import torch or the GR00T model code.

Observation format returned by every adapter (one env step)::

    {"video": {camera_key: uint8[H, W, 3]}, "state": {state_key: float32[D]}, "language": str}

Actions are absolute (decoded) robot actions ``{action_key: float32 [D]}`` for one step.
"""
