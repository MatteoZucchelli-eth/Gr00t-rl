"""Check that MolmoSpaces observations match GR00T's OXE_DROID observation space.

Runs in the MolmoSpaces venv (numpy only, no model). For each task it resets the env,
holds the initial pose for a few steps, and reports:

* image shapes/dtypes and a side-by-side PNG of both cameras (compare with DROID frames);
* each state dim against the checkpoint's DROID statistics (z-score, and whether it
  falls inside the [q01, q99] training range);
* the EEF rotation-convention fit: all 24 axis-aligned corrections ``C`` in
  ``R_droid = R_tcp @ C`` ranked by how typical the resulting rot6d is under the DROID
  statistics (the default is GR00T's DROID_EEF_ROTATION_CORRECT);
* joint drift while holding the pose (should be ~0 for a position-controlled arm).

Usage (from the repo root, inside the gr00t container)::

    gr00t/eval/sim/MolmoSpaces/molmospaces_uv/.venv/bin/python scripts/rl/check_droid_spaces.py \
        --statistics /path/to/GR00T-N1.7-3B/statistics.json --tasks pick/0 pick/1 pick/2
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from gr00t.rl.envs.molmospaces import DROID_EEF_ROTATION_CORRECT, MolmoSpacesAdapter
import numpy as np


TAG = "oxe_droid_relative_eef_relative_joint"


def axis_rotations() -> list[np.ndarray]:
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            m = np.zeros((3, 3))
            for r, (c, s) in enumerate(zip(perm, signs)):
                m[r, c] = s
            if np.isclose(np.linalg.det(m), 1.0):
                mats.append(m)
    return mats


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--statistics", required=True, help="statistics.json of the GR00T checkpoint")
    ap.add_argument("--tasks", nargs="+", default=["pick/0", "pick/1", "pick/2"])
    ap.add_argument(
        "--benchmarks-json", default=None, help='JSON {"name": "/path/to/benchmark_dir"}'
    )
    ap.add_argument("--hold-steps", type=int, default=10)
    ap.add_argument("--eef-frame", default="arm_base", choices=["arm_base", "robot"])
    ap.add_argument("--tcp-offset-z", type=float, default=0.0)
    ap.add_argument("--out-dir", default="outputs/check_droid_spaces")
    args = ap.parse_args()

    stats = json.loads(Path(args.statistics).read_text())[TAG]["state"]
    benchmarks = json.loads(args.benchmarks_json) if args.benchmarks_json else None
    env = MolmoSpacesAdapter(
        benchmarks=benchmarks, eef_frame=args.eef_frame, tcp_offset_z=args.tcp_offset_z
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tcp_rots, states = [], []
    for task in args.tasks:
        obs, info = env.reset(task, seed=0)
        tcp_rots.append(env.tcp_in_arm_base()[:3, :3])
        print(f"\n=== {task}: '{obs['language']}'  success at reset: {info['success']}")
        for k, v in obs["video"].items():
            print(f"  video.{k}: {v.shape} {v.dtype} mean {v.mean():.1f}")
        hold = {
            "joint_position": obs["state"]["joint_position"],
            "gripper_position": obs["state"]["gripper_position"],
        }
        first = obs["state"]["joint_position"].copy()
        for _ in range(args.hold_steps):
            obs, info = env.step(hold)
        drift = np.abs(obs["state"]["joint_position"] - first).max()
        print(f"  joint drift after holding {args.hold_steps} steps: {drift:.4f} rad")
        states.append(obs["state"])
        try:
            from PIL import Image

            img = np.concatenate(
                [obs["video"]["exterior_image_1_left"], obs["video"]["wrist_image_left"]], axis=1
            )
            Image.fromarray(img).save(out / f"{task.replace('/', '_')}.png")
        except Exception as e:  # pragma: no cover
            print("  (could not save image:", e, ")")

    print("\n=== state vs DROID statistics (z = (x - mean) / std; in = within [q01, q99])")
    for key in ["eef_9d", "gripper_position", "joint_position"]:
        x = np.stack([s[key] for s in states])
        mean, std = np.asarray(stats[key]["mean"]), np.asarray(stats[key]["std"])
        q01, q99 = np.asarray(stats[key]["q01"]), np.asarray(stats[key]["q99"])
        z = (x - mean) / np.maximum(std, 1e-6)
        inside = ((x >= q01) & (x <= q99)).mean(0)
        print(f"  {key}:")
        print(f"    mean |z| per dim: {np.round(np.abs(z).mean(0), 2)}")
        print(f"    fraction in [q01,q99]: {np.round(inside, 2)}")
        if key == "eef_9d":
            print(f"    xyz here (mean over tasks): {np.round(x[:, :3].mean(0), 3)}")
            print(
                f"    DROID xyz mean {np.round(mean[:3], 3)}  q01 {np.round(q01[:3], 3)}  q99 {np.round(q99[:3], 3)}"
            )
            print(
                f"    offset to DROID mean (if a constant frame/TCP offset): {np.round(mean[:3] - x[:, :3].mean(0), 3)}"
            )

    print("\n=== EEF rotation convention: R_droid = R_tcp @ C (lower score = more typical)")
    mean = np.asarray(stats["eef_9d"]["mean"])[3:9]
    std = np.maximum(np.asarray(stats["eef_9d"]["std"])[3:9], 1e-6)
    scored = []
    for C in axis_rotations():
        r6 = np.stack([(R @ C)[:2, :].reshape(6) for R in tcp_rots])
        scored.append((float((((r6 - mean) / std) ** 2).mean()), C))
    scored.sort(key=lambda t: t[0])
    default = np.asarray(DROID_EEF_ROTATION_CORRECT, dtype=float)
    for rank, (score, C) in enumerate(scored[:5]):
        tag = "  <- current default" if np.allclose(C, default) else ""
        print(f"  #{rank + 1} score {score:7.2f}  C = {C.astype(int).tolist()}{tag}")
    d_score = next(s for s, C in scored if np.allclose(C, default))
    print(
        f"  current default score {d_score:.2f}. If another C fits much better, pass it as "
        "eef_rotation_correction in --env.backend-kwargs-json."
    )
    env.close()


if __name__ == "__main__":
    main()
