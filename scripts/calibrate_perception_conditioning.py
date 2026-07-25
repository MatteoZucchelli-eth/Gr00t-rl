# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Derive the perception-conditioning teacher-forcing noise from a closed-loop eval.

The perception-conditioned action model teacher-forces its "where" token on the
ground-truth label plus noise. That noise must match the error the perception
model ACTUALLY makes in the loop — noise smaller than reality teaches the policy
to trust perfect perception and it is brittle on the robot. This script measures
that error and prints the three flags to pass.

RE-RUN IT FOR EVERY NEW PERCEPTION CHECKPOINT: the numbers are a property of the
checkpoint, not of the task.

Input: the planner eval CSVs (columns pred_rel_*, act_rel_*, dist_target_actual,
rel_err_rot_deg), which record the planner's estimate of the tip->target vector
against the true one, in the drill-tip frame.

CRITICAL — those CSVs contain TWO models. The coarse model drives the approach
while the arm is moving; the perception model only fires when the arm PARKS to
accumulate detections. Pooling them measures the coarse model (its error is ~10x
larger) and would grossly over-noise training. The parked stretches are found from
the trace itself: `dist_target_actual` goes flat when the arm stops to accumulate
(--flat-thresh-mm over a 0.7 s window), and phase 0 is excluded as the coarse leg.

Within a stop, successive detections are averaged, so only the FIRST detection of
each stop is a single-frame measurement — which is what the new model does every
step. The script reports both.

The conversion from those metres back into the model's own units (processed-image
px and camera-frame depth) needs the camera intrinsics and the camera->tip
extrinsics. Both are RECOVERED FROM THE DATASET rather than assumed:

  focal length   the heatmap sidecar carries both `depth_z` (Z) and `ray_dist`
                 (sqrt(X^2+Y^2+Z^2)) for the same labelled pixel, so
                 (ray/Z)^2 - 1 = ((u-cx)^2 + (v-cy)^2) / f^2  solves for f.
  extrinsics     the sidecar's (u, v, depth_z) give the target in the camera
                 frame; `next.tip_target_9d` gives the same point in the tip
                 frame. Kabsch on the two point sets gives the rigid transform
                 (it IS rigid: the wrist camera and the tip share a link).

The result is a BEST CASE and is inflated before being recommended: the perception
model is only exercised here with a stationary camera and only over the near part
of the training range, whereas the new model runs it every frame, while moving,
out to the full fine-phase distance.

Usage:
    python scripts/calibrate_perception_conditioning.py \
        --eval-dir vla_eval_results/1500_perception_planner \
        --dataset-path ./g1_dataset
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


DIST_BINS = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16]


def recover_focal(labels: pd.DataFrame, cx: float, cy: float) -> float:
    """Solve (ray/Z)^2 - 1 = r_px^2 / f^2 for f (least squares through the origin)."""
    r2 = (labels["u"] - cx) ** 2 + (labels["v"] - cy) ** 2
    lhs = (labels["ray_dist"] / labels["depth_z"]) ** 2 - 1.0
    ok = (lhs > 1e-9) & (r2 > 1.0)
    inv_f2 = (lhs[ok] * r2[ok]).sum() / (r2[ok] ** 2).sum()
    return float(1.0 / np.sqrt(inv_f2))


def recover_extrinsics(cam_pts: np.ndarray, tip_pts: np.ndarray):
    """Kabsch: rigid (R, t) with tip ~= cam @ R.T + t. Returns (R, t, rms_residual_m)."""
    cm, tm = cam_pts.mean(0), tip_pts.mean(0)
    u, _, vt = np.linalg.svd((cam_pts - cm).T @ (tip_pts - tm))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    trans = tm - rot @ cm
    resid = np.linalg.norm(tip_pts - (cam_pts @ rot.T + trans), axis=1)
    return rot, trans, float(np.sqrt((resid**2).mean()))


def load_dataset_geometry(dataset_path: Path, max_episodes: int, cx: float, cy: float):
    label_files = sorted(
        glob.glob(str(dataset_path / "heatmap_labels" / "chunk-*" / "episode_*.parquet"))
    )
    if not label_files:
        raise FileNotFoundError(f"No heatmap_labels/ parquets under {dataset_path}")
    step = max(1, len(label_files) // max_episodes)
    sample = label_files[::step][:max_episodes]

    cols = [
        "frame_index",
        "u",
        "v",
        "depth_z",
        "ray_dist",
        "tip_target_dist",
        "fine_phase",
        "label_ok",
        "visible",
    ]
    labels, cam_pts, tip_pts, train_dist = [], [], [], []
    for label_path in sample:
        lab = pd.read_parquet(label_path, columns=cols)
        fine = lab[lab["fine_phase"] & lab["label_ok"]].dropna(subset=["tip_target_dist"])
        train_dist.append(fine["tip_target_dist"])
        vis = fine[fine["visible"]].dropna(subset=["u", "v", "depth_z", "ray_dist"])
        if vis.empty:
            continue
        labels.append(vis)
        data_path = Path(str(label_path).replace("heatmap_labels/", "data/"))
        if not data_path.exists():
            continue
        dat = pd.read_parquet(data_path, columns=["next.tip_target_9d"]).reset_index(
            names="frame_index"
        )
        merged = vis.merge(dat, on="frame_index")
        if merged.empty:
            continue
        tip = np.stack(merged["next.tip_target_9d"].to_numpy())[:, :3]
        keep = np.isfinite(tip).all(axis=1)
        z = merged["depth_z"].to_numpy()[keep]
        cam_pts.append(
            np.stack(
                [
                    (merged["u"].to_numpy()[keep] - cx) * z,
                    (merged["v"].to_numpy()[keep] - cy) * z,
                    z,
                ],
                axis=1,
            )
        )
        tip_pts.append(tip[keep])
    return (
        pd.concat(labels, ignore_index=True),
        np.concatenate(cam_pts),
        np.concatenate(tip_pts),
        pd.concat(train_dist, ignore_index=True),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--eval-dir", type=Path, required=True, help="Directory of planner eval CSVs.")
    ap.add_argument("--dataset-path", type=Path, default=Path("./g1_dataset"))
    ap.add_argument("--image-target-size", type=int, default=720)
    ap.add_argument("--image-crop-size", type=int, default=648)
    ap.add_argument("--camera-size", type=int, default=720, help="Raw wrist-camera edge, px.")
    ap.add_argument(
        "--max-episodes", type=int, default=60, help="Episodes sampled for the geometry fit."
    )
    ap.add_argument(
        "--flat-thresh-mm",
        type=float,
        default=0.5,
        help="Max tip->target travel over a 0.7 s window for the arm to count as parked.",
    )
    ap.add_argument(
        "--coarse-phase",
        type=float,
        default=0.0,
        help="Value of `phase` that is driven by the coarse model (excluded).",
    )
    args = ap.parse_args()

    cx = cy = args.camera_size / 2.0
    labels, cam_pts_scaled, tip_pts, train_dist = load_dataset_geometry(
        args.dataset_path, args.max_episodes, cx, cy
    )

    focal = recover_focal(labels, cx, cy)
    # cam_pts were built with X = (u-cx)*Z, needing the /f that was unknown until now.
    cam_pts = cam_pts_scaled.copy()
    cam_pts[:, 0] /= focal
    cam_pts[:, 1] /= focal
    rot, trans, resid = recover_extrinsics(cam_pts, tip_pts)

    crop_scale = args.image_crop_size / args.image_target_size
    print("=== recovered geometry (from the dataset, not assumed) ===")
    print(
        f"  focal length          {focal:.2f} px  @ {args.camera_size}px camera "
        f"(FOV {2 * np.degrees(np.arctan(cx / focal)):.1f} deg)"
    )
    print(f"  1 camera px           {1000 / focal:.3f} mm per metre of depth")
    print(
        f"  camera->tip fit       rigid, rms residual {resid * 1000:.2f} mm "
        f"over {len(cam_pts)} points"
    )
    print(f"  optical axis in tip frame  {np.round(rot[:, 2], 4)}")
    print(
        f"  processed px          1 camera px = {1 / crop_scale:.3f} processed px "
        f"(crop {args.image_crop_size}/{args.image_target_size})"
    )

    csvs = sorted(glob.glob(str(args.eval_dir / "*.csv")))
    if not csvs:
        raise FileNotFoundError(f"No CSVs in {args.eval_dir}")
    frames = []
    for path in csvs:
        run = pd.read_csv(path)
        run["scen"] = Path(path).stem
        pred = run[["pred_rel_x", "pred_rel_y", "pred_rel_z"]].to_numpy()
        # A fresh model call = the estimate actually changed (it is held in between).
        run["fresh"] = np.r_[True, np.abs(np.diff(pred, axis=0)).sum(axis=1) > 1e-12] & np.isfinite(
            pred
        ).all(axis=1)
        roll = run["dist_target_actual"].rolling(7, center=True)
        run["parked"] = (roll.max() - roll.min()) * 1000 < args.flat_thresh_mm
        run["stop_id"] = (run["parked"].astype(int).diff().fillna(0) != 0).cumsum()
        frames.append(run)
    ev = pd.concat(frames, ignore_index=True).dropna(subset=["pred_rel_x"])

    def to_camera(p_tip):
        return (p_tip - trans) @ rot

    def to_pixels(p_cam):
        return np.stack(
            [cx + focal * p_cam[:, 0] / p_cam[:, 2], cy + focal * p_cam[:, 1] / p_cam[:, 2]], axis=1
        )

    cam_pred = to_camera(ev[["pred_rel_x", "pred_rel_y", "pred_rel_z"]].to_numpy())
    cam_act = to_camera(ev[["act_rel_x", "act_rel_y", "act_rel_z"]].to_numpy())
    duv = (to_pixels(cam_pred) - to_pixels(cam_act)) / crop_scale  # processed px
    ev = ev.assign(du=duv[:, 0], dv=duv[:, 1], dz=cam_pred[:, 2] - cam_act[:, 2])

    # --- split the two models ---
    perception = ev[ev.parked & ev.fresh & (ev.phase != args.coarse_phase)].copy()
    perception["k"] = perception.groupby(["scen", "stop_id"]).cumcount()
    single = perception[perception.k == 0]  # one frame, no accumulation
    coarse = ev[(~ev.parked) & ev.fresh & (ev.phase == args.coarse_phase)]

    def report(x, label):
        e = np.concatenate([x.du, x.dv])
        print(
            f"  {label:34s} n={len(x):4d} | uv rms={np.sqrt((e**2).mean()):5.2f} "
            f"std={e.std():5.2f} px | depth rms={np.sqrt((x.dz**2).mean()) * 1000:5.2f} "
            f"std={x.dz.std() * 1000:5.2f} bias={x.dz.mean() * 1000:+6.2f} mm"
        )

    print("\n=== the two models in this eval ===")
    report(coarse, "COARSE, in motion (ignore)")
    report(perception, "perception, all stop detections")
    report(single, "perception, FIRST of each stop")
    if len(single) < 25:
        print(f"  NOTE: only {len(single)} single-frame samples — treat these as indicative.")
    print(
        f"  perception fired at tip distances "
        f"{perception.dist_target_actual.min() * 1000:.1f}-"
        f"{perception.dist_target_actual.max() * 1000:.1f} mm; training spans "
        f"{train_dist.min() * 1000:.1f}-{train_dist.max() * 1000:.1f} mm"
    )

    uv = float(np.sqrt((np.concatenate([single.du, single.dv]) ** 2).mean()))
    dz_std = float(single.dz.std())
    dz_bias = float(perception.dz.mean())
    ms = float((perception.dz**2).mean())
    rot_dev = float(
        np.radians(perception["rel_err_rot_deg"] - perception["rel_err_rot_deg"].median()).std()
    )

    print("\n=== depth error is mostly BIAS, which a zero-mean noise cannot reproduce ===")
    print(
        f"  bias {dz_bias * 1000:+.2f} mm vs spread {perception.dz.std() * 1000:.2f} mm "
        f"-> bias is {100 * dz_bias**2 / ms:.0f}% of the mean-square depth error"
    )

    # The measurement is a best case: stationary camera, and only over the near part of
    # the training range. Running perception every frame while moving is harder.
    margin = 3.0
    print("\n=== PASS THESE ===")
    print(
        f"  --perception-cond-noise-uv-px       {max(1.0, np.ceil(uv * margin)):.0f}     "
        f"(single-frame rms {uv:.2f} px x{margin:.0f} margin for motion blur and the "
        f"unmeasured far range)"
    )
    print(
        f"  --perception-cond-noise-depth-m     {dz_std:.3f}  (single-frame spread "
        f"{dz_std * 1000:.2f} mm, bias excluded)"
    )
    print(
        f"  --perception-cond-bias-depth-m      {dz_bias:.4f} (systematic {dz_bias * 1000:+.2f} mm)"
    )
    print(
        f"  --perception-cond-noise-normal      {rot_dev / np.sqrt(2):.2f}   "
        f"(rotation spread {np.degrees(rot_dev):.2f} deg; least reliable of the four)"
    )
    print(
        "\nIf you have the perception run's own `uv_err_px_metric` from wandb, prefer it "
        "for the\nuv figure: it covers every fine-phase frame, moving ones included, "
        "which this eval does not."
    )


if __name__ == "__main__":
    main()
