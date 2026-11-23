"""GPU smoke test of the RL adapter on a real GR00T N1.7 checkpoint.

Checks (prints PASS/FAIL per item, exits non-zero on failure):
  1. the refactored action head reproduces the pre-refactor ``get_action`` exactly
     (old head source taken from git ref ``--old-ref``);
  2. ``FlowPolicy.sample(mode="ode")`` matches ``get_action`` given the same noise;
  3. SDE log-probs recomputed by ``traj_log_prob`` match those from sampling (ratio ~ 1);
  4. decode -> absolute actions -> re-encode round-trips (BC / critic targets);
  5. the BC loss backpropagates into the trainable parameters;
  6. one tiny OGPO update runs and starts at ratio 1.

Usage (inside the gr00t container, repo root)::

    python scripts/rl/smoke_test_policy.py --model-path nvidia/GR00T-N1.7-3B
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
import time

from gr00t.rl.config import TrainConfig
from gr00t.rl.flow_policy import FlowPolicy
from gr00t.rl.ogpo import OGPOLearner
import numpy as np
import torch


FAILED = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILED.append(name)


def fake_droid_observation(policy: FlowPolicy, rng: np.random.Generator, language: str) -> dict:
    T = len(policy.video_delta_indices)
    video = {
        k: rng.integers(0, 255, size=(T, 180, 320, 3), dtype=np.uint8) for k in policy.camera_keys
    }
    stats = policy.processor.state_action_processor.norm_params[policy.embodiment_tag.value][
        "state"
    ]
    state = {}
    for k in policy.state_keys:
        p = stats[k]
        lo = np.asarray(p["min"], dtype=np.float32).reshape(-1)
        hi = np.asarray(p["max"], dtype=np.float32).reshape(-1)
        state[k] = (lo + (hi - lo) * rng.uniform(0.3, 0.7, size=lo.shape)).astype(np.float32)[
            None, :
        ]
    if "eef_9d" in state:  # valid rotation
        state["eef_9d"][0, 3:9] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    return {"video": video, "state": state, "language": {policy.language_key: [language]}}


def load_old_head_class(ref: str):
    src = subprocess.check_output(
        ["git", "-c", "safe.directory=*", "show", f"{ref}:gr00t/model/gr00t_n1d7/gr00t_n1d7.py"],
        text=True,
    )
    src = "\n".join(
        line
        for line in src.splitlines()
        if not line.startswith(("AutoConfig.register", "AutoModel.register"))
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("old_gr00t_n1d7", f.name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Gr00tN1d7ActionHead


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="nvidia/GR00T-N1.7-3B")
    ap.add_argument("--embodiment-tag", default="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT")
    ap.add_argument("--old-ref", default="main", help="git ref with the pre-refactor action head")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument(
        "--train-diffusion-model",
        action="store_true",
        help="also train the DiT (needs more memory)",
    )
    ap.add_argument(
        "--trainable-regex",
        default=r"action_decoder",
        help="restrict trained head params (default keeps a 16 GB GPU within memory); '' for all",
    )
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    from huggingface_hub import snapshot_download

    path = args.model_path
    if not path.startswith("/") and not path.startswith("."):
        path = snapshot_download(path)
    t0 = time.time()
    policy = FlowPolicy.from_pretrained(
        path,
        args.embodiment_tag,
        device="cuda",
        sigma=0.1,
        train_diffusion_model=args.train_diffusion_model,
        train_vlln=args.train_diffusion_model,
        train_projector=True,
        trainable_regex=args.trainable_regex or None,
    )
    n_train = sum(p.numel() for p in policy.trainable_parameters())
    n_head = sum(p.numel() for p in policy.head.parameters())
    print(
        f"loaded in {time.time() - t0:.1f}s: horizon {policy.model_horizon} (valid {policy.valid_horizon}), "
        f"action dim {policy.action_dim}/{policy.max_action_dim}, state dim {policy.state_dim}, K={policy.num_steps}, "
        f"prefix {policy.prefix_dim}, head params {n_head / 1e6:.0f}M (trainable {n_train / 1e6:.1f}M), "
        f"dropout layers removed {policy.num_dropout_removed}, video deltas {policy.video_delta_indices}"
    )
    rng = np.random.default_rng(0)
    obs = [fake_droid_observation(policy, rng, "pick up the red cup") for _ in range(args.batch)]
    t0 = time.time()
    collated = policy.prepare_inputs(obs)
    t_prep = time.time() - t0
    t0 = time.time()
    feats = policy.encode_prepared(collated)
    torch.cuda.synchronize()
    print(
        f"prepare {t_prep * 1000 / args.batch:.0f} ms/obs, encode {(time.time() - t0) * 1000 / args.batch:.0f} ms/obs, "
        f"tokens {feats.backbone_output['backbone_features'].shape[1]}"
    )
    check("prefix embedding shape", tuple(feats.prefix.shape) == (args.batch, policy.prefix_dim))

    B, Hm, Dm = args.batch, policy.model_horizon, policy.max_action_dim
    noise = torch.randn(B, Hm, Dm, device="cuda", generator=torch.Generator("cuda").manual_seed(0))

    # 1 + 2: refactor equivalence and ODE path.
    model = policy.model
    inputs = {k: v for k, v in collated.items()}
    backbone_inputs, action_inputs = model.prepare_input(inputs)
    with torch.no_grad():
        bo = model.backbone(backbone_inputs)

    def run_get_action(head):
        orig = torch.randn

        def fixed(*a, **kw):
            return noise.to(kw.get("dtype", torch.float32)).clone()

        torch.randn = fixed
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = head.get_action(
                    type(bo)(data=dict(bo.items())),
                    type(action_inputs)(data=dict(action_inputs.items())),
                )
        finally:
            torch.randn = orig
        return out["action_pred"].float()

    new_actions = run_get_action(policy.head)
    try:
        OldHead = load_old_head_class(args.old_ref)
        old_head = OldHead(model.config).to("cuda")
        old_head.load_state_dict(policy.head.state_dict())
        old_head.to(torch.bfloat16)
        for (n, p_new), p_old in zip(policy.head.named_parameters(), old_head.parameters()):
            p_old.data = p_old.data.to(p_new.dtype)
        old_head.eval()
        from gr00t.rl.flow_policy import disable_dropout

        disable_dropout(old_head)
        old_actions = run_get_action(old_head)
        diff = (old_actions - new_actions).abs().max().item()
        check(
            "refactored get_action == pre-refactor get_action",
            diff == 0.0,
            f"(max |diff| {diff:.2e})",
        )
        del old_head
    except (subprocess.CalledProcessError, FileNotFoundError):
        # No git history (e.g. code rsynced to Euler without .git): verified locally.
        print("[SKIP] refactored get_action == pre-refactor get_action (no git history)")
    ode = policy.sample(feats, mode="ode", init_noise=noise).actions
    vh, ad = policy.valid_horizon, policy.action_dim
    d = (ode[:, :vh, :ad] - new_actions[:, :vh, :ad]).abs().max().item()
    check(
        "FlowPolicy ODE == get_action (same noise)",
        d < 5e-2,
        f"(max |diff| {d:.2e}, valid dims; bf16 vs fp32 state)",
    )

    # 3: SDE log-prob consistency.
    out = policy.sample(feats, mode="sde", repeats=args.group, return_trajectory=True)
    with torch.no_grad():
        lp = policy.traj_log_prob(feats, out.trajectory, repeats=args.group)
    dlp = (lp.sum(0) - out.log_probs.sum(0)).abs().max().item()
    check(
        "SDE log-prob recompute matches sampling",
        dlp < 1.0,
        f"(max |d log p| {dlp:.3f} over {int(policy.logprob_mask.sum())} dims x K)",
    )
    with torch.no_grad():
        lp2 = policy.traj_log_prob(feats, out.trajectory, repeats=args.group)
    d2 = (lp2 - lp).abs().max().item()
    check("traj_log_prob deterministic (no dropout)", d2 == 0.0, f"(max diff {d2:.2e})")
    spread = (
        (out.actions[:, :vh, :ad] - ode.repeat_interleave(args.group, 0)[:, :vh, :ad]).std().item()
    )
    print(f"SDE vs ODE action spread (std of difference): {spread:.4f}")

    # 4: decode / re-encode round trip.
    states = [o["state"] for o in obs]
    decoded = policy.decode_actions(ode, states)
    re = np.stack([policy.encode_action_chunk(s, a) for s, a in zip(states, decoded)])
    orig = np.clip(ode[:, :vh, :ad].cpu().numpy(), -1, 1)
    bounds = np.cumsum([0] + policy.action_dims)
    per_key = {
        k: float(np.abs(re - orig)[..., bounds[i] : bounds[i + 1]].max())
        for i, k in enumerate(policy.action_keys)
    }
    print("round-trip max |diff| per action key:", {k: f"{v:.2e}" for k, v in per_key.items()})
    executed = [k for k in policy.action_keys if k != "eef_9d"]
    rt = max(per_key[k] for k in executed)
    check(
        "decode -> encode round trip (executed keys)",
        rt < 1e-3,
        f"(max |diff| {rt:.2e} on {executed})",
    )
    print("decoded action keys/shapes:", {k: v.shape for k, v in decoded[0].items()})
    ns = policy.normalize_state(states[0])
    check("normalized state shape", ns.shape == (policy.state_dim,), str(ns.shape))

    # 5: BC loss backward.
    policy.model.zero_grad(set_to_none=True)
    bc = policy.bc_loss(feats, torch.as_tensor(re, device="cuda"))
    bc.backward()
    grads = [p.grad for p in policy.trainable_parameters() if p.grad is not None]
    gnorm = torch.sqrt(sum((g.float() ** 2).sum() for g in grads)).item() if grads else 0.0
    check(
        "BC loss backprop",
        np.isfinite(bc.item()) and gnorm > 0,
        f"(loss {bc.item():.4f}, grad norm {gnorm:.3e})",
    )
    policy.model.zero_grad(set_to_none=True)

    # 6: one OGPO update.
    cfg = TrainConfig()
    cfg.critic.horizon = 16
    cfg.critic.hidden_dim = 256
    cfg.critic.num_qs = 4
    cfg.actor.group_size = args.group
    cfg.actor.micro_batch = args.group
    cfg.actor.warmup_steps = 0
    learner = OGPOLearner(
        policy, cfg, policy.prefix_dim, policy.state_dim, policy.action_dim, "cuda"
    )
    state_t = torch.as_tensor(np.stack([policy.normalize_state(s) for s in states]), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    info = learner.update_policy(
        feats, feats.prefix, state_t, feats, torch.as_tensor(re, device="cuda")
    )
    torch.cuda.synchronize()
    check(
        "OGPO update starts at ratio 1",
        abs(info["actor/ratio_mean"].item() - 1) < 1e-3,
        f"(ratio mean {info['actor/ratio_mean'].item():.6f}, max {info['actor/ratio_max'].item():.6f})",
    )
    print(
        f"OGPO update: {time.time() - t0:.2f}s for {B}x{args.group} trajectories, peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
    )
    print({k: round(float(v), 5) for k, v in info.items()})

    print("\nFAILED:" if FAILED else "\nALL CHECKS PASSED", FAILED if FAILED else "")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
