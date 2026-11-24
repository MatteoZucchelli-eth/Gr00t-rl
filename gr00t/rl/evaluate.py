"""Evaluate a GR00T checkpoint in remote simulators (no training).

Used for the zero-shot pilot (PLAN.md milestone 2: base N1.7 on every candidate task,
32 episodes each) and for evaluating any checkpoint on held-out tasks. Writes per-task
success rates to ``<output_dir>/eval.json`` and prints a summary table.

Example::

    python -m gr00t.rl.evaluate \
        --model-path nvidia/GR00T-N1.7-3B \
        --env.backend molmospaces \
        --env.python gr00t/eval/sim/MolmoSpaces/molmospaces_uv/.venv/bin/python \
        --env.num-envs 16 --episodes-per-task 32 \
        --tasks pick/0 pick/1 pick/2 --output-dir outputs/pilot_pick
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time

import torch
import tyro

from gr00t.rl.config import EnvConfig, expand_task_specs
from gr00t.rl.envs.client import RemoteVectorEnv
from gr00t.rl.flow_policy import FlowPolicy
from gr00t.rl.rollout import Collector, RolloutStats, cap_num_envs


@dataclass
class EvalConfig:
    model_path: str = "nvidia/GR00T-N1.7-3B"
    embodiment_tag: str = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
    tasks: list[str] = field(default_factory=list)
    episodes_per_task: int = 32
    execute_steps: int = 8
    mode: str = "ode"
    """ode (deployment) or sde (with --sigma)."""
    sigma: float = 0.1
    seed: int = 10_000_019
    """Default is disjoint from the training-collection seeds used by gr00t.rl.train."""
    output_dir: str = "outputs/eval"
    partial_every: int = 25
    """Write eval_partial.json (per-task counts so far) every n episodes."""
    use_wandb: bool = False
    """Log the summary and a per-task table (the Euler entrypoint enables it with WANDB_API_KEY)."""
    wandb_project: str = "groot-rl-transfer"
    wandb_entity: str | None = None
    run_name: str | None = None
    device: str = "cuda"
    env: EnvConfig = field(default_factory=EnvConfig)


def main(cfg: EvalConfig) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg.tasks = expand_task_specs(cfg.tasks)
    if not cfg.tasks:
        raise ValueError("--tasks is empty")
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    policy = FlowPolicy.from_pretrained(
        cfg.model_path, cfg.embodiment_tag, device=cfg.device, sigma=cfg.sigma
    )
    e = cfg.env
    num_envs = (
        cap_num_envs(e.num_envs, e.gpu_mem_per_env_gb, cfg.device, e.render_devices)
        if e.cap_envs_by_gpu_mem
        else e.num_envs
    )
    env = RemoteVectorEnv(
        num_envs,
        e.backend,
        host=e.host,
        base_port=e.base_port,
        spawn=e.spawn,
        python=e.python,
        render_devices=e.render_devices,
        backend_kwargs=json.loads(e.backend_kwargs_json),
        max_episode_steps=e.max_episode_steps,
        timeout_s=e.step_timeout_s,
        mem_fraction=e.mem_fraction,
        log_dir=str(out / "env_logs"),
    )
    try:
        collector = Collector(env, policy, learner=None, execute_steps=cfg.execute_steps)
        total = len(cfg.tasks) * cfg.episodes_per_task
        t_start = time.time()
        partial_path = out / "eval_partial.json"
        prev = RolloutStats()
        if partial_path.exists():
            d = json.loads(partial_path.read_text())
            same = (
                d.get("tasks") == cfg.tasks
                and d.get("episodes_per_task") == cfg.episodes_per_task
                and d.get("seed") == cfg.seed
                and d.get("mode") == cfg.mode
            )
            if same:
                prev = RolloutStats.from_dict(d["stats"])
                logging.info(
                    "resuming: %d/%d episodes already done (%s)",
                    sum(prev.episodes.values()),
                    total,
                    partial_path,
                )
            else:
                stale = partial_path.with_suffix(f".stale.{int(time.time())}.json")
                partial_path.rename(stale)
                logging.warning("eval_partial.json is from a different config; moved to %s", stale)

        def save_partial(stats) -> None:
            stats = prev.merged(stats)
            n = sum(stats.episodes.values())
            if n % cfg.partial_every and n != total:
                return
            stats.wall_time_s = prev.wall_time_s + (time.time() - t_start)
            partial = {
                "tasks": cfg.tasks,
                "episodes_per_task": cfg.episodes_per_task,
                "seed": cfg.seed,
                "mode": cfg.mode,
                "episodes_done": n,
                "episodes_total": total,
                "mean_success_so_far": stats.success_rate,
                "stats": stats.to_dict(),
            }
            tmp = out / "eval_partial.json.tmp"
            tmp.write_text(json.dumps(partial, indent=2))
            tmp.replace(partial_path)
            done_here = n - sum(prev.episodes.values())
            rate = done_here / max(time.time() - t_start, 1e-6)
            logging.info(
                "eval progress: %d/%d episodes, success so far %.3f, %.2f episodes/min, ETA %.1f h",
                n,
                total,
                stats.success_rate,
                60 * rate,
                (total - n) / max(rate, 1e-9) / 3600,
            )

        new_stats = collector.run(
            cfg.tasks,
            cfg.episodes_per_task,
            mode=cfg.mode,
            best_of_n=1,
            buffer=None,
            seed=cfg.seed,
            on_episode=save_partial,
            skip=dict(prev.episodes),
        )
        stats = prev.merged(new_stats)
    finally:
        env.close()
    result = {
        "model_path": cfg.model_path,
        "episodes_per_task": cfg.episodes_per_task,
        "mean_success": stats.success_rate,
        "per_task": stats.per_task(),
        "tasks_with_any_success": sum(1 for v in stats.per_task().values() if v > 0),
        "num_tasks": len(cfg.tasks),
        "env_steps": stats.env_steps,
        "wall_time_s": stats.wall_time_s,
        "worker_failures": stats.worker_failures,
    }
    (out / "eval.json").write_text(json.dumps(result, indent=2))
    if cfg.use_wandb:
        import dataclasses

        import wandb

        name = cfg.run_name or out.name
        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=name,
            id=name.replace("/", "_"),
            resume="allow",
            job_type="eval",
            config=dataclasses.asdict(cfg),
            dir=str(out),
        )
        run.summary.update({k: v for k, v in result.items() if not isinstance(v, dict)})
        table = wandb.Table(columns=["task", "success_rate"])
        for t, sr in sorted(result["per_task"].items()):
            table.add_data(t, sr)
        run.log({"per_task": table})
        run.finish()
    print(
        f"\nmean success {result['mean_success']:.3f} | tasks with >=1 success: "
        f"{result['tasks_with_any_success']}/{result['num_tasks']} | {stats.env_steps} env steps in {stats.wall_time_s:.0f}s"
    )
    for t, sr in sorted(result["per_task"].items()):
        print(f"  {t:40s} {sr:.3f}")
    return result


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
