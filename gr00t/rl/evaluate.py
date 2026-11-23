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

import torch
import tyro

from gr00t.rl.config import EnvConfig, expand_task_specs
from gr00t.rl.envs.client import RemoteVectorEnv
from gr00t.rl.flow_policy import FlowPolicy
from gr00t.rl.rollout import Collector


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
    env = RemoteVectorEnv(
        e.num_envs,
        e.backend,
        host=e.host,
        base_port=e.base_port,
        spawn=e.spawn,
        python=e.python,
        render_devices=e.render_devices,
        backend_kwargs=json.loads(e.backend_kwargs_json),
        max_episode_steps=e.max_episode_steps,
        timeout_s=e.step_timeout_s,
        log_dir=str(out / "env_logs"),
    )
    try:
        collector = Collector(env, policy, learner=None, execute_steps=cfg.execute_steps)
        stats = collector.run(
            cfg.tasks, cfg.episodes_per_task, mode=cfg.mode, best_of_n=1, buffer=None, seed=cfg.seed
        )
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
