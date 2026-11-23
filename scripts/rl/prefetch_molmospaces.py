"""Install MolmoSpaces assets for a set of tasks before training/evaluation.

Runs in the MolmoSpaces venv. Instantiates the resource manager (robots, benchmarks)
and resets each task once, which downloads its scene and objects. Doing this up front
keeps parallel env workers from all downloading the same assets at job start.

    $RL_MOLMOSPACES_PYTHON scripts/rl/prefetch_molmospaces.py --tasks pick/0:40 open/0:40
"""

from __future__ import annotations

import argparse
import time

from gr00t.rl.config import expand_task_specs


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tasks", nargs="*", default=[])
    args = ap.parse_args()

    from molmo_spaces.molmo_spaces_constants import get_resource_manager

    get_resource_manager()
    print("[prefetch] base resources installed", flush=True)
    tasks = expand_task_specs(args.tasks)
    if not tasks:
        return
    from gr00t.rl.envs.molmospaces import MolmoSpacesAdapter

    env = MolmoSpacesAdapter()
    failed = []
    t0 = time.time()
    for i, task in enumerate(tasks):
        try:
            env.reset(task, seed=0)
        except Exception as e:  # keep going; report at the end
            failed.append((task, f"{type(e).__name__}: {e}"))
        if (i + 1) % 10 == 0 or i + 1 == len(tasks):
            print(f"[prefetch] {i + 1}/{len(tasks)} tasks ({time.time() - t0:.0f}s)", flush=True)
    env.close()
    for task, err in failed:
        print(f"[prefetch] FAILED {task}: {err}")
    print(f"[prefetch] done, {len(failed)} failures")


if __name__ == "__main__":
    main()
