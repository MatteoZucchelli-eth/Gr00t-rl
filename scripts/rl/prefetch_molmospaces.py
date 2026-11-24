"""Install MolmoSpaces assets for a set of tasks before training/evaluation.

Runs in the MolmoSpaces venv. Instantiates the resource manager (robots, benchmarks)
and resets each task once, which downloads its scene and objects. Doing this up front
keeps parallel env workers from all downloading the same assets at job start.

    $RL_MOLMOSPACES_PYTHON scripts/rl/prefetch_molmospaces.py --tasks pick/0:40 open/0:40
"""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import socket
import time

from gr00t.rl.config import expand_task_specs


class TaskTimeout(BaseException):
    """BaseException so library code with broad ``except Exception`` cannot swallow it
    (MolmoSpaces re-raises errors during scene setup as HouseInvalidForTask); ``with``
    blocks, e.g. the asset cache lock, are still exited cleanly."""


def _on_alarm(signum, frame):
    raise TaskTimeout()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tasks", nargs="*", default=[])
    ap.add_argument(
        "--socket-timeout", type=float, default=120.0,
        help="seconds without data before a network read fails (stalled proxy connections)",
    )
    ap.add_argument(
        "--task-timeout", type=int, default=2400,
        help="hard limit per task in seconds (a scene archive can be several GB)",
    )
    ap.add_argument("--retries", type=int, default=2, help="extra passes over failed tasks")
    ap.add_argument(
        "--done-file", default=None,
        help="record finished tasks here and skip them on later runs (resume after a time limit)",
    )
    args = ap.parse_args()

    # Behind Euler's proxy a download can stall with the connection still open; without a
    # read timeout the process then waits forever. Layer 1: default timeout for sockets
    # created without one. Layer 2: a per-task alarm, in case a client disables timeouts.
    socket.setdefaulttimeout(args.socket_timeout)
    signal.signal(signal.SIGALRM, _on_alarm)

    from molmo_spaces.molmo_spaces_constants import get_resource_manager

    get_resource_manager()
    print("[prefetch] base resources installed", flush=True)
    pending = expand_task_specs(args.tasks)
    done_path = Path(args.done_file) if args.done_file else None
    if done_path is not None and done_path.exists():
        done = set(done_path.read_text().split())
        skipped = [t for t in pending if t in done]
        pending = [t for t in pending if t not in done]
        print(f"[prefetch] skipping {len(skipped)} tasks already done ({done_path})", flush=True)
    if not pending:
        print("[prefetch] done, 0 failures (nothing left)", flush=True)
        return
    from gr00t.rl.envs.molmospaces import MolmoSpacesAdapter

    t0 = time.time()
    failed: dict[str, str] = {}
    for attempt in range(args.retries + 1):
        if attempt:
            print(f"[prefetch] retry pass {attempt}: {len(pending)} tasks", flush=True)
        env = MolmoSpacesAdapter()
        failed = {}
        for i, task in enumerate(pending):
            t_task = time.time()
            print(f"[prefetch] start {task}", flush=True)
            signal.alarm(args.task_timeout)
            try:
                env.reset(task, seed=0)
            except TaskTimeout:
                failed[task] = f"timed out after {args.task_timeout}s"
                env = MolmoSpacesAdapter()  # the old one may be mid-download; start clean
            except Exception as e:  # keep going; retried and reported at the end
                failed[task] = f"{type(e).__name__}: {e}"
            finally:
                signal.alarm(0)
            if done_path is not None and task not in failed:
                with open(done_path, "a") as f:
                    f.write(task + "\n")
            dt = time.time() - t_task
            if task in failed or dt > 120:
                print(f"[prefetch]   {task}: {failed.get(task, 'ok')} ({dt:.0f}s)", flush=True)
            if (i + 1) % 10 == 0 or i + 1 == len(pending):
                print(
                    f"[prefetch] {i + 1}/{len(pending)} tasks ({time.time() - t0:.0f}s), "
                    f"{len(failed)} failed this pass",
                    flush=True,
                )
        pending = list(failed)
        if not pending:
            break
    for task, err in failed.items():
        print(f"[prefetch] FAILED {task}: {err}")
    print(f"[prefetch] done, {len(failed)} failures")


if __name__ == "__main__":
    main()
