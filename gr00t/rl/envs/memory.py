"""Host-memory accounting for env workers.

MolmoSpaces workers grow by a scene's worth of memory on (almost) every reset to a new
task, and the memory is not given back. On Euler, 20 workers reach the job's cgroup limit
after ~1.5 h, after which the kernel OOM-kills one worker every ~15 min. The client
recycles a worker at an episode boundary once it uses more than its share of the job's
memory (see ``RemoteVectorEnv.needs_recycle``).
"""

from __future__ import annotations

import os
from pathlib import Path


_GB = 2**30


def process_mem_gb(pid: int | None = None) -> float:
    """Proportional set size of a process (shared pages split among sharers), in GB.

    PSS sums correctly over processes that share libraries / EGL mappings; falls back to
    RSS where ``smaps_rollup`` is unavailable. Returns 0.0 if the process is gone.
    """
    base = Path(f"/proc/{pid if pid is not None else 'self'}")
    try:
        for line in (base / "smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) * 1024 / _GB
    except OSError:
        pass
    try:
        for line in (base / "status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024 / _GB
    except OSError:
        pass
    return 0.0


def _cgroup_limit_bytes() -> int | None:
    limits = []
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hier, controllers, rel = parts
        if hier == "0" and controllers == "":  # cgroup v2: walk up, the tightest limit wins
            p = Path("/sys/fs/cgroup") / rel.lstrip("/")
            while True:
                f = p / "memory.max"
                try:
                    v = f.read_text().strip()
                    if v != "max":
                        limits.append(int(v))
                except (OSError, ValueError):
                    pass
                if p == Path("/sys/fs/cgroup") or p == p.parent:
                    break
                p = p.parent
        elif "memory" in controllers.split(","):  # cgroup v1
            for p in (
                Path("/sys/fs/cgroup/memory") / rel.lstrip("/"),
                Path("/sys/fs/cgroup/memory"),
            ):
                try:
                    v = int((p / "memory.limit_in_bytes").read_text().strip())
                    if v < 2**60:
                        limits.append(v)
                        break
                except (OSError, ValueError):
                    pass
    return min(limits) if limits else None


def job_memory_limit_gb() -> float:
    """Memory this job may use: the tightest of the cgroup limit, the SLURM allocation
    (inherited by the container) and physical RAM."""
    cands = []
    cg = _cgroup_limit_bytes()
    if cg:
        cands.append(cg / _GB)
    env = os.environ
    if env.get("SLURM_MEM_PER_NODE", "").isdigit():
        cands.append(int(env["SLURM_MEM_PER_NODE"]) / 1024)
    elif env.get("SLURM_MEM_PER_CPU", "").isdigit():
        cpus = env.get("SLURM_CPUS_PER_TASK") or env.get("SLURM_CPUS_ON_NODE") or ""
        if cpus.isdigit():
            cands.append(int(env["SLURM_MEM_PER_CPU"]) * int(cpus) / 1024)
    try:
        cands.append(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / _GB)
    except (ValueError, OSError):
        pass
    return min(cands) if cands else float("inf")
