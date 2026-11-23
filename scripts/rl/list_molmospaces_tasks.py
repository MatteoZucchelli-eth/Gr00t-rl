"""List MolmoSpaces benchmark episodes (task ids) for designing family splits.

Runs in the MolmoSpaces venv. Prints the number of episodes per benchmark and, with
``--detail``, one line per episode: task id, scene, and the object/fixture involved.
``--write DIR`` writes ``DIR/<benchmark>.txt`` (one task id per line), usable as
``--collect.tasks @DIR/open.txt`` or ``--tasks @DIR/open.txt``.

    gr00t/eval/sim/MolmoSpaces/molmospaces_uv/.venv/bin/python scripts/rl/list_molmospaces_tasks.py \
        --benchmarks pick pnp open close --detail
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def describe(ep) -> str:
    """Best-effort one-line summary of a benchmark episode (schema varies by task type)."""
    d = ep.model_dump() if hasattr(ep, "model_dump") else dict(getattr(ep, "__dict__", {}))
    parts = []
    for key in ("scene_dataset", "house_index"):
        if key in d:
            parts.append(f"{key}={d[key]}")
    text = json.dumps(d, default=str)
    for key in ("pickup_obj_name", "place_receptacle_name", "articulated_obj_name", "object_name"):
        i = text.find(f'"{key}"')
        if i >= 0:
            parts.append(text[i : i + 80].split(",")[0])
    return "  ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--benchmarks", nargs="+", default=["pick", "pnp", "pick_hard", "pnp_hard", "open", "close"]
    )
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--write", default=None, help="directory for <benchmark>.txt task lists")
    args = ap.parse_args()

    from gr00t.rl.envs.molmospaces import MolmoSpacesAdapter

    env = MolmoSpacesAdapter()
    for bench in args.benchmarks:
        try:
            eps = env.episodes(bench)
        except Exception as e:  # missing benchmark etc.
            print(f"{bench:12s} ERROR {e}")
            continue
        print(f"{bench:12s} {len(eps):5d} episodes  ({env.benchmark_dir(bench)})")
        if args.detail:
            for i, ep in enumerate(eps):
                print(f"    {bench}/{i:<5d} {describe(ep)}")
        if args.write:
            out = Path(args.write)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{bench}.txt").write_text("".join(f"{bench}/{i}\n" for i in range(len(eps))))


if __name__ == "__main__":
    main()
