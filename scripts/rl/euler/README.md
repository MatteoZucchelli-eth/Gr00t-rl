# Running GR00T RL on Euler

Same pattern as `submit_euler.sh` (Docker → Apptainer `.sif` → `tar.gz` on `/cluster/work`,
extracted to `$TMPDIR` per job, code bind-mounted from the synced repo). Every run is
described by a small preset in `rl_experiments/`, and one command submits it.

## One-time setup

Locally (repo root):

```bash
scripts/rl/euler/build_container.sh --upload   # docker build gr00t-rl -> groot-rl.tar.gz -> euler
scripts/rl/euler/sync.sh --assets              # code -> euler:~/groot-rl, + local MolmoSpaces assets
```

On Euler:

```bash
cd ~/groot-rl
cp .env.sample .env    # HF_TOKEN (required), WANDB_API_KEY (optional)
```

Rebuild/upload the container only when dependencies change; for code changes `sync.sh` is enough.
`--assets` is optional (jobs download missing MolmoSpaces assets themselves); the
`assets` preset prefetches scenes for a task list on Euler.

## Submitting

From `~/groot-rl` on Euler:

```bash
scripts/rl/euler/submit.sh rl_experiments/smoke.sh            # ~30 min end-to-end check
scripts/rl/euler/submit.sh rl_experiments/list_tasks.sh       # benchmark episode lists
scripts/rl/euler/submit.sh rl_experiments/pilot_zero_shot.sh  # M2: zero-shot gate
scripts/rl/euler/submit.sh rl_experiments/ogpo_arm_c.sh       # arm C (chained 24 h jobs)
scripts/rl/euler/submit.sh rl_experiments/bon_arm_e.sh        # arm E
scripts/rl/euler/submit.sh rl_experiments/libero_ogpo.sh      # M3 validation on LIBERO
```

`submit.sh` takes `--chain N` (override the preset's `CHAIN`), `--dry-run`, and any extra
`sbatch` flags (e.g. `--time=04:00:00`). It refuses presets that still contain a
`REPLACE_WITH_*` placeholder or reference an empty task list. Plain
`sbatch submit_rl_euler.sh rl_experiments/<preset>.sh` also works (single job).

Default resources (`submit_rl_euler.sh` header): 1× A100 80 GB, 32 CPUs, 256 GB RAM,
24 h, 150 GB `$TMPDIR`. Presets lower them via `SBATCH_ARGS`.

## Presets

A preset is a bash file:

```bash
MODE=train                      # smoke | assets | space_check | list_tasks | eval | train
RUN_NAME=ogpo_arm_c_seed0       # run dir name; resubmitting the same name RESUMES
CHAIN=6                         # chained 24 h jobs (submit.sh)
SBATCH_ARGS="--time=02:00:00"   # optional
ARGS=( --model-path /groot/runs/<B run> --collect.tasks @rl_experiments/splits/train.txt ... )
```

`ARGS` go straight to `python -m gr00t.rl.train` / `gr00t.rl.evaluate` (see
`gr00t/rl/config.py` for every option). Task lists accept ranges and files:
`open/0:40`, `@rl_experiments/splits/train.txt`. Benchmark short names: `pick`, `pnp`
(Droid mini benches, Korniak's), `pick_hard`, `pnp_hard`, `open`, `close` (procthor-10k
hard benches); any `<scene-set>/<Benchmark>` path works too, e.g. `ithor/FrankaOpenHardBench/3`.

Fill `rl_experiments/splits/train.txt` and `heldout.txt` after the pilot (PLAN.md §5) and
set `MODEL_PATH` in the arm presets to the warm-start checkpoint B: runs from
`submit_euler.sh` live in `/cluster/scratch/$USER/groot/runs/<run>`, which the RL job sees
as `/groot/runs/<run>`.

## Where things go

| What | Path on Euler |
|---|---|
| Container | `/cluster/work/mrl/$USER/containers/groot-rl.tar.gz` |
| Run dir (replay buffer, round checkpoints, metrics) | `/cluster/scratch/$USER/groot/rl_runs/<RUN_NAME>` |
| Results (final policy, metrics, eval json) | `/cluster/project/mrl/$USER/results/groot-rl/<RUN_NAME>` |
| HF weights, kernel caches | `/cluster/scratch/$USER/groot/cache` |
| MolmoSpaces assets | `/cluster/work/mrl/$USER/mlspaces/{assets,cache}` (bound at `/mlspaces`) |
| Logs | `~/groot-rl/logs/rl-<RUN_NAME>-<jobid>.{out,err}` |

Scratch is purged after ~2 weeks of inactivity: results that matter are copied to
`/cluster/project` when a run finishes. Fetch them locally with

```bash
scripts/rl/euler/fetch_run.sh ogpo_arm_c_seed0            # metrics + eval json
scripts/rl/euler/fetch_run.sh ogpo_arm_c_seed0 --policy   # + final_policy weights
```

`final_policy/` is a normal GR00T checkpoint: use it as `--base-model-path` for the
adaptation sweep with the existing finetuning pipeline.

## Notes

* Resume: each round writes `rounds/round_XXX/` (policy, learner, replay buffer). A job
  killed by the time limit loses at most the current round; the next chained job
  restarts it. A job whose run already has `final_eval.json` exits immediately.
* Rendering: MuJoCo uses EGL on the job's GPU (`apptainer --nv`, the image sets
  `NVIDIA_DRIVER_CAPABILITIES=all`). Set `MUJOCO_GL=osmesa` via `EXTRA_ENV="--env MUJOCO_GL=osmesa"` to fall back to CPU rendering.
* Local dry run of the exact job script (needs the image built and apptainer):
  `TMPDIR=/tmp/rljob CONTAINER_TAR=groot-rl.tar.gz GROOT_DIR=/tmp/rljob/groot RESULTS_ROOT=/tmp/rljob/results MLSPACES_DIR=$HOME/mlspaces EXTRA_ENV="--env SMOKE_TRAINABLE_REGEX=action_decoder" bash submit_rl_euler.sh rl_experiments/smoke.sh`
