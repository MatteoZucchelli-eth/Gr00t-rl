"""Configuration for batch-online OGPO / Best-of-N training of GR00T.

Defaults follow Korniak et al. 2026 (Table 5, OGPO column) where they apply, and
Patil et al. 2026 (OGPO) otherwise. See ``groot-rl-transfer/PLAN.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EnvConfig:
    """Simulator workers (each runs ``gr00t.rl.envs.server`` in the simulator's venv)."""

    backend: Literal["molmospaces", "libero", "fake"] = "molmospaces"
    """Environment adapter run by every worker."""
    python: str = "python"
    """Interpreter of the simulator venv used to launch workers (ignored if spawn=False)."""
    num_envs: int = 8
    """Number of parallel environment workers."""
    host: str = "127.0.0.1"
    base_port: int = 6100
    """Worker i listens on base_port + i."""
    spawn: bool = True
    """Launch workers as subprocesses; set False to connect to already running servers."""
    render_devices: list[int] = field(default_factory=lambda: [0])
    """GPU ids for EGL rendering, assigned round-robin to workers."""
    backend_kwargs_json: str = "{}"
    """JSON dict of extra keyword arguments for the backend adapter."""
    max_episode_steps: int = 450
    """Time limit in env steps (truncation, not termination)."""
    step_timeout_s: float = 600.0


@dataclass
class CollectConfig:
    tasks: list[str] = field(default_factory=list)
    """Training task ids, in the backend's task-id format."""
    eval_tasks: list[str] = field(default_factory=list)
    """Tasks evaluated after training (and optionally between rounds)."""
    episodes_per_task: int = 20
    """K: episodes collected per task and round (Korniak: 20)."""
    eval_episodes_per_task: int = 32
    execute_steps: int = 8
    """Actions executed open-loop from each predicted chunk before re-querying."""
    best_of_n: int = 32
    """Candidates per query during collection, ranked by the target critic (1 = off)."""
    sample_mode: Literal["sde", "ode"] = "sde"
    """How rollout candidates are sampled (OGPO collects with SDE samples of the EMA policy)."""
    eval_mode: Literal["ode", "sde"] = "ode"
    eval_best_of_n: int = 1
    """Korniak evaluates with N = 1."""
    plain_first_round: bool = True
    """Round 0 collects without Best-of-N (the critic is untrained)."""
    eval_every_rounds: int = 0
    """Run the eval suite every n rounds (0: only at the end)."""
    seed_offset: int = 0


@dataclass
class CriticConfig:
    horizon: int = 16
    """H: action-chunk length the critic scores and TD step size, in env steps."""
    hidden_dim: int = 1024
    depth: int = 2
    num_qs: int = 10
    num_vs: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 512
    target_decay: float = 0.95
    """Polyak decay of the target critic (Korniak: 0.95, i.e. tau = 0.05)."""
    aggregation: Literal["subsample", "min", "mean"] = "subsample"
    """Ensemble reduction for TD targets and Best-of-N (min over a random pair)."""
    bootstrap: Literal["policy", "sarsa"] = "policy"
    """policy: TD target averages Q_targ over num_vr actions sampled from the slow policy
    at s' (OGPO, Eq. 14). sarsa: Q -> r + gamma^H V(s'), V -> Q(s, a_buffer) (Korniak's
    Best-of-N critic; no VLM pass on s', much cheaper)."""
    num_vr: int = 16
    """Q-variance reduction: actions averaged in the TD target (bootstrap=policy)."""
    success_batch: bool = True
    """Second critic update per step on a batch drawn from successful episodes only."""
    grad_clip: float = 10.0
    updates_per_step: int = 1


@dataclass
class ActorConfig:
    lr: float = 2.5e-5
    weight_decay: float = 0.0
    batch_states: int = 512
    """Replay states per policy update (Korniak: batch 512)."""
    group_size: int = 32
    """G: SDE denoising trajectories sampled per state."""
    micro_batch: int = 512
    """Trajectories per forward/backward chunk (gradient accumulation)."""
    clip_eps: float = 0.01
    advantage: Literal["conservative", "mean"] = "conservative"
    """conservative: OGPO+CA (Eq. 4.5) over all ensemble members; mean: group-relative
    advantage of the aggregated target Q."""
    bc_coeff: float = 1.0
    """Weight of the success-buffer flow-matching loss (OGPO+)."""
    bc_batch: int = 512
    ema_decay: float = 0.995
    """Slow policy: PPO reference, rollout policy, and TD bootstrap policy."""
    sigma: float = 0.1
    """sigma_init (tapered schedule) or per-step std (constant schedule)."""
    noise_schedule: Literal["tapered", "constant"] = "tapered"
    logprob_reduce: Literal["sum", "mean"] = "sum"
    logprob_horizon: int | None = None
    """Restrict the likelihood to the first n chunk steps (None: full valid horizon)."""
    train_vlln: bool | None = None
    train_projector: bool | None = None
    train_diffusion_model: bool | None = None
    """None: use the checkpoint's tune_* flags. The VLM backbone is always frozen."""
    trainable_regex: str | None = None
    """Further restrict trained action-head parameters to names matching this regex."""
    grad_clip: float = 1.0
    warmup_steps: int = 100
    update_every: int = 1
    """Critic steps per policy update."""
    kl_every: int = 50
    """Log the path-space KL to the starting policy every n policy steps (0: off)."""
    kl_states: int = 32


@dataclass
class TrainConfig:
    model_path: str = "nvidia/GR00T-N1.7-3B"
    embodiment_tag: str = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
    output_dir: str = "outputs/ogpo"
    algo: Literal["ogpo", "bon"] = "ogpo"
    """ogpo: critic + policy updates (arm C). bon: critic only, policy frozen, Best-of-N
    at collection and evaluation (arm E)."""
    seed: int = 0
    rounds: int = 10
    """M: collection rounds (batch-online protocol, policy frozen during collection)."""
    steps_per_round: int = 1000
    gamma: float = 0.995
    buffer_capacity: int = 1_000_000
    frame_codec: Literal["jpeg", "png", "raw"] = "jpeg"
    frame_quality: int = 95
    num_workers: int = 8
    """CPU workers that decode frames and run the processor for training batches."""
    prefix_batch: int = 64
    """Batch size when computing missing prefix embeddings after a collection round."""
    encode_chunk: int = 64
    """Observations per backbone pass when encoding training batches."""
    log_every: int = 25
    device: str = "cuda"
    use_wandb: bool = False
    wandb_project: str = "groot-rl-transfer"
    run_name: str | None = None
    resume: bool = True
    """Resume from the latest round checkpoint in output_dir if present."""
    save_buffer: bool = True
    """Persist the replay buffer with each round checkpoint (needed to resume)."""
    kill_switch_min_gain: float = 0.10
    """Stop if training-task success has not improved by this much over round 0 ..."""
    kill_switch_after_rounds: int = 0
    """... after this many rounds (0 disables). PLAN.md: >=10 pts within ~200 GPU-h."""
    env: EnvConfig = field(default_factory=EnvConfig)
    collect: CollectConfig = field(default_factory=CollectConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
