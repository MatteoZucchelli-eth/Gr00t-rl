"""GR00T N1.7 flow-matching policy, adapted for critic-guided RL (OGPO / Best-of-N).

What this adds on top of ``Gr00tN1d7``:

* ``encode``: run the (frozen) VLM backbone once per observation and keep its outputs, so
  many action samples can share one backbone pass. Also returns the critic's
  observation embedding: the backbone tokens mean-pooled over valid positions.
* ``sample``: K-step denoising, deterministic (ODE, as at deployment) or stochastic
  (marginal-preserving SDE, see ``gr00t.rl.sde``), returning the full denoising
  trajectory and per-step Gaussian log-likelihoods.
* ``traj_log_prob``: per-step log-likelihoods of a given trajectory under the current
  parameters, differentiable w.r.t. the trainable action-head parameters.
* ``bc_loss``: the standard flow-matching loss on (normalized) action chunks.
* ``decode_actions`` / ``encode_action_chunk`` / ``normalize_state``: conversions
  between the model's normalized, relative action space and absolute robot actions.
* ``EMAPolicyParams``: a slowly updated copy of the trainable parameters (OGPO's
  reference policy for the PPO ratio and the rollout policy), swapped in on demand.

Dropout is disabled in the action head (it would corrupt the likelihood ratio, see
pi-RL App. H), and the VLM is frozen, matching Korniak et al.'s OGPO setup. Trainable
action-head parameters are held in float32; compute runs under bfloat16 autocast.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from gr00t.rl.sde import gaussian_log_prob, sde_step_params


@dataclass
class PolicyFeatures:
    """Backbone outputs for a batch of observations (frozen VLM, no grad)."""

    backbone_output: BatchFeature  # raw backbone outputs (before vlln)
    action_input: BatchFeature  # state, embodiment_id (+ action_mask)
    prefix: torch.Tensor  # [B, P] float32 critic embedding

    @property
    def batch_size(self) -> int:
        return int(self.prefix.shape[0])

    def index(self, idx: torch.Tensor | slice) -> "PolicyFeatures":
        return PolicyFeatures(
            backbone_output=_map_batch(self.backbone_output, lambda t: t[idx]),
            action_input=_map_batch(self.action_input, lambda t: t[idx]),
            prefix=self.prefix[idx],
        )

    def repeat_interleave(self, n: int) -> "PolicyFeatures":
        if n == 1:
            return self
        return PolicyFeatures(
            backbone_output=_map_batch(self.backbone_output, lambda t: t.repeat_interleave(n, 0)),
            action_input=_map_batch(self.action_input, lambda t: t.repeat_interleave(n, 0)),
            prefix=self.prefix.repeat_interleave(n, 0),
        )


@dataclass
class SampleOutput:
    actions: torch.Tensor  # [B, H, D] final (normalized) actions, float32
    trajectory: torch.Tensor | None  # [K + 1, B, H, D] denoising states x_0 (noise) .. x_K
    log_probs: torch.Tensor | None  # [K, B] per-step log-likelihoods (SDE only)


def _map_batch(bf: BatchFeature, fn) -> BatchFeature:
    data = {}
    for k, v in bf.items():
        data[k] = fn(v) if isinstance(v, torch.Tensor) and v.dim() > 0 else v
    return BatchFeature(data=data)


def _copy_batch(bf: BatchFeature) -> BatchFeature:
    """Shallow copy: the action head reassigns keys in place (e.g. vlln'd features)."""
    return BatchFeature(data=dict(bf.items()))


def disable_dropout(module: nn.Module) -> int:
    """Replace every ``nn.Dropout`` with ``nn.Identity``; returns how many were replaced."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Dropout):
            setattr(module, name, nn.Identity())
            count += 1
        else:
            count += disable_dropout(child)
    return count


class FlowPolicy:
    """RL view of a GR00T N1.7 checkpoint for one embodiment."""

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        embodiment_tag: Any,
        *,
        sigma: float = 0.1,
        noise_schedule: str = "tapered",
        logprob_reduce: str = "sum",
        logprob_horizon: int | None = None,
        train_vlln: bool | None = None,
        train_projector: bool | None = None,
        train_diffusion_model: bool | None = None,
        trainable_regex: str | None = None,
        autocast_dtype: torch.dtype | None = torch.bfloat16,
    ):
        from gr00t.data.embodiment_tags import EmbodimentTag

        if isinstance(embodiment_tag, str):
            embodiment_tag = EmbodimentTag.resolve(embodiment_tag)
        self.model = model
        self.head = model.action_head
        self.processor = processor
        self.embodiment_tag = embodiment_tag
        self.sigma = float(sigma)
        self.noise_schedule = noise_schedule
        self.logprob_reduce = logprob_reduce
        self.autocast_dtype = autocast_dtype
        self.device = next(model.parameters()).device

        modality = processor.get_modality_configs()[embodiment_tag.value]
        self.modality_configs = {k: v for k, v in modality.items() if k != "rl_info"}
        self.camera_keys = list(self.modality_configs["video"].modality_keys)
        self.video_delta_indices = list(self.modality_configs["video"].delta_indices)
        self.state_keys = list(self.modality_configs["state"].modality_keys)
        self.action_keys = list(self.modality_configs["action"].modality_keys)
        self.language_key = self.modality_configs["language"].modality_keys[0]
        sap = processor.state_action_processor
        norm = sap.norm_params[embodiment_tag.value]
        self.state_dims = [int(norm["state"][k]["dim"].item()) for k in self.state_keys]
        self.action_dims = [int(norm["action"][k]["dim"].item()) for k in self.action_keys]
        self.action_dim = sum(self.action_dims)  # valid (unpadded) action dims
        self.state_dim = sap.get_state_dim(embodiment_tag.value, include_sincos_expansion=True)
        self.valid_horizon = len(self.modality_configs["action"].delta_indices)
        self.model_horizon = int(self.head.action_horizon)
        self.max_action_dim = int(self.head.action_dim)
        self.num_steps = int(self.head.num_inference_timesteps)
        self.prefix_dim = int(model.config.backbone_embedding_dim)

        # Dims that are sampled stochastically and enter the likelihood.
        mask = torch.zeros(self.model_horizon, self.max_action_dim)
        lp_h = (
            self.valid_horizon
            if logprob_horizon is None
            else min(logprob_horizon, self.valid_horizon)
        )
        mask[:lp_h, : self.action_dim] = 1.0
        self.logprob_mask = mask.to(self.device)
        # Dims supervised by the BC loss (full valid chunk).
        bc_mask = torch.zeros(self.model_horizon, self.max_action_dim)
        bc_mask[: self.valid_horizon, : self.action_dim] = 1.0
        self.bc_mask = bc_mask.to(self.device)

        self._configure_trainable(
            train_vlln, train_projector, train_diffusion_model, trainable_regex
        )

    # ------------------------------------------------------------------ setup

    @classmethod
    def from_pretrained(
        cls, model_path: str, embodiment_tag: str, device: str | int = "cuda", **kwargs
    ) -> "FlowPolicy":
        """Load model + processor the same way ``Gr00tPolicy`` does."""
        from transformers import AutoModel, AutoProcessor

        import gr00t.model  # noqa: F401  (registers the model classes)

        model_dir = Path(model_path)
        if not model_dir.is_dir():  # Hugging Face repo id, e.g. nvidia/GR00T-N1.7-3B
            from huggingface_hub import snapshot_download

            model_dir = Path(snapshot_download(str(model_path)))
        model = AutoModel.from_pretrained(model_dir)
        model.to(device=device, dtype=torch.bfloat16)
        processor_dir = (
            model_dir / "processor"
            if (model_dir / "processor").is_dir()
            and not (model_dir / "processor_config.json").exists()
            else model_dir
        )
        processor = AutoProcessor.from_pretrained(processor_dir)
        processor.eval()
        return cls(model, processor, embodiment_tag, **kwargs)

    def _configure_trainable(
        self, train_vlln, train_projector, train_diffusion_model, trainable_regex=None
    ) -> None:
        head = self.head
        if getattr(head, "use_tip_target_head", False):
            raise NotImplementedError(
                "RL fine-tuning of checkpoints with the tip-target head is not supported."
            )
        cfg = self.model.config
        train_vlln = cfg.tune_vlln if train_vlln is None else train_vlln
        train_projector = cfg.tune_projector if train_projector is None else train_projector
        train_diffusion_model = (
            cfg.tune_diffusion_model if train_diffusion_model is None else train_diffusion_model
        )

        self.model.requires_grad_(False)
        self.model.eval()  # no dropout anywhere; eval() also disables functional dropout
        self.num_dropout_removed = disable_dropout(head)
        head.set_trainable_parameters(train_projector, train_diffusion_model, train_vlln)
        self.model.backbone.requires_grad_(False)
        if trainable_regex:
            import re

            pattern = re.compile(trainable_regex)
            for name, p in head.named_parameters():
                if p.requires_grad and not pattern.search(name):
                    p.requires_grad_(False)
        # fp32 master weights for everything we train; the frozen rest stays bf16.
        for p in head.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.head.parameters() if p.requires_grad]

    def named_trainable_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(n, p) for n, p in self.head.named_parameters() if p.requires_grad]

    def _autocast(self):
        if self.autocast_dtype is None or self.device.type != "cuda":
            return torch.autocast(device_type=self.device.type, enabled=False)
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    # ------------------------------------------------------------------ inputs

    def prepare_inputs(self, observations: Sequence[dict]) -> dict:
        """Process unbatched observations into collated CPU model inputs.

        Each observation: ``{"video": {cam: uint8 [T,H,W,C]}, "state": {key: f32 [T,D]},
        "language": {language_key: [str]}}``. Safe to call in DataLoader workers.
        """
        return prepare_inputs(self.processor, self.embodiment_tag, self.language_key, observations)

    @torch.no_grad()
    def encode_prepared(self, collated: dict | Sequence[dict]) -> PolicyFeatures:
        """Backbone pass on collated inputs (or a list of collated chunks, concatenated)."""
        if not isinstance(collated, dict):
            return cat_features([self.encode_prepared(c) for c in collated])
        inputs = {k: v for k, v in collated.items()}
        backbone_inputs, action_inputs = self.model.prepare_input(inputs)
        with self._autocast():
            backbone_output = self.model.backbone(backbone_inputs)
        prefix = masked_mean(
            backbone_output["backbone_features"].float(),
            backbone_output.get("backbone_attention_mask", None),
        )
        # Keep only what the action head reads; e.g. pixel_values is indexed by image
        # patch, not by sample, and must not be sliced or repeated with the batch.
        action_input = BatchFeature(
            data={k: action_inputs[k] for k in ("state", "embodiment_id") if k in action_inputs}
        )
        backbone_output = BatchFeature(
            data={
                k: backbone_output[k]
                for k in ("backbone_features", "backbone_attention_mask", "image_mask")
                if k in backbone_output
            }
        )
        return PolicyFeatures(
            backbone_output=backbone_output, action_input=action_input, prefix=prefix
        )

    def encode(self, observations: Sequence[dict], chunk_size: int | None = None) -> PolicyFeatures:
        if chunk_size is None or len(observations) <= chunk_size:
            return self.encode_prepared(self.prepare_inputs(observations))
        chunks = [observations[i : i + chunk_size] for i in range(0, len(observations), chunk_size)]
        return self.encode_prepared([self.prepare_inputs(c) for c in chunks])

    # ------------------------------------------------------------------ denoising

    def _condition(self, feats: PolicyFeatures, repeats: int = 1):
        """vlln'd VL tokens and state-stream tokens (differentiable if those modules train).

        Computed once per state, then repeated ``repeats`` times along the batch (so a
        group of samples per state shares one conditioning pass).
        """
        bo = _copy_batch(feats.backbone_output)
        ai = _copy_batch(feats.action_input)
        enc = self.head._encode_features(bo, ai)
        vl = enc.backbone_features
        sa_state, _ = self.head.build_denoising_condition(vl, enc.state_features, bo)
        emb_id = ai.embodiment_id
        if repeats > 1:
            vl = vl.repeat_interleave(repeats, 0)
            sa_state = sa_state.repeat_interleave(repeats, 0)
            emb_id = emb_id.repeat_interleave(repeats, 0)
            bo = _map_batch(bo, lambda t: t.repeat_interleave(repeats, 0))
        return vl, sa_state, bo, emb_id

    def _velocity(self, cond, x: torch.Tensor, tau: float) -> torch.Tensor:
        vl, sa_state, bo, emb_id = cond
        v = self.head.predict_velocity(x.to(vl.dtype), tau, vl, sa_state, emb_id, bo)
        return v.float()

    def sample(
        self,
        feats: PolicyFeatures,
        mode: str = "ode",
        return_trajectory: bool = False,
        repeats: int = 1,
        generator: torch.Generator | None = None,
        init_noise: torch.Tensor | None = None,
    ) -> SampleOutput:
        """Sample ``repeats`` action chunks per row of ``feats`` (state-major order).

        ``mode="ode"`` reproduces ``get_action`` (Euler integration); ``mode="sde"``
        uses the marginal-preserving SDE and also returns per-step log-likelihoods.
        ``init_noise`` optionally fixes the starting noise [B*repeats, H, D].
        """
        if mode not in ("ode", "sde"):
            raise ValueError(f"mode must be 'ode' or 'sde', got {mode!r}")
        K = self.num_steps
        dt = 1.0 / K
        B = feats.batch_size * repeats
        with torch.no_grad(), self._autocast():
            cond = self._condition(feats, repeats)
            if init_noise is not None:
                x = init_noise.to(self.device, torch.float32)
            else:
                x = torch.randn(
                    (B, self.model_horizon, self.max_action_dim),
                    device=self.device,
                    dtype=torch.float32,
                    generator=generator,
                )
            traj = [x] if return_trajectory or mode == "sde" else None
            log_probs = []
            mask = self.logprob_mask.unsqueeze(0)
            for k in range(K):
                tau = k / K
                v = self._velocity(cond, x, tau)
                if mode == "ode":
                    x = x + v * dt
                else:
                    mean, std = sde_step_params(x, v, tau, dt, self.sigma, self.noise_schedule)
                    eps = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
                    x_next = torch.where(mask.bool(), mean + std * eps, x + v * dt)
                    log_probs.append(
                        gaussian_log_prob(x_next, mean, std, mask, self.logprob_reduce)
                    )
                    x = x_next
                if traj is not None:
                    traj.append(x)
        return SampleOutput(
            actions=x,
            trajectory=torch.stack(traj) if traj is not None else None,
            log_probs=torch.stack(log_probs) if log_probs else None,
        )

    def traj_log_prob(
        self, feats: PolicyFeatures, trajectory: torch.Tensor, repeats: int = 1
    ) -> torch.Tensor:
        """Per-step log-likelihoods [K, B*repeats] of ``trajectory`` [K+1, B*repeats, H, D].

        Differentiable w.r.t. the trainable parameters; ``repeats`` as in :meth:`sample`.
        """
        K = self.num_steps
        dt = 1.0 / K
        mask = self.logprob_mask.unsqueeze(0)
        out = []
        with self._autocast():
            cond = self._condition(feats, repeats)
            for k in range(K):
                tau = k / K
                x = trajectory[k]
                v = self._velocity(cond, x, tau)
                mean, std = sde_step_params(x, v, tau, dt, self.sigma, self.noise_schedule)
                out.append(
                    gaussian_log_prob(trajectory[k + 1], mean, std, mask, self.logprob_reduce)
                )
        return torch.stack(out)

    def bc_loss(self, feats: PolicyFeatures, actions: torch.Tensor) -> torch.Tensor:
        """Flow-matching loss on normalized chunks ``actions`` [B, H_valid, Da] (unpadded)."""
        B = actions.shape[0]
        padded = torch.zeros(B, self.model_horizon, self.max_action_dim, device=self.device)
        padded[:, : actions.shape[1], : actions.shape[2]] = actions.to(self.device, torch.float32)
        ai = _copy_batch(feats.action_input)
        ai["action"] = padded
        ai["action_mask"] = self.bc_mask.unsqueeze(0).expand(B, -1, -1)
        bo = _copy_batch(feats.backbone_output)
        with self._autocast():
            out = self.head(bo, ai)
        return out["loss"].float()

    # ------------------------------------------------------------------ action spaces

    def valid_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Crop padded normalized actions [B, H, D] to the valid [B, H_valid, Da]."""
        return actions[:, : self.valid_horizon, : self.action_dim]

    def decode_actions(
        self, actions: torch.Tensor | np.ndarray, states: Sequence[dict]
    ) -> list[dict]:
        """Normalized actions -> list of absolute action dicts ``{key: [H_valid, Dk]}``."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().float().cpu().numpy()
        batched_states = {k: np.stack([s[k] for s in states]) for k in self.state_keys}
        decoded = self.processor.decode_action(actions, self.embodiment_tag, batched_states)
        B = actions.shape[0]
        return [{k: decoded[k][i].astype(np.float32) for k in self.action_keys} for i in range(B)]

    def encode_action_chunk(self, state: dict, chunk: dict) -> np.ndarray:
        """Absolute chunk ``{key: [H, Dk]}`` + state ``{key: [1, D]}`` -> normalized [H, Da]."""
        norm = self.processor.state_action_processor.apply_action(
            chunk, self.embodiment_tag.value, state=state
        )
        return np.concatenate([norm[k] for k in self.action_keys], axis=-1).astype(np.float32)

    def normalize_state(self, state: dict) -> np.ndarray:
        """Raw state ``{key: [1, D]}`` -> normalized flat [S] (the critic's state input)."""
        norm = self.processor.state_action_processor.apply_state(state, self.embodiment_tag.value)
        return np.concatenate([norm[k].reshape(-1) for k in self.state_keys]).astype(np.float32)

    # ------------------------------------------------------------------ checkpoints

    def save_pretrained(self, output_dir: str | Path) -> None:
        """Save model + processor in the layout ``Gr00tPolicy`` / ``launch_finetune`` load."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Restore bf16 storage for the saved copy without touching the live fp32 weights.
        state = {
            k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
            for k, v in self.model.state_dict().items()
        }
        self.model.save_pretrained(output_dir, state_dict=state, safe_serialization=True)
        self.processor.save_pretrained(output_dir / "processor")


def cat_features(parts: Sequence[PolicyFeatures]) -> PolicyFeatures:
    """Concatenate feature chunks along the batch, left-padding token sequences.

    The processor pads text on the left, so chunks collated separately have different
    sequence lengths; padded positions are masked out (attention / image masks False).
    """
    if len(parts) == 1:
        return parts[0]
    seq = max(p.backbone_output["backbone_features"].shape[1] for p in parts)

    def pad_left(t: torch.Tensor) -> torch.Tensor:
        missing = seq - t.shape[1]
        if missing == 0:
            return t
        pad = torch.zeros((t.shape[0], missing, *t.shape[2:]), dtype=t.dtype, device=t.device)
        return torch.cat([pad, t], dim=1)

    bo = {}
    for k in parts[0].backbone_output.keys():
        bo[k] = torch.cat([pad_left(p.backbone_output[k]) for p in parts], dim=0)
    ai = {
        k: torch.cat([p.action_input[k] for p in parts], dim=0)
        for k in parts[0].action_input.keys()
    }
    return PolicyFeatures(
        backbone_output=BatchFeature(data=bo),
        action_input=BatchFeature(data=ai),
        prefix=torch.cat([p.prefix for p in parts], dim=0),
    )


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean over the sequence dim of [B, T, D], counting only ``mask`` [B, T] positions."""
    if mask is None:
        return x.mean(dim=1)
    m = mask.to(x.dtype).unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp_min(1.0)


def prepare_inputs(
    processor, embodiment_tag, language_key: str, observations: Sequence[dict]
) -> dict:
    """Mirror of ``Gr00tPolicy._get_action`` steps 1-3 for unbatched observations."""
    from gr00t.data.types import MessageType, VLAStepData

    processed = []
    for obs in observations:
        step = VLAStepData(
            images=obs["video"],
            states={k: np.asarray(v, dtype=np.float32) for k, v in obs["state"].items()},
            actions={},
            text=obs["language"][language_key][0],
            embodiment=embodiment_tag,
        )
        processed.append(processor([{"type": MessageType.EPISODE_STEP.value, "content": step}]))
    collated = processor.collator(processed)
    inputs = collated["inputs"] if "inputs" in collated else collated
    return {
        k: (v.to(torch.bfloat16) if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
        for k, v in inputs.items()
    }


class EMAPolicyParams:
    """Polyak-averaged copy of a policy's trainable parameters (OGPO's slow policy)."""

    def __init__(self, params: Sequence[nn.Parameter], decay: float):
        self.params = list(params)
        self.decay = decay
        self.shadow = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def update(self) -> None:
        for s, p in zip(self.shadow, self.params):
            s.lerp_(p.detach().to(s.dtype), 1.0 - self.decay)

    @contextmanager
    def swapped(self) -> Iterator[None]:
        """Temporarily load the EMA weights into the live parameters (no copies)."""
        online = [p.data for p in self.params]
        for p, s in zip(self.params, self.shadow):
            p.data = s
        try:
            yield
        finally:
            for i, p in enumerate(self.params):
                self.shadow[i] = p.data
                p.data = online[i]

    def state_dict(self) -> list[torch.Tensor]:
        return [s.detach().cpu() for s in self.shadow]

    def load_state_dict(self, tensors: Sequence[torch.Tensor]) -> None:
        for s, t in zip(self.shadow, tensors):
            s.copy_(t.to(s.device, s.dtype))
