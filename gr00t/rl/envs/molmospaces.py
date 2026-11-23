"""MolmoSpaces (Franka FR3, DROID configuration) adapter for GR00T's DROID embodiment.

Port of Korniak et al. 2026's ``src/envs/molmo.py`` (github.com/lasgroup/generalist-
improvement, MIT), extended to several benchmarks and to the GR00T
``OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`` observation/action layout:

* video: ``exterior_image_1_left`` <- exo camera, ``wrist_image_left`` <- wrist camera,
  resized with padding to ``image_size`` (DROID frames are 180x320);
* state: ``eef_9d`` = TCP xyz + rot6d (first two rows of ``R_tcp @ correction``), with
  the TCP (gripper grasp site) expressed in the arm base frame ``fr3_link0`` like DROID
  (MolmoSpaces' own ``tcp_pose`` sensor is relative to the robot mount),
  ``gripper_position`` = finger qpos / 0.824033 clipped to [0, 1] (as MolmoSpaces'
  pi policy does), ``joint_position`` = 7 arm joints;
* action: absolute ``joint_position`` targets (7) and ``gripper_position`` (binarised
  to 0 / 255 at 0.5); ``eef_9d`` is ignored (the env is joint-position controlled at
  66 ms, i.e. DROID's 15 Hz).

Task ids are ``"<benchmark>/<episode_index>"``, where ``<benchmark>`` is a key of the
``benchmarks`` mapping (name -> benchmark directory), or a bare episode index for the
default benchmark. Episodes are MolmoSpaces benchmark JSON episodes; each reset
re-randomises the object XY and the robot's initial joints (Korniak's defaults).

The TCP frame convention relative to DROID's is not guaranteed; check it with
``scripts/rl/check_droid_spaces.py`` and override ``eef_rotation_correction`` if needed.
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import logging
import os
from pathlib import Path
import warnings

import numpy as np


# GR00T's DROID client post-multiplies the robot EEF rotation by this matrix to match the
# OXE DROID training pipeline (examples/DROID/main_gr00t.py).
DROID_EEF_ROTATION_CORRECT = [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
GRIPPER_QPOS_OPEN_TO_CLOSED = 0.824033
BENCHMARK_ROOT_SUBPATH = "benchmarks/molmospaces-bench-v1"
# Short names usable in task ids ("open/12"). Any other "<scene-set>/<Benchmark>" path
# relative to BENCHMARK_ROOT_SUBPATH (or an absolute directory) works too, e.g.
# "ithor/FrankaOpenHardBench/3".
BUILTIN_BENCHMARKS = {
    "pick": "procthor-10k/FrankaPickDroidMiniBench",  # Korniak et al.'s pick suite
    "pnp": "procthor-10k/FrankaPickandPlaceDroidMiniBench",
    "pick_hard": "procthor-10k/FrankaPickHardBench",
    "pnp_hard": "procthor-10k/FrankaPickandPlaceHardBench",
    "open": "procthor-10k/FrankaOpenHardBench",
    "close": "procthor-10k/FrankaCloseHardBench",
}


def resolve_benchmark_dir(spec: str, assets_dir: Path) -> Path:
    """Benchmark spec -> directory holding ``benchmark.json``.

    Relative specs are resolved under ``<assets>/benchmarks/molmospaces-bench-v1``; a
    directory that only wraps one dated ``*json_benchmark*`` folder is descended into.
    """
    p = Path(spec).expanduser()
    if not p.is_absolute():
        p = Path(assets_dir) / BENCHMARK_ROOT_SUBPATH / p
    if p.is_dir() and not (p / "benchmark.json").exists() and not any(p.glob("house_*")):
        subs = sorted(d for d in p.iterdir() if d.is_dir() and "json_benchmark" in d.name)
        if len(subs) == 1:
            p = subs[0]
    return p


@contextmanager
def _quiet():
    logging.getLogger("molmo_spaces").setLevel(logging.ERROR)
    with open(os.devnull, "w") as devnull, warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"Using objaverse data version .*", category=UserWarning
        )
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def tcp_pose_to_eef_9d(tcp_pose: np.ndarray, correction: np.ndarray) -> np.ndarray:
    """MolmoSpaces ``tcp_pose`` (x, y, z, qw, qx, qy, qz) -> DROID ``eef_9d``."""
    tcp_pose = np.asarray(tcp_pose, dtype=np.float64).reshape(7)
    rot = quat_wxyz_to_matrix(tcp_pose[3:]) @ correction
    return np.concatenate([tcp_pose[:3], rot[:2, :].reshape(6)]).astype(np.float32)


def pose_to_eef_9d(
    pose: np.ndarray, correction: np.ndarray, tcp_offset_z: float = 0.0
) -> np.ndarray:
    """4x4 TCP pose -> DROID ``eef_9d``; ``tcp_offset_z`` shifts the point along tool z."""
    pose = np.asarray(pose, dtype=np.float64)
    pos = pose[:3, 3] + pose[:3, 2] * tcp_offset_z
    rot = pose[:3, :3] @ correction
    return np.concatenate([pos, rot[:2, :].reshape(6)]).astype(np.float32)


def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Letterbox-resize (same as GR00T's DROID client / openpi ``resize_with_pad``)."""
    from PIL import Image

    if image.shape[:2] == (height, width):
        return image
    img = Image.fromarray(image)
    cur_w, cur_h = img.size
    ratio = max(cur_w / width, cur_h / height)
    new_h, new_w = int(cur_h / ratio), int(cur_w / ratio)
    img = img.resize((new_w, new_h), resample=Image.BILINEAR)
    canvas = Image.new(img.mode, (width, height), 0)
    canvas.paste(img, ((width - new_w) // 2, (height - new_h) // 2))
    return np.asarray(canvas)


class _PolicyStub:
    """Minimal policy object some MolmoSpaces sensors query."""

    def __init__(self):
        self.target_poses = {"grasp": np.eye(4, dtype=np.float32)}
        self.task = None
        self.retry_count = 0

    def reset(self):
        return

    def get_phase(self):
        return "inference"

    def get_all_phases(self):
        return {"inference": 0}


class MolmoSpacesAdapter:
    def __init__(
        self,
        render_device: int = 0,
        benchmarks: dict[str, str] | None = None,
        default_benchmark: str | None = None,
        image_size: tuple[int, int] = (180, 320),
        eef_rotation_correction: list[list[float]] | None = None,
        eef_frame: str = "arm_base",
        tcp_offset_z: float = 0.0,
        randomize_init_positions: bool = True,
        init_object_position_noise_xy: float = 0.01,
        init_qpos_noise: float = 0.01,
        randomization_max_attempts: int = 20,
        drop_sensor_uuids: tuple[str, ...] = ("object_image_points",),
        exo_camera_keys: tuple[str, ...] = ("droid_shoulder_light_randomization", "exo_camera_1"),
        wrist_camera_keys: tuple[str, ...] = ("wrist_camera_zed_mini", "wrist_camera"),
        gripper_threshold: float = 0.5,
    ):
        with _quiet():
            from molmo_spaces.evaluation.benchmark_schema import load_all_episodes
            from molmo_spaces.molmo_spaces_constants import ASSETS_DIR

        self._load_all_episodes = load_all_episodes
        self._assets_dir = Path(ASSETS_DIR)
        specs = dict(BUILTIN_BENCHMARKS)
        if os.environ.get("MLSPACES_BENCHMARK_DIR"):  # Korniak's override of the default
            specs["pick"] = os.environ["MLSPACES_BENCHMARK_DIR"]
        specs.update(benchmarks or {})
        self.benchmark_specs = specs
        self.benchmark_dirs: dict[str, Path] = {}  # resolved lazily
        self.default_benchmark = default_benchmark or "pick"
        self._episodes: dict[str, list] = {}
        self.render_device = render_device
        self.image_size = tuple(image_size)
        self.correction = np.asarray(
            eef_rotation_correction or DROID_EEF_ROTATION_CORRECT, dtype=np.float64
        )
        if eef_frame not in ("arm_base", "robot"):
            raise ValueError("eef_frame must be 'arm_base' or 'robot'")
        self.eef_frame = eef_frame
        self.tcp_offset_z = float(tcp_offset_z)
        self.randomize = randomize_init_positions
        self.obj_noise = init_object_position_noise_xy
        self.qpos_noise = init_qpos_noise
        self.max_attempts = randomization_max_attempts
        self.drop_sensor_uuids = tuple(drop_sensor_uuids)
        self.exo_camera_keys = tuple(exo_camera_keys)
        self.wrist_camera_keys = tuple(wrist_camera_keys)
        self.gripper_threshold = gripper_threshold

        self._sampler = None
        self._task = None
        self._loaded = None  # (benchmark, episode_index)
        self._prompt_sampler = None
        self._language = ""

    # ------------------------------------------------------------------ spec

    def spec(self) -> dict:
        return {
            "camera_keys": ["exterior_image_1_left", "wrist_image_left"],
            "state_keys": ["eef_9d", "gripper_position", "joint_position"],
            "image_size": list(self.image_size),
            "benchmarks": dict(self.benchmark_specs),
        }

    def benchmark_dir(self, benchmark: str) -> Path:
        if benchmark not in self.benchmark_dirs:
            spec = self.benchmark_specs.get(benchmark, benchmark)
            path = resolve_benchmark_dir(spec, self._assets_dir)
            if not path.is_dir():
                raise KeyError(
                    f"Benchmark {benchmark!r} -> {path} does not exist; "
                    f"short names: {sorted(self.benchmark_specs)}"
                )
            self.benchmark_dirs[benchmark] = path
        return self.benchmark_dirs[benchmark]

    def episodes(self, benchmark: str) -> list:
        if benchmark not in self._episodes:
            path = self.benchmark_dir(benchmark)
            with _quiet():
                eps = self._load_all_episodes(path)
            if not eps:
                raise ValueError(f"No benchmark episodes found in {path}")
            self._episodes[benchmark] = eps
        return self._episodes[benchmark]

    def _parse_task(self, task_id: str) -> tuple[str, int]:
        task_id = str(task_id)
        if "/" in task_id:
            bench, idx = task_id.rsplit("/", 1)
        elif task_id.startswith("molmo_"):  # Korniak's naming
            bench, idx = self.default_benchmark, task_id.split("_")[-1]
        else:
            bench, idx = self.default_benchmark, task_id
        self.benchmark_dir(bench)  # validates
        return bench, int(idx)

    # ------------------------------------------------------------------ episodes

    def _close_episode(self) -> None:
        if self._sampler is not None:
            self._sampler.close()
        self._sampler = None
        self._task = None

    def reset(self, task_id: str, seed: int | None = None):
        bench, ep_idx = self._parse_task(task_id)
        episodes = self.episodes(bench)
        if not 0 <= ep_idx < len(episodes):
            raise ValueError(
                f"Episode {ep_idx} out of range for {bench} ({len(episodes)} episodes)"
            )
        episode = episodes[ep_idx]
        if seed is not None:
            np.random.seed(seed % (2**32))

        if self._sampler is None or self._loaded != (bench, ep_idx):
            with _quiet():
                from molmo_spaces.evaluation.configs.evaluation_configs import PiPolicyEvalConfig
                from molmo_spaces.policy.learned_policy.utils import PromptSampler
                from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler

            self._close_episode()
            cfg = PiPolicyEvalConfig()
            cfg.scene_dataset = episode.scene_dataset
            cfg.data_split = episode.data_split
            ts = cfg.task_sampler_config
            ts.render_device = self.render_device
            ts.randomize_init_object_positions = self.randomize
            ts.randomize_init_qpos = self.randomize
            ts.init_object_position_noise_xy = self.obj_noise
            ts.init_qpos_noise = self.qpos_noise
            ts.init_position_randomization_max_attempts = self.max_attempts
            self._prompt_sampler = PromptSampler(
                task_type=cfg.task_type,
                prompt_templates=cfg.policy_config.prompt_templates,
                prompt_object_word_num=cfg.policy_config.prompt_object_word_num,
            )
            with _quiet():
                self._sampler = JsonEvalTaskSampler(cfg, episode)
            self._loaded = (bench, ep_idx)

        self._prompt_sampler.next()
        with _quiet():
            self._task = self._sampler.sample_task(
                force_advance_scene=False, house_index=episode.house_index
            )
        if self._task is None:
            raise RuntimeError("JsonEvalTaskSampler returned no task")
        suite = getattr(self._task, "_sensor_suite", None)
        if suite is not None and hasattr(suite, "sensors"):
            for uuid in self.drop_sensor_uuids:
                suite.sensors.pop(uuid, None)
        if self._task.env.n_batch != 1:
            raise ValueError(f"Expected n_batch=1, got {self._task.env.n_batch}")
        self._task.register_policy(_PolicyStub())
        self._language = self._prompt_sampler.get_prompt(self._task).lower()
        observations, infos = self._task.reset()
        return self._convert(observations[0]), {"success": bool(infos[0].get("success", False))}

    def step(self, action: dict):
        joints = np.asarray(action["joint_position"], dtype=np.float32).reshape(-1)[:7]
        grip = float(np.asarray(action["gripper_position"]).reshape(-1)[0])
        env_action = {
            "arm": joints,
            "gripper": np.array(
                [255.0 if grip > self.gripper_threshold else 0.0], dtype=np.float32
            ),
        }
        observations, _rewards, _terminated, _truncated, infos = self._task.step(env_action)
        return self._convert(observations[0]), {"success": bool(infos[0]["success"])}

    def close(self) -> None:
        self._close_episode()

    # ------------------------------------------------------------------ conversion

    def tcp_in_arm_base(self) -> np.ndarray:
        """Gripper grasp-site pose relative to the arm's root body ``fr3_link0`` (4x4).

        MolmoSpaces' ``tcp_pose`` sensor is relative to the robot *mount* body, whereas
        DROID reports the end effector in the Franka base (link0) frame.
        """
        rv = self._task.env.robots[0].robot_view
        data = rv.mj_data
        model = data.model
        ns = getattr(rv, "_namespace", "")
        base_id = model.body(f"{ns}fr3_link0").id
        site_id = model.site(f"{ns}gripper/grasp_site").id
        t_base = np.eye(4)
        t_base[:3, 3] = data.xpos[base_id]
        t_base[:3, :3] = data.xmat[base_id].reshape(3, 3)
        t_site = np.eye(4)
        t_site[:3, 3] = data.site_xpos[site_id]
        t_site[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
        return np.linalg.inv(t_base) @ t_site

    def _pick(self, obs: dict, keys: tuple[str, ...]) -> np.ndarray:
        for k in keys:
            if k in obs:
                return np.asarray(obs[k])
        raise KeyError(f"None of the camera keys {keys} in observation (have {sorted(obs)})")

    def _convert(self, obs: dict) -> dict:
        h, w = self.image_size
        exo = resize_with_pad(self._pick(obs, self.exo_camera_keys).astype(np.uint8), h, w)
        wrist = resize_with_pad(self._pick(obs, self.wrist_camera_keys).astype(np.uint8), h, w)
        qpos = obs["qpos"]
        arm = np.asarray(qpos["arm"], dtype=np.float32)[:7]
        grip = np.clip(
            float(np.asarray(qpos["gripper"]).reshape(-1)[0]) / GRIPPER_QPOS_OPEN_TO_CLOSED,
            0.0,
            1.0,
        )
        if self.eef_frame == "arm_base":
            eef = pose_to_eef_9d(self.tcp_in_arm_base(), self.correction, self.tcp_offset_z)
        else:
            eef = tcp_pose_to_eef_9d(np.asarray(obs["tcp_pose"]), self.correction)
        return {
            "video": {"exterior_image_1_left": exo, "wrist_image_left": wrist},
            "state": {
                "eef_9d": eef,
                "gripper_position": np.array([grip], dtype=np.float32),
                "joint_position": arm,
            },
            "language": self._language,
        }
