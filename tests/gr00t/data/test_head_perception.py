"""Head training must join and sample head sidecars, including visibility negatives."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
from unittest.mock import patch

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.launch_finetune import count_perception_label_steps
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def head_config():
    path = Path(__file__).resolve().parents[3] / "register_head_perception_modality.py"
    with patch.dict(MODALITY_CONFIGS):
        MODALITY_CONFIGS.pop(EmbodimentTag.NEW_EMBODIMENT.value, None)
        runpy.run_path(str(path))
        yield deepcopy(MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value])


@pytest.fixture
def dataset(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    files = {
        "info.json": {
            "chunks_size": 1000,
            "fps": 30,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        },
        "episodes.jsonl": {"episode_index": 0, "length": 10},
        "tasks.jsonl": {"task_index": 0, "task": "drill in the red circle"},
        "stats.json": {},
        "modality.json": {
            "video": {
                "ego_view": {"original_key": "observation.images.ego_view"},
                "wrist_view": {"original_key": "observation.images.wrist_view"},
            }
        },
    }
    for name, value in files.items():
        (meta / name).write_text(json.dumps(value) + "\n")
    for directory, view, frames, u in [
        ("heatmap_labels", "wrist_view", [0, 1, 2, 3], 360.0),
        ("heatmap_labels_head", "ego_view", [1, 3], 228.0),
    ]:
        root = tmp_path / directory
        (root / "chunk-000").mkdir(parents=True)
        (root / "labels_info.json").write_text(
            json.dumps({"video_key": f"observation.images.{view}"})
        )
        pd.DataFrame(
            {
                "frame_index": frames,
                "u": u,
                "v": 246.0,
                "depth_z": 0.5,
                "visible": [True] * (len(frames) - 1) + [False],
                "label_ok": True,
                "wall_normal_cam_0": 0.0,
                "wall_normal_cam_1": 0.0,
                "wall_normal_cam_2": 1.0,
            }
        ).to_parquet(root / "chunk-000/episode_000000.parquet", index=False)
    return tmp_path


def test_head_sparse_join_sampling_and_epoch_count(dataset, head_config):
    data = ShardedSingleStepDataset(
        dataset_path=str(dataset),
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        modality_configs=head_config,
        perception_labels_only=True,
        heatmap_labels_dir="heatmap_labels_head",
    )
    assert head_config["video"].modality_keys == ["ego_view"]
    np.testing.assert_array_equal(data.get_valid_step_indices(0), [1, 3])
    labels = data.episode_loader._load_heatmap_labels(0, 10)
    np.testing.assert_array_equal(labels.target_uv.iloc[1], [228, 246])
    assert np.isnan(labels.target_uv.iloc[2]).all()
    assert not labels.target_visible.iloc[3]  # Occluded frames still get sampled.
    assert count_perception_label_steps(str(dataset), 1, "heatmap_labels_head") == 2
    assert count_perception_label_steps(str(dataset), 1) == 4  # Wrist default unchanged.


def test_head_rejects_wrist_labels(dataset, head_config):
    with pytest.raises(ValueError, match="model reads"):
        ShardedSingleStepDataset(
            dataset_path=str(dataset),
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            modality_configs=head_config,
            perception_labels_only=True,
        )
