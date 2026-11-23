"""Task-spec expansion and simulator-interpreter resolution (CPU)."""

import sys

from gr00t.rl.config import expand_task_specs
from gr00t.rl.envs.client import resolve_sim_python
import pytest


def test_expand_ranges_files_and_dedup(tmp_path):
    f = tmp_path / "split.txt"
    f.write_text("# comment\npick/0:3\n\nopen/7  # trailing comment\n")
    out = expand_task_specs(["@" + str(f), "pick/1", "libero_10/2:4", "molmo_5"])
    assert out == ["pick/0", "pick/1", "pick/2", "open/7", "libero_10/2", "libero_10/3", "molmo_5"]


def test_expand_nested_scene_path():
    assert expand_task_specs(["ithor/FrankaOpenHardBench/0:2"]) == [
        "ithor/FrankaOpenHardBench/0",
        "ithor/FrankaOpenHardBench/1",
    ]


def test_resolve_sim_python(monkeypatch):
    assert resolve_sim_python("fake") == sys.executable
    assert resolve_sim_python("molmospaces", "/x/python") == "/x/python"
    monkeypatch.setenv("RL_MOLMOSPACES_PYTHON", "/opt/sim/molmospaces/.venv/bin/python")
    assert resolve_sim_python("molmospaces") == "/opt/sim/molmospaces/.venv/bin/python"
    monkeypatch.delenv("RL_LIBERO_PYTHON", raising=False)
    monkeypatch.setattr(
        "gr00t.rl.envs.client._REPO_ROOT", __import__("pathlib").Path("/nonexistent")
    )
    with pytest.raises(FileNotFoundError):
        resolve_sim_python("libero")
