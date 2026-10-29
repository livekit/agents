"""Release updates must reach OpenSpline's build metadata and marked extra."""

import importlib.util
import runpy
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_openspline_release_updates_metadata_and_python_marker(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    plugin = Path("livekit-plugins/livekit-plugins-openspline")
    paths = [
        Path("livekit-agents/pyproject.toml"),
        Path("livekit-agents/livekit/agents/version.py"),
        plugin / "pyproject.toml",
        plugin / "livekit/plugins/openspline/version.py",
    ]
    for path in paths:
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / path, destination)

    spec = importlib.util.spec_from_file_location(
        "openspline_release_updater", root / ".github/update_versions.py"
    )
    assert spec is not None and spec.loader is not None
    updater = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(updater)
    plugin_version_path = plugin / "livekit/plugins/openspline/version.py"
    next_plugin_version = updater.bump_version(
        updater.read_version(root / plugin_version_path), "patch"
    )
    next_agents_version = updater.bump_version(updater.read_version(root / paths[1]), "patch")
    original_agents = (root / paths[0]).read_text()
    openai_extra = next(
        line for line in original_agents.splitlines() if line.startswith("openai =")
    )
    monkeypatch.chdir(tmp_path)
    updater.update_versions("patch")

    version = runpy.run_path(str(plugin_version_path))["__version__"]
    assert version == next_plugin_version
    manifest = (plugin / "pyproject.toml").read_text()
    assert 'dynamic = ["version"]' in manifest
    assert 'path = "livekit/plugins/openspline/version.py"' in manifest
    assert f'"livekit-agents>={next_agents_version}"' in manifest
    agents = Path("livekit-agents/pyproject.toml").read_text()
    assert (
        f"\"livekit-plugins-openspline>={next_plugin_version}; python_version < '3.13'\"" in agents
    )
    assert openai_extra in agents
