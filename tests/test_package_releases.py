"""Release automation updates the distribution that carries the 2.0 API."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_RELEASE_SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "update_versions.py"


@pytest.mark.parametrize(
    "arguments, expected, plugin_floor",
    [(["--pre", "b", "--agents-only"], "2.0.0b2", "1.8.6"), ([], "2.0.1", "2.0.1")],
)
def test_release_updates_the_v2_distribution(
    tmp_path: Path, arguments: list[str], expected: str, plugin_floor: str
) -> None:
    package = tmp_path / "livekit-agents-next"
    version = package / "livekit" / "agents" / "version.py"
    version.parent.mkdir(parents=True)
    version.write_text('__version__ = "2.0.0b1"\n')
    manifest = package / "pyproject.toml"
    manifest.write_text('[project]\nname = "livekit-agents-next"\nversion = "2.0.0b1"\n')
    plugin = tmp_path / "livekit-plugins" / "livekit-plugins-example"
    plugin.mkdir(parents=True)
    plugin_manifest = plugin / "pyproject.toml"
    plugin_manifest.write_text(
        '[project]\nname = "livekit-plugins-example"\ndependencies = ["livekit-agents-next>=1.8.6"]\n'
    )

    subprocess.run([sys.executable, str(_RELEASE_SCRIPT), *arguments], cwd=tmp_path, check=True)

    assert version.read_text() == f'__version__ = "{expected}"\n'
    assert tomllib.loads(manifest.read_text())["project"]["version"] == expected
    assert f'"livekit-agents-next>={plugin_floor}"' in plugin_manifest.read_text()
