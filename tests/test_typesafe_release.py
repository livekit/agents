from __future__ import annotations

import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.version import Version

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_PATH = Path("livekit-plugins/livekit-plugins-typesafe")


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["--bump-type", "patch"], "0.1.1"),
        (["--pre", "rc"], "0.1.0rc1"),
        (["--pre", "dev"], "0.1.0.dev0"),
    ],
)
def test_release_updater_includes_typesafe(
    tmp_path: Path,
    arguments: list[str],
    expected: str,
) -> None:
    # Never run the release updater against the working tree. Give the isolated
    # copy a fixed baseline so future releases do not change these test cases.
    plugin = tmp_path / PLUGIN_PATH
    shutil.copytree(REPO_ROOT / PLUGIN_PATH, plugin, ignore=shutil.ignore_patterns("__pycache__"))
    version_file = plugin / "livekit/plugins/typesafe/version.py"
    assert version_file.exists(), "The release updater requires the plugin's version.py"
    version_file.write_text('__version__ = "0.1.0"\n')
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / ".github/update_versions.py"), *arguments],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert Version(runpy.run_path(str(version_file))["__version__"]) == Version(expected)
