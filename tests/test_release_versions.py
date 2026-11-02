"""Releases keep Rime's tested core minimum while advancing package versions."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("current", "operation", "expected"),
    [
        ("1.8.10", "patch", "1.8.11"),
        ("1.8.10", "rc", "1.8.10.rc1"),
        ("1.8.10.rc2", "release", "1.8.10"),
    ],
)
def test_release_preserves_rime_core_minimum(tmp_path, monkeypatch, current, operation, expected):
    spec = importlib.util.spec_from_file_location(
        "update_versions", ROOT / ".github/update_versions.py"
    )
    assert spec is not None and spec.loader is not None
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)

    def write(path, text):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        return target

    core_version = write("livekit-agents/livekit/agents/version.py", f'__version__ = "{current}"\n')
    core_project = write(
        "livekit-agents/pyproject.toml",
        f'[project.optional-dependencies]\nrime = ["livekit-plugins-rime>={current}"]\n',
    )
    projects = {}
    versions = {}
    for name in ("rime", "openai"):
        directory = f"livekit-plugins/livekit-plugins-{name}"
        # Exercise the actual Rime dependency declaration, including its extras.
        text = (
            (ROOT / directory / "pyproject.toml").read_text()
            if name == "rime"
            else f'[project]\nname = "livekit-plugins-openai"\n'
            f'dependencies = ["livekit-agents[codecs]>={current}"]\n'
        )
        projects[name] = write(f"{directory}/pyproject.toml", text)
        versions[name] = write(
            f"{directory}/livekit/plugins/{name}/version.py", f'__version__ = "{current}"\n'
        )
    original_rime_project = projects["rime"].read_text()
    monkeypatch.chdir(tmp_path)

    if operation == "rc":
        release.update_prerelease(operation)
    else:
        release.update_versions(operation)

    assert release.read_version(core_version) == expected
    assert all(release.read_version(path) == expected for path in versions.values())
    assert projects["rime"].read_text() == original_rime_project
    assert '"livekit-agents[codecs]>=1.7.1"' in projects["rime"].read_text()
    assert f'"livekit-agents[codecs]>={expected}"' in projects["openai"].read_text()
    assert f'rime = ["livekit-plugins-rime>={expected}"]' in core_project.read_text()
