"""The type gate must reject invalid provider implementations throughout the fleet."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_types import get_packages

pytestmark = pytest.mark.unit


def test_invalid_nvidia_and_rtzr_providers_fail_the_type_gate(tmp_path: Path) -> None:
    sources = [tmp_path]
    for package in ("livekit/agents", "tests/type_checks"):
        path = tmp_path / package
        path.mkdir(parents=True)
        (path / "__init__.py").write_text("")
    for provider in ("nvidia", "rtzr"):
        source = tmp_path / "livekit-plugins" / f"livekit-plugins-{provider}"
        package = source / "livekit/plugins" / provider
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("def transcript() -> str:\n    return 42\n")
        (package / "py.typed").touch()
        sources.append(source)
    config = tmp_path / "mypy.ini"
    config.write_text("[mypy]\nstrict = true\n")
    targets = [argument for package in get_packages(tmp_path) for argument in ("-p", package)]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(config),
            "--no-site-packages",
            "--cache-dir",
            str(tmp_path / "cache"),
            *targets,
        ],
        cwd=tmp_path,
        env={**os.environ, "MYPYPATH": os.pathsep.join(map(str, sources))},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    for provider in ("nvidia", "rtzr"):
        assert f"{provider}/__init__.py" in result.stdout
    assert result.stdout.count("[return-value]") == 2, result.stdout
