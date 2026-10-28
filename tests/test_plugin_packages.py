"""The supported distributions exclude retired integrations."""

from pathlib import Path

import pytest
import tomllib

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("package", ["livekit-plugins-hamming", "livekit-durable"])
def test_retired_plugin_cannot_be_installed_from_supported_packages(package: str) -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "livekit-plugins" / package).exists()

    for manifest in (root / "pyproject.toml", root / "livekit-agents-next/pyproject.toml"):
        metadata = tomllib.loads(manifest.read_text())
        sources = metadata.get("tool", {}).get("uv", {}).get("sources", {})
        assert package not in sources
        for dependencies in metadata.get("project", {}).get("optional-dependencies", {}).values():
            assert not any(dependency.startswith(package) for dependency in dependencies)
