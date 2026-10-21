from __future__ import annotations

import importlib
import pkgutil

import pytest

import livekit.agents

pytestmark = pytest.mark.unit


def _packages_with_all() -> list[str]:
    names = ["livekit.agents"]
    for info in pkgutil.walk_packages(livekit.agents.__path__, prefix="livekit.agents."):
        if info.ispkg:
            names.append(info.name)
    return [name for name in names if hasattr(importlib.import_module(name), "__all__")]


@pytest.mark.parametrize("package", _packages_with_all())
def test_star_import_resolves_every_all_entry(package: str) -> None:
    # `from package import *` looks up every `__all__` entry on the package, so a name
    # that isn't bound there (or a submodule that doesn't exist) breaks the star import.
    exec(f"from {package} import *", {})
