"""Build the host for the prepared engine, including universal2 interpreters."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import TypeGuard

from setuptools import setup


def _mapping(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def _record(value: object) -> TypeGuard[dict[str, object]]:
    return _mapping(value) and all(isinstance(key, str) for key in value)


def _configure_target() -> None:
    metadata: object = json.loads(
        (Path(__file__).parent / "src/livekit/local_inference/_vendor/metadata.json").read_text()
    )
    if not _record(metadata):
        message = "Build requires the prepared engine metadata"
        raise ValueError(message)
    target = metadata.get("platform")
    if sys.platform == "linux" and (target == "linux-x64-gnu" or target == "linux-arm64-gnu"):
        return
    minimum = metadata.get("minimum_macos")
    if target != "darwin-arm64" or minimum != "11.0" or sys.platform != "darwin":
        message = "Build requires a supported prepared engine and matching operating system"
        raise ValueError(message)
    # The official engine is ARM64 only; the build interpreter can be universal2.
    # Keep wheel's normal compiled minimum-OS scan rather than overriding its tag.
    os.environ["ARCHFLAGS"] = "-arch arm64"
    os.environ["_PYTHON_HOST_PLATFORM"] = f"macosx-{minimum}-arm64"


_configure_target()
setup()
