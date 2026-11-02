"""Import the forkserver image in order, refusing failures before any job can fork."""

import os

from .launch import _FORKSERVER_PRELOAD_MODULES, load_preload_modules

load_preload_modules(
    [
        "livekit.agents.ipc._preload",
        # Import SDK/exporter code once into the frozen Linux forkserver image.
        # Provider setup, clients and exporter threads still belong to each child.
        "livekit.agents.telemetry._providers",
        "livekit.agents.telemetry.traces",
        *filter(None, os.environ[_FORKSERVER_PRELOAD_MODULES].split(os.pathsep)),
    ]
)

# This must follow all framework, plugin and application imports.
from . import _preload_freeze  # noqa: E402, F401
