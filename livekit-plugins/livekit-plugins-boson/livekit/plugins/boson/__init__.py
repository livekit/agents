"""Boson plugin for LiveKit Agents."""

from __future__ import annotations

from livekit.agents import Plugin

from . import realtime
from .log import logger
from .version import __version__

__all__ = ["realtime", "__version__"]


class BosonPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(BosonPlugin())
