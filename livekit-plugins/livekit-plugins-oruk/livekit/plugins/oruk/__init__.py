"""Hosted Oruk speech-to-text for LiveKit Agents."""

import logging

from livekit.agents import Plugin

from .stt import STT
from .version import __version__

__all__ = ["STT", "__version__"]


class OrukPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logging.getLogger(__name__))


Plugin.register_plugin(OrukPlugin())
