"""Hosted Oruk speech-to-text for LiveKit Agents."""

import logging

from livekit.agents import Plugin

from .realtime import RealtimeSTT
from .realtime_bridge import vad_stream_node
from .stt import STT
from .version import __version__

__all__ = ["STT", "RealtimeSTT", "vad_stream_node", "__version__"]


class OrukPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logging.getLogger(__name__))


Plugin.register_plugin(OrukPlugin())
