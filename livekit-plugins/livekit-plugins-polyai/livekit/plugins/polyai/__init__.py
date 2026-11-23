"""PolyAI plugin for LiveKit Agents

PolyAI's Dialog-RSN-1 as a realtime model.
"""

from . import realtime
from .realtime.realtime_model import KnowledgeTopic, RealtimeModel, RealtimeSession
from .version import __version__

__all__ = [
    "realtime",
    "KnowledgeTopic",
    "RealtimeModel",
    "RealtimeSession",
    "__version__",
]

from livekit.agents import Plugin

from .log import logger


class PolyAIPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(PolyAIPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
