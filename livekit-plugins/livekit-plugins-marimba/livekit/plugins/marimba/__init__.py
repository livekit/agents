"""Marimba audio-only turn detection for LiveKit Agents."""

from livekit.agents import Plugin

from .log import logger
from .turn_detector import TurnDetector, TurnDetectorStream
from .version import __version__

__all__ = ["TurnDetector", "TurnDetectorStream", "__version__"]


class MarimbaPlugin(Plugin):
    """Register Marimba's package metadata and logger with LiveKit Agents."""

    def __init__(self) -> None:
        """Initialize the plugin registration metadata."""
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(MarimbaPlugin())

_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]
__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
