"""
LiveKit Whisper plugin
"""

from .log import logger
from .version import __version__
from .whisper_stt import WhisperSTT

__all__ = [
    "logger",
    "__version__",
    "WhisperSTT",
]

# Plugin registration with LiveKit Agents, if available
try:
    from livekit.agents import Plugin  # type: ignore

    class WhisperPlugin(Plugin):
        def __init__(self):
            # Fallback to empty string if __package__ is None for runtime safety
            super().__init__(__name__, __version__, __package__ or "")

    Plugin.register_plugin(WhisperPlugin())
except ImportError:
    # livekit.agents is not available; skip registration
    pass