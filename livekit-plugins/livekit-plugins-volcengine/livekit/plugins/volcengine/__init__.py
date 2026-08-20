"""Volcengine text-to-speech plugin for LiveKit Agents.

See https://www.volcengine.com/docs/6561/1329505 for API documentation.
"""

from livekit.agents import Plugin

from .log import logger
from .tts import TTS, SynthesizeStream
from .version import __version__

__all__ = ["SynthesizeStream", "TTS", "__version__"]


class VolcenginePlugin(Plugin):
    """Register the Volcengine provider with LiveKit Agents."""

    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(VolcenginePlugin())

_module = dir()
NOT_IN_ALL = [module_name for module_name in _module if module_name not in __all__]

__pdoc__ = dict.fromkeys(NOT_IN_ALL, False)
