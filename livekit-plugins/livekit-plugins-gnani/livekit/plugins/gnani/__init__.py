"""Gnani Vachana plugin for LiveKit Agents

Support for speech-to-text and text-to-speech with [Gnani's Vachana platform](https://gnani.ai/).

Vachana provides high-accuracy STT and low-latency TTS for Indian languages,
including multilingual and code-switching scenarios.

For API access, email speechstack@gnani.ai
"""

from .stt import STT
from .tts import TTS
from .version import __version__

__all__ = ["STT", "TTS", "__version__"]


from livekit.agents import Plugin

from .log import logger


class GnaniPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(GnaniPlugin())

_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
