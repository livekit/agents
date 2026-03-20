"""FonadaLabs plugin for LiveKit Agents

Provides high-quality Text-to-Speech for Indian languages via the FonadaLabs API.
See https://fonada.in for more information.
"""

from .tts import TTS
from .version import __version__

__all__ = ["TTS", "__version__"]

from livekit.agents import Plugin

from .log import logger


class FonadaLabsPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(FonadaLabsPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False