"""LiveKit plugin for Mistral AI models. Supports Chat and STT models"""

from livekit.agents import Plugin

from .llm import LLM
from .log import logger
from .stt import STT, SpeechStream
from .version import __version__

__all__ = ["LLM", "STT", "SpeechStream", "__version__"]


class MistralAIPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(MistralAIPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
