from livekit.agents import Plugin

from .log import logger
from .version import __version__
from .llm import LLM, LLMStream
from .models import ChatModels


__all__ = [
    "LLM",
    "LLMStream",
    "logger",
    "__version__",
]

class DifyPlugin(Plugin):
    def __init__(self):
        super().__init__(__name__, __version__, __package__, logger)

Plugin.register_plugin(DifyPlugin())
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False

def greet():
    return 'hi, im dify'


