"""xAI plugin for LiveKit Agents"""

from . import realtime, responses
from .tools import FileSearch, WebSearch, XSearch
from .version import __version__

__all__ = [
    "realtime",
    "responses",
    "WebSearch",
    "XSearch",
    "FileSearch",
    "__version__",
]


from livekit.agents import Plugin

from .log import logger


class XAIPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(XAIPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
