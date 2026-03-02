"""PersonaPlex plugin for LiveKit Agents

Support for NVIDIA PersonaPlex full-duplex conversational AI model.
"""

from . import realtime
from .models import PersonaplexVoice
from .realtime.realtime_model import RealtimeModel, RealtimeSession
from .version import __version__

__all__ = [
    "PersonaplexVoice",
    "realtime",
    "RealtimeModel",
    "RealtimeSession",
    "__version__",
]

from livekit.agents import Plugin

from .log import logger


class PersonaplexPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(PersonaplexPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__: dict[str, bool] = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
