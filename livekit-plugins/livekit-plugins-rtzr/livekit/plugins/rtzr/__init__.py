"""RTZR plugin for LiveKit Agents

See Streaming STT docs at: https://developers.rtzr.ai/docs/en/

Environment variables used:
- `RTZR_CLIENT_ID` / `RTZR_CLIENT_SECRET` for authentication (required)
"""

from livekit.agents import Plugin

from .log import logger
from .stt import STT
from .version import __version__

__all__ = ["STT", "__version__"]


class RTZRPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(RTZRPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
