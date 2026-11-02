"""Protoface avatar plugin for LiveKit Agents."""

from .avatar import DEFAULT_STOCK_AVATAR_ID, AvatarSession
from .errors import ProtofaceException
from .version import __version__

__all__ = [
    "DEFAULT_STOCK_AVATAR_ID",
    "AvatarSession",
    "ProtofaceException",
    "__version__",
]

from livekit.agents import Plugin

from .log import logger


class ProtofacePlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(ProtofacePlugin())
