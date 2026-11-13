"""LiveAvatar avatar plugin for LiveKit Agents

Provides LiveAvatar interactive avatar integration similar to Tavus.
"""

from .avatar import AvatarSession, LiveAvatarException
from .version import __version__

__all__ = [
    "LiveAvatarException",
    "AvatarSession",
    "__version__",
]

from livekit.agents import Plugin

from .log import logger


class LiveAvatarPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __name__, logger)


Plugin.register_plugin(LiveAvatarPlugin())
