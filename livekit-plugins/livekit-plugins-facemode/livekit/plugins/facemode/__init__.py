"""FaceMode avatar plugin for LiveKit Agents."""

from .avatar import AvatarSession
from .errors import (
    FaceModeAPIError,
    FaceModeError,
    FaceModeNotReadyError,
    FaceModeProtocolError,
)
from .version import __version__

__all__ = [
    "AvatarSession",
    "FaceModeAPIError",
    "FaceModeError",
    "FaceModeNotReadyError",
    "FaceModeProtocolError",
    "__version__",
]

from livekit.agents import Plugin

from .log import logger


class FaceModePlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(FaceModePlugin())
