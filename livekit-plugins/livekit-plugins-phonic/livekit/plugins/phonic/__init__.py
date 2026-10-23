from . import realtime
from .version import __version__

__all__ = ["__version__", "realtime"]

from livekit.agents import Plugin

from .log import logger


class PhonicPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(PhonicPlugin())
