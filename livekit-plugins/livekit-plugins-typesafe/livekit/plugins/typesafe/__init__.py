"""TypeSafe Jev decision models for LiveKit Agents."""

import logging
from importlib.metadata import version

from livekit.agents import Plugin

from .jev import Jev

__version__ = version("livekit-plugins-typesafe")
__all__ = ["Jev", "__version__"]


class TypeSafePlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logging.getLogger(__name__))


Plugin.register_plugin(TypeSafePlugin())
