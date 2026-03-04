"""Browser plugin for LiveKit Agents

Support for Chromium Embedded Framework (CEF).
"""

from livekit.agents import Plugin

# Re-export from livekit-browser for convenience
from livekit.browser import (  # type: ignore[import-untyped]
    AudioData,
    BrowserContext,
    BrowserPage,
    PaintData,
)

from .log import logger
from .session import BrowserSession
from .version import __version__

__all__ = [
    "AudioData",
    "BrowserContext",
    "BrowserPage",
    "BrowserSession",
    "PaintData",
]


class BrowserPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)

    def download_files(self) -> None:
        from livekit.browser import download

        download()


Plugin.register_plugin(BrowserPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
