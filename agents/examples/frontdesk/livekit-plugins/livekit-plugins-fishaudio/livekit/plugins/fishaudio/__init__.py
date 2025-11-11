"""Fish Audio plugin for LiveKit Agents

See https://docs.fish.audio for more information.

Environment variables used:
- `FISH_API_KEY` for authentication (required)
"""

from fish_audio_sdk.schemas import Backends  # type: ignore[import-untyped]

from livekit.agents import Plugin

from .log import logger
from .models import LatencyMode, OutputFormat
from .tts import TTS
from .version import __version__

__all__ = [
    "TTS",
    "Backends",
    "OutputFormat",
    "LatencyMode",
    "__version__",
]


class FishAudioPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__ or "", logger)


Plugin.register_plugin(FishAudioPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
