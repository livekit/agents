"""Vui plugin for LiveKit Agents

Local, streaming text-to-speech with Vui Nano — a small, context-aware model
trained on real conversations (219M active parameters, Apache 2.0). Runs
in-process on CUDA or on MLX on Apple Silicon.

See https://github.com/fluxions-ai/vui for more information.
"""

from .tts import TTS, ChunkedStream, SynthesizeStream
from .version import __version__

__all__ = ["TTS", "ChunkedStream", "SynthesizeStream", "__version__"]

from livekit.agents import Plugin

from .log import logger


class VuiPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(VuiPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
