"""BreezeBlue text-to-speech for LiveKit Agents."""

import logging

from livekit.agents import Plugin

from .tts import TTS
from .version import __version__

__all__ = ["TTS", "__version__"]

Plugin.register_plugin(Plugin("BreezeBlue", __version__, __package__, logging.getLogger(__name__)))
