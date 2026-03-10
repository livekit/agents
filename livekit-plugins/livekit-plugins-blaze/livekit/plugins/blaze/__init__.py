"""
livekit-plugins-blaze

LiveKit Agent Framework plugin for Blaze AI services (STT, TTS, LLM).

This package provides LiveKit agent plugins that interface with Blaze's
speech-to-text, text-to-speech, and conversational AI services.

Example:
    >>> from livekit.plugins import blaze
    >>>
    >>> # Create plugins with environment variables (BLAZE_*)
    >>> stt = blaze.STT(language="vi")
    >>> tts = blaze.TTS(speaker_id="speaker-1")
    >>> llm = blaze.LLM(bot_id="my-chatbot")
    >>>
    >>> # Or use shared configuration
    >>> config = blaze.BlazeConfig(
    ...     api_url="https://api.example.com",
    ...     auth_token="my-token",
    ... )
    >>> stt = blaze.STT(config=config)
    >>> tts = blaze.TTS(config=config, speaker_id="custom-voice")
    >>> llm = blaze.LLM(config=config, bot_id="enterprise-bot")

Environment Variables:
    BLAZE_API_URL: Base URL for Blaze API gateway
    BLAZE_API_TOKEN: Bearer token for API authentication
"""

from .version import __version__
from ._config import BlazeConfig
from .stt import STT, STTError
from .tts import TTS, TTSError
from .llm import LLM, LLMStream, LLMError

from livekit.agents import Plugin

from .log import logger

__all__ = [
    # Version
    "__version__",
    # Configuration
    "BlazeConfig",
    # Plugins
    "STT",
    "TTS",
    "LLM",
    "LLMStream",
    # Exceptions
    "STTError",
    "TTSError",
    "LLMError",
]


class BlazePlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(BlazePlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
