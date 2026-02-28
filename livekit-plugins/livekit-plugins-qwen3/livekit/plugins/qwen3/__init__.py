"""Qwen3 TTS and STT plugins for LiveKit Agents

This plugin provides Text-to-Speech and Speech-to-Text capabilities using
Alibaba Cloud's Qwen3 (DashScope) service via WebSocket streaming.

See https://www.alibabacloud.com/help/en/model-studio/qwen-tts for TTS information.
See https://www.alibabacloud.com/help/en/model-studio/qwen-real-time-speech-recognition for ASR information.
"""

from .models import (
    DEFAULT_BASE_URL,
    DEFAULT_LANGUAGE,
    DEFAULT_MODE,
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_VOICE,
    INTL_BASE_URL,
    TTSLanguage,
    TTSMode,
    TTSModel,
    TTSVoice,
)
from .stt import (
    DEFAULT_STT_LANGUAGE,
    DEFAULT_STT_MODEL,
    DEFAULT_STT_SAMPLE_RATE,
    STT,
)
from .tts import TTS
from .version import __version__

__all__ = [
    # TTS
    "TTS",
    "__version__",
    "TTSModel",
    "TTSVoice",
    "TTSLanguage",
    "TTSMode",
    "DEFAULT_MODEL",
    "DEFAULT_VOICE",
    "DEFAULT_LANGUAGE",
    "DEFAULT_MODE",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_BASE_URL",
    "INTL_BASE_URL",
    # STT
    "STT",
    "DEFAULT_STT_MODEL",
    "DEFAULT_STT_LANGUAGE",
    "DEFAULT_STT_SAMPLE_RATE",
]

from livekit.agents import Plugin

from .log import logger


class Qwen3Plugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(Qwen3Plugin())

# pdoc configuration to hide internal modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
