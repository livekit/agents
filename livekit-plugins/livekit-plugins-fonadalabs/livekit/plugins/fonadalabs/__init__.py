"""FonadaLabs plugin for LiveKit Agents

Text-to-Speech (Klone V2 Flash / Pro) and Speech-to-Text (Kapture V3 / V2)
for Indian languages via the FonadaLabs API.
See https://fonadalabs.ai/docs for more information.
"""

from .models import (
    KAPTURE_LANGUAGES,
    KLONE_V2_FLASH_LANGUAGES,
    KLONE_V2_PRO_LANGUAGES,
    STTLanguages,
    STTModels,
    TTSFlashLanguages,
    TTSModels,
)
from .stt import STT
from .tts import TTS, SynthesizeStream
from .version import __version__

__all__ = [
    "KAPTURE_LANGUAGES",
    "KLONE_V2_FLASH_LANGUAGES",
    "KLONE_V2_PRO_LANGUAGES",
    "STT",
    "TTS",
    "STTLanguages",
    "STTModels",
    "SynthesizeStream",
    "TTSFlashLanguages",
    "TTSModels",
    "__version__",
]

from livekit.agents import Plugin

from .log import logger


class FonadaLabsPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(FonadaLabsPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
