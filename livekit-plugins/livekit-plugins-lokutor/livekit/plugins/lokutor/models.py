from __future__ import annotations

from typing import Literal

VoiceID = Literal[
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
]

TTSLanguage = Literal[
    "en",
    "es",
    "fr",
    "pt",
    "ko",
]

TTSModels = Literal["versa-1.0"]

DEFAULT_VOICE_ID: VoiceID = "F1"
DEFAULT_LANGUAGE: TTSLanguage = "en"
DEFAULT_SAMPLE_RATE: int = 44100
