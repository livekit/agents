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

# Lokutor's 32 supported languages (ISO codes). See
# https://docs.lokutor.com/voices-languages-models#languages
TTSLanguage = Literal[
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "ja",
    "ko",
    "zh",
    "ar",
    "bg",
    "hr",
    "cs",
    "da",
    "nl",
    "et",
    "fi",
    "el",
    "hi",
    "hu",
    "id",
    "lv",
    "lt",
    "pl",
    "ro",
    "ru",
    "sk",
    "sl",
    "sv",
    "tr",
    "uk",
    "vi",
]

TTSModels = Literal["versa-1.0"]

DEFAULT_VOICE_ID: VoiceID = "F1"
DEFAULT_LANGUAGE: TTSLanguage = "en"
DEFAULT_SAMPLE_RATE: int = 44100
