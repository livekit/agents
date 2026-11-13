from __future__ import annotations

from typing import Literal

# Voice slots: five female, five male. Each language has its own native speaker in
# each slot, so the speaker heard depends on the language too. See
# https://docs.lokutor.com/voices-languages-models#voices
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

# The nine languages Lokutor speaks and recognises. A language outside this list
# uses the English speakers for synthesis and is detected for recognition. See
# https://docs.lokutor.com/voices-languages-models#languages
TTSLanguage = Literal["en", "es", "ca", "gl", "eu", "pt", "fr", "it", "de"]
STTLanguage = TTSLanguage
STT_LANGUAGES: frozenset[str] = frozenset(("en", "es", "ca", "gl", "eu", "pt", "fr", "it", "de"))

TTSModels = Literal["versa-2.0"]
TTS_MODEL: TTSModels = "versa-2.0"
STT_MODEL = "conv-2.0"

DEFAULT_VOICE_ID: VoiceID = "F1"
DEFAULT_LANGUAGE: TTSLanguage = "en"
DEFAULT_SAMPLE_RATE: int = 44100
DEFAULT_BASE_URL = "wss://api.lokutor.com"
