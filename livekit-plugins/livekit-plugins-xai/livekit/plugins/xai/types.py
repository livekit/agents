from typing import Literal

STTAudioEncoding = Literal["pcm", "mulaw", "alaw"]

STTLanguages = Literal[
    "ar",
    "cs",
    "da",
    "nl",
    "en",
    "fil",
    "fr",
    "de",
    "hi",
    "id",
    "it",
    "ja",
    "ko",
    "mk",
    "ms",
    "fa",
    "pl",
    "pt",
    "ro",
    "ru",
    "es",
    "sv",
    "th",
    "tr",
    "vi",
]

GrokVoices = Literal[
    "Ara",
    "Eve",
    "Leo",
    "Rex",
    "Sal",
]

TTSLanguages = Literal[
    "auto",
    "en",
    "ar-EG",
    "ar-SA",
    "ar-AE",
    "bn",
    "zh",
    "fr",
    "de",
    "hi",
    "id",
    "it",
    "ja",
    "ko",
    "pt-BR",
    "pt-PT",
    "ru",
    "es-MX",
    "es-ES",
    "tr",
    "vi",
]
