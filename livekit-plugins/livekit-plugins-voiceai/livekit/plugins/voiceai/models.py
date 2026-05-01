from typing import Literal

# Voice.ai TTS Models
TTSModels = Literal[
    "voiceai-tts-v1-latest",
    "voiceai-tts-v1-2025-12-19",
    "voiceai-tts-multilingual-v1-latest",
    "voiceai-tts-multilingual-v1-2025-01-14",
]

# Audio output formats (all at 32kHz sample rate)
TTSEncoding = Literal[
    "mp3",  # Compressed, smallest size
    "wav",  # Uncompressed with headers
    "pcm",  # Raw 16-bit signed little-endian
]

# Supported languages (ISO 639-1 codes)
TTSLanguages = Literal[
    "en",  # English (non-multilingual model)
    "ca",  # Catalan (multilingual)
    "sv",  # Swedish (multilingual)
    "es",  # Spanish (multilingual)
    "fr",  # French (multilingual)
    "de",  # German (multilingual)
    "it",  # Italian (multilingual)
    "pt",  # Portuguese (multilingual)
    "pl",  # Polish (multilingual)
    "ru",  # Russian (multilingual)
    "nl",  # Dutch (multilingual)
]
