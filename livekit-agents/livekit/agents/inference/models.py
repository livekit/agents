from typing import Literal

TTSModels = Literal[
    "elevenlabs/eleven_flash_v2",
    "...",
]

STTModels = Literal[
    "cartesia/ink-whisper",
    "deepgram/nova-3",
    "assemblyai",
]

STTLanguages = Literal["en", "de", "es", "fr", "ja", "pt", "zh"]

LLMModels = Literal[
    "openai/gpt4o",
    "...",
]
