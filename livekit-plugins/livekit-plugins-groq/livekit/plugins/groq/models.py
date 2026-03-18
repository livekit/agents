from typing import Literal

# listing production models from https://console.groq.com/docs/models

STTModels = Literal[
    "whisper-large-v3",
    "whisper-large-v3-turbo",
    "distil-whisper-large-v3-en",
]

LLMModels = Literal[
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "moonshotai/kimi-k2-instruct-0905",
    "qwen/qwen3-32b",
]

TTSModels = Literal[
    "canopylabs/orpheus-v1-english",
    "canopylabs/orpheus-arabic-saudi",
]

TTSVoices = Literal[
    # english voices
    "autumn",
    "diana",
    "hannah",
    "austin",
    "daniel",
    "troy",
    # arabic voices
    "fahad",
    "sultan",
    "lulwa",
    "noura",
]
