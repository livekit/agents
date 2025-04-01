from typing import Literal

# listing production models from https://console.groq.com/docs/models

STTModels = Literal[
    "whisper-large-v3",
    "whisper-large-v3-turbo",
    "distil-whisper-large-v3-en",
]

LLMModels = Literal[
    "llama3-8b-8192",
    "llama3-70b-8192",
    "llama-guard-3-8b",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
]

TTSModels = Literal[
    "playai-tts",
    "playai-tts-arabic",
]

TTSVoices = Literal[
    # english voices
    "Arista-PlayAI",
    "Atlas-PlayAI",
    "Basil-PlayAI",
    "Briggs-PlayAI",
    "Calum-PlayAI",
    "Celeste-PlayAI",
    "Cheyenne-PlayAI",
    "Chip-PlayAI",
    "Cillian-PlayAI",
    "Deedee-PlayAI",
    "Fritz-PlayAI",
    "Gail-PlayAI",
    "Indigo-PlayAI",
    "Mamaw-PlayAI",
    "Mason-PlayAI",
    "Mikail-PlayAI",
    "Mitch-PlayAI",
    "Quinn-PlayAI",
    "Thunder-PlayAI",
    # arabic voices
    "Nasser-PlayAI",
    "Khalid-PlayAI",
    "Amira-PlayAI",
    "Ahmad-PlayAI",
]
