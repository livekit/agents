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
    "Aaliyah-PlayAI",
    "Abigail-PlayAI",
    "Angelo-PlayAI",
    "Arthur-PlayAI",
    "Aurora-PlayAI",
    "Autumn-PlayAI",
    "Ayla-Meditation-PlayAI",
    "Ayla-Advertising-PlayAI",
    "Bryan-PlayAI",
    "Chuck-PlayAI",
    "Darrell-PlayAI",
    "Dexter-PlayAI",
    "Donovan-PlayAI",
    "Eileen-PlayAI",
    "Eleanor-PlayAI",
    "Erasmo-PlayAI",
    "Hudson-PlayAI",
    "Inara-PlayAI",
    "Luna-PlayAI",
    "Nia-PlayAI",
    "Phoebe-PlayAI",
    "Ranger-PlayAI",
    "Sophia-PlayAI",
    "Waylon-PlayAI",
    "William-Training-PlayAI",
    "Nasser-PlayAI",
    "Khalid-PlayAI",
    "Amira-PlayAI",
    "Ahmad-PlayAI",
]
