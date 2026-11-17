from __future__ import annotations

from typing import Literal

# Dialects the Hamsa TTS endpoint accepts. Pick a speaker recorded in the same
# dialect. See https://docs.tryhamsa.com/websocket/websocket-tts
TTSDialect = Literal[
    "pls",  # Palestinian
    "egy",  # Egyptian
    "syr",  # Syrian
    "irq",  # Iraqi
    "jor",  # Jordanian
    "leb",  # Lebanese
    "ksa",  # Saudi
    "uae",  # Emirati
    "bah",  # Bahraini
    "qat",  # Qatari
    "kuw",  # Kuwaiti
    "oma",  # Omani
    "msa",  # Modern Standard Arabic (Fusha)
    "ar-sa",  # Gulf Arabic
    "en",  # English
]
TTS_DIALECTS: frozenset[str] = frozenset(
    (
        "pls",
        "egy",
        "syr",
        "irq",
        "jor",
        "leb",
        "ksa",
        "uae",
        "bah",
        "qat",
        "kuw",
        "oma",
        "msa",
        "ar-sa",
        "en",
    )
)

# Example pre-built speakers per dialect, from Hamsa's documentation. The full list
# is in the Hamsa dashboard; a cloned voice is addressed by its UUID instead.
EXAMPLE_SPEAKERS: dict[str, tuple[str, ...]] = {
    "pls": ("Amjad", "Layan"),
    "egy": ("Mariam", "Samir"),
    "syr": ("Dalal", "Mais"),
    "irq": ("Lyali", "Fatma"),
    "jor": ("Lana", "Jasem"),
    "leb": ("Carla", "Majd"),
    "ksa": ("Hiba", "Fahd"),
    "uae": ("Salma", "Dima"),
    "bah": ("Mazen", "Ruba"),
    "qat": ("Deema", "Faisal"),
    "kuw": ("Mai", "Hatem"),
    "oma": ("Aisha", "Jaber"),
    "msa": ("Salem", "Tamim"),
    "ar-sa": ("Khalid", "Rahma"),
    "en": ("Emma", "James"),
}

TTSSampleRate = Literal[8000, 16000]

DEFAULT_SPEAKER = "Mariam"
DEFAULT_DIALECT: TTSDialect = "egy"
DEFAULT_SAMPLE_RATE: TTSSampleRate = 16000
DEFAULT_BASE_URL = "wss://api.tryhamsa.com"
TTS_MODEL = "hamsa-tts"
