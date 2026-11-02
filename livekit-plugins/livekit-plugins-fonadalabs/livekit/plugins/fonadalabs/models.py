"""FonadaLabs model and language definitions.

Language lists are taken from the FonadaLabs API docs:

- Klone V2 Flash: https://fonadalabs.ai/docs/text-to-speech/klone-v2-flash
- Klone V2 Pro:   https://fonadalabs.ai/docs/text-to-speech/klone-v2-pro
                  (``GET https://api.fonada.ai/v1/voice-clone/languages``)
- Kapture V3:     https://fonadalabs.ai/docs/speech-to-text/kapture-v3

Klone V2 voices are selected only by their 8-character Voice Arena ``share_id``.
"""

from __future__ import annotations

from typing import Literal

# ---------------------------------------------------------------------------
# Text-to-Speech
# ---------------------------------------------------------------------------

TTSModels = Literal["klone_v2_flash", "klone_v2_pro"]
"""``klone_v2_flash`` streams over a WebSocket (recommended for agents);
``klone_v2_pro`` is higher quality but one-shot HTTP only."""

KLONE_V2_FLASH_LANGUAGES: dict[str, str] = {
    "bn": "Bengali",
    "en": "English",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}

KLONE_V2_PRO_LANGUAGES: dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "arb": "Standard Arabic",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "id": "Indonesian",
    "de": "German",
    "ko": "Korean",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "it": "Italian",
    "th": "Thai",
    "pl": "Polish",
    "my": "Burmese",
    "fa": "Persian",
    "sw": "Swahili",
    "jv": "Javanese",
    "ms": "Malay",
    "fil": "Filipino",
    "nl": "Dutch",
    "uk": "Ukrainian",
    "ro": "Romanian",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
    "as": "Assamese",
    "ory": "Odia",
    "mai": "Maithili",
    "bho": "Bhojpuri",
    "sa": "Sanskrit",
    "sd": "Sindhi",
    "ks": "Kashmiri",
    "dgo": "Dogri",
    "sat": "Santali",
    "mni": "Manipuri",
    "brx": "Bodo",
    "gom": "Goan Konkani",
    "knn": "Konkani",
}

TTS_LANGUAGES: dict[str, dict[str, str]] = {
    "klone_v2_flash": KLONE_V2_FLASH_LANGUAGES,
    "klone_v2_pro": KLONE_V2_PRO_LANGUAGES,
}

TTSFlashLanguages = Literal[
    "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur",
    "Bengali", "English", "Gujarati", "Hindi", "Kannada", "Malayalam",
    "Marathi", "Odia", "Punjabi", "Tamil", "Telugu", "Urdu",
]  # fmt: skip
"""Languages accepted by Klone V2 Flash (ISO code or display name)."""

# ---------------------------------------------------------------------------
# Speech-to-Text
# ---------------------------------------------------------------------------

STTModels = Literal["v3"]
"""``v3`` = Kapture V3, the only documented FonadaLabs ASR model."""

KAPTURE_LANGUAGES: dict[str, str] = {
    "en": "English",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
}

STT_LANGUAGES: dict[str, dict[str, str]] = {
    "v3": KAPTURE_LANGUAGES,
}

STTLanguages = Literal["en", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
"""Language codes accepted by Kapture V3."""


def resolve_language(
    language: str, supported: dict[str, str], aliases: dict[str, str] | None = None
) -> str | None:
    """Return the API language code for *language* (code or display name), or ``None``."""
    value = language.strip()
    if value in supported:
        return value
    lowered = value.lower()
    if aliases and lowered in aliases and aliases[lowered] in supported:
        return aliases[lowered]
    for code, name in supported.items():
        if lowered == code.lower() or lowered == name.lower():
            return code
    return None


# Odia is "or" on Flash and "ory" on Pro; accept either for both models.
LANGUAGE_ALIASES: dict[str, dict[str, str]] = {
    "klone_v2_flash": {"ory": "or"},
    "klone_v2_pro": {"or": "ory"},
}
