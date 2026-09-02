from __future__ import annotations

from typing import Literal

from livekit.agents.language import LanguageCode

from .log import logger

MuseModels = Literal["muse-voice-transcribe-1.0"]

MuseEncoding = Literal["PCM_24KHZ", "PCM_16KHZ"]
"""Signed 16-bit little-endian mono PCM, at 24 kHz or 16 kHz."""

SAMPLE_RATES: dict[str, int] = {"PCM_24KHZ": 24000, "PCM_16KHZ": 16000}

MuseMode = Literal["PUSH_TO_TALK", "ENDPOINTING", "DIARIZATION"]
"""PUSH_TO_TALK: the caller delimits the turn. ENDPOINTING: the model detects turn
boundaries, one turn per speech segment. DIARIZATION: adds speaker attribution."""

MusePartialMode = Literal["CUMULATIVE", "DELTA"]
"""CUMULATIVE: every ``transcript`` event carries the whole current hypothesis and
replaces the previous one. DELTA: per-chunk text, for the file endpoint's SSE mode."""

# The 25 languages Meta lists as evaluated for muse-voice-transcribe-1.0. Muse takes
# `languageBias` as English language *names*, not BCP-47 codes, so this is keyed by the
# base code LanguageCode.language produces ("en-US" and "cmn-Hans-CN" both reduce here).
SUPPORTED_LANGUAGES: frozenset[str] = frozenset(
    {
        "ar",
        "bn",
        "de",
        "en",
        "es",
        "fr",
        "he",
        "hi",
        "id",
        "it",
        "ja",
        "kn",
        "ko",
        "mr",
        "ms",
        "nl",
        "pl",
        "pt",
        "ta",
        "te",
        "th",
        "tl",
        "tr",
        "vi",
        "zh",
    }
)

# LanguageCode.to_language_name() covers 23 of the 25 (lowercased); these two it does
# not: it calls zh "chinese" where Meta's list says "Mandarin Chinese", and it has no
# name for Filipino, which is Tagalog on Meta's side.
_LANGUAGE_NAME_OVERRIDES: dict[str, str] = {"zh": "Mandarin Chinese", "fil": "Tagalog"}


def to_language_bias(languages: list[str]) -> list[str]:
    """Map BCP-47 codes onto the English language names Muse's `languageBias` expects.

    Codes Muse does not list are dropped with a warning rather than forwarded: the
    handshake is rejected wholesale on an unknown bias entry, which would take down
    recognition entirely instead of merely losing the hint.
    """
    names: list[str] = []
    for lang in languages:
        code = LanguageCode(lang)
        base = code.language
        name = _LANGUAGE_NAME_OVERRIDES.get(base)
        if name is None and base in SUPPORTED_LANGUAGES:
            resolved = code.to_language_name()
            name = resolved.title() if resolved else None
        if name is None:
            logger.warning(
                "language is not supported by Muse Voice Transcribe, dropping from languageBias",
                extra={"language": lang},
            )
            continue
        if name not in names:
            names.append(name)
    return names
