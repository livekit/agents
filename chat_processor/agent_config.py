# filler_agent/agent_config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Set, Dict, Optional

from dotenv import load_dotenv

# Load .env so LIVEKIT_* and OPENAI_API_KEY are available
load_dotenv()


@dataclass
class Settings:
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    openai_api_key: str

    # default language code (e.g. "en")
    default_language: str

    # Global unions (used as safe fallback)
    ignored_filler_words: Set[str]
    interrupt_command_words: Set[str]

    # Per-language mappings, e.g. {"en": {...}, "hi": {...}}
    ignored_filler_words_by_lang: Dict[str, Set[str]]
    interrupt_command_words_by_lang: Dict[str, Set[str]]

    filler_confidence_threshold: float

    # Optional path to a JSON file for dynamic runtime updates
    dynamic_config_path: Optional[str] = None


def _parse_word_list(value: str) -> Set[str]:
    """
    Turn a comma-separated string like "uh,umm,hmm"
    into a set {"uh", "umm", "hmm"}.
    """
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def get_settings() -> Settings:
    """
    Read configuration from environment variables.
    """
    livekit_url = os.environ["LIVEKIT_URL"]
    livekit_api_key = os.environ["LIVEKIT_API_KEY"]
    livekit_api_secret = os.environ["LIVEKIT_API_SECRET"]
    openai_api_key = os.environ["OPENAI_API_KEY"]

    # Base/global lists (language agnostic)
    ignored_words_str = os.getenv("IGNORED_FILLER_WORDS", "uh,umm,hmm,haan")
    interrupt_words_str = os.getenv("INTERRUPT_COMMAND_WORDS", "wait,stop,no,hold on")
    threshold_str = os.getenv("FILLER_CONFIDENCE_THRESHOLD", "0.6")

    ignored_global = _parse_word_list(ignored_words_str)
    interrupt_global = _parse_word_list(interrupt_words_str)

    # --- Multi-language specific lists (EN / HI for this assignment) ---
    default_language = os.getenv("DEFAULT_LANGUAGE", "en").lower()

    ignored_en = _parse_word_list(
        os.getenv("IGNORED_FILLER_WORDS_EN", "")
    ) or ignored_global
    ignored_hi = _parse_word_list(
        os.getenv("IGNORED_FILLER_WORDS_HI", "")
    ) or ignored_global

    interrupt_en = _parse_word_list(
        os.getenv("INTERRUPT_COMMAND_WORDS_EN", "")
    ) or interrupt_global
    interrupt_hi = _parse_word_list(
        os.getenv("INTERRUPT_COMMAND_WORDS_HI", "")
    ) or interrupt_global

    ignored_by_lang: Dict[str, Set[str]] = {
        "en": ignored_en,
        "hi": ignored_hi,
    }
    commands_by_lang: Dict[str, Set[str]] = {
        "en": interrupt_en,
        "hi": interrupt_hi,
    }

    # Global unions across all languages
    global_ignored: Set[str] = set().union(*ignored_by_lang.values())
    global_interrupt: Set[str] = set().union(*commands_by_lang.values())

    try:
        threshold = float(threshold_str)
    except ValueError:
        threshold = 0.6

    dynamic_config_path = os.getenv("FILLER_CONFIG_PATH")

    return Settings(
        livekit_url=livekit_url,
        livekit_api_key=livekit_api_key,
        livekit_api_secret=livekit_api_secret,
        openai_api_key=openai_api_key,
        default_language=default_language,
        ignored_filler_words=global_ignored,
        interrupt_command_words=global_interrupt,
        ignored_filler_words_by_lang=ignored_by_lang,
        interrupt_command_words_by_lang=commands_by_lang,
        filler_confidence_threshold=threshold,
        dynamic_config_path=dynamic_config_path,
    )
