"""Clause-level sentence tokenizer with multi-language support.

Splits text at clause boundaries (commas, conjunctions, sentence endings)
instead of only at sentence endings.  This produces smaller segments that
can be dispatched to TTS sooner, reducing perceived latency without
sacrificing naturalness.

Drop-in replacement for ``basic.SentenceTokenizer`` -- just swap:

    # before
    tokenizer = tokenize.basic.SentenceTokenizer()

    # after
    tokenizer = tokenize.clause.SentenceTokenizer(language="en")
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from . import _clause_sent, token_stream, tokenizer
from ._clause_sent import ENGLISH, FRENCH, GERMAN, ITALIAN, TURKISH, LanguageProfile

__all__ = [
    "SentenceTokenizer",
    "LanguageProfile",
    "ENGLISH",
    "FRENCH",
    "GERMAN",
    "ITALIAN",
    "TURKISH",
]

_PROFILES: dict[str, LanguageProfile] = {
    "en": ENGLISH,
    "english": ENGLISH,
    "fr": FRENCH,
    "french": FRENCH,
    "de": GERMAN,
    "german": GERMAN,
    "it": ITALIAN,
    "italian": ITALIAN,
    "tr": TURKISH,
    "turkish": TURKISH,
}


@dataclass
class _TokenizerOptions:
    profile: LanguageProfile
    min_clause_len: int
    stream_context_len: int
    retain_format: bool


class SentenceTokenizer(tokenizer.SentenceTokenizer):
    """Multi-language clause tokenizer for TTS latency optimization.

    Splits text at clause boundaries using language-specific rules
    (conjunctions, abbreviations, number formats).  Finer-grained than
    ``basic.SentenceTokenizer`` while still producing natural-sounding
    segments.

    Args:
        language: Language code ("en", "english", "tr", "turkish") or a
            ``LanguageProfile`` instance for custom languages.
        min_clause_len: Minimum clause length in characters.  Shorter
            clauses are merged with the next one.
        stream_context_len: Minimum buffer size for streaming mode.
        retain_format: Preserve original whitespace/newlines in output.
    """

    def __init__(
        self,
        *,
        language: str | LanguageProfile = "en",
        min_clause_len: int = 15,
        stream_context_len: int = 10,
        retain_format: bool = False,
    ) -> None:
        if isinstance(language, LanguageProfile):
            profile = language
        else:
            _prof = _PROFILES.get(language.lower())
            if _prof is None:
                raise ValueError(
                    f"unknown language {language!r}, "
                    f"available: {sorted(_PROFILES.keys())}. "
                    f"Pass a LanguageProfile instance for custom languages."
                )
            profile = _prof

        self._config = _TokenizerOptions(
            profile=profile,
            min_clause_len=min_clause_len,
            stream_context_len=stream_context_len,
            retain_format=retain_format,
        )

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        return [
            tok[0]
            for tok in _clause_sent.split_clauses(
                text,
                profile=self._config.profile,
                min_clause_len=self._config.min_clause_len,
                retain_format=self._config.retain_format,
            )
        ]

    def stream(self, *, language: str | None = None) -> tokenizer.SentenceStream:
        return token_stream.BufferedSentenceStream(
            tokenizer=functools.partial(
                _clause_sent.split_clauses,
                profile=self._config.profile,
                min_clause_len=self._config.min_clause_len,
                retain_format=self._config.retain_format,
            ),
            min_token_len=self._config.min_clause_len,
            min_ctx_len=self._config.stream_context_len,
        )
