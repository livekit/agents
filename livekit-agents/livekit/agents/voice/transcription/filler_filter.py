from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Iterable, Sequence

from livekit.agents.tokenize.basic import split_words

DEFAULT_FILLER_TOKENS: tuple[str, ...] = (
    "um",
    "uh",
    "uhm",
    "uhh",
    "erm",
    "er",
    "ah",
    "eh",
    "hmm",
    "mm",
    "mmm",
    "uhhuh",
)


@dataclass(frozen=True)
class InterruptionFilterResult:
    tokens: tuple[str, ...]
    is_filler_only: bool

    @property
    def token_count(self) -> int:
        return len(self.tokens)


def _normalize_text(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for raw_word, _, _ in split_words(text, ignore_punctuation=True, split_character=True):
        normalized = raw_word.strip().lower()
        if normalized:
            tokens.append(normalized)
    return tuple(tokens)


def _normalize_token_sequence(tokens: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for token in tokens:
        normalized.extend(_normalize_text(token))

    # deduplicate while preserving insertion order
    seen: dict[str, None] = {}
    for tok in normalized:
        seen.setdefault(tok, None)
    return tuple(seen.keys())


class FillerOnlyTranscriptFilter:
    def __init__(
        self,
        *,
        tokens: Sequence[str] | None = None,
        enabled: bool = True,
    ) -> None:
        normalized = (
            _normalize_token_sequence(tokens) if tokens is not None else DEFAULT_FILLER_TOKENS
        )

        self._lock = RLock()
        self._tokens: tuple[str, ...] = normalized
        self._enabled = enabled

    @property
    def tokens(self) -> tuple[str, ...]:
        with self._lock:
            return self._tokens

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        with self._lock:
            self._enabled = value

    def evaluate(self, transcript: str) -> InterruptionFilterResult:
        tokens = _normalize_text(transcript)
        with self._lock:
            enabled = self._enabled
            filler_tokens = set(self._tokens)

        if not tokens:
            return InterruptionFilterResult(tokens=tokens, is_filler_only=True)

        if not enabled:
            return InterruptionFilterResult(tokens=tokens, is_filler_only=False)

        is_filler_only = all(token in filler_tokens for token in tokens)
        return InterruptionFilterResult(tokens=tokens, is_filler_only=is_filler_only)

    def update_tokens(self, tokens: Iterable[str]) -> None:
        normalized = _normalize_token_sequence(list(tokens))
        with self._lock:
            self._tokens = normalized

    def extend_tokens(self, tokens: Iterable[str]) -> None:
        normalized = _normalize_token_sequence(list(tokens))
        if not normalized:
            return

        with self._lock:
            current = list(self._tokens)
            for token in normalized:
                if token not in current:
                    current.append(token)
            self._tokens = tuple(current)
