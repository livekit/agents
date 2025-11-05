from __future__ import annotations

import functools
import re
from dataclasses import dataclass

from livekit import blingfire

from . import token_stream, tokenizer

__all__ = [
    "SentenceTokenizer",
]


def _split_sentences(
    text: str, min_sentence_len: int, *, retain_format: bool = False
) -> list[tuple[str, int, int]]:
    _, offsets = blingfire.text_to_sentences_with_offsets(text)

    # Added boundaries where punctuation is immediately followed by '<' or a capital letter.
    ends: list[int] = [end for _, end in offsets]
    for m in re.finditer(r"[.!?](?=(?:<|[A-Z]))", text):
        ends.append(m.end())

    # Normalize and sort unique boundaries
    ends = sorted({e for e in ends if 0 < e <= len(text)})

    merged_sentences = []
    start = 0

    for end in ends:
        if end <= start:
            continue
        raw_sentence = text[start:end]
        sentence = re.sub(r"\s*\n+\s*", " ", raw_sentence).strip()
        if not sentence or len(sentence) < min_sentence_len:
            continue

        if retain_format:
            merged_sentences.append((raw_sentence, start, end))
        else:
            merged_sentences.append((sentence, start, end))
        start = end

    if start < len(text):
        raw_sentence = text[start:]
        if retain_format:
            merged_sentences.append((raw_sentence, start, len(text)))
        elif sentence := raw_sentence.strip():
            merged_sentences.append((sentence, start, len(text)))

    return merged_sentences


@dataclass
class _TokenizerOptions:
    min_sentence_len: int
    stream_context_len: int
    retain_format: bool


class SentenceTokenizer(tokenizer.SentenceTokenizer):
    def __init__(
        self,
        *,
        min_sentence_len: int = 20,
        stream_context_len: int = 10,
        retain_format: bool = False,
    ) -> None:
        self._config = _TokenizerOptions(
            min_sentence_len=min_sentence_len,
            stream_context_len=stream_context_len,
            retain_format=retain_format,
        )

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        return [
            tok[0]
            for tok in _split_sentences(
                text,
                min_sentence_len=self._config.min_sentence_len,
                retain_format=self._config.retain_format,
            )
        ]

    def stream(self, *, language: str | None = None) -> tokenizer.SentenceStream:
        return token_stream.BufferedSentenceStream(
            tokenizer=functools.partial(
                _split_sentences,
                min_sentence_len=self._config.min_sentence_len,
                retain_format=self._config.retain_format,
            ),
            min_token_len=self._config.min_sentence_len,
            min_ctx_len=self._config.stream_context_len,
        )
