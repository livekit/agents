from __future__ import annotations

import functools
from dataclasses import dataclass

from livekit import blingfire

from . import token_stream, tokenizer

__all__ = [
    "SentenceTokenizer",
]


def _split_sentences(text: str, min_sentence_len: int) -> list[tuple[str, int, int]]:
    bf_sentences, offsets = blingfire.text_to_sentences_with_offsets(text)
    raw_sentences = bf_sentences.split("\n")

    merged_sentences = []
    buffer = ""
    buffer_start = None

    for i, (sentence, (start, end)) in enumerate(zip(raw_sentences, offsets)):
        sentence = sentence.strip()
        if not sentence:
            continue

        if buffer:
            buffer += " " + sentence
            buffer_end = end
        else:
            buffer = sentence
            buffer_start = start
            buffer_end = end

        if len(buffer) >= min_sentence_len or i == len(offsets) - 1:
            merged_sentences.append((buffer, buffer_start, buffer_end))
            buffer = ""
            buffer_start = None

    return merged_sentences


@dataclass
class _TokenizerOptions:
    min_sentence_len: int
    stream_context_len: int


class SentenceTokenizer(tokenizer.SentenceTokenizer):
    def __init__(
        self,
        *,
        min_sentence_len: int = 20,
        stream_context_len: int = 10,
    ) -> None:
        self._config = _TokenizerOptions(
            min_sentence_len=min_sentence_len, stream_context_len=stream_context_len
        )

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        return [
            tok[0] for tok in _split_sentences(text, min_sentence_len=self._config.min_sentence_len)
        ]

    def stream(self, *, language: str | None = None) -> tokenizer.SentenceStream:
        return token_stream.BufferedSentenceStream(
            tokenizer=functools.partial(
                _split_sentences, min_sentence_len=self._config.min_sentence_len
            ),
            min_token_len=self._config.min_sentence_len,
            min_ctx_len=self._config.stream_context_len,
        )
