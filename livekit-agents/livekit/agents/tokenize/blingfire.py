from __future__ import annotations

import functools
import re
from dataclasses import dataclass

from livekit import blingfire

from . import token_stream, tokenizer
from .token_stream import _has_unclosed_xml_tags

__all__ = [
    "SentenceTokenizer",
]


def _split_sentences(
    text: str, min_sentence_len: int, *, retain_format: bool = False
) -> list[tuple[str, int, int]]:
    _, offsets = blingfire.text_to_sentences_with_offsets(text)

    sentences: list[tuple[str, int, int]] = []
    start = 0

    for _, end in offsets:
        raw_sentence = text[start:end]
        sentence = re.sub(r"\s*\n+\s*", " ", raw_sentence).strip()
        if not sentence or len(sentence) < min_sentence_len:
            continue

        if retain_format:
            sentences.append((raw_sentence, start, end))
        else:
            sentences.append((sentence, start, end))
        start = end

    if start < len(text):
        raw_sentence = text[start:]
        if retain_format:
            sentences.append((raw_sentence, start, len(text)))
        elif sentence := raw_sentence.strip():
            sentences.append((sentence, start, len(text)))

    # merge sentences where a split landed inside an XML tag
    if sentences:
        merged: list[tuple[str, int, int]] = [sentences[0]]
        for sent_text, s_start, s_end in sentences[1:]:
            prev_text, prev_start, prev_end = merged[-1]
            if _has_unclosed_xml_tags(prev_text):
                gap = text[prev_end:s_start] if retain_format else " "
                merged[-1] = (prev_text + gap + sent_text, prev_start, s_end)
            else:
                merged.append((sent_text, s_start, s_end))
        sentences = merged

    return sentences


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
