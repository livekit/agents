from __future__ import annotations

import functools
from dataclasses import dataclass

from . import (
    _basic_hyphenator,
    _basic_paragraph,
    _basic_sent,
    _basic_word,
    token_stream,
    tokenizer,
)

# Really naive implementation of SentenceTokenizer, WordTokenizer + hyphenate_word
# The basic tokenizer is rule-based and only English is really tested

__all__ = [
    "SentenceTokenizer",
    "WordTokenizer",
    "hyphenate_word",
    "tokenize_paragraphs",
]


@dataclass
class _TokenizerOptions:
    language: str
    min_sentence_len: int
    stream_context_len: int


class SentenceTokenizer(tokenizer.SentenceTokenizer):
    def __init__(
        self,
        *,
        language: str = "english",
        min_sentence_len: int = 20,
        stream_context_len: int = 10,
    ) -> None:
        self._config = _TokenizerOptions(
            language=language,
            min_sentence_len=min_sentence_len,
            stream_context_len=stream_context_len,
        )

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        return [
            tok[0]
            for tok in _basic_sent.split_sentences(
                text, min_sentence_len=self._config.min_sentence_len
            )
        ]

    def stream(self, *, language: str | None = None) -> tokenizer.SentenceStream:
        return token_stream.BufferedSentenceStream(
            tokenizer=functools.partial(
                _basic_sent.split_sentences,
                min_sentence_len=self._config.min_sentence_len,
            ),
            min_token_len=self._config.min_sentence_len,
            min_ctx_len=self._config.stream_context_len,
        )


class WordTokenizer(tokenizer.WordTokenizer):
    def __init__(self, *, ignore_punctuation: bool = True) -> None:
        self._ignore_punctuation = ignore_punctuation

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        return [
            tok[0]
            for tok in _basic_word.split_words(
                text, ignore_punctuation=self._ignore_punctuation
            )
        ]

    def stream(self, *, language: str | None = None) -> tokenizer.WordStream:
        return token_stream.BufferedWordStream(
            tokenizer=functools.partial(
                _basic_word.split_words, ignore_punctuation=self._ignore_punctuation
            ),
            min_token_len=1,
            min_ctx_len=1,  # ignore
        )


def hyphenate_word(word: str) -> list[str]:
    return _basic_hyphenator.hyphenate_word(word)


def tokenize_paragraphs(text: str) -> list[str]:
    return [tok[0] for tok in _basic_paragraph.split_paragraphs(text)]
