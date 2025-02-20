"""
Basic text tokenization utilities for English text processing.

Provides rule-based implementations for:
- Sentence splitting
- Word tokenization
- Paragraph detection
- Hyphenation

Note: Primarily designed for English text processing.
"""

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
    """Configuration options for basic tokenizers."""
    language: str          # Language code (primarily 'english')
    min_sentence_len: int  # Minimum characters to consider a sentence
    stream_context_len: int  # Buffer size for streaming processing


class SentenceTokenizer(tokenizer.SentenceTokenizer):
    """Rule-based sentence tokenizer optimized for English text."""
    
    def __init__(
        self,
        *,
        language: str = "english",
        min_sentence_len: int = 20,
        stream_context_len: int = 10,
    ) -> None:
        """
        Initialize sentence tokenizer with configurable thresholds.
        
        Args:
            language: Supported language (default: english)
            min_sentence_len: Minimum character length for a valid sentence
            stream_context_len: Buffer size for streaming input
        """
        self._config = _TokenizerOptions(
            language=language,
            min_sentence_len=min_sentence_len,
            stream_context_len=stream_context_len,
        )

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        """Split text into sentences using rule-based detection."""
        return [
            tok[0]
            for tok in _basic_sent.split_sentences(
                text, min_sentence_len=self._config.min_sentence_len
            )
        ]

    def stream(self, *, language: str | None = None) -> tokenizer.SentenceStream:
        """Create streaming sentence tokenizer instance."""
        return token_stream.BufferedSentenceStream(
            tokenizer=functools.partial(
                _basic_sent.split_sentences,
                min_sentence_len=self._config.min_sentence_len,
            ),
            min_token_len=self._config.min_sentence_len,
            min_ctx_len=self._config.stream_context_len,
        )


class WordTokenizer(tokenizer.WordTokenizer):
    """Basic word tokenizer with punctuation handling options."""
    
    def __init__(self, *, ignore_punctuation: bool = True) -> None:
        """
        Initialize word tokenizer.
        
        Args:
            ignore_punctuation: Whether to exclude punctuation from tokens
        """
        self._ignore_punctuation = ignore_punctuation

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        """Split text into words, optionally ignoring punctuation."""
        return [
            tok[0]
            for tok in _basic_word.split_words(
                text, ignore_punctuation=self._ignore_punctuation
            )
        ]

    def stream(self, *, language: str | None = None) -> tokenizer.WordStream:
        """Create streaming word tokenizer instance."""
        return token_stream.BufferedWordStream(
            tokenizer=functools.partial(
                _basic_word.split_words, ignore_punctuation=self._ignore_punctuation
            ),
            min_token_len=1,
            min_ctx_len=1,  # Process immediately
        )


def hyphenate_word(word: str) -> list[str]:
    """Split word into hyphenation components using basic rules."""
    return _basic_hyphenator.hyphenate_word(word)


def split_words(
    text: str, ignore_punctuation: bool = True
) -> list[tuple[str, int, int]]:
    """Split text into words with position information."""
    return _basic_word.split_words(text, ignore_punctuation=ignore_punctuation)


def tokenize_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs using newline detection."""
    return [tok[0] for tok in _basic_paragraph.split_paragraphs(text)]
