import functools
from dataclasses import dataclass

from . import _basic_hyphenator, _basic_sent, token_stream, tokenizer

# Really naive implementation of SentenceTokenizer, WordTokenizer + hyphenate_word
# The basic tokenizer is rule-based and only English is really tested

__all__ = [
    "SentenceTokenizer",
    "WordTokenizer",
    "hyphenate_word",
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

    def tokenize(self, *, text: str, language: str | None = None) -> list[str]:
        return _basic_sent.split_sentences(
            text, min_sentence_len=self._config.min_sentence_len
        )

    def stream(self, *, language: str | None = None) -> tokenizer.SentenceStream:
        return token_stream.BufferedTokenStream(
            tokenizer=functools.partial(
                _basic_sent.split_sentences,
                min_sentence_len=self._config.min_sentence_len,
            ),
            min_token_len=self._config.min_sentence_len,
            ctx_len=self._config.stream_context_len,
        )


class WordTokenizer(tokenizer.WordTokenizer):
    pass


hyphenate_word = _basic_hyphenator.hyphenate_word
