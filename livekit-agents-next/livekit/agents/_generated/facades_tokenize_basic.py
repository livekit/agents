# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import _bindings, _core
from ..tokenize import _CoreSentenceTokenizer, _CoreWordTokenizer


class WordTokenizer(_CoreWordTokenizer):
    """Really naive implementation of a word tokenizer: words split on whitespace, punctuation
    stripped unless asked to keep it.
    """

    def __init__(
        self,
        *,
        ignore_punctuation: bool = True,
        split_character: bool = False,
        retain_format: bool = False,
    ) -> None:
        super().__init__()
        self._native = _core.BasicWordTokenizer(
            ignore_punctuation=ignore_punctuation,
            split_character=split_character,
            retain_format=retain_format,
        )

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        return self._native.tokenize(text, language=language)


class SentenceTokenizer(_CoreSentenceTokenizer):
    """Really naive implementation of a sentence tokenizer: rule-based, and only English is really
    tested.
    """

    def __init__(
        self,
        *,
        language: str = "english",
        min_sentence_len: int = 20,
        stream_context_len: int = 10,
        retain_format: bool = False,
    ) -> None:
        super().__init__()
        self._native = _core.BasicSentenceTokenizer(
            language=language,
            min_sentence_len=min_sentence_len,
            stream_context_len=stream_context_len,
            retain_format=retain_format,
        )

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        return self._native.tokenize(text, language=language)


WordTokenizer.__module__ = "livekit.agents.tokenize.basic"
SentenceTokenizer.__module__ = "livekit.agents.tokenize.basic"
