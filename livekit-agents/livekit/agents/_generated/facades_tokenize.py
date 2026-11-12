# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import _bindings, _core
from ..tokenize import _SentenceTokenizerBase, _WordTokenizerBase


class WordTokenizer(_WordTokenizerBase):
    """Splits text into words: at whitespace, with each Chinese or Japanese character a word of its
    own. Thai, Lao, Khmer and Burmese split only at whitespace.

    The stream emits a word once the next one proves it whole. Subclass it for a custom
    tokenizer: override `tokenize`, and the inherited `stream` emits each of its words once the
    next one starts; override `stream` too to stream them yourself.

    Args:
        ignore_punctuation: drop punctuation from the words; a word of punctuation alone is
            dropped.
        retain_format: keep the whitespace before each word, so the words join back to the text.
    """

    def __init__(self, *, ignore_punctuation: bool = True, retain_format: bool = False) -> None:
        super().__init__()
        self._native = _core.SpeechWordTokenizer(
            ignore_punctuation=ignore_punctuation,
            retain_format=retain_format,
        )

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        return self._native.tokenize(text, language=language)


class SentenceTokenizer(_SentenceTokenizerBase):
    """Splits text into sentences for speech, batched so short ones are not spoken alone.

    The stream emits a sentence as soon as no text to come could change where it ends. Subclass
    it for a custom tokenizer: override `tokenize`, and the inherited `stream` emits each of its
    sentences once the next one starts; override `stream` too to stream them yourself.

    Args:
        min_words: sentences are batched until a chunk holds this many spoken words. A Chinese or
            Japanese character counts as a word, and Thai about every three characters.
        clause_words: a sentence running past this many words without an end breaks at its next
            clause mark (``,`` ``;`` ``:`` and their fullwidth forms, never inside a number such as
            ``1,000``). ``None`` breaks only at sentence ends.
        retain_format: keep original whitespace/formatting in emitted tokens.
        max_token_len: cap on a chunk's length in characters: batching stops before a sentence
            that would pass it. A longer sentence is cut at its last clause mark or word
            boundary before the cap (1000 characters when unset).
    """

    def __init__(
        self,
        *,
        min_words: int = 4,
        clause_words: int | None = None,
        retain_format: bool = False,
        max_token_len: int | None = None,
    ) -> None:
        super().__init__()
        self._native = _core.SpeechSentenceTokenizer(
            min_words=min_words,
            clause_words=clause_words,
            retain_format=retain_format,
            max_token_len=max_token_len,
        )

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        """The sentences of `text`, batched as the stream batches them."""
        return self._native.tokenize(text, language=language)


WordTokenizer.__module__ = "livekit.agents.tokenize"
SentenceTokenizer.__module__ = "livekit.agents.tokenize"
