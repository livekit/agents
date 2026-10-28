# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the declarations in:
#     rust/crates/lk-agents-abi/src/tokenize/bindings.rs
#     rust/crates/lk-agents-voice/src/transcription/synchronizer.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from .. import _core
from .._core import hyphenate_word, split_words, tokenize_paragraphs
from ..tokenize import _CoreSentenceTokenizer, _CoreWordTokenizer

__all__ = [
    "WordTokenizer",
    "SentenceTokenizer",
    "hyphenate_word",
    "split_words",
    "tokenize_paragraphs",
]


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
        self._core = _core.BasicWordTokenizer(
            ignore_punctuation=ignore_punctuation,
            split_character=split_character,
            retain_format=retain_format,
        )


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
        self._core = _core.BasicSentenceTokenizer(
            language=language,
            min_sentence_len=min_sentence_len,
            stream_context_len=stream_context_len,
            retain_format=retain_format,
        )


WordTokenizer.__module__ = "livekit.agents.tokenize.basic"
SentenceTokenizer.__module__ = "livekit.agents.tokenize.basic"
