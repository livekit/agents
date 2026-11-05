# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the declarations in:
#     rust/crates/lk-agents-abi/src/tokenize/bindings.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from .. import _core
from ..tokenize import _CoreSentenceTokenizer

__all__ = [
    "SentenceTokenizer",
]


class SentenceTokenizer(_CoreSentenceTokenizer):
    """Sentences split by the BlingFire model, batched for speech.

    Args:
        min_sentence_len: minimum length for a span to be treated as its own sentence; shorter
            spans are merged forward into the next one.
        stream_context_len: minimum buffered text before the stream emits.
        retain_format: keep original whitespace/formatting in emitted tokens.
        max_token_len: hard cap on emitted token length; a token is flushed before appending a
            sentence that would exceed it.
        min_token_len: minimum length a token must reach before it is emitted. Sentences are
            batched until the running token reaches this length, so raising it (e.g. toward
            ``max_token_len``) yields larger, fewer chunks. Defaults to ``min_sentence_len``.
        xml_aware: treat XML markup as atomic: never split a tag across tokens and keep tags
            attached to the following sentence. Only enable when the input carries markup (e.g.
            expressive TTS): a stray "<" in plain text can otherwise hold back streaming.
    """

    def __init__(
        self,
        *,
        min_sentence_len: int = 20,
        stream_context_len: int = 10,
        retain_format: bool = False,
        max_token_len: int | None = None,
        min_token_len: int | None = None,
        xml_aware: bool = False,
    ) -> None:
        self._core = _core.BlingfireSentenceTokenizer(
            min_sentence_len=min_sentence_len,
            stream_context_len=stream_context_len,
            retain_format=retain_format,
            max_token_len=max_token_len,
            min_token_len=min_token_len,
            xml_aware=xml_aware,
        )


SentenceTokenizer.__module__ = "livekit.agents.tokenize.blingfire"
