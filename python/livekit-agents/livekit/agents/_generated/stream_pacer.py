# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/tts/stream_pacer.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations


@dataclass
class SentenceStreamPacer:
    """Paces the text sent to a TTS: the first chunk goes at once; each later batch once the audio
    stopped growing and at most ``min_remaining_audio`` seconds of it are left to play, of whole
    chunks joined by a space up to ``max_text_length`` characters (never past the provider's
    request limit). It wastes less on interruptions and gives the TTS more context.
    """

    min_remaining_audio: float = 5.0
    """The audio left, in seconds, below which the next batch goes."""
    max_text_length: int = 300
    """The longest batch sent at once, in characters."""


SentenceStreamPacer.__annotations__ = get_annotations(SentenceStreamPacer, eval_str=True)
SentenceStreamPacer.__module__ = "livekit.agents.tts"
