# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/tts/transcript.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations


@dataclass
class TranscriptChunk:
    """A transcript word or chunk with optional alignment, confidence and speaker metadata.

    Times are seconds on the owning audio stream's timeline. Stream adapters apply the source
    offset when normalizing provider events; `start_time_offset` records that offset and must
    not be added again to normalized start/end times. Missing metadata is `None`.
    """

    text: str
    start_time: float | None = None
    """Start time in seconds on the audio stream's timeline."""
    end_time: float | None = None
    """End time in seconds on the audio stream's timeline."""
    confidence: float | None = None
    """Provider-reported confidence, when available."""
    start_time_offset: float | None = None
    """The source offset already applied by the stream adapter."""
    speaker_id: str | None = None


TranscriptChunk.__annotations__ = get_annotations(TranscriptChunk, eval_str=True)
TranscriptChunk.__module__ = "livekit.agents.types"
