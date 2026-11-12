# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/io.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations

from .io_events import PlaybackSegment


@dataclass
class PlaybackFinishedEvent:
    playback_position: float
    """How much of the audio was played back"""
    interrupted: bool
    """Interrupted is True if playback was interrupted (clear_buffer() was called)"""
    synchronized_transcript: str | None = None
    """Transcript synced with playback; may be partial if the audio was interrupted"""
    segment: PlaybackSegment | None = None
    """The captured segment key on emitted playback observations."""


@dataclass
class PlaybackStartedEvent:
    created_at: float
    segment: PlaybackSegment
    """The timestamp (time.time()) when the playback started"""


@dataclass
class PlaybackProgressedEvent:
    """A stretch of the current segment that has played; never revised."""

    started_at: float
    offset: float
    duration: float
    segment: PlaybackSegment


PlaybackFinishedEvent.__annotations__ = get_annotations(PlaybackFinishedEvent, eval_str=True)
PlaybackFinishedEvent.__module__ = "livekit.agents.voice.io"
PlaybackStartedEvent.__annotations__ = get_annotations(PlaybackStartedEvent, eval_str=True)
PlaybackStartedEvent.__module__ = "livekit.agents.voice.io"
PlaybackProgressedEvent.__annotations__ = get_annotations(PlaybackProgressedEvent, eval_str=True)
PlaybackProgressedEvent.__module__ = "livekit.agents.voice.io"
