# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/tts/segments.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations
from typing import TypeAlias

from .transcript import TranscriptChunk
from .tts import SpeechChunk
from .usage import TokenUsage


@dataclass
class SegmentStart:
    """A segment opens; its text follows. The id is the framework's, unique to the segment."""

    segment_id: str


@dataclass
class SegmentText:
    """Text of a segment: `chunk.text` in the provider's syntax, `chunk.said` its words."""

    segment_id: str
    chunk: SpeechChunk


@dataclass
class SegmentEnd:
    """A segment's text ended: the provider owes its audio, then its `SegmentDone`."""

    segment_id: str


@dataclass
class SegmentAudio:
    """Audio of a segment, encoded as the stream's `mime_type`."""

    segment_id: str
    data: bytes


@dataclass
class SegmentWords:
    """Word timings of a segment, from its start."""

    segment_id: str
    words: list[TranscriptChunk]


@dataclass
class SegmentDone:
    """The provider finished a segment: its audio is complete."""

    segment_id: str


@dataclass
class SegmentUsage:
    """What the provider billed for a segment: its tokens, its characters, or both. Each report is
    one provider request's share; it rides on the segment's final audio into its metrics.
    """

    segment_id: str
    usage: TokenUsage | None = None
    characters_count: int | None = None


SegmentInput: TypeAlias = SegmentStart | SegmentText | SegmentEnd
"""What a socket TTS stream's `encode` receives for its segments."""


SegmentOutput: TypeAlias = SegmentAudio | SegmentWords | SegmentUsage | SegmentDone
"""What a WebSocket TTS stream's `on_message` returns for a segment."""


SegmentStart.__annotations__ = get_annotations(SegmentStart, eval_str=True)
SegmentStart.__module__ = "livekit.agents.tts"
SegmentText.__annotations__ = get_annotations(SegmentText, eval_str=True)
SegmentText.__module__ = "livekit.agents.tts"
SegmentEnd.__annotations__ = get_annotations(SegmentEnd, eval_str=True)
SegmentEnd.__module__ = "livekit.agents.tts"
SegmentAudio.__annotations__ = get_annotations(SegmentAudio, eval_str=True)
SegmentAudio.__module__ = "livekit.agents.tts"
SegmentWords.__annotations__ = get_annotations(SegmentWords, eval_str=True)
SegmentWords.__module__ = "livekit.agents.tts"
SegmentDone.__annotations__ = get_annotations(SegmentDone, eval_str=True)
SegmentDone.__module__ = "livekit.agents.tts"
SegmentUsage.__annotations__ = get_annotations(SegmentUsage, eval_str=True)
SegmentUsage.__module__ = "livekit.agents.tts"
