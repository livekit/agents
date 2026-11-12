# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/llm/duplex.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations
from typing import TypeAlias

from .realtime_wire import Event
from .rtc_frames import AudioFrame


@dataclass
class DuplexCapabilities:
    """What varies between duplex providers."""

    user_transcription: bool
    auto_tool_reply_generation: bool
    mutable_chat_context: bool = False
    mutable_instructions: bool = False
    mutable_tools: bool = False


@dataclass
class DuplexAudioFrame:
    """One frame of the model's output audio; `start_ms` places it on the model's timeline."""

    frame: AudioFrame
    start_ms: int | None = None


@dataclass
class DuplexOutputTranscriptDelta:
    """A fragment of the model's transcript of its own speech."""

    text: str
    start_ms: int | None = None
    end_ms: int | None = None


@dataclass
class DuplexInputGap:
    """Caller audio the input dropped to stay current, as the seconds it covered: silence to the model."""

    duration: float


@dataclass
class DuplexInputMute:
    """The caller's input is muted, or unmuted, from this point of the audio on."""

    muted: bool


DuplexEvent: TypeAlias = DuplexOutputTranscriptDelta | Event
"""One event of a plugin session: its transcript, or what a realtime session reports."""


DuplexInputItem: TypeAlias = AudioFrame | DuplexInputGap | DuplexInputMute
"""What a plugin pulls from its input, in order: the caller's audio, a gap, a mute or an unmute."""


DuplexCapabilities.__annotations__ = get_annotations(DuplexCapabilities, eval_str=True)
DuplexCapabilities.__module__ = "livekit.agents.llm.duplex"
DuplexAudioFrame.__annotations__ = get_annotations(DuplexAudioFrame, eval_str=True)
DuplexAudioFrame.__module__ = "livekit.agents.llm.duplex"
DuplexOutputTranscriptDelta.__annotations__ = get_annotations(DuplexOutputTranscriptDelta, eval_str=True)
DuplexOutputTranscriptDelta.__module__ = "livekit.agents.llm.duplex"
DuplexInputGap.__annotations__ = get_annotations(DuplexInputGap, eval_str=True)
DuplexInputGap.__module__ = "livekit.agents.llm.duplex"
DuplexInputMute.__annotations__ = get_annotations(DuplexInputMute, eval_str=True)
DuplexInputMute.__module__ = "livekit.agents.llm.duplex"
