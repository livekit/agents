# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/room_io.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations

from livekit import rtc


@dataclass
class TextInputEvent:
    text: str
    info: rtc.TextStreamInfo | None = None
    participant: rtc.RemoteParticipant | None = None


@dataclass
class NoiseCancellationParams:
    participant: rtc.Participant
    track: rtc.Track


TextInputEvent.__annotations__ = get_annotations(TextInputEvent, eval_str=True)
TextInputEvent.__module__ = "livekit.agents.voice.room_io"
NoiseCancellationParams.__annotations__ = get_annotations(NoiseCancellationParams, eval_str=True)
NoiseCancellationParams.__module__ = "livekit.agents.voice.room_io"
