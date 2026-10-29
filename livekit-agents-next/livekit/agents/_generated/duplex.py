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
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from livekit.rtc import AudioFrame


@dataclass(frozen=True, kw_only=True, slots=True)
class DuplexCapabilities:
    """What varies between duplex providers."""

    user_transcription: bool
    auto_tool_reply_generation: bool
    mutable_chat_context: bool = False
    mutable_instructions: bool = False
    mutable_tools: bool = False


@dataclass(frozen=True, kw_only=True, slots=True)
class DuplexAudio:
    """One frame of the model's output audio; `start_ms` places it on the model's timeline."""

    frame: AudioFrame
    start_ms: int | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class DuplexTranscript:
    """A fragment of the model's transcript of its own speech."""

    text: str
    start_ms: int | None = None
    end_ms: int | None = None


DuplexEvent: TypeAlias = DuplexTranscript | dict[str, object]
"""One event of a plugin session: its transcript, or what a realtime session reports."""
