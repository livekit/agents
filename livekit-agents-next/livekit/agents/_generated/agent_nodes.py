# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/nodes.rs

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from livekit.agents.types import FlushSentinel

from .llm import ChatChunk

if TYPE_CHECKING:
    from livekit.agents.types import TimedString
    from livekit.rtc import AudioFrame


@dataclass(frozen=True, kw_only=True, slots=True)
class NodeFrame:
    """A frame the default `tts_node` yields, with what the reference stamps on its `userdata`: the
    words it covers and when its synthesis started (wall-clock seconds).
    """

    frame: AudioFrame
    timed_transcript: list[TimedString]
    started_time: float | None


NodeChunk: TypeAlias = str | FlushSentinel | ChatChunk
"""What an `llm_node` yields: text it answered itself (spoken, never metered as a model call), a
`FlushSentinel` (the spoken segment ends here), or a model's chunk.
"""
