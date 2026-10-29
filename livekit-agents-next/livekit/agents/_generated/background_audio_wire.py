# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/background_audio.rs

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from livekit.agents.voice.background_audio import _FrameStream


@dataclass(frozen=True, kw_only=True, slots=True)
class Sound:
    """One candidate sound, as the host hands it over: a file path (a builtin clip arrives as its
    path) or a host stream, and how to play it.
    """

    path: str | None
    stream: _FrameStream | None
    volume: float
    """0.0-1.0."""
    probability: float
    """The chance this one is picked when several are offered. 0.0-1.0."""
    fade_in: float
    """Seconds to ramp from 0 up to `volume` when playback starts. `0` starts at full volume."""
    fade_out: float
    """Seconds to ramp back down to 0 when `stop()` is called. `0` cuts immediately."""
