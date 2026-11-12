# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/avatar_queue.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass

from .io_events import PlaybackSegment


@dataclass
class QueueFinished:
    segment: PlaybackSegment
    playback_position: float
    interrupted: bool


@dataclass
class QueueEffects:
    started: list[PlaybackSegment]
    finished: list[QueueFinished]
    clear: bool
    flushed: bool
    reset: bool
