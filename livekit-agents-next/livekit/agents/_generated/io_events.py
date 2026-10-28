# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/io.rs

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True, slots=True)
class PlaybackSegment:
    """Correlation supplied at segment admission and retained by every transport operation.
    A delayed report from an earlier attachment must not settle its replacement.
    """

    attachment: int
    segment: int


@dataclass(frozen=True, kw_only=True, slots=True)
class AvatarPlaybackSegment:
    """An avatar receiver retains the source key and, for RTC, its existing byte-stream id."""

    playback: PlaybackSegment
    stream_id: str | None
