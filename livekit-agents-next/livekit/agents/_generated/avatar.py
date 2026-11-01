# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/utils/video.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import TypeAlias

from .rtc_frames import AudioFrame, VideoFrame


class AudioSegmentEnd:
    """Marks the end of one avatar audio segment."""

    pass


MediaFrame: TypeAlias = AudioFrame | VideoFrame
"""A frame the SDK's synchronizer takes: audio or video."""


AvatarFrame: TypeAlias = AudioFrame | VideoFrame | AudioSegmentEnd
"""What an avatar's video generator yields: a frame, or the end of a segment
(`AudioSegmentEnd`).
"""


AudioReceiverFrame: TypeAlias = AudioFrame | AudioSegmentEnd
"""An avatar audio receiver yields audio or the end of its segment, never video."""


AudioSegmentEnd.__module__ = "livekit.agents.voice.avatar"
