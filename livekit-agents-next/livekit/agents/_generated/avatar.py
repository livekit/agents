# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/utils/video.rs
#     rust/crates/lk-agents-abi/src/voice/avatar.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from .rtc_frames import AudioFrame, VideoFrame

if TYPE_CHECKING:
    from .io_events import AvatarPlaybackSegment


class AudioSegmentEnd:
    """Marks the end of one avatar audio segment."""

    pass


@dataclass(kw_only=True, slots=True)
class AudioReceiverObservation:
    """What the avatar runner reads from its receiver: the clear epoch the frame was captured in,
    whether this first frame opened a segment, the segment, and the frame (`None` at the end).
    """

    epoch: int
    starts_segment: bool
    segment: AvatarPlaybackSegment | None
    frame: AudioReceiverFrame | None


MediaFrame: TypeAlias = AudioFrame | VideoFrame
"""A frame the SDK's synchronizer takes: audio or video."""


AvatarFrame: TypeAlias = AudioFrame | VideoFrame | AudioSegmentEnd
"""What an avatar's video generator yields: a frame, or the end of a segment
(`AudioSegmentEnd`).
"""


AudioReceiverFrame: TypeAlias = AudioFrame | AudioSegmentEnd
"""An avatar audio receiver yields audio or the end of its segment, never video."""


AudioSegmentEnd.__module__ = "livekit.agents.voice.avatar"
