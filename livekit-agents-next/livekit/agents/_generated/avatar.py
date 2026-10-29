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

from livekit.agents.voice.avatar import AudioSegmentEnd
from livekit.rtc import AudioFrame

AudioReceiverFrame: TypeAlias = AudioFrame | AudioSegmentEnd
"""An avatar audio receiver yields audio or the end of its segment, never video."""
