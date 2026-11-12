# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/utils/audio.rs
#     rust/crates/lk-agents-abi/src/utils/video.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from livekit.rtc import AudioFrame as _HostAudioFrame, VideoFrame as _HostVideoFrame

AudioFrame = _HostAudioFrame


VideoFrame = _HostVideoFrame
