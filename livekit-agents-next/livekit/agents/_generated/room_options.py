# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/room_io.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from livekit.agents.voice.room_io import (
    AudioInputOptions as _HostAudioInputConfig,
    AudioOutputOptions as _HostAudioOutputConfig,
    AudioOutputOptions as _HostAudioOutputOptionsWire,
    RoomOptions as _HostRoomIoConfig,
    TextOutputOptions as _HostTextOutputConfig,
)
from livekit.rtc import TrackPublishOptions as _HostAudioTrackPublishOptions

AudioInputConfig = _HostAudioInputConfig


AudioOutputConfig = _HostAudioOutputConfig


TextOutputConfig = _HostTextOutputConfig


RoomIoConfig = _HostRoomIoConfig


AudioTrackPublishOptions = _HostAudioTrackPublishOptions


AudioOutputOptionsWire = _HostAudioOutputOptionsWire
