# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/room_io.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from livekit.agents.voice.room_io import (
        AudioInputOptions as _HostAudioInputConfig,
        AudioOutputOptions as _HostAudioOutputConfig,
        AudioOutputOptions as _HostAudioOutputOptionsWire,
        RoomOptions as _HostRoomIoConfig,
        TextOutputOptions as _HostTextOutputConfig,
    )
    from livekit.rtc import TrackPublishOptions as _HostAudioTrackPublishOptions

AudioInputConfig: TypeAlias = "_HostAudioInputConfig"


AudioOutputConfig: TypeAlias = "_HostAudioOutputConfig"


TextOutputConfig: TypeAlias = "_HostTextOutputConfig"


RoomIoConfig: TypeAlias = "_HostRoomIoConfig"


AudioTrackPublishOptions: TypeAlias = "_HostAudioTrackPublishOptions"


AudioOutputOptionsWire: TypeAlias = "_HostAudioOutputOptionsWire"
