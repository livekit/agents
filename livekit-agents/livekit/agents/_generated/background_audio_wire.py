# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/background_audio.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum, unique
from inspect import get_annotations
from typing import TypeAlias

from livekit.rtc import AudioFrame


@unique
class BuiltinAudioClip(StrEnum):
    """A sound packaged with livekit-agents."""

    CITY_AMBIENCE = "city-ambience.ogg"
    FOREST_AMBIENCE = "forest-ambience.ogg"
    OFFICE_AMBIENCE = "office-ambience.ogg"
    CROWDED_ROOM = "crowded-room.ogg"
    KEYBOARD_TYPING = "keyboard-typing.ogg"
    KEYBOARD_TYPING2 = "keyboard-typing2.ogg"
    HOLD_MUSIC = "hold_music.ogg"

    def path(self) -> str:
        """The packaged file, on disk for the life of the process."""
        from .. import _bindings

        return _bindings.builtin_audio_clip_path(self)


@dataclass(frozen=True, slots=True)
class AudioConfig:
    """The audio to be played in the background."""

    source: AudioSource
    volume: float = 1.0
    """The volume of the audio (0.0-1.0)."""
    probability: float = 1.0
    """The chance this one is picked when a list of sounds is offered (0.0-1.0)."""
    fade_in: float = 0.0
    """Seconds to ramp from 0 up to `volume` when playback starts. `0` starts at full volume."""
    fade_out: float = 0.0
    """Seconds to ramp back down to 0 when `PlayHandle.stop()` is called. `0` cuts immediately."""


AudioSource: TypeAlias = BuiltinAudioClip | str | AsyncIterator[AudioFrame]
"""What a sound plays: a builtin clip (read as its packaged file), a file path, which can loop,
or a host `AsyncIterator[rtc.AudioFrame]`.
"""


Sounds: TypeAlias = list[AudioConfig] | AudioConfig | AudioSource
"""One sound, or a list of sounds one is drawn from by probability (a total below 1.0 may draw
none: silence).
"""


BuiltinAudioClip.__module__ = "livekit.agents.voice.background_audio"
AudioConfig.__annotations__ = get_annotations(AudioConfig, eval_str=True)
AudioConfig.__module__ = "livekit.agents.voice.background_audio"
