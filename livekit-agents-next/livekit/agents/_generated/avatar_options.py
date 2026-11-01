# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/inference/avatar_session.rs
#     rust/crates/lk-agents-abi/src/voice/avatar.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations

from typing_extensions import TypedDict


class LemonSliceOptions(TypedDict, total=False):
    """Provider-specific options for a LemonSlice inference avatar.

        Mirrors the per-provider option dicts in :mod:`inference.stt` /
        :mod:`inference.tts` (e.g. ``CartesiaOptions``); passed via ``extra_kwargs``.
        All keys are optional.
    """

    image_url: str
    prompt: str
    idle_prompt: str
    idle_timeout: int


@dataclass
class AvatarOptions:
    video_width: int
    video_height: int
    video_fps: float
    audio_sample_rate: int
    audio_channels: int


LemonSliceOptions.__module__ = "livekit.agents.inference.avatar"
AvatarOptions.__annotations__ = get_annotations(AvatarOptions, eval_str=True)
AvatarOptions.__module__ = "livekit.agents.voice.avatar"
