# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/session.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations


@dataclass(kw_only=True)
class VoiceActivityVideoSampler:
    """Sample the camera at ``speaking_fps`` while the user speaks, ``silent_fps`` otherwise."""

    speaking_fps: float = 1.0
    silent_fps: float = 0.3


VoiceActivityVideoSampler.__annotations__ = get_annotations(VoiceActivityVideoSampler, eval_str=True)
VoiceActivityVideoSampler.__module__ = "livekit.agents.voice.agent_session"
