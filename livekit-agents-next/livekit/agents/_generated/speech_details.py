# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/speech_handle.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations
from typing import Literal


@dataclass
class InputDetails:
    modality: Literal['text', 'audio']


InputDetails.__annotations__ = get_annotations(InputDetails, eval_str=True)
InputDetails.__module__ = "livekit.agents.voice.io"
