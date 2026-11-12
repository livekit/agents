# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/io.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

import time
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class AudioOutputError(BaseModel):
    """`AudioOutputError`: the session's audio output raised on a frame. The speech it was playing
    ended interrupted there; the output stays the session's.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["audio_output_error"] = "audio_output_error"
    timestamp: float = Field(default_factory=time.time)
    label: str
    error: Annotated[Exception, Field(exclude=True)]
    recoverable: bool = True


AudioOutputError.__module__ = "livekit.agents.voice.io"
