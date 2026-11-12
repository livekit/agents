# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/room_io.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

import time
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class AudioProcessingError(BaseModel):
    """An input audio processor failed on a span; the session forwarded that span unprocessed
        and kept listening.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["audio_processing_error"] = "audio_processing_error"
    timestamp: float = Field(default_factory=time.time)
    label: str
    error: Annotated[Exception, Field(exclude=True)]
    recoverable: bool = True


AudioProcessingError.__module__ = "livekit.agents.voice.room_io"
