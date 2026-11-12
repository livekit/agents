# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/avatar.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

import time
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class AvatarError(BaseModel):
    """`AvatarError`: the avatar session failed while the call went on (it did not join in time)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["avatar_error"] = "avatar_error"
    timestamp: float = Field(default_factory=time.time)
    label: str = "avatar"
    error: Annotated[Exception, Field(exclude=True)]
    recoverable: bool = True


AvatarError.__module__ = "livekit.agents.voice.avatar"
