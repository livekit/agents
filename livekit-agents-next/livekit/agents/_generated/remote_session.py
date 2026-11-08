# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/worker/remote_session.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .chat_items import ChatItem


@dataclass(kw_only=True, slots=True)
class AgentInfo:
    """The current agent: `None` instructions differ from `""`; tools are flattened across
    `Toolset`s, names only.
    """

    id: str
    instructions: str | None
    tools: list[str]
    chat_ctx: list[ChatItem]


@dataclass(kw_only=True, slots=True)
class RemoteSessionState:
    """The session's state as the host holds it: an unknown state word answers `AS_IDLE` /
    `US_LISTENING`; `options` keys are `STATE_OPTION_FIELDS`; `started_at` is epoch seconds,
    `None` or `0` stamped with the answer time.
    """

    agent_state: str
    user_state: str
    agent_id: str
    options: dict[str, str]
    started_at: float | None


@dataclass(kw_only=True, slots=True)
class RtcStats:
    """Each stat is `MessageToDict` of one SDK stat."""

    publisher_stats: list[object]
    subscriber_stats: list[object]


@dataclass(kw_only=True, slots=True)
class FrameworkInfo:
    """The SDK the session runs on and its version."""

    sdk: str
    sdk_version: str


@dataclass(kw_only=True, slots=True)
class IoUpdate:
    """The channels to switch; `None` leaves a channel alone."""

    input_audio: bool | None
    input_video: bool | None
    output_audio: bool | None
    output_video: bool | None
    output_transcription: bool | None
