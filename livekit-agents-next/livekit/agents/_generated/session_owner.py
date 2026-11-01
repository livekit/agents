# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/session_owner.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..types import NOT_GIVEN

if TYPE_CHECKING:
    from livekit.agents import VAD
    from livekit.agents._core import AgentNodes
    from livekit.agents.stt import STT
    from livekit.agents.tts import TTS

    from ..types import NotGivenOr


@dataclass(frozen=True, kw_only=True, slots=True)
class SessionOverride:
    layer: int
    providers: dict[str, object | None]


@dataclass(frozen=True, kw_only=True, slots=True)
class ProviderBindings:
    stt: NotGivenOr[STT | None] = NOT_GIVEN
    tts: NotGivenOr[TTS | None] = NOT_GIVEN
    vad: NotGivenOr[VAD | None] = NOT_GIVEN
    stt_node: AgentNodes | None = None
    tts_node: AgentNodes | None = None
