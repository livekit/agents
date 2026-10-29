# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/llm/realtime.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from livekit.agents.llm.tool_context import ToolChoice
    from livekit.agents.types import APIConnectOptions


@dataclass(frozen=True, kw_only=True, slots=True)
class RealtimeJsonOption:
    """A supplied JSON option, including explicit null. An absent wrapper means omitted."""

    value: object


@dataclass(frozen=True, kw_only=True, slots=True)
class RealtimeToolChoice:
    value: ToolChoice | None


@dataclass(frozen=True, kw_only=True, slots=True)
class RealtimeDurationOption:
    value: float | None


@dataclass(frozen=True, kw_only=True, slots=True)
class RealtimeOptions:
    """Omission preserves a setting; a supplied nullable wrapper clears it explicitly."""

    input_audio_transcription: RealtimeJsonOption | None = None
    input_audio_noise_reduction: RealtimeJsonOption | None = None
    turn_detection: RealtimeJsonOption | None = None
    tool_choice: RealtimeToolChoice | None = None
    speed: float | None = None
    tracing: RealtimeJsonOption | None = None
    truncation: RealtimeJsonOption | None = None
    reasoning: RealtimeJsonOption | None = None
    max_session_duration: RealtimeDurationOption | None = None
    voice: str | None = None
    max_response_output_tokens: RealtimeJsonOption | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class NativeRealtimeConfig:
    api_key: str | None
    base_url: str
    conn_options: APIConnectOptions
    options: RealtimeOptions
    dialect: str | None = None
    model: str | None = None
    voice: str | None = None
    modalities: list[str] | None = None
    azure_deployment: str | None = None
    entra_token: str | None = None
    azure_defaults: bool = False
    api_secret: str | None = None
    inference_provider: str | None = None
    inference_class: str | None = None
    label: str | None = None
