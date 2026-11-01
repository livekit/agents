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

from ..types import NOT_GIVEN

if TYPE_CHECKING:
    from livekit.agents.llm.tool_context import ToolChoice

    from ..types import NotGivenOr
    from .connect import APIConnectOptions


@dataclass(frozen=True, kw_only=True, slots=True)
class RealtimeOptions:
    """A session update: an omitted option (`NOT_GIVEN`) keeps its setting, an explicit `None`
    clears it.
    """

    input_audio_transcription: NotGivenOr[object | None] = NOT_GIVEN
    input_audio_noise_reduction: NotGivenOr[object | None] = NOT_GIVEN
    turn_detection: NotGivenOr[object | None] = NOT_GIVEN
    tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN
    speed: NotGivenOr[float] = NOT_GIVEN
    tracing: NotGivenOr[object | None] = NOT_GIVEN
    truncation: NotGivenOr[object | None] = NOT_GIVEN
    reasoning: NotGivenOr[object | None] = NOT_GIVEN
    max_session_duration: NotGivenOr[float | None] = NOT_GIVEN
    voice: NotGivenOr[str] = NOT_GIVEN
    max_response_output_tokens: NotGivenOr[object | None] = NOT_GIVEN


@dataclass(frozen=True, kw_only=True, slots=True)
class OpenAIRealtimeConfig:
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
