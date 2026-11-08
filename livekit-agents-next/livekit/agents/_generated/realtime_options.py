# GENERATED FILE -- DO NOT EDIT.
# From ABI declarations and authored plugin bases:
#     rust/crates/lk-agents-abi/src/llm/realtime.rs

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from ..types import NOT_GIVEN

if TYPE_CHECKING:
    from livekit.agents.llm.tool_context import ToolChoice

    from ..types import NotGivenOr
    from .connect import APIConnectOptions


# Record declarations:


@dataclass(kw_only=True, slots=True)
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
    voice: NotGivenOr[object] = NOT_GIVEN
    """A voice name, or the provider's voice object."""
    max_response_output_tokens: NotGivenOr[object | None] = NOT_GIVEN


@dataclass(kw_only=True, slots=True)
class OpenAIRealtimeConfig:
    api_key: str | None
    """The caller's key. Outside the gateway and Azure, `OPENAI_API_KEY` when not given; for
    `azure_defaults` (`with_azure`), `AZURE_OPENAI_API_KEY`.
    """
    conn_options: APIConnectOptions
    options: RealtimeOptions
    dialect: str | None = None
    model: str | None = None
    voice: object | None = None
    modalities: list[str] | None = None
    azure_deployment: str | None = None
    entra_token: str | None = None
    azure_defaults: bool = False
    """Azure's defaults and credential rules (`with_azure`)."""
    base_url: str | None = None
    """The caller's base URL. Not given: for Azure, `{azure_endpoint}/openai` with
    `azure_endpoint` else `AZURE_OPENAI_ENDPOINT`; otherwise `OPENAI_BASE_URL`, else OpenAI's.
    """
    azure_endpoint: str | None = None
    """`with_azure`'s endpoint; exclusive with `base_url`."""
    api_secret: str | None = None
    inference_provider: str | None = None
    inference_class: str | None = None
    label: str | None = None


SAMPLE_RATE: Final = 24000
"""The OpenAI Realtime protocol's audio rate, in and out."""
NUM_CHANNELS: Final = 1
OPENAI_BASE_URL: Final = "https://api.openai.com/v1"
"""The OpenAI API's base URL, the realtime model's default."""
DEFAULT_VOICE: Final = "marin"
"""The OpenAI realtime model's default voice."""
