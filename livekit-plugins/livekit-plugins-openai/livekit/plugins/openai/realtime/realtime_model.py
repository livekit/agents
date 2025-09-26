from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
import json
import os
import time
import weakref
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union, cast, overload
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import aiohttp
from pydantic import BaseModel, ValidationError

from livekit import rtc
from livekit.agents import APIConnectionError, APIError, io, llm, utils
from livekit.agents.llm.tool_context import (
    get_function_info,
    get_raw_function_info,
    is_function_tool,
    is_raw_function_tool,
)
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from openai.types import realtime
from openai.types.beta.realtime.session import (
    InputAudioNoiseReduction,
    InputAudioTranscription,
    TurnDetection,
)
from openai.types.realtime import (
    AudioTranscription,
    ConversationItemAdded,
    ConversationItemCreateEvent,
    ConversationItemDeletedEvent,
    ConversationItemDeleteEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionFailedEvent,
    ConversationItemTruncateEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferClearEvent,
    InputAudioBufferCommitEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    NoiseReductionType,
    RealtimeAudioConfig,
    RealtimeAudioConfigInput,
    RealtimeAudioConfigOutput,
    RealtimeAudioInputTurnDetection,
    RealtimeClientEvent,
    RealtimeConversationItemFunctionCall,
    RealtimeErrorEvent,
    RealtimeFunctionTool,
    RealtimeResponseCreateParams,
    RealtimeSessionCreateRequest,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCancelEvent,
    ResponseContentPartAddedEvent,
    ResponseCreatedEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    SessionUpdateEvent,
)
from openai.types.realtime.realtime_audio_config_input import NoiseReduction
from openai.types.realtime.realtime_session_create_response import (
    Tracing,
)

from ..log import logger
from ..models import RealtimeModels
from .realtime_model_beta import RealtimeModelBeta
from .utils import (
    AZURE_DEFAULT_INPUT_AUDIO_TRANSCRIPTION,
    AZURE_DEFAULT_TURN_DETECTION,
    DEFAULT_MAX_RESPONSE_OUTPUT_TOKENS,
    DEFAULT_MAX_SESSION_DURATION,
    livekit_item_to_openai_item,
    openai_item_to_livekit_item,
    to_audio_transcription,
    to_noise_reduction,
    to_oai_tool_choice,
    to_turn_detection,
)

# When a response is created with the OpenAI Realtime API, those events are sent in this order:
# 1. response.created (contains resp_id)
# 2. response.output_item.added (contains item_id)
# 3. conversation.item.added
# 4. response.content_part.added (type audio/text)
# 5. response.output_audio_transcript.delta (x2, x3, x4, etc)
# 6. response.output_audio.delta (x2, x3, x4, etc)
# 7. response.content_part.done
# 8. response.output_item.done (contains item_status: "completed/incomplete")
# 9. response.done (contains status_details for cancelled/failed/turn_detected/content_filter)
#
# Ourcode assumes a response will generate only one item with type "message"


SAMPLE_RATE = 24000
NUM_CHANNELS = 1
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_VOICE = "marin"

lk_oai_debug = int(os.getenv("LK_OPENAI_DEBUG", 0))


@dataclass
class _RealtimeOptions:
    model: str
    voice: str
    tool_choice: llm.ToolChoice | None
    input_audio_transcription: AudioTranscription | None
    input_audio_noise_reduction: NoiseReduction | None
    turn_detection: RealtimeAudioInputTurnDetection | None
    max_response_output_tokens: int | Literal["inf"] | None
    tracing: Tracing | None
    api_key: str | None
    base_url: str
    is_azure: bool
    azure_deployment: str | None
    entra_token: str | None
    api_version: str | None
    modalities: list[Literal["text", "audio"]]
    max_session_duration: float | None
    """reset the connection after this many seconds if provided"""
    conn_options: APIConnectOptions
    speed: float = 1.0


@dataclass
class _MessageGeneration:
    message_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    modalities: asyncio.Future[list[Literal["text", "audio"]]]
    audio_transcript: str = ""


@dataclass
class _ResponseGeneration:
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]

    messages: dict[str, _MessageGeneration]

    _done_fut: asyncio.Future[None]
    _created_timestamp: float
    """timestamp when the response was created"""
    _first_token_timestamp: float | None = None
    """timestamp when the first token was received"""


class RealtimeModel(llm.RealtimeModel):
    @overload
    def __init__(
        self,
        *,
        model: RealtimeModels | str = "gpt-realtime",
        voice: str = DEFAULT_VOICE,
        modalities: NotGivenOr[list[Literal["text", "audio"]]] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[
            AudioTranscription | InputAudioTranscription | None
        ] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[
            NoiseReductionType | NoiseReduction | InputAudioNoiseReduction | None
        ] = NOT_GIVEN,
        turn_detection: NotGivenOr[
            RealtimeAudioInputTurnDetection | TurnDetection | None
        ] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Tracing | None] = NOT_GIVEN,
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        temperature: NotGivenOr[float] = NOT_GIVEN,  # deprecated, unused in v1
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        azure_deployment: str | None = None,
        entra_token: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        voice: str = DEFAULT_VOICE,
        modalities: NotGivenOr[list[Literal["text", "audio"]]] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[
            AudioTranscription | InputAudioTranscription | None
        ] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[
            NoiseReductionType | NoiseReduction | InputAudioNoiseReduction | None
        ] = NOT_GIVEN,
        turn_detection: NotGivenOr[
            RealtimeAudioInputTurnDetection | TurnDetection | None
        ] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Tracing | None] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        temperature: NotGivenOr[float] = NOT_GIVEN,  # deprecated, unused in v1
    ) -> None: ...

    def __init__(
        self,
        *,
        model: str = "gpt-realtime",
        voice: str = DEFAULT_VOICE,
        modalities: NotGivenOr[list[Literal["text", "audio"]]] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[
            AudioTranscription | InputAudioTranscription | None
        ] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[
            NoiseReductionType | NoiseReduction | InputAudioNoiseReduction | None
        ] = NOT_GIVEN,
        turn_detection: NotGivenOr[
            RealtimeAudioInputTurnDetection | TurnDetection | None
        ] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Tracing | None] = NOT_GIVEN,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        azure_deployment: str | None = None,
        entra_token: str | None = None,
        api_version: str | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        temperature: NotGivenOr[float] = NOT_GIVEN,  # deprecated, unused in v1
    ) -> None:
        """
        Initialize a Realtime model client for OpenAI or Azure OpenAI.

        Args:
            model (str): Realtime model name, e.g., "gpt-realtime".
            voice (str): Voice used for audio responses. Defaults to "marin".
            modalities (list[Literal["text", "audio"]] | NotGiven): Modalities to enable. Defaults to ["text", "audio"] if not provided.
            tool_choice (llm.ToolChoice | None | NotGiven): Tool selection policy for responses.
            base_url (str | NotGiven): HTTP base URL of the OpenAI/Azure API. If not provided, uses OPENAI_BASE_URL for OpenAI; for Azure, constructed from AZURE_OPENAI_ENDPOINT.
            input_audio_transcription (AudioTranscription | None | NotGiven): Options for transcribing input audio.
            input_audio_noise_reduction (NoiseReductionType | NoiseReduction | InputAudioNoiseReduction | None | NotGiven): Input audio noise reduction settings.
            turn_detection (RealtimeAudioInputTurnDetection | None | NotGiven): Server-side turn-detection options.
            speed (float | NotGiven): Audio playback speed multiplier.
            tracing (Tracing | None | NotGiven): Tracing configuration for OpenAI Realtime.
            api_key (str | None): OpenAI API key. If None and not using Azure, read from OPENAI_API_KEY.
            http_session (aiohttp.ClientSession | None): Optional shared HTTP session.
            azure_deployment (str | None): Azure deployment name. Presence of any Azure-specific option enables Azure mode.
            entra_token (str | None): Azure Entra token auth (alternative to api_key).
            api_version (str | None): Azure OpenAI API version appended as query parameter.
            max_session_duration (float | None | NotGiven): Seconds before recycling the connection.
            conn_options (APIConnectOptions): Retry/backoff and connection settings.
            temperature (float | NotGiven): Deprecated; ignored by Realtime v1.

        Raises:
            ValueError: If OPENAI_API_KEY is missing in non-Azure mode, or if Azure endpoint cannot be determined when in Azure mode.

        Examples:
            Basic OpenAI usage:

            ```python
            from livekit.plugins.openai.realtime import RealtimeModel
            from openai.types import realtime

            model = RealtimeModel(
                voice="marin",
                modalities=["audio"],
                input_audio_transcription=realtime.AudioTranscription(
                    model="gpt-4o-transcribe",
                ),
                input_audio_noise_reduction="near_field",
                turn_detection=realtime.realtime_audio_input_turn_detection.SemanticVad(
                    type="semantic_vad",
                    create_response=True,
                    eagerness="auto",
                    interrupt_response=True,
                ),
            )
            session = AgentSession(llm=model)
            ```
        """
        modalities = modalities if is_given(modalities) else ["text", "audio"]
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=True,
                turn_detection=turn_detection is not None,
                user_transcription=input_audio_transcription is not None,
                auto_tool_reply_generation=False,
                audio_output="audio" in modalities,
                manual_function_calls=True,
            )
        )

        is_azure = (
            api_version is not None or entra_token is not None or azure_deployment is not None
        )

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if api_key is None and not is_azure:
            raise ValueError(
                "The api_key client option must be set either by passing api_key "
                "to the client or by setting the OPENAI_API_KEY environment variable"
            )

        if is_given(base_url):
            base_url_val = base_url
        else:
            if is_azure:
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
                if azure_endpoint is None:
                    raise ValueError(
                        "Missing Azure endpoint. Please pass base_url "
                        "or set AZURE_OPENAI_ENDPOINT environment variable."
                    )
                base_url_val = f"{azure_endpoint.rstrip('/')}/openai"
            else:
                base_url_val = OPENAI_BASE_URL

        self._opts = _RealtimeOptions(
            model=model,
            voice=voice,
            tool_choice=tool_choice or None,
            modalities=modalities,
            input_audio_transcription=to_audio_transcription(input_audio_transcription),
            input_audio_noise_reduction=to_noise_reduction(input_audio_noise_reduction),
            turn_detection=to_turn_detection(turn_detection),
            api_key=api_key,
            base_url=base_url_val,
            is_azure=is_azure,
            azure_deployment=azure_deployment,
            entra_token=entra_token,
            api_version=api_version,
            max_response_output_tokens=DEFAULT_MAX_RESPONSE_OUTPUT_TOKENS,  # type: ignore
            speed=speed if is_given(speed) else 1.0,
            tracing=cast(Union[Tracing, None], tracing) if is_given(tracing) else None,
            max_session_duration=max_session_duration
            if is_given(max_session_duration)
            else DEFAULT_MAX_SESSION_DURATION,
            conn_options=conn_options,
        )
        self._http_session = http_session
        self._http_session_owned = False
        self._sessions = weakref.WeakSet[RealtimeSession]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        from urllib.parse import urlparse

        return urlparse(self._opts.base_url).netloc

    @classmethod
    def with_azure(
        cls,
        *,
        azure_deployment: str,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        entra_token: str | None = None,
        base_url: str | None = None,
        voice: str = DEFAULT_VOICE,
        modalities: NotGivenOr[list[Literal["text", "audio"]]] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[
            AudioTranscription | InputAudioTranscription | None
        ] = NOT_GIVEN,
        input_audio_noise_reduction: NoiseReductionType | InputAudioNoiseReduction | None = None,
        turn_detection: NotGivenOr[
            RealtimeAudioInputTurnDetection | TurnDetection | None
        ] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Tracing | None] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        temperature: NotGivenOr[float] = NOT_GIVEN,  # deprecated, unused in v1
    ) -> RealtimeModel | RealtimeModelBeta:
        """
        Create a RealtimeModelBeta configured for Azure OpenAI. Azure does not currently support the GA API,
        so we return RealtimeModelBeta instead of RealtimeModel.

        Args:
            azure_deployment (str): Azure OpenAI deployment name.
            azure_endpoint (str | None): Azure endpoint URL; if None, taken from AZURE_OPENAI_ENDPOINT.
            api_version (str | None): Azure API version; if None, taken from OPENAI_API_VERSION.
            api_key (str | None): Azure API key; if None, taken from AZURE_OPENAI_API_KEY. Omit if using `entra_token`.
            entra_token (str | None): Azure Entra token for AAD auth. Provide instead of `api_key`.
            base_url (str | None): Explicit base URL. Mutually exclusive with `azure_endpoint`. If provided, used as-is.
            voice (str): Voice used for audio responses.
            modalities (list[Literal["text", "audio"]] | NotGiven): Modalities to enable. Defaults to ["text", "audio"] if not provided.
            input_audio_transcription (AudioTranscription | InputAudioTranscription | None | NotGiven): Transcription options; defaults to Azure-optimized values when not provided.
            input_audio_noise_reduction (NoiseReductionType | InputAudioNoiseReduction | None): Input noise reduction settings. Defaults to None.
            turn_detection (RealtimeAudioInputTurnDetection | TurnDetection | None | NotGiven): Server-side VAD; defaults to Azure-optimized values when not provided.
            speed (float | NotGiven): Audio playback speed multiplier.
            tracing (Tracing | None | NotGiven): Tracing configuration for OpenAI Realtime.
            http_session (aiohttp.ClientSession | None): Optional shared HTTP session.
            temperature (float | NotGiven): Deprecated; ignored by Realtime v1.

        Returns:
            RealtimeModelBeta: Configured client for Azure OpenAI Realtime.

        Raises:
            ValueError: If credentials are missing, `api_version` is not provided, Azure endpoint cannot be determined, or both `base_url` and `azure_endpoint` are provided.

        Examples:
            Azure usage with api-version 2024-10-01-preview:

            ```python
            from livekit.plugins.openai.realtime import RealtimeModel
            from openai.types.beta import realtime

            model = openai.realtime.RealtimeModel.with_azure(
                azure_deployment="gpt-realtime",
                azure_endpoint="https://yourendpoint.azure.com",
                api_version="2024-10-01-preview",
                api_key="your-api-key",
                modalities=["text", "audio"],
                input_audio_transcription=realtime.session.InputAudioTranscription(
                    model="gpt-4o-transcribe",
                ),
                input_audio_noise_reduction=realtime.session.InputAudioNoiseReduction(
                    type="near_field",
                ),
                turn_detection=realtime.session.TurnDetection(
                    type="semantic_vad",
                    create_response=True,
                    eagerness="auto",
                    interrupt_response=True,
                ),
            )
            ```

            Azure usage with api-version 2025-08-28:
            ```python
            from livekit.plugins.openai.realtime import RealtimeModel
            from openai.types import realtime

            model = RealtimeModel(
                azure_deployment="gpt-realtime",
                azure_endpoint="https://yourendpoint.azure.com",
                api_version="2024-10-01-preview",
                api_key="your-api-key",
                input_audio_transcription=realtime.AudioTranscription(
                    model="gpt-4o-transcribe",
                ),
                input_audio_noise_reduction="near_field",
                turn_detection=realtime.realtime_audio_input_turn_detection.SemanticVad(
                    type="semantic_vad",
                    create_response=True,
                    eagerness="auto",
                    interrupt_response=True,
                ),
            )
            ```
        """
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if api_key is None and entra_token is None:
            raise ValueError(
                "Missing credentials. Please pass one of `api_key`, `entra_token`, "
                "or the `AZURE_OPENAI_API_KEY` environment variable."
            )

        api_version = api_version or os.getenv("OPENAI_API_VERSION")
        if api_version is None:
            raise ValueError(
                "Must provide either the `api_version` argument or the "
                "`OPENAI_API_VERSION` environment variable"
            )

        if base_url is None:
            azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            if azure_endpoint is None:
                raise ValueError(
                    "Missing Azure endpoint. Please pass the `azure_endpoint` "
                    "parameter or set the `AZURE_OPENAI_ENDPOINT` environment variable."
                )

            base_url = f"{azure_endpoint.rstrip('/')}/openai"
        elif azure_endpoint is not None:
            raise ValueError("base_url and azure_endpoint are mutually exclusive")

        if not is_given(input_audio_transcription):
            input_audio_transcription = AZURE_DEFAULT_INPUT_AUDIO_TRANSCRIPTION

        if not is_given(turn_detection):
            turn_detection = AZURE_DEFAULT_TURN_DETECTION

        if is_given(input_audio_transcription) and not isinstance(
            input_audio_transcription, InputAudioTranscription
        ):
            raise ValueError(
                f"input_audio_transcription must be an instance of InputAudioTranscription for api-version {api_version}"
            )
        if is_given(turn_detection) and not isinstance(turn_detection, TurnDetection):
            raise ValueError(
                f"turn_detection must be an instance of TurnDetection for api-version {api_version}"
            )
        if input_audio_noise_reduction is not None and not isinstance(
            input_audio_noise_reduction, InputAudioNoiseReduction
        ):
            raise ValueError(
                f"input_audio_noise_reduction must be an instance of InputAudioNoiseReduction for api-version {api_version}"
            )

        return RealtimeModelBeta(
            voice=voice,
            modalities=modalities,
            input_audio_transcription=input_audio_transcription,  # type: ignore
            input_audio_noise_reduction=input_audio_noise_reduction,
            turn_detection=turn_detection,  # type: ignore
            temperature=temperature,
            speed=speed,
            tracing=tracing,  # type: ignore
            api_key=api_key,
            http_session=http_session,
            azure_deployment=azure_deployment,
            api_version=api_version,
            entra_token=entra_token,
            base_url=base_url,
        )

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: NotGivenOr[
            RealtimeAudioInputTurnDetection | TurnDetection | None
        ] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[
            InputAudioTranscription | AudioTranscription | None
        ] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[
            NoiseReduction | NoiseReductionType | InputAudioNoiseReduction | None
        ] = NOT_GIVEN,
        max_response_output_tokens: NotGivenOr[int | Literal["inf"] | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Tracing | None] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,  # deprecated, unused in v1
    ) -> None:
        if is_given(voice):
            self._opts.voice = voice

        if is_given(turn_detection):
            self._opts.turn_detection = to_turn_detection(turn_detection)  # type: ignore

        if is_given(tool_choice):
            self._opts.tool_choice = cast(Optional[llm.ToolChoice], tool_choice)

        if is_given(input_audio_transcription):
            self._opts.input_audio_transcription = to_audio_transcription(input_audio_transcription)  # type: ignore

        if is_given(input_audio_noise_reduction):
            self._opts.input_audio_noise_reduction = to_noise_reduction(input_audio_noise_reduction)  # type: ignore

        if is_given(max_response_output_tokens):
            self._opts.max_response_output_tokens = max_response_output_tokens  # type: ignore

        if is_given(speed):
            self._opts.speed = speed

        if is_given(tracing):
            self._opts.tracing = cast(Union[Tracing, None], tracing)

        for sess in self._sessions:
            sess.update_options(
                voice=voice,
                turn_detection=self._opts.turn_detection,
                tool_choice=tool_choice,
                input_audio_transcription=self._opts.input_audio_transcription,
                input_audio_noise_reduction=self._opts.input_audio_noise_reduction,
                max_response_output_tokens=max_response_output_tokens,
                speed=speed,
                tracing=tracing,
            )

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if not self._http_session:
            try:
                self._http_session = utils.http_context.http_session()
            except RuntimeError:
                self._http_session = aiohttp.ClientSession()
                self._http_session_owned = True

        return self._http_session

    def session(self) -> RealtimeSession:
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None:
        if self._http_session_owned and self._http_session:
            await self._http_session.close()


def process_base_url(
    url: str,
    model: str,
    is_azure: bool = False,
    azure_deployment: str | None = None,
    api_version: str | None = None,
) -> str:
    if url.startswith("http"):
        url = url.replace("http", "ws", 1)

    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)

    # ensure "/realtime" is added if the path is empty OR "/v1"
    if not parsed_url.path or parsed_url.path.rstrip("/") in ["", "/v1", "/openai"]:
        path = parsed_url.path.rstrip("/") + "/realtime"
    else:
        path = parsed_url.path

    if is_azure:
        if api_version:
            query_params["api-version"] = [api_version]
        if azure_deployment:
            query_params["deployment"] = [azure_deployment]

    else:
        if "model" not in query_params:
            query_params["model"] = [model]

    new_query = urlencode(query_params, doseq=True)
    new_url = urlunparse((parsed_url.scheme, parsed_url.netloc, path, "", new_query, ""))

    return new_url


class RealtimeSession(
    llm.RealtimeSession[Literal["openai_server_event_received", "openai_client_event_queued"]]
):
    """
    A session for the OpenAI Realtime API.

    This class is used to interact with the OpenAI Realtime API.
    It is responsible for sending events to the OpenAI Realtime API and receiving events from it.

    It exposes two more events:
    - openai_server_event_received: expose the raw server events from the OpenAI Realtime API
    - openai_client_event_queued: expose the raw client events sent to the OpenAI Realtime API
    """

    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model: RealtimeModel = realtime_model
        self._tools = llm.ToolContext.empty()
        self._msg_ch = utils.aio.Chan[Union[RealtimeClientEvent, dict[str, Any]]]()
        self._input_resampler: rtc.AudioResampler | None = None

        self._instructions: str | None = None
        self._main_atask = asyncio.create_task(self._main_task(), name="RealtimeSession._main_task")
        self.send_event(self._create_session_update_event())

        self._response_created_futures: dict[str, asyncio.Future[llm.GenerationCreatedEvent]] = {}
        self._item_delete_future: dict[str, asyncio.Future] = {}
        self._item_create_future: dict[str, asyncio.Future] = {}

        self._current_generation: _ResponseGeneration | None = None
        self._remote_chat_ctx = llm.remote_chat_context.RemoteChatContext()

        self._update_chat_ctx_lock = asyncio.Lock()
        self._update_fnc_ctx_lock = asyncio.Lock()

        # 100ms chunks
        self._bstream = utils.audio.AudioByteStream(
            SAMPLE_RATE, NUM_CHANNELS, samples_per_channel=SAMPLE_RATE // 10
        )
        self._pushed_duration_s: float = 0  # duration of audio pushed to the OpenAI Realtime API

    def send_event(self, event: RealtimeClientEvent | dict[str, Any]) -> None:
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(event)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        num_retries: int = 0
        max_retries = self._realtime_model._opts.conn_options.max_retry

        async def _reconnect() -> None:
            logger.debug(
                "reconnecting to OpenAI Realtime API",
                extra={"max_session_duration": self._realtime_model._opts.max_session_duration},
            )

            events: list[RealtimeClientEvent] = []

            # options and instructions
            events.append(self._create_session_update_event())

            # tools
            tools = list(self._tools.function_tools.values())
            if tools:
                events.append(self._create_tools_update_event(tools))

            # chat context
            chat_ctx = self.chat_ctx.copy(
                exclude_function_call=True,
                exclude_instructions=True,
                exclude_empty_message=True,
            )
            old_chat_ctx = self._remote_chat_ctx
            self._remote_chat_ctx = llm.remote_chat_context.RemoteChatContext()
            events.extend(self._create_update_chat_ctx_events(chat_ctx))

            try:
                for ev in events:
                    msg = ev.model_dump(by_alias=True, exclude_unset=True, exclude_defaults=False)
                    self.emit("openai_client_event_queued", msg)
                    await ws_conn.send_str(json.dumps(msg))
            except Exception as e:
                self._remote_chat_ctx = old_chat_ctx  # restore the old chat context
                raise APIConnectionError(
                    message=(
                        "Failed to send message to OpenAI Realtime API during session re-connection"
                    ),
                ) from e

            logger.debug("reconnected to OpenAI Realtime API")
            self.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())

        reconnecting = False
        while not self._msg_ch.closed:
            try:
                ws_conn = await self._create_ws_conn()
                if reconnecting:
                    await _reconnect()
                    num_retries = 0  # reset the retry counter
                await self._run_ws(ws_conn)

            except APIError as e:
                if max_retries == 0 or not e.retryable:
                    self._emit_error(e, recoverable=False)
                    raise
                elif num_retries == max_retries:
                    self._emit_error(e, recoverable=False)
                    raise APIConnectionError(
                        f"OpenAI Realtime API connection failed after {num_retries} attempts",
                    ) from e
                else:
                    self._emit_error(e, recoverable=True)

                    retry_interval = self._realtime_model._opts.conn_options._interval_for_retry(
                        num_retries
                    )
                    logger.warning(
                        f"OpenAI Realtime API connection failed, retrying in {retry_interval}s",
                        exc_info=e,
                        extra={"attempt": num_retries, "max_retries": max_retries},
                    )
                    await asyncio.sleep(retry_interval)
                num_retries += 1

            except Exception as e:
                self._emit_error(e, recoverable=False)
                raise

            reconnecting = True

    async def _create_ws_conn(self) -> aiohttp.ClientWebSocketResponse:
        headers = {"User-Agent": "LiveKit Agents"}
        if self._realtime_model._opts.is_azure:
            if self._realtime_model._opts.entra_token:
                headers["Authorization"] = f"Bearer {self._realtime_model._opts.entra_token}"

            if self._realtime_model._opts.api_key:
                headers["api-key"] = self._realtime_model._opts.api_key
        else:
            headers["Authorization"] = f"Bearer {self._realtime_model._opts.api_key}"

        url = process_base_url(
            self._realtime_model._opts.base_url,
            self._realtime_model._opts.model,
            is_azure=self._realtime_model._opts.is_azure,
            api_version=self._realtime_model._opts.api_version,
            azure_deployment=self._realtime_model._opts.azure_deployment,
        )

        if lk_oai_debug:
            logger.debug(f"connecting to Realtime API: {url}")

        try:
            return await asyncio.wait_for(
                self._realtime_model._ensure_http_session().ws_connect(url=url, headers=headers),
                self._realtime_model._opts.conn_options.timeout,
            )
        except asyncio.TimeoutError as e:
            raise APIConnectionError(
                message="OpenAI Realtime API connection timed out",
            ) from e

    async def _run_ws(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        closing = False

        @utils.log_exceptions(logger=logger)
        async def _send_task() -> None:
            nonlocal closing
            async for msg in self._msg_ch:
                try:
                    if isinstance(msg, BaseModel):
                        msg = msg.model_dump(
                            by_alias=True, exclude_unset=True, exclude_defaults=False
                        )

                    self.emit("openai_client_event_queued", msg)
                    await ws_conn.send_str(json.dumps(msg))

                    if lk_oai_debug:
                        msg_copy = msg.copy()
                        if msg_copy["type"] == "input_audio_buffer.append":
                            msg_copy = {**msg_copy, "audio": "..."}

                        logger.debug(f">>> {msg_copy}")
                except Exception:
                    break

            closing = True
            await ws_conn.close()

        @utils.log_exceptions(logger=logger)
        async def _recv_task() -> None:
            while True:
                msg = await ws_conn.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing:  # closing is expected, see _send_task
                        return

                    # this will trigger a reconnection
                    raise APIConnectionError(message="OpenAI S2S connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                event = json.loads(msg.data)

                # emit the raw json dictionary instead of the BaseModel because different
                # providers can have different event types that are not part of the OpenAI Realtime API  # noqa: E501
                self.emit("openai_server_event_received", event)

                try:
                    if lk_oai_debug:
                        event_copy = event.copy()
                        if event_copy["type"] == "response.output_audio.delta":
                            event_copy = {**event_copy, "delta": "..."}

                        logger.debug(f"<<< {event_copy}")

                    if event["type"] == "input_audio_buffer.speech_started":
                        self._handle_input_audio_buffer_speech_started(
                            InputAudioBufferSpeechStartedEvent.construct(**event)
                        )
                    elif event["type"] == "input_audio_buffer.speech_stopped":
                        self._handle_input_audio_buffer_speech_stopped(
                            InputAudioBufferSpeechStoppedEvent.construct(**event)
                        )
                    elif event["type"] == "response.created":
                        self._handle_response_created(ResponseCreatedEvent.construct(**event))
                    elif event["type"] == "response.output_item.added":
                        self._handle_response_output_item_added(
                            ResponseOutputItemAddedEvent.construct(**event)
                        )
                    elif event["type"] == "response.content_part.added":
                        self._handle_response_content_part_added(
                            ResponseContentPartAddedEvent.construct(**event)
                        )
                    elif event["type"] == "conversation.item.added":
                        self._handle_conversion_item_added(ConversationItemAdded.construct(**event))
                    elif event["type"] == "conversation.item.deleted":
                        self._handle_conversion_item_deleted(
                            ConversationItemDeletedEvent.construct(**event)
                        )
                    elif event["type"] == "conversation.item.input_audio_transcription.delta":
                        # currently incoming transcripts are transcribed only after the user stops speaking
                        # it's not very useful to emit these as the transcribe process takes place within ~100ms
                        # when they handle streaming transcriptions, we'll handle it then.
                        pass
                    elif event["type"] == "conversation.item.input_audio_transcription.completed":
                        self._handle_conversion_item_input_audio_transcription_completed(
                            ConversationItemInputAudioTranscriptionCompletedEvent.construct(**event)
                        )
                    elif event["type"] == "conversation.item.input_audio_transcription.failed":
                        self._handle_conversion_item_input_audio_transcription_failed(
                            ConversationItemInputAudioTranscriptionFailedEvent.construct(**event)
                        )
                    elif event["type"] == "response.output_text.delta":
                        self._handle_response_text_delta(ResponseTextDeltaEvent.construct(**event))
                    elif event["type"] == "response.output_text.done":
                        self._handle_response_text_done(ResponseTextDoneEvent.construct(**event))
                    elif event["type"] == "response.output_audio_transcript.delta":
                        self._handle_response_audio_transcript_delta(event)
                    elif event["type"] == "response.output_audio.delta":
                        self._handle_response_audio_delta(
                            ResponseAudioDeltaEvent.construct(**event)
                        )
                    elif event["type"] == "response.output_audio_transcript.done":
                        self._handle_response_audio_transcript_done(
                            ResponseAudioTranscriptDoneEvent.construct(**event)
                        )
                    elif event["type"] == "response.output_audio.done":
                        self._handle_response_audio_done(ResponseAudioDoneEvent.construct(**event))
                    elif event["type"] == "response.output_item.done":
                        self._handle_response_output_item_done(
                            ResponseOutputItemDoneEvent.construct(**event)
                        )
                    elif event["type"] == "response.done":
                        self._handle_response_done(ResponseDoneEvent.construct(**event))
                    elif event["type"] == "error":
                        self._handle_error(RealtimeErrorEvent.construct(**event))
                    elif lk_oai_debug:
                        logger.debug(f"unhandled event: {event['type']}", extra={"event": event})
                except Exception:
                    if event["type"] == "response.output_audio.delta":
                        event["delta"] = event["delta"][:10] + "..."
                    logger.exception("failed to handle event", extra={"event": event})

        tasks = [
            asyncio.create_task(_recv_task(), name="_recv_task"),
            asyncio.create_task(_send_task(), name="_send_task"),
        ]
        wait_reconnect_task: asyncio.Task | None = None
        if self._realtime_model._opts.max_session_duration is not None:
            wait_reconnect_task = asyncio.create_task(
                asyncio.sleep(self._realtime_model._opts.max_session_duration),
                name="_timeout_task",
            )
            tasks.append(wait_reconnect_task)
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            # propagate exceptions from completed tasks
            for task in done:
                if task != wait_reconnect_task:
                    task.result()

            if wait_reconnect_task and wait_reconnect_task in done and self._current_generation:
                # wait for the current generation to complete before reconnecting
                await self._current_generation._done_fut
                closing = True

        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await ws_conn.close()

    def _create_session_update_event(self) -> SessionUpdateEvent:
        audio_format = realtime.realtime_audio_formats.AudioPCM(rate=SAMPLE_RATE, type="audio/pcm")
        # they do not support both text and audio modalities, it'll respond in audio + transcript
        modality = "audio" if "audio" in self._realtime_model._opts.modalities else "text"

        session = RealtimeSessionCreateRequest(
            type="realtime",
            model=self._realtime_model._opts.model,
            output_modalities=[modality],
            audio=RealtimeAudioConfig(
                input=RealtimeAudioConfigInput(
                    format=audio_format,
                    noise_reduction=self._realtime_model._opts.input_audio_noise_reduction,
                    transcription=self._realtime_model._opts.input_audio_transcription,
                    turn_detection=self._realtime_model._opts.turn_detection,
                ),
                output=RealtimeAudioConfigOutput(
                    format=audio_format,
                    speed=self._realtime_model._opts.speed,
                    voice=self._realtime_model._opts.voice,
                ),
            ),
            max_output_tokens=self._realtime_model._opts.max_response_output_tokens,
            tool_choice=to_oai_tool_choice(self._realtime_model._opts.tool_choice),
            tracing=self._realtime_model._opts.tracing,
        )
        if self._instructions is not None:
            session.instructions = self._instructions

        # initial session update
        return SessionUpdateEvent(
            type="session.update",
            # Using model_construct since OpenAI restricts voices to those defined in the BaseModel.  # noqa: E501
            # Other providers support different voices, so we need to accommodate that.
            session=session,
            event_id=utils.shortuuid("session_update_"),
        )

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._remote_chat_ctx.to_chat_ctx()

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools.copy()

    def update_options(
        self,
        *,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: NotGivenOr[RealtimeAudioInputTurnDetection | None] = NOT_GIVEN,
        max_response_output_tokens: NotGivenOr[int | Literal["inf"] | None] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[AudioTranscription | None] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[
            NoiseReductionType | NoiseReduction | InputAudioNoiseReduction | None
        ] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Tracing | None] = NOT_GIVEN,
    ) -> None:
        session = RealtimeSessionCreateRequest(
            type="realtime",
        )
        has_changes = False

        if is_given(tool_choice):
            tool_choice = cast(Optional[llm.ToolChoice], tool_choice)
            self._realtime_model._opts.tool_choice = tool_choice
            session.tool_choice = to_oai_tool_choice(tool_choice)
            has_changes = True

        if is_given(max_response_output_tokens):
            self._realtime_model._opts.max_response_output_tokens = max_response_output_tokens  # type: ignore
            session.max_output_tokens = max_response_output_tokens  # type: ignore
            has_changes = True

        if is_given(tracing):
            self._realtime_model._opts.tracing = cast(Union[Tracing, None], tracing)
            session.tracing = cast(Union[Tracing, None], tracing)  # type: ignore
            has_changes = True

        has_audio_config = False
        audio_output = RealtimeAudioConfigOutput()
        audio_input = RealtimeAudioConfigInput()
        audio_config = RealtimeAudioConfig(
            output=audio_output,
            input=audio_input,
        )

        if is_given(voice):
            self._realtime_model._opts.voice = voice
            audio_output.voice = voice
            has_audio_config = True

        if is_given(turn_detection):
            self._realtime_model._opts.turn_detection = turn_detection  # type: ignore
            audio_input.turn_detection = turn_detection  # type: ignore
            has_audio_config = True

        if is_given(input_audio_transcription):
            self._realtime_model._opts.input_audio_transcription = input_audio_transcription
            audio_input.transcription = input_audio_transcription
            has_audio_config = True

        if is_given(input_audio_noise_reduction):
            input_audio_noise_reduction = to_noise_reduction(input_audio_noise_reduction)  # type: ignore
            self._realtime_model._opts.input_audio_noise_reduction = input_audio_noise_reduction
            audio_input.noise_reduction = input_audio_noise_reduction
            has_audio_config = True

        if is_given(speed):
            self._realtime_model._opts.speed = speed
            audio_output.speed = speed
            has_audio_config = True

        if has_audio_config:
            session.audio = audio_config
            has_changes = True

        if has_changes:
            self.send_event(
                SessionUpdateEvent(
                    type="session.update",
                    session=session,
                    event_id=utils.shortuuid("options_update_"),
                )
            )

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        async with self._update_chat_ctx_lock:
            events = self._create_update_chat_ctx_events(chat_ctx)
            futs: list[asyncio.Future[None]] = []

            for ev in events:
                futs.append(f := asyncio.Future[None]())
                if isinstance(ev, ConversationItemDeleteEvent):
                    self._item_delete_future[ev.item_id] = f
                elif isinstance(ev, ConversationItemCreateEvent):
                    assert ev.item.id is not None
                    self._item_create_future[ev.item.id] = f
                self.send_event(ev)

            if not futs:
                return
            try:
                await asyncio.wait_for(asyncio.gather(*futs, return_exceptions=True), timeout=5.0)
            except asyncio.TimeoutError:
                raise llm.RealtimeError("update_chat_ctx timed out.") from None

    def _create_update_chat_ctx_events(
        self, chat_ctx: llm.ChatContext
    ) -> list[ConversationItemCreateEvent | ConversationItemDeleteEvent]:
        events: list[ConversationItemCreateEvent | ConversationItemDeleteEvent] = []
        remote_ctx = self._remote_chat_ctx.to_chat_ctx()
        diff_ops = llm.utils.compute_chat_ctx_diff(remote_ctx, chat_ctx)

        def _delete_item(msg_id: str) -> None:
            events.append(
                ConversationItemDeleteEvent(
                    type="conversation.item.delete",
                    item_id=msg_id,
                    event_id=utils.shortuuid("chat_ctx_delete_"),
                )
            )

        def _create_item(previous_msg_id: str | None, msg_id: str) -> None:
            chat_item = chat_ctx.get_by_id(msg_id)
            assert chat_item is not None
            events.append(
                ConversationItemCreateEvent(
                    type="conversation.item.create",
                    item=livekit_item_to_openai_item(chat_item),
                    previous_item_id=("root" if previous_msg_id is None else previous_msg_id),
                    event_id=utils.shortuuid("chat_ctx_create_"),
                )
            )

        def _is_content_empty(msg_id: str) -> bool:
            remote_item = remote_ctx.get_by_id(msg_id)
            if remote_item and remote_item.type == "message" and not remote_item.content:
                return True
            return False

        for msg_id in diff_ops.to_remove:
            # we don't have content synced down for some types of content (audio/images)
            # these won't be present in the Agent's view of the context
            # so in those cases, we do not want to remove them from the server context
            if _is_content_empty(msg_id):
                continue
            _delete_item(msg_id)

        for previous_msg_id, msg_id in diff_ops.to_create:
            _create_item(previous_msg_id, msg_id)

        # update the items with the same id but different content
        for previous_msg_id, msg_id in diff_ops.to_update:
            # likewise, empty content almost always means the content is not synced down
            # we don't want to recreate these items there
            if _is_content_empty(msg_id):
                continue
            _delete_item(msg_id)
            _create_item(previous_msg_id, msg_id)

        return events

    async def update_tools(self, tools: list[llm.FunctionTool | llm.RawFunctionTool]) -> None:
        async with self._update_fnc_ctx_lock:
            ev = self._create_tools_update_event(tools)
            self.send_event(ev)

            assert isinstance(ev.session, RealtimeSessionCreateRequest)
            assert ev.session.tools is not None
            retained_tool_names: set[str] = set()
            for t in ev.session.tools:
                if isinstance(t, RealtimeFunctionTool) and t.name is not None:
                    retained_tool_names.add(t.name)
                # TODO(dz): handle MCP tools
            retained_tools = [
                tool
                for tool in tools
                if (is_function_tool(tool) and get_function_info(tool).name in retained_tool_names)
                or (
                    is_raw_function_tool(tool)
                    and get_raw_function_info(tool).name in retained_tool_names
                )
            ]
            self._tools = llm.ToolContext(retained_tools)

    def _create_tools_update_event(
        self, tools: list[llm.FunctionTool | llm.RawFunctionTool]
    ) -> SessionUpdateEvent:
        oai_tools: list[RealtimeFunctionTool] = []
        retained_tools: list[llm.FunctionTool | llm.RawFunctionTool] = []

        for tool in tools:
            if is_function_tool(tool):
                tool_desc = llm.utils.build_legacy_openai_schema(tool, internally_tagged=True)
            elif is_raw_function_tool(tool):
                tool_info = get_raw_function_info(tool)
                tool_desc = tool_info.raw_schema
                tool_desc.pop("meta", None)  # meta is not supported by OpenAI Realtime API
                tool_desc["type"] = "function"  # internally tagged
            else:
                logger.error(
                    "OpenAI Realtime API doesn't support this tool type", extra={"tool": tool}
                )
                continue

            try:
                session_tool = RealtimeFunctionTool.model_validate(tool_desc)
                oai_tools.append(session_tool)
                retained_tools.append(tool)
            except ValidationError:
                logger.error(
                    "OpenAI Realtime API doesn't support this tool",
                    extra={"tool": tool_desc},
                )
                continue

        return SessionUpdateEvent(
            type="session.update",
            session=RealtimeSessionCreateRequest.model_construct(
                type="realtime",
                model=self._realtime_model._opts.model,
                tools=oai_tools,  # type: ignore
            ),
            event_id=utils.shortuuid("tools_update_"),
        )

    async def update_instructions(self, instructions: str) -> None:
        event_id = utils.shortuuid("instructions_update_")
        # f = asyncio.Future()
        # self._response_futures[event_id] = f
        self.send_event(
            SessionUpdateEvent(
                type="session.update",
                session=RealtimeSessionCreateRequest.model_construct(
                    type="realtime",
                    instructions=instructions,
                ),
                event_id=event_id,
            )
        )
        self._instructions = instructions

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        for f in self._resample_audio(frame):
            data = f.data.tobytes()
            for nf in self._bstream.write(data):
                self.send_event(
                    InputAudioBufferAppendEvent(
                        type="input_audio_buffer.append",
                        audio=base64.b64encode(nf.data).decode("utf-8"),
                    )
                )
                self._pushed_duration_s += nf.duration

    def push_video(self, frame: rtc.VideoFrame) -> None:
        message = llm.ChatMessage(
            role="user",
            content=[llm.ImageContent(image=frame)],
        )
        oai_item = livekit_item_to_openai_item(message)
        self.send_event(
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item=oai_item,
                event_id=utils.shortuuid("video_"),
            )
        )

    def commit_audio(self) -> None:
        if self._pushed_duration_s > 0.1:  # OpenAI requires at least 100ms of audio
            self.send_event(InputAudioBufferCommitEvent(type="input_audio_buffer.commit"))
            self._pushed_duration_s = 0

    def clear_audio(self) -> None:
        self.send_event(InputAudioBufferClearEvent(type="input_audio_buffer.clear"))
        self._pushed_duration_s = 0

    def generate_reply(
        self, *, instructions: NotGivenOr[str] = NOT_GIVEN
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        event_id = utils.shortuuid("response_create_")
        fut = asyncio.Future[llm.GenerationCreatedEvent]()
        self._response_created_futures[event_id] = fut
        self.send_event(
            ResponseCreateEvent(
                type="response.create",
                event_id=event_id,
                response=RealtimeResponseCreateParams(
                    instructions=instructions or None,
                    metadata={"client_event_id": event_id},
                ),
            )
        )

        def _on_timeout() -> None:
            if fut and not fut.done():
                fut.set_exception(llm.RealtimeError("generate_reply timed out."))

        handle = asyncio.get_event_loop().call_later(5.0, _on_timeout)
        fut.add_done_callback(lambda _: handle.cancel())
        return fut

    def interrupt(self) -> None:
        self.send_event(ResponseCancelEvent(type="response.cancel"))

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if "audio" in modalities:
            self.send_event(
                ConversationItemTruncateEvent(
                    type="conversation.item.truncate",
                    content_index=0,
                    item_id=message_id,
                    audio_end_ms=audio_end_ms,
                )
            )
        elif utils.is_given(audio_transcript):
            # sync the forwarded text to the remote chat ctx
            chat_ctx = self.chat_ctx.copy()
            if (idx := chat_ctx.index_by_id(message_id)) is not None:
                new_item = copy.copy(chat_ctx.items[idx])
                assert new_item.type == "message"

                new_item.content = [audio_transcript]
                chat_ctx.items[idx] = new_item
                events = self._create_update_chat_ctx_events(chat_ctx)
                for ev in events:
                    self.send_event(ev)

    async def aclose(self) -> None:
        self._msg_ch.close()
        await self._main_atask

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler:
            if frame.sample_rate != self._input_resampler._input_rate:
                # input audio changed to a different sample rate
                self._input_resampler = None

        if self._input_resampler is None and (
            frame.sample_rate != SAMPLE_RATE or frame.num_channels != NUM_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )

        if self._input_resampler:
            # TODO(long): flush the resampler when the input source is changed
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    def _handle_input_audio_buffer_speech_started(
        self, _: InputAudioBufferSpeechStartedEvent
    ) -> None:
        self.emit("input_speech_started", llm.InputSpeechStartedEvent())

    def _handle_input_audio_buffer_speech_stopped(
        self, _: InputAudioBufferSpeechStoppedEvent
    ) -> None:
        user_transcription_enabled = (
            self._realtime_model._opts.input_audio_transcription is not None
        )
        self.emit(
            "input_speech_stopped",
            llm.InputSpeechStoppedEvent(user_transcription_enabled=user_transcription_enabled),
        )

    def _handle_response_created(self, event: ResponseCreatedEvent) -> None:
        assert event.response.id is not None, "response.id is None"

        self._current_generation = _ResponseGeneration(
            message_ch=utils.aio.Chan(),
            function_ch=utils.aio.Chan(),
            messages={},
            _created_timestamp=time.time(),
            _done_fut=asyncio.Future(),
        )

        generation_ev = llm.GenerationCreatedEvent(
            message_stream=self._current_generation.message_ch,
            function_stream=self._current_generation.function_ch,
            user_initiated=False,
            response_id=event.response.id,
        )

        if (
            isinstance(event.response.metadata, dict)
            and (client_event_id := event.response.metadata.get("client_event_id"))
            and (fut := self._response_created_futures.pop(client_event_id, None))
        ):
            if not fut.done():
                generation_ev.user_initiated = True
                fut.set_result(generation_ev)
            else:
                logger.warning("response of generate_reply received after it's timed out.")

        self.emit("generation_created", generation_ev)

    def _handle_response_output_item_added(self, event: ResponseOutputItemAddedEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"
        assert (item_id := event.item.id) is not None, "item.id is None"
        assert (item_type := event.item.type) is not None, "item.type is None"

        if item_type == "message":
            item_generation = _MessageGeneration(
                message_id=item_id,
                text_ch=utils.aio.Chan(),
                audio_ch=utils.aio.Chan(),
                modalities=asyncio.Future(),
            )
            if not self._realtime_model.capabilities.audio_output:
                item_generation.audio_ch.close()
                item_generation.modalities.set_result(["text"])

            self._current_generation.message_ch.send_nowait(
                llm.MessageGeneration(
                    message_id=item_id,
                    text_stream=item_generation.text_ch,
                    audio_stream=item_generation.audio_ch,
                    modalities=item_generation.modalities,
                )
            )
            self._current_generation.messages[item_id] = item_generation

    def _handle_response_content_part_added(self, event: ResponseContentPartAddedEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"
        assert (item_id := event.item_id) is not None, "item_id is None"
        assert (item_type := event.part.type) is not None, "part.type is None"

        if item_type == "text" and self._realtime_model.capabilities.audio_output:
            logger.warning("Text response received from OpenAI Realtime API in audio modality.")

        with contextlib.suppress(asyncio.InvalidStateError):
            self._current_generation.messages[item_id].modalities.set_result(
                ["text"] if item_type == "text" else ["audio", "text"]
            )

    def _handle_conversion_item_added(self, event: ConversationItemAdded) -> None:
        assert event.item.id is not None, "item.id is None"

        try:
            self._remote_chat_ctx.insert(
                event.previous_item_id, openai_item_to_livekit_item(event.item)
            )
        except ValueError as e:
            logger.warning(
                f"failed to insert item `{event.item.id}`: {str(e)}",
            )

        if fut := self._item_create_future.pop(event.item.id, None):
            fut.set_result(None)

    def _handle_conversion_item_deleted(self, event: ConversationItemDeletedEvent) -> None:
        assert event.item_id is not None, "item_id is None"

        try:
            self._remote_chat_ctx.delete(event.item_id)
        except ValueError as e:
            logger.warning(
                f"failed to delete item `{event.item_id}`: {str(e)}",
            )

        if fut := self._item_delete_future.pop(event.item_id, None):
            fut.set_result(None)

    def _handle_conversion_item_input_audio_transcription_completed(
        self, event: ConversationItemInputAudioTranscriptionCompletedEvent
    ) -> None:
        if remote_item := self._remote_chat_ctx.get(event.item_id):
            assert isinstance(remote_item.item, llm.ChatMessage)
            remote_item.item.content.append(event.transcript)

        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=event.item_id,
                transcript=event.transcript,
                is_final=True,
            ),
        )

    def _handle_conversion_item_input_audio_transcription_failed(
        self, event: ConversationItemInputAudioTranscriptionFailedEvent
    ) -> None:
        logger.error(
            "OpenAI Realtime API failed to transcribe input audio",
            extra={"error": event.error},
        )

    def _handle_response_text_delta(self, event: ResponseTextDeltaEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"
        item_generation = self._current_generation.messages[event.item_id]
        if (
            item_generation.audio_ch.closed
            and self._current_generation._first_token_timestamp is None
        ):
            # only if audio is not available
            self._current_generation._first_token_timestamp = time.time()

        item_generation.text_ch.send_nowait(event.delta)
        item_generation.audio_transcript += event.delta

    def _handle_response_text_done(self, event: ResponseTextDoneEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"

    def _handle_response_audio_transcript_delta(self, event: dict[str, Any]) -> None:
        assert self._current_generation is not None, "current_generation is None"

        item_id = event["item_id"]
        delta = event["delta"]

        if (start_time := event.get("start_time")) is not None:
            delta = io.TimedString(delta, start_time=start_time)

        item_generation = self._current_generation.messages[item_id]
        item_generation.text_ch.send_nowait(delta)
        item_generation.audio_transcript += delta

    def _handle_response_audio_delta(self, event: ResponseAudioDeltaEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"
        item_generation = self._current_generation.messages[event.item_id]
        if self._current_generation._first_token_timestamp is None:
            self._current_generation._first_token_timestamp = time.time()

        if not item_generation.modalities.done():
            item_generation.modalities.set_result(["audio", "text"])

        data = base64.b64decode(event.delta)
        item_generation.audio_ch.send_nowait(
            rtc.AudioFrame(
                data=data,
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=len(data) // 2,
            )
        )

    def _handle_response_audio_transcript_done(
        self, event: ResponseAudioTranscriptDoneEvent
    ) -> None:
        assert self._current_generation is not None, "current_generation is None"
        # also need to sync existing item's context
        remote_item = self._remote_chat_ctx.get(event.item_id)
        if remote_item and event.transcript and isinstance(remote_item.item, llm.ChatMessage):
            remote_item.item.content.append(event.transcript)

    def _handle_response_audio_done(self, _: ResponseAudioDoneEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"

    def _handle_response_output_item_done(self, event: ResponseOutputItemDoneEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"
        assert (item_id := event.item.id) is not None, "item.id is None"
        assert (item_type := event.item.type) is not None, "item.type is None"

        if item_type == "function_call" and isinstance(
            event.item, RealtimeConversationItemFunctionCall
        ):
            item = event.item
            assert item.call_id is not None, "call_id is None"
            assert item.name is not None, "name is None"
            assert item.arguments is not None, "arguments is None"

            self._current_generation.function_ch.send_nowait(
                llm.FunctionCall(
                    call_id=item.call_id,
                    name=item.name,
                    arguments=item.arguments,
                )
            )
        elif item_type == "message":
            item_generation = self._current_generation.messages[item_id]
            item_generation.text_ch.close()
            item_generation.audio_ch.close()
            if not item_generation.modalities.done():
                # in case message modalities is not set, this shouldn't happen
                item_generation.modalities.set_result(self._realtime_model._opts.modalities)

    def _handle_response_done(self, event: ResponseDoneEvent) -> None:
        if self._current_generation is None:
            return  # OpenAI has a race condition where we could receive response.done without any previous response.created (This happens generally during interruption)  # noqa: E501

        assert self._current_generation is not None, "current_generation is None"

        created_timestamp = self._current_generation._created_timestamp
        first_token_timestamp = self._current_generation._first_token_timestamp

        for generation in self._current_generation.messages.values():
            # close all messages that haven't been closed yet
            if not generation.text_ch.closed:
                generation.text_ch.close()
            if not generation.audio_ch.closed:
                generation.audio_ch.close()
            if not generation.modalities.done():
                generation.modalities.set_result(self._realtime_model._opts.modalities)

        self._current_generation.function_ch.close()
        self._current_generation.message_ch.close()
        for item_id, item_generation in self._current_generation.messages.items():
            if (remote_item := self._remote_chat_ctx.get(item_id)) and isinstance(
                remote_item.item, llm.ChatMessage
            ):
                remote_item.item.content.append(item_generation.audio_transcript)

        with contextlib.suppress(asyncio.InvalidStateError):
            self._current_generation._done_fut.set_result(None)
        self._current_generation = None

        # calculate metrics
        usage = (
            event.response.usage.model_dump(exclude_defaults=True) if event.response.usage else {}
        )
        ttft = first_token_timestamp - created_timestamp if first_token_timestamp else -1
        duration = time.time() - created_timestamp
        metrics = RealtimeModelMetrics(
            timestamp=created_timestamp,
            request_id=event.response.id or "",
            ttft=ttft,
            duration=duration,
            cancelled=event.response.status == "cancelled",
            label=self._realtime_model.label,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            tokens_per_second=usage.get("output_tokens", 0) / duration,
            input_token_details=RealtimeModelMetrics.InputTokenDetails(
                audio_tokens=usage.get("input_token_details", {}).get("audio_tokens", 0),
                cached_tokens=usage.get("input_token_details", {}).get("cached_tokens", 0),
                text_tokens=usage.get("input_token_details", {}).get("text_tokens", 0),
                cached_tokens_details=RealtimeModelMetrics.CachedTokenDetails(
                    text_tokens=usage.get("input_token_details", {})
                    .get("cached_tokens_details", {})
                    .get("text_tokens", 0),
                    audio_tokens=usage.get("input_token_details", {})
                    .get("cached_tokens_details", {})
                    .get("audio_tokens", 0),
                    image_tokens=usage.get("input_token_details", {})
                    .get("cached_tokens_details", {})
                    .get("image_tokens", 0),
                ),
                image_tokens=0,
            ),
            output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                text_tokens=usage.get("output_token_details", {}).get("text_tokens", 0),
                audio_tokens=usage.get("output_token_details", {}).get("audio_tokens", 0),
                image_tokens=0,
            ),
            metadata=Metadata(
                model_name=self._realtime_model.model, model_provider=self._realtime_model.provider
            ),
        )
        self.emit("metrics_collected", metrics)
        self._handle_response_done_but_not_complete(event)

    def _handle_response_done_but_not_complete(self, event: ResponseDoneEvent) -> None:
        """Handle response done but not complete, i.e. cancelled, incomplete or failed.

        For example this method will emit an error if we receive a "failed" status, e.g.
        with type "invalid_request_error" due to code "inference_rate_limit_exceeded".

        In other failures it will emit a debug level log.
        """
        if event.response.status == "completed":
            return

        if event.response.status == "failed":
            if event.response.status_details and hasattr(event.response.status_details, "error"):
                error_type = getattr(event.response.status_details.error, "type", "unknown")
                error_body = event.response.status_details.error
                message = f"OpenAI Realtime API response failed with error type: {error_type}"
            else:
                error_body = None
                message = "OpenAI Realtime API response failed with unknown error"
            self._emit_error(
                APIError(
                    message=message,
                    body=error_body,
                    retryable=True,
                ),
                # all possible faulures undocumented by openai,
                # so we assume optimistically all retryable/recoverable
                recoverable=True,
            )
        elif event.response.status in {"cancelled", "incomplete"}:
            logger.debug(
                "OpenAI Realtime API response done but not complete with status: %s",
                event.response.status,
                extra={
                    "event_id": event.response.id,
                    "event_response_status": event.response.status,
                },
            )
        else:
            logger.debug("Unknown response status: %s", event.response.status)

    def _handle_error(self, event: RealtimeErrorEvent) -> None:
        if event.error.message.startswith("Cancellation failed"):
            return

        logger.error(
            "OpenAI Realtime API returned an error",
            extra={"error": event.error},
        )
        self._emit_error(
            APIError(
                message="OpenAI Realtime API returned an error",
                body=event.error,
                retryable=True,
            ),
            recoverable=True,
        )

        # TODO: set exception for the response future if it exists

    def _emit_error(self, error: Exception, recoverable: bool) -> None:
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self._realtime_model._label,
                error=error,
                recoverable=recoverable,
            ),
        )
