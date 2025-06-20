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
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from openai.types.beta.realtime import (
    ConversationItem,
    ConversationItemContent,
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    ConversationItemDeletedEvent,
    ConversationItemDeleteEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionFailedEvent,
    ConversationItemTruncateEvent,
    ErrorEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferClearEvent,
    InputAudioBufferCommitEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    RealtimeClientEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCancelEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    SessionUpdateEvent,
    session_update_event,
)
from openai.types.beta.realtime.response_create_event import Response
from openai.types.beta.realtime.session import (
    InputAudioNoiseReduction,
    InputAudioTranscription,
    Tracing,
    TracingTracingConfiguration,
    TurnDetection,
)

from ..log import logger

# When a response is created with the OpenAI Realtime API, those events are sent in this order:
# 1. response.created (contains resp_id)
# 2. response.output_item.added (contains item_id)
# 3. conversation.item.created
# 4. response.content_part.added (type audio/text)
# 5. response.audio_transcript.delta (x2, x3, x4, etc)
# 6. response.audio.delta (x2, x3, x4, etc)
# 7. response.content_part.done
# 8. response.output_item.done (contains item_status: "completed/incomplete")
# 9. response.done (contains status_details for cancelled/failed/turn_detected/content_filter)
#
# Ourcode assumes a response will generate only one item with type "message"


SAMPLE_RATE = 24000
NUM_CHANNELS = 1
OPENAI_BASE_URL = "https://api.openai.com/v1"

lk_oai_debug = int(os.getenv("LK_OPENAI_DEBUG", 0))


@dataclass
class _RealtimeOptions:
    model: str
    voice: str
    temperature: float
    tool_choice: llm.ToolChoice | None
    input_audio_transcription: InputAudioTranscription | None
    input_audio_noise_reduction: InputAudioNoiseReduction | None
    turn_detection: TurnDetection | None
    max_response_output_tokens: int | Literal["inf"] | None
    speed: float | None
    tracing: Tracing | None
    api_key: str | None
    base_url: str
    is_azure: bool
    azure_deployment: str | None
    entra_token: str | None
    api_version: str | None
    max_session_duration: float | None
    """reset the connection after this many seconds if provided"""
    conn_options: APIConnectOptions


@dataclass
class _MessageGeneration:
    message_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
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


@dataclass
class _CreateResponseHandle:
    instructions: NotGivenOr[str]
    done_fut: asyncio.Future[llm.GenerationCreatedEvent]
    timeout: asyncio.TimerHandle | None = None

    def timeout_start(self) -> None:
        if self.timeout or self.done_fut is None or self.done_fut.done():
            return

        def _on_timeout() -> None:
            if not self.done_fut.done():
                self.done_fut.set_exception(llm.RealtimeError("generate_reply timed out."))

        def _on_done(_: asyncio.Future[llm.GenerationCreatedEvent]) -> None:
            if self.timeout:
                self.timeout.cancel()

        self.timeout = asyncio.get_event_loop().call_later(5.0, _on_timeout)
        self.done_fut.add_done_callback(_on_done)


_MOCK_AUDIO_ID_PREFIX = "lk_mock_audio_item_"

# default values got from a "default" session from their API
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TURN_DETECTION = TurnDetection(
    type="server_vad",
    threshold=0.5,
    prefix_padding_ms=300,
    silence_duration_ms=200,
    create_response=True,
    interrupt_response=True,
)
DEFAULT_INPUT_AUDIO_TRANSCRIPTION = InputAudioTranscription(
    model="gpt-4o-mini-transcribe",
)
DEFAULT_TOOL_CHOICE = "auto"
DEFAULT_MAX_RESPONSE_OUTPUT_TOKENS = "inf"

AZURE_DEFAULT_TURN_DETECTION = TurnDetection(
    type="server_vad",
    threshold=0.5,
    prefix_padding_ms=300,
    silence_duration_ms=200,
    create_response=True,
)

AZURE_DEFAULT_INPUT_AUDIO_TRANSCRIPTION = InputAudioTranscription(
    model="whisper-1",
)

DEFAULT_MAX_SESSION_DURATION = 20 * 60  # 20 minutes


class RealtimeModel(llm.RealtimeModel):
    @overload
    def __init__(
        self,
        *,
        model: str = "gpt-4o-realtime-preview",
        voice: str = "alloy",
        input_audio_transcription: NotGivenOr[InputAudioTranscription | None] = NOT_GIVEN,
        input_audio_noise_reduction: InputAudioNoiseReduction | None = None,
        turn_detection: NotGivenOr[TurnDetection | None] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Tracing | None] = NOT_GIVEN,
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
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
        voice: str = "alloy",
        input_audio_transcription: NotGivenOr[InputAudioTranscription | None] = NOT_GIVEN,
        input_audio_noise_reduction: InputAudioNoiseReduction | None = None,
        turn_detection: NotGivenOr[TurnDetection | None] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Tracing | None] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None: ...

    def __init__(
        self,
        *,
        model: str = "gpt-4o-realtime-preview",
        voice: str = "alloy",
        temperature: NotGivenOr[float] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[InputAudioTranscription | None] = NOT_GIVEN,
        input_audio_noise_reduction: InputAudioNoiseReduction | None = None,
        turn_detection: NotGivenOr[TurnDetection | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Tracing | None] = NOT_GIVEN,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        azure_deployment: str | None = None,
        entra_token: str | None = None,
        api_version: str | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=True,
                turn_detection=turn_detection is not None,
                user_transcription=input_audio_transcription is not None,
                auto_tool_reply_generation=False,
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
            temperature=temperature if is_given(temperature) else DEFAULT_TEMPERATURE,
            tool_choice=tool_choice or None,
            input_audio_transcription=input_audio_transcription
            if is_given(input_audio_transcription)
            else DEFAULT_INPUT_AUDIO_TRANSCRIPTION,
            input_audio_noise_reduction=input_audio_noise_reduction,
            turn_detection=turn_detection if is_given(turn_detection) else DEFAULT_TURN_DETECTION,
            api_key=api_key,
            base_url=base_url_val,
            is_azure=is_azure,
            azure_deployment=azure_deployment,
            entra_token=entra_token,
            api_version=api_version,
            max_response_output_tokens=DEFAULT_MAX_RESPONSE_OUTPUT_TOKENS,  # type: ignore
            speed=speed if is_given(speed) else None,
            tracing=cast(Union[Tracing, None], tracing) if is_given(tracing) else None,
            max_session_duration=max_session_duration
            if is_given(max_session_duration)
            else DEFAULT_MAX_SESSION_DURATION,
            conn_options=conn_options,
        )
        self._http_session = http_session
        self._sessions = weakref.WeakSet[RealtimeSession]()

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
        voice: str = "alloy",
        input_audio_transcription: NotGivenOr[InputAudioTranscription | None] = NOT_GIVEN,
        input_audio_noise_reduction: InputAudioNoiseReduction | None = None,
        turn_detection: NotGivenOr[TurnDetection | None] = NOT_GIVEN,
        temperature: float = 0.8,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Tracing | None] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> RealtimeModel:
        """
        Create a RealtimeClient instance configured for Azure OpenAI Service.

        Args:
            azure_deployment (str): The name of your Azure OpenAI deployment.
            azure_endpoint (str or None, optional): The endpoint URL for your Azure OpenAI resource. If None, will attempt to read from the environment variable AZURE_OPENAI_ENDPOINT.
            api_version (str or None, optional): API version to use with Azure OpenAI Service. If None, will attempt to read from the environment variable OPENAI_API_VERSION.
            api_key (str or None, optional): Azure OpenAI API key. If None, will attempt to read from the environment variable AZURE_OPENAI_API_KEY.
            entra_token (str or None, optional): Azure Entra authentication token. Required if not using API key authentication.
            base_url (str or None, optional): Base URL for the API endpoint. If None, constructed from the azure_endpoint.
            voice (api_proto.Voice, optional): Voice setting for audio outputs. Defaults to "alloy".
            input_audio_transcription (InputTranscriptionOptions, optional): Options for transcribing input audio. Defaults to DEFAULT_INPUT_AUDIO_TRANSCRIPTION.
            input_audio_noise_reduction (InputAudioNoiseReduction or None, optional): Options for input audio noise reduction. `near_field` is for close-talking microphones such as headphones, `far_field` is for far-field microphones such as laptop or conference room microphones. Defaults to None.
            turn_detection (ServerVadOptions, optional): Options for server-based voice activity detection (VAD). Defaults to DEFAULT_SERVER_VAD_OPTIONS.
            temperature (float, optional): Sampling temperature for response generation. Defaults to 0.8.
            max_response_output_tokens (int or Literal["inf"], optional): Maximum number of tokens in the response. Defaults to "inf".
            http_session (aiohttp.ClientSession or None, optional): Async HTTP session to use for requests. If None, a new session will be created.

        Returns:
            RealtimeClient: An instance of RealtimeClient configured for Azure OpenAI Service.

        Raises:
            ValueError: If required Azure parameters are missing or invalid.
        """  # noqa: E501
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

        return cls(
            voice=voice,
            input_audio_transcription=input_audio_transcription,
            input_audio_noise_reduction=input_audio_noise_reduction,
            turn_detection=turn_detection,
            temperature=temperature,
            speed=speed,
            tracing=tracing,
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
        temperature: NotGivenOr[float] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetection | None] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[InputAudioTranscription | None] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[InputAudioNoiseReduction | None] = NOT_GIVEN,
        max_response_output_tokens: NotGivenOr[int | Literal["inf"] | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Tracing | None] = NOT_GIVEN,
    ) -> None:
        if is_given(voice):
            self._opts.voice = voice

        if is_given(temperature):
            self._opts.temperature = temperature

        if is_given(turn_detection):
            self._opts.turn_detection = turn_detection

        if is_given(tool_choice):
            self._opts.tool_choice = cast(Optional[llm.ToolChoice], tool_choice)

        if is_given(input_audio_transcription):
            self._opts.input_audio_transcription = input_audio_transcription

        if is_given(input_audio_noise_reduction):
            self._opts.input_audio_noise_reduction = input_audio_noise_reduction

        if is_given(max_response_output_tokens):
            self._opts.max_response_output_tokens = max_response_output_tokens  # type: ignore

        if is_given(speed):
            self._opts.speed = speed

        if is_given(tracing):
            self._opts.tracing = cast(Union[Tracing, None], tracing)

        for sess in self._sessions:
            sess.update_options(
                voice=voice,
                temperature=temperature,
                turn_detection=turn_detection,
                tool_choice=tool_choice,
                input_audio_transcription=input_audio_transcription,
                max_response_output_tokens=max_response_output_tokens,
                speed=speed,
                tracing=tracing,
            )

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if not self._http_session:
            self._http_session = utils.http_context.http_session()

        return self._http_session

    def session(self) -> RealtimeSession:
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None: ...


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

        self._response_created_futures: dict[str, _CreateResponseHandle] = {}
        self._text_mode_recovery_atask: asyncio.Task[None] | None = None
        self._text_mode_recovery_retries: int = 0

        self._item_delete_future: dict[str, asyncio.Future[None]] = {}
        self._item_create_future: dict[str, asyncio.Future[None]] = {}

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
            old_chat_ctx_copy = copy.deepcopy(self._remote_chat_ctx)
            self._remote_chat_ctx = llm.remote_chat_context.RemoteChatContext()
            events.extend(self._create_update_chat_ctx_events(chat_ctx))

            try:
                for ev in events:
                    msg = ev.model_dump(by_alias=True, exclude_unset=True, exclude_defaults=False)
                    self.emit("openai_client_event_queued", msg)
                    await ws_conn.send_str(json.dumps(msg))
            except Exception as e:
                self._remote_chat_ctx = old_chat_ctx_copy  # restore the old chat context
                raise APIConnectionError(
                    message=(
                        "Failed to send message to OpenAI Realtime API during session re-connection"
                    ),
                ) from e

            logger.debug("reconnected to OpenAI Realtime API")
            self.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())

        reconnecting = False
        while not self._msg_ch.closed:
            ws_conn = await self._create_ws_conn()

            try:
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
            headers["OpenAI-Beta"] = "realtime=v1"

        url = process_base_url(
            self._realtime_model._opts.base_url,
            self._realtime_model._opts.model,
            is_azure=self._realtime_model._opts.is_azure,
            api_version=self._realtime_model._opts.api_version,
            azure_deployment=self._realtime_model._opts.azure_deployment,
        )

        if lk_oai_debug:
            logger.debug(f"connecting to Realtime API: {url}")

        return await asyncio.wait_for(
            self._realtime_model._ensure_http_session().ws_connect(url=url, headers=headers),
            self._realtime_model._opts.conn_options.timeout,
        )

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
                        if event_copy["type"] == "response.audio.delta":
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
                    elif event["type"] == "conversation.item.created":
                        self._handle_conversion_item_created(
                            ConversationItemCreatedEvent.construct(**event)
                        )
                    elif event["type"] == "conversation.item.deleted":
                        self._handle_conversion_item_deleted(
                            ConversationItemDeletedEvent.construct(**event)
                        )
                    elif event["type"] == "conversation.item.input_audio_transcription.completed":
                        self._handle_conversion_item_input_audio_transcription_completed(
                            ConversationItemInputAudioTranscriptionCompletedEvent.construct(**event)
                        )
                    elif event["type"] == "conversation.item.input_audio_transcription.failed":
                        self._handle_conversion_item_input_audio_transcription_failed(
                            ConversationItemInputAudioTranscriptionFailedEvent.construct(**event)
                        )
                    elif event["type"] == "response.content_part.added":
                        self._handle_response_content_part_added(
                            ResponseContentPartAddedEvent.construct(**event)
                        )
                    elif event["type"] == "response.content_part.done":
                        self._handle_response_content_part_done(
                            ResponseContentPartDoneEvent.construct(**event)
                        )
                    elif event["type"] == "response.audio_transcript.delta":
                        self._handle_response_audio_transcript_delta(event)
                    elif event["type"] == "response.audio.delta":
                        self._handle_response_audio_delta(
                            ResponseAudioDeltaEvent.construct(**event)
                        )
                    elif event["type"] == "response.audio_transcript.done":
                        self._handle_response_audio_transcript_done(
                            ResponseAudioTranscriptDoneEvent.construct(**event)
                        )
                    elif event["type"] == "response.audio.done":
                        self._handle_response_audio_done(ResponseAudioDoneEvent.construct(**event))
                    elif event["type"] == "response.output_item.done":
                        self._handle_response_output_item_done(
                            ResponseOutputItemDoneEvent.construct(**event)
                        )
                    elif event["type"] == "response.done":
                        self._handle_response_done(ResponseDoneEvent.construct(**event))
                    elif event["type"] == "error":
                        self._handle_error(ErrorEvent.construct(**event))
                except Exception:
                    if event["type"] == "response.audio.delta":
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

        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await ws_conn.close()

    def _create_session_update_event(self) -> SessionUpdateEvent:
        input_audio_transcription_opts = self._realtime_model._opts.input_audio_transcription
        input_audio_transcription = (
            session_update_event.SessionInputAudioTranscription.model_validate(
                input_audio_transcription_opts.model_dump(
                    by_alias=True,
                    exclude_unset=True,
                    exclude_defaults=True,
                )
            )
            if input_audio_transcription_opts
            else None
        )

        turn_detection_opts = self._realtime_model._opts.turn_detection
        turn_detection = (
            session_update_event.SessionTurnDetection.model_validate(
                turn_detection_opts.model_dump(
                    by_alias=True,
                    exclude_unset=True,
                    exclude_defaults=True,
                )
            )
            if turn_detection_opts
            else None
        )

        tracing_opts = self._realtime_model._opts.tracing
        if isinstance(tracing_opts, TracingTracingConfiguration):
            tracing: session_update_event.SessionTracing | None = (
                session_update_event.SessionTracingTracingConfiguration.model_validate(
                    tracing_opts.model_dump(
                        by_alias=True,
                        exclude_unset=True,
                        exclude_defaults=True,
                    )
                )
            )
        else:
            tracing = tracing_opts

        kwargs: dict[str, Any] = {
            "model": self._realtime_model._opts.model,
            "voice": self._realtime_model._opts.voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "modalities": ["text", "audio"],
            "turn_detection": turn_detection,
            "input_audio_transcription": input_audio_transcription,
            "input_audio_noise_reduction": self._realtime_model._opts.input_audio_noise_reduction,
            "temperature": self._realtime_model._opts.temperature,
            "tool_choice": _to_oai_tool_choice(self._realtime_model._opts.tool_choice),
        }
        if self._instructions is not None:
            kwargs["instructions"] = self._instructions

        if self._realtime_model._opts.speed is not None:
            kwargs["speed"] = self._realtime_model._opts.speed

        if tracing:
            kwargs["tracing"] = tracing

        # initial session update
        return SessionUpdateEvent(
            type="session.update",
            # Using model_construct since OpenAI restricts voices to those defined in the BaseModel.  # noqa: E501
            # Other providers support different voices, so we need to accommodate that.
            session=session_update_event.Session.model_construct(**kwargs),
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
        temperature: NotGivenOr[float] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetection | None] = NOT_GIVEN,
        max_response_output_tokens: NotGivenOr[int | Literal["inf"] | None] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[InputAudioTranscription | None] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[InputAudioNoiseReduction | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Tracing | None] = NOT_GIVEN,
    ) -> None:
        kwargs: dict[str, Any] = {}

        if is_given(tool_choice):
            tool_choice = cast(Optional[llm.ToolChoice], tool_choice)
            self._realtime_model._opts.tool_choice = tool_choice
            kwargs["tool_choice"] = _to_oai_tool_choice(tool_choice)

        if is_given(voice):
            self._realtime_model._opts.voice = voice
            kwargs["voice"] = voice

        if is_given(temperature):
            self._realtime_model._opts.temperature = temperature
            kwargs["temperature"] = temperature

        if is_given(turn_detection):
            self._realtime_model._opts.turn_detection = turn_detection
            kwargs["turn_detection"] = turn_detection

        if is_given(max_response_output_tokens):
            self._realtime_model._opts.max_response_output_tokens = max_response_output_tokens  # type: ignore
            kwargs["max_response_output_tokens"] = max_response_output_tokens

        if is_given(input_audio_transcription):
            self._realtime_model._opts.input_audio_transcription = input_audio_transcription
            kwargs["input_audio_transcription"] = input_audio_transcription

        if is_given(input_audio_noise_reduction):
            self._realtime_model._opts.input_audio_noise_reduction = input_audio_noise_reduction
            kwargs["input_audio_noise_reduction"] = input_audio_noise_reduction

        if is_given(speed):
            self._realtime_model._opts.speed = speed
            kwargs["speed"] = speed

        if is_given(tracing):
            self._realtime_model._opts.tracing = cast(Union[Tracing, None], tracing)
            kwargs["tracing"] = cast(Union[Tracing, None], tracing)

        if kwargs:
            self.send_event(
                SessionUpdateEvent(
                    type="session.update",
                    session=session_update_event.Session.model_construct(**kwargs),
                    event_id=utils.shortuuid("options_update_"),
                )
            )

    async def update_chat_ctx(
        self, chat_ctx: llm.ChatContext, *, _add_mock_audio: bool = False
    ) -> None:
        async with self._update_chat_ctx_lock:
            events = self._create_update_chat_ctx_events(chat_ctx, _add_mock_audio=_add_mock_audio)
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
        self, chat_ctx: llm.ChatContext, *, _add_mock_audio: bool = False
    ) -> list[ConversationItemCreateEvent | ConversationItemDeleteEvent]:
        chat_ctx = chat_ctx.copy()
        if _add_mock_audio:
            chat_ctx.items.append(_create_mock_audio_item())
        else:
            # clean up existing mock audio items
            chat_ctx.items[:] = [
                item for item in chat_ctx.items if not item.id.startswith(_MOCK_AUDIO_ID_PREFIX)
            ]

        events: list[ConversationItemCreateEvent | ConversationItemDeleteEvent] = []
        diff_ops = llm.utils.compute_chat_ctx_diff(self._remote_chat_ctx.to_chat_ctx(), chat_ctx)

        for msg_id in diff_ops.to_remove:
            events.append(
                ConversationItemDeleteEvent(
                    type="conversation.item.delete",
                    item_id=msg_id,
                    event_id=utils.shortuuid("chat_ctx_delete_"),
                )
            )

        for previous_msg_id, msg_id in diff_ops.to_create:
            chat_item = chat_ctx.get_by_id(msg_id)
            assert chat_item is not None
            events.append(
                ConversationItemCreateEvent(
                    type="conversation.item.create",
                    item=_livekit_item_to_openai_item(chat_item),
                    previous_item_id=("root" if previous_msg_id is None else previous_msg_id),
                    event_id=utils.shortuuid("chat_ctx_create_"),
                )
            )

        return events

    async def update_tools(self, tools: list[llm.FunctionTool | llm.RawFunctionTool]) -> None:
        async with self._update_fnc_ctx_lock:
            ev = self._create_tools_update_event(tools)
            self.send_event(ev)

            assert ev.session.tools is not None
            retained_tool_names = {name for t in ev.session.tools if (name := t.name) is not None}
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
        oai_tools: list[session_update_event.SessionTool] = []
        retained_tools: list[llm.FunctionTool | llm.RawFunctionTool] = []

        for tool in tools:
            if is_function_tool(tool):
                tool_desc = llm.utils.build_legacy_openai_schema(tool, internally_tagged=True)
            elif is_raw_function_tool(tool):
                tool_info = get_raw_function_info(tool)
                tool_desc = tool_info.raw_schema
                tool_desc["type"] = "function"  # internally tagged
            else:
                logger.error(
                    "OpenAI Realtime API doesn't support this tool type", extra={"tool": tool}
                )
                continue

            try:
                session_tool = session_update_event.SessionTool.model_validate(tool_desc)
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
            session=session_update_event.Session.model_construct(
                model=self._realtime_model._opts.model,
                tools=oai_tools,
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
                session=session_update_event.Session.model_construct(instructions=instructions),
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
        pass

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
        handle = self._create_response(instructions=instructions, user_initiated=True)
        self._text_mode_recovery_retries = 0  # reset the counter
        return handle.done_fut

    def interrupt(self) -> None:
        self.send_event(ResponseCancelEvent(type="response.cancel"))

    def truncate(self, *, message_id: str, audio_end_ms: int) -> None:
        self.send_event(
            ConversationItemTruncateEvent(
                type="conversation.item.truncate",
                content_index=0,
                item_id=message_id,
                audio_end_ms=audio_end_ms,
            )
        )

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

    def _create_response(
        self,
        *,
        user_initiated: bool,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        old_handle: _CreateResponseHandle | None = None,
    ) -> _CreateResponseHandle:
        handle = old_handle or _CreateResponseHandle(
            instructions=instructions, done_fut=asyncio.Future()
        )
        if old_handle and utils.is_given(instructions):
            handle.instructions = instructions

        event_id = utils.shortuuid("response_create_")
        if user_initiated:
            self._response_created_futures[event_id] = handle

        self.send_event(
            ResponseCreateEvent(
                type="response.create",
                event_id=event_id,
                response=Response(
                    instructions=handle.instructions or None,
                    metadata={"client_event_id": event_id} if user_initiated else None,
                ),
            )
        )
        if user_initiated:
            handle.timeout_start()
        return handle

    def _emit_generation_event(self, response_id: str) -> None:
        # called when the generation is a function call or a audio message
        assert self._current_generation is not None, "current_generation is None"

        generation_ev = llm.GenerationCreatedEvent(
            message_stream=self._current_generation.message_ch,
            function_stream=self._current_generation.function_ch,
            user_initiated=False,
        )

        if handle := self._response_created_futures.pop(response_id, None):
            generation_ev.user_initiated = True
            try:
                handle.done_fut.set_result(generation_ev)
            except asyncio.InvalidStateError:
                # in case the generation comes after the reply timeout
                logger.warning(
                    "response received after timeout", extra={"response_id": response_id}
                )

        self.emit("generation_created", generation_ev)

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

        if (
            isinstance(event.response.metadata, dict)
            and (client_event_id := event.response.metadata.get("client_event_id"))
            and (handle := self._response_created_futures.pop(client_event_id, None))
        ):
            # set key to the response id
            self._response_created_futures[event.response.id] = handle

        # the generation_created event is emitted when
        # 1. the response is not a message on response.output_item.added event
        # 2. the content is audio on response.content_part.added event
        # will try to recover from text response on response.content_part.done event

    def _handle_response_output_item_added(self, event: ResponseOutputItemAddedEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"
        assert (item_type := event.item.type) is not None, "item.type is None"
        assert (response_id := event.response_id) is not None, "response_id is None"

        if item_type != "message":
            # emit immediately if it's not a message, otherwise wait response.content_part.added
            self._emit_generation_event(response_id)
            self._text_mode_recovery_retries = 0

    def _handle_response_content_part_added(self, event: ResponseContentPartAddedEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"
        assert (item_id := event.item_id) is not None, "item_id is None"
        assert (item_type := event.part.type) is not None, "part.type is None"
        assert (response_id := event.response_id) is not None, "response_id is None"

        if item_type == "audio":
            self._emit_generation_event(response_id)
            if self._text_mode_recovery_retries > 0:
                logger.info(
                    "recovered from text-only response",
                    extra={"retried_times": self._text_mode_recovery_retries},
                )
                self._text_mode_recovery_retries = 0

            item_generation = _MessageGeneration(
                message_id=item_id,
                text_ch=utils.aio.Chan(),
                audio_ch=utils.aio.Chan(),
            )
            self._current_generation.message_ch.send_nowait(
                llm.MessageGeneration(
                    message_id=item_id,
                    text_stream=item_generation.text_ch,
                    audio_stream=item_generation.audio_ch,
                )
            )
            self._current_generation.messages[item_id] = item_generation
            self._current_generation._first_token_timestamp = time.time()
        else:
            self.interrupt()
            if self._text_mode_recovery_retries == 0:
                logger.warning("received text-only response from realtime API")

    def _handle_response_content_part_done(self, event: ResponseContentPartDoneEvent) -> None:
        if event.part.type != "text":
            return

        # try to recover from text-only response on response.content_part_done event
        assert self._current_generation is not None, "current_generation is None"
        assert (item_id := event.item_id) is not None, "item_id is None"
        assert (response_id := event.response_id) is not None, "response_id is None"

        async def _retry_generation(
            item_id: str, response_handle: _CreateResponseHandle | None
        ) -> None:
            """Recover from text-only response to audio mode.

            When chat history is loaded, OpenAI Realtime API may respond with text only.
            This method recovers by:
            1. Deleting the text response
            2. Creating an empty user audio message
            3. Requesting a new response to trigger audio mode
            """

            # remove the text item
            chat_ctx = self.chat_ctx
            idx = chat_ctx.index_by_id(item_id)
            if idx is not None:
                chat_ctx.items.pop(idx)
            await self.update_chat_ctx(chat_ctx, _add_mock_audio=True)

            if response_handle and response_handle.done_fut.done():
                if response_handle.done_fut.exception() is not None:
                    logger.error("generate_reply timed out, cancel recovery")
                return

            self._create_response(
                old_handle=response_handle,
                user_initiated=response_handle is not None,
            )

        if self._text_mode_recovery_retries >= 5:
            logger.error(
                "failed to recover from text-only response",
                extra={"retried_times": self._text_mode_recovery_retries},
            )
            self._text_mode_recovery_retries = 0
            return

        handle = self._response_created_futures.pop(response_id, None)
        if handle and handle.done_fut.done():
            if handle.done_fut.exception() is not None:
                logger.error("generate_reply timed out, cancel recovery")
            self._text_mode_recovery_retries = 0
            return

        self._text_mode_recovery_retries += 1
        logger.warning(
            "trying to recover from text-only response",
            extra={"retries": self._text_mode_recovery_retries},
        )

        if self._text_mode_recovery_atask and not self._text_mode_recovery_atask.done():
            self._text_mode_recovery_atask.cancel()
        self._text_mode_recovery_atask = asyncio.create_task(
            _retry_generation(item_id=item_id, response_handle=handle)
        )

    def _handle_conversion_item_created(self, event: ConversationItemCreatedEvent) -> None:
        assert event.item.id is not None, "item.id is None"

        try:
            self._remote_chat_ctx.insert(
                event.previous_item_id, _openai_item_to_livekit_item(event.item)
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

        data = base64.b64decode(event.delta)
        item_generation.audio_ch.send_nowait(
            rtc.AudioFrame(
                data=data,
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=len(data) // 2,
            )
        )

    def _handle_response_audio_transcript_done(self, _: ResponseAudioTranscriptDoneEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"

    def _handle_response_audio_done(self, _: ResponseAudioDoneEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"

    def _handle_response_output_item_done(self, event: ResponseOutputItemDoneEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"
        assert (item_id := event.item.id) is not None, "item.id is None"
        assert (item_type := event.item.type) is not None, "item.type is None"

        if item_type == "function_call":
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
        elif item_type == "message" and (
            item_generation := self._current_generation.messages.get(item_id)
        ):
            # text response doesn't have item_generation
            item_generation.text_ch.close()
            item_generation.audio_ch.close()

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
            label=self._realtime_model._label,
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
        )
        self.emit("metrics_collected", metrics)

    def _handle_error(self, event: ErrorEvent) -> None:
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


def _livekit_item_to_openai_item(item: llm.ChatItem) -> ConversationItem:
    conversation_item = ConversationItem(
        id=item.id,
    )

    if item.type == "function_call":
        conversation_item.type = "function_call"
        conversation_item.call_id = item.call_id
        conversation_item.name = item.name
        conversation_item.arguments = item.arguments

    elif item.type == "function_call_output":
        conversation_item.type = "function_call_output"
        conversation_item.call_id = item.call_id
        conversation_item.output = item.output

    elif item.type == "message":
        role = "system" if item.role == "developer" else item.role
        conversation_item.type = "message"
        conversation_item.role = role

        content_list: list[ConversationItemContent] = []
        for c in item.content:
            if isinstance(c, str):
                content_list.append(
                    ConversationItemContent(
                        type=("text" if role == "assistant" else "input_text"),
                        text=c,
                    )
                )

            elif isinstance(c, llm.ImageContent):
                continue  # not supported for now
            elif isinstance(c, llm.AudioContent):
                if conversation_item.role == "user":
                    encoded_audio = base64.b64encode(rtc.combine_audio_frames(c.frame).data).decode(
                        "utf-8"
                    )

                    content_list.append(
                        ConversationItemContent(
                            type="input_audio",
                            audio=encoded_audio,
                            transcript=c.transcript,
                        )
                    )

        conversation_item.content = content_list

    return conversation_item


def _openai_item_to_livekit_item(item: ConversationItem) -> llm.ChatItem:
    assert item.id is not None, "id is None"

    if item.type == "function_call":
        assert item.call_id is not None, "call_id is None"
        assert item.name is not None, "name is None"
        assert item.arguments is not None, "arguments is None"

        return llm.FunctionCall(
            id=item.id,
            call_id=item.call_id,
            name=item.name,
            arguments=item.arguments,
        )

    if item.type == "function_call_output":
        assert item.call_id is not None, "call_id is None"
        assert item.output is not None, "output is None"

        return llm.FunctionCallOutput(
            id=item.id,
            call_id=item.call_id,
            output=item.output,
            is_error=False,
        )

    if item.type == "message":
        assert item.role is not None, "role is None"
        assert item.content is not None, "content is None"

        content: list[llm.ChatContent] = []

        for c in item.content:
            if c.type == "text" or c.type == "input_text":
                assert c.text is not None, "text is None"
                content.append(c.text)

        return llm.ChatMessage(
            id=item.id,
            role=item.role,
            content=content,
        )

    raise ValueError(f"unsupported item type: {item.type}")


def _create_mock_audio_item(duration: float = 2) -> llm.ChatMessage:
    audio_data = b"\x00\x00" * int(SAMPLE_RATE * duration)
    return llm.ChatMessage(
        id=utils.shortuuid(_MOCK_AUDIO_ID_PREFIX),
        role="user",
        content=[
            llm.AudioContent(
                frame=[
                    rtc.AudioFrame(
                        data=audio_data,
                        sample_rate=SAMPLE_RATE,
                        num_channels=1,
                        samples_per_channel=len(audio_data) // 2,
                    )
                ]
            )
        ],
    )


def _to_oai_tool_choice(tool_choice: llm.ToolChoice | None) -> str:
    if isinstance(tool_choice, str):
        return tool_choice

    elif isinstance(tool_choice, dict) and tool_choice["type"] == "function":
        return tool_choice["function"]["name"]

    return "auto"
