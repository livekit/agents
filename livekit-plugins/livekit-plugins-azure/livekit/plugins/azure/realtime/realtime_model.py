from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import time
import weakref
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal, cast
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import aiohttp

from azure.ai.voicelive.models import (
    AudioInputTranscriptionOptions,
    AzureStandardVoice,
    TurnDetection,
)
from livekit import rtc
from livekit.agents import APIConnectionError, APIError, llm, utils
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from ..log import logger
from .utils import (
    DEFAULT_API_VERSION,
    DEFAULT_INPUT_AUDIO_FORMAT,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODALITIES,
    DEFAULT_OUTPUT_AUDIO_FORMAT,
    DEFAULT_TEMPERATURE,
    build_voice_config,
    livekit_item_to_azure_item,
    livekit_tool_to_azure_tool,
    to_audio_transcription,
    to_azure_tool_choice,
    to_turn_detection,
)

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
BYTES_PER_SAMPLE = NUM_CHANNELS * 2  # 2 bytes per sample for PCM16

# Voice type must match the model family: OpenAI models (gpt-*) require OpenAI
# voices like "alloy"; mixing them yields "Cannot update voice from OpenAIVoice
# to AzureVoice" from Azure and a rejected session.update.
_DEFAULT_OPENAI_VOICE = "alloy"
_DEFAULT_AZURE_VOICE = "en-US-AvaMultilingualNeural"


def _default_voice_for_model(model: str) -> str:
    return _DEFAULT_OPENAI_VOICE if model.startswith("gpt-") else _DEFAULT_AZURE_VOICE


@dataclass
class _RealtimeOptions:
    endpoint: str
    model: str
    api_version: str
    voice: str | AzureStandardVoice
    input_audio_transcription: AudioInputTranscriptionOptions | None
    tool_choice: llm.ToolChoice | None
    turn_detection: TurnDetection | None
    input_audio_format: str
    output_audio_format: str
    modalities: list[str]
    temperature: float
    max_output_tokens: int
    api_key: str | None
    use_default_credential: bool
    conn_options: APIConnectOptions


@dataclass
class _MessageGeneration:
    message_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    modalities: asyncio.Future[list[Literal["text", "audio"]]]
    audio_transcript: str = ""


@dataclass
class _FunctionCallGeneration:
    """Tracks a function call as its arguments are streamed in."""

    item_id: str
    call_id: str
    name: str
    arguments: str = ""  # Accumulated via delta events


@dataclass
class _ResponseGeneration:
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]
    messages: dict[str, _MessageGeneration]
    function_calls: dict[str, _FunctionCallGeneration]  # Track function calls being streamed
    _done_fut: asyncio.Future[None]
    _created_timestamp: float
    _first_token_timestamp: float | None = None


_ENTRA_SCOPE = "https://ai.azure.com/.default"


async def _fetch_entra_token() -> str:
    try:
        from azure.identity.aio import DefaultAzureCredential  # type: ignore
    except ImportError as e:
        raise ImportError(
            "use_default_credential=True requires the 'entra' extra. "
            "Install with: pip install livekit-plugins-azure[entra]"
        ) from e

    async with DefaultAzureCredential() as credential:
        token = await credential.get_token(_ENTRA_SCOPE)
        return str(token.token)


def _build_ws_url(endpoint: str, api_version: str, model: str) -> str:
    """Build the Azure Voice Live realtime websocket URL from a service endpoint."""
    parsed = urlparse(endpoint)
    if parsed.scheme in ("https", "wss"):
        scheme = "wss"
    elif parsed.scheme in ("http", "ws"):
        scheme = "ws"
    else:
        scheme = parsed.scheme or "wss"

    params: dict[str, str] = {"api-version": api_version, "model": model}
    for key, value_list in parse_qs(parsed.query).items():
        if key not in params and value_list:
            params[key] = value_list[0]

    path = parsed.path.rstrip("/") + "/voice-live/realtime"
    return urlunparse(
        (scheme, parsed.netloc, path, parsed.params, urlencode(params), parsed.fragment)
    )


class RealtimeModel(llm.RealtimeModel):
    def __init__(
        self,
        *,
        endpoint: str | None = None,
        model: str | None = None,
        api_version: str = DEFAULT_API_VERSION,
        voice: NotGivenOr[str | AzureStandardVoice] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[AudioInputTranscriptionOptions | None] = NOT_GIVEN,
        modalities: NotGivenOr[list[Literal["text", "audio"]]] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetection | None] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_output_tokens: NotGivenOr[int] = NOT_GIVEN,
        api_key: str | None = None,
        use_default_credential: bool = False,
        http_session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        Initialize Azure Voice Live Realtime model.

        Args:
            endpoint: Azure Voice Live service endpoint.
                If None, reads from ``AZURE_VOICE_LIVE_ENDPOINT``.
            model: Model name. If None, reads from ``AZURE_VOICE_LIVE_MODEL``
                (default: ``"gpt-realtime"``).
            api_version: Service API version (default: ``"2025-10-01"``).
            voice: Voice name or ``AzureStandardVoice``. When not given, defaults
                to ``"alloy"`` for OpenAI models (``gpt-*``) and
                ``"en-US-AvaMultilingualNeural"`` for Azure models. Mixing voice
                types with the wrong model family will be rejected by the
                server with ``"Cannot update voice from OpenAIVoice to AzureVoice"``.
            input_audio_transcription: Input transcription config. NOT_GIVEN uses
                ``whisper-1``; None disables transcription.
            modalities: Modalities to enable (default: ``["text", "audio"]``).
            turn_detection: Turn detection config. Default is ``ServerVad(threshold=0.5)``;
                pass None to disable.
            tool_choice: Tool selection policy.
            temperature: Sampling temperature (default: 0.8).
            max_output_tokens: Maximum output tokens (default: 4096).
            api_key: Azure API key. If None, reads from ``AZURE_VOICE_LIVE_API_KEY``.
                Ignored when ``use_default_credential`` is True.
            use_default_credential: Authenticate with ``DefaultAzureCredential``
                (Entra ID / managed identity) instead of an API key. Requires the
                ``entra`` optional extra: ``pip install livekit-plugins-azure[entra]``.
            http_session: Optional shared aiohttp session.
            conn_options: Connection retry/timeout options.
        """
        modalities_list: list[str] = (
            list(modalities) if is_given(modalities) else list(DEFAULT_MODALITIES)
        )
        turn_detection_val = to_turn_detection(turn_detection)
        input_audio_transcription_val = to_audio_transcription(input_audio_transcription)

        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=turn_detection_val is not None,
                user_transcription=input_audio_transcription_val is not None,
                auto_tool_reply_generation=False,  # Tool responses handled via generate_reply
                audio_output="audio" in modalities_list,
                manual_function_calls=True,
                mutable_chat_context=False,
                mutable_instructions=True,
                mutable_tools=True,
            )
        )

        endpoint_val = endpoint or os.environ.get("AZURE_VOICE_LIVE_ENDPOINT")
        if not endpoint_val:
            raise ValueError(
                "Azure Voice Live endpoint must be provided via 'endpoint' parameter "
                "or AZURE_VOICE_LIVE_ENDPOINT environment variable"
            )

        model_val = model or os.environ.get("AZURE_VOICE_LIVE_MODEL") or "gpt-realtime"

        api_key_val: str | None = None
        if not use_default_credential:
            api_key_val = api_key or os.environ.get("AZURE_VOICE_LIVE_API_KEY")
            if not api_key_val:
                raise ValueError(
                    "Azure Voice Live API key must be provided via 'api_key' parameter "
                    "or AZURE_VOICE_LIVE_API_KEY environment variable, "
                    "or set use_default_credential=True"
                )

        tool_choice_val: llm.ToolChoice | None = (
            cast(llm.ToolChoice, tool_choice) if is_given(tool_choice) else None
        )
        voice_val: str | AzureStandardVoice = (
            voice if is_given(voice) else _default_voice_for_model(model_val)
        )
        self._opts = _RealtimeOptions(
            endpoint=endpoint_val,
            model=model_val,
            api_version=api_version,
            voice=voice_val,
            input_audio_transcription=input_audio_transcription_val,
            tool_choice=tool_choice_val,
            turn_detection=turn_detection_val,
            input_audio_format=DEFAULT_INPUT_AUDIO_FORMAT,
            output_audio_format=DEFAULT_OUTPUT_AUDIO_FORMAT,
            modalities=modalities_list,
            temperature=temperature if is_given(temperature) else DEFAULT_TEMPERATURE,
            max_output_tokens=max_output_tokens
            if is_given(max_output_tokens)
            else DEFAULT_MAX_OUTPUT_TOKENS,
            api_key=api_key_val,
            use_default_credential=use_default_credential,
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
        return "azure-voicelive"

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
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


class RealtimeSession(
    llm.RealtimeSession[Literal["azure_server_event_received", "azure_client_event_sent"]]
):
    """
    Azure Voice Live Realtime API session.

    Manages a direct WebSocket connection to Azure Voice Live and handles:
    - Audio streaming (input/output)
    - Text generation
    - Function calling
    - Turn detection (VAD)
    - Session management

    Emits additional events:
    - azure_server_event_received: Raw server events from Azure
    - azure_client_event_sent: Raw client events sent to Azure
    """

    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model: RealtimeModel = realtime_model
        self._tools = llm.ToolContext.empty()
        self._instructions: str | None = None

        self._closing = False
        self._reconnecting = False
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._msg_ch = utils.aio.Chan[dict[str, Any]]()
        self._connection_ready = asyncio.Event()
        self._main_atask = asyncio.create_task(
            self._main_task(), name="AzureRealtimeSession._main_task"
        )

        self._current_generation: _ResponseGeneration | None = None
        self._remote_chat_ctx = llm.remote_chat_context.RemoteChatContext()

        self._update_chat_ctx_lock = asyncio.Lock()
        self._update_fnc_ctx_lock = asyncio.Lock()

        self._bstream = utils.audio.AudioByteStream(
            SAMPLE_RATE, NUM_CHANNELS, samples_per_channel=SAMPLE_RATE // 10
        )
        self._input_resampler: rtc.AudioResampler | None = None
        self._pushed_duration_s: float = 0

        self._response_id: str | None = None
        self._session_id: str | None = None

        # Maps client_event_id -> Future resolved when response.created arrives
        self._response_created_futures: dict[str, asyncio.Future[llm.GenerationCreatedEvent]] = {}

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._remote_chat_ctx.to_chat_ctx()

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools

    @property
    def tools_ctx(self) -> llm.ToolContext:
        return self._tools

    # ------------------------------------------------------------------ #
    # Outbound event helpers
    # ------------------------------------------------------------------ #

    def _send_event(self, event: dict[str, Any]) -> None:
        """Queue a client event to be sent over the websocket."""
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(event)
        self.emit("azure_client_event_sent", event)

    # ------------------------------------------------------------------ #
    # Connection lifecycle
    # ------------------------------------------------------------------ #

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        """Drive the websocket connection with reconnection/retry."""
        num_retries: int = 0
        max_retries = self._realtime_model._opts.conn_options.max_retry

        while not self._closing:
            try:
                ws = await self._create_ws_conn()
                self._ws = ws
                try:
                    if self._reconnecting:
                        await self._resync_chat_ctx()
                    await self._run_ws(ws)
                finally:
                    self._ws = None
                    self._connection_ready.clear()
                    self._reconnecting = True
                    with contextlib.suppress(Exception):
                        await ws.close()

                # Connection closed cleanly (e.g. session ended or shutdown)
                if self._closing:
                    break
                # Otherwise treat unexpected close as retryable
                raise APIConnectionError("Azure Voice Live connection closed unexpectedly")

            except APIError as e:
                if max_retries == 0 or not e.retryable:
                    self._emit_error(e, recoverable=False)
                    raise
                if num_retries >= max_retries:
                    self._emit_error(e, recoverable=False)
                    raise APIConnectionError(
                        f"Azure Voice Live connection failed after {num_retries} attempts"
                    ) from e
                self._emit_error(e, recoverable=True)
                retry_interval = self._realtime_model._opts.conn_options._interval_for_retry(
                    num_retries
                )
                logger.warning(
                    f"Azure Voice Live connection failed, retrying in {retry_interval}s",
                    exc_info=e,
                    extra={"attempt": num_retries, "max_retries": max_retries},
                )
                await asyncio.sleep(retry_interval)
                num_retries += 1

            except asyncio.CancelledError:
                raise

            except Exception as e:
                self._emit_error(e, recoverable=False)
                raise

    async def _create_ws_conn(self) -> aiohttp.ClientWebSocketResponse:
        opts = self._realtime_model._opts
        url = _build_ws_url(opts.endpoint, opts.api_version, opts.model)
        headers: dict[str, str] = {"User-Agent": "LiveKit Agents"}

        if opts.use_default_credential:
            headers["Authorization"] = f"Bearer {await _fetch_entra_token()}"
        else:
            assert opts.api_key is not None
            headers["api-key"] = opts.api_key

        try:
            return await asyncio.wait_for(
                self._realtime_model._ensure_http_session().ws_connect(
                    url, headers=headers, max_msg_size=4 * 1024 * 1024, heartbeat=30
                ),
                opts.conn_options.timeout,
            )
        except aiohttp.ClientError as e:
            raise APIConnectionError("Azure Voice Live client connection error") from e
        except asyncio.TimeoutError as e:
            raise APIConnectionError("Azure Voice Live connection timed out") from e

    async def _run_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Run the outbound and inbound loops until the socket closes."""
        # Configure the session first
        self._send_event(self._build_session_update_event())
        # Azure Voice Live's session.update silently drops input_audio_transcription
        # for gpt-realtime; transcription must be configured through the separate
        # transcription_session.update event.
        if (transcription := self._build_transcription_session_event()) is not None:
            self._send_event(transcription)

        closing = False

        @utils.log_exceptions(logger=logger)
        async def _send_task() -> None:
            nonlocal closing
            async for event in self._msg_ch:
                try:
                    await ws.send_str(json.dumps(event))
                except Exception:
                    logger.exception("Failed to send event")
            closing = True
            with contextlib.suppress(Exception):
                await ws.close()

        @utils.log_exceptions(logger=logger)
        async def _recv_task() -> None:
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing or self._closing:
                        return
                    raise APIConnectionError("Azure Voice Live connection closed unexpectedly")
                if msg.type == aiohttp.WSMsgType.ERROR:
                    raise APIConnectionError(
                        f"Azure Voice Live websocket error: {ws.exception()!r}"
                    )
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    event = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.exception("Failed to decode Azure Voice Live event")
                    continue

                try:
                    await self._handle_event(event)
                except Exception:
                    logger.exception(
                        "Failed to handle Azure Voice Live event",
                        extra={"event_type": event.get("type")},
                    )

        send = asyncio.create_task(_send_task(), name="_send_task")
        recv = asyncio.create_task(_recv_task(), name="_recv_task")
        try:
            done, _ = await asyncio.wait({send, recv}, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                # Propagate exceptions from completed tasks
                task.result()
        finally:
            await utils.aio.cancel_and_wait(send, recv)

    async def _resync_chat_ctx(self) -> None:
        """Re-send chat context on reconnection.

        After a reconnect, the new Azure session starts with an empty
        conversation. Reset _remote_chat_ctx and re-create each item so the
        diff sees them as new — mirrors the OpenAI plugin's _reconnect path.
        """
        old_remote_ctx = self._remote_chat_ctx
        self._remote_chat_ctx = llm.remote_chat_context.RemoteChatContext()

        chat_ctx = old_remote_ctx.to_chat_ctx().copy(
            exclude_function_call=True,
            exclude_instructions=True,
            exclude_empty_message=True,
            exclude_handoff=True,
            exclude_config_update=True,
        )

        if not chat_ctx.items:
            return

        logger.info(
            f"[RESYNC] Re-sending {len(chat_ctx.items)} chat context items after reconnection"
        )
        for chat_item in chat_ctx.items:
            try:
                azure_item = livekit_item_to_azure_item(chat_item)
                self._send_event({"type": "conversation.item.create", "item": azure_item})
                prev_id = (
                    self._remote_chat_ctx._tail.item.id if self._remote_chat_ctx._tail else None
                )
                self._remote_chat_ctx.insert(prev_id, chat_item)
            except Exception:
                logger.exception(f"[RESYNC] Failed to re-send item {chat_item.id}")

        for fut in self._response_created_futures.values():
            if not fut.done():
                fut.set_exception(
                    llm.RealtimeError("pending response discarded due to session reconnection")
                )
        self._response_created_futures.clear()

    # ------------------------------------------------------------------ #
    # Session config
    # ------------------------------------------------------------------ #

    def _build_session_update_event(self) -> dict[str, Any]:
        opts = self._realtime_model._opts
        tools_list: list[dict[str, Any]] = []
        for tool in self._tools.flatten():
            converted = livekit_tool_to_azure_tool(tool)
            if converted is not None:
                tools_list.append(converted)

        language = (
            opts.input_audio_transcription.language if opts.input_audio_transcription else None
        )

        session: dict[str, Any] = {
            "modalities": list(opts.modalities),
            "instructions": self._instructions or "You are a helpful assistant.",
            "voice": build_voice_config(opts.voice, language),
            "input_audio_format": opts.input_audio_format,
            "output_audio_format": opts.output_audio_format,
            "temperature": opts.temperature,
            "max_response_output_tokens": opts.max_output_tokens,
            "tool_choice": to_azure_tool_choice(opts.tool_choice),
        }
        if opts.turn_detection is not None:
            session["turn_detection"] = opts.turn_detection.as_dict()
        if opts.input_audio_transcription is not None:
            session["input_audio_transcription"] = opts.input_audio_transcription.as_dict()
        if tools_list:
            session["tools"] = tools_list

        return {"type": "session.update", "session": session}

    def _build_transcription_session_event(self) -> dict[str, Any] | None:
        transcription = self._realtime_model._opts.input_audio_transcription
        if transcription is None:
            return None
        return {"type": "transcription_session.update", "session": transcription.as_dict()}

    # ------------------------------------------------------------------ #
    # Inbound event handling
    # ------------------------------------------------------------------ #

    async def _handle_event(self, event: dict[str, Any]) -> None:
        self.emit("azure_server_event_received", event)

        event_type = event.get("type", "")

        if event_type not in ("response.audio.delta", "response.audio_transcript.delta"):
            logger.debug(f"[EVENT] Received: {event_type}")

        if event_type == "session.updated" or event_type == "session.created":
            await self._handle_session_updated(event)

        elif event_type == "input_audio_buffer.speech_started":
            self.emit("input_speech_started", llm.InputSpeechStartedEvent())
            if self._current_generation:
                await self._cancel_response()

        elif event_type == "input_audio_buffer.speech_stopped":
            self.emit(
                "input_speech_stopped",
                llm.InputSpeechStoppedEvent(
                    user_transcription_enabled=self._realtime_model._opts.input_audio_transcription
                    is not None
                ),
            )

        elif event_type == "conversation.item.input_audio_transcription.delta":
            # Streaming partial transcripts — keep around in case the framework
            # wants them later. The OpenAI plugin currently ignores deltas too.
            pass

        elif event_type == "conversation.item.input_audio_transcription.completed":
            item_id = event.get("item_id", "") or ""
            transcript = event.get("transcript", "") or ""
            self.emit(
                "input_audio_transcription_completed",
                llm.InputTranscriptionCompleted(
                    item_id=item_id,
                    transcript=transcript,
                    is_final=True,
                ),
            )

        elif event_type == "conversation.item.input_audio_transcription.failed":
            err = event.get("error") or {}
            err_msg = err.get("message", "Unknown error") if isinstance(err, dict) else str(err)
            logger.warning(
                "Azure Voice Live input audio transcription failed",
                extra={"item_id": event.get("item_id"), "error": err_msg},
            )

        elif event_type == "response.created":
            await self._handle_response_created(event)

        elif event_type == "response.output_item.added":
            await self._handle_output_item_added(event)

        elif event_type == "response.content_part.added":
            await self._handle_content_part_added(event)

        elif event_type == "response.audio.delta":
            await self._handle_audio_delta(event)

        elif event_type in ("response.audio_transcript.delta", "response.text.delta"):
            await self._handle_text_delta(event)

        elif event_type == "response.function_call_arguments.delta":
            await self._handle_function_call_arguments_delta(event)

        elif event_type == "response.function_call_arguments.done":
            await self._handle_function_call_arguments_done(event)

        elif event_type == "response.done":
            await self._handle_response_done(event)

        elif event_type == "error":
            error = event.get("error") or {}
            error_msg = (
                error.get("message", "Unknown error") if isinstance(error, dict) else str(error)
            )
            error_lower = error_msg.lower()
            # Suppress benign races where cancel arrives after the response ended
            if "no active response" in error_lower or "response_cancel_not_active" in error_lower:
                logger.debug(f"Azure Voice Live (suppressed): {error_msg}")
                return
            logger.error(f"Azure Voice Live error: {error_msg}")
            self._emit_error(APIError(error_msg), recoverable=True)

    async def _handle_session_updated(self, event: dict[str, Any]) -> None:
        session = event.get("session") or {}
        new_session_id = session.get("id") if isinstance(session, dict) else None
        old_session_id = self._session_id
        self._session_id = new_session_id

        if isinstance(session, dict):
            logger.info(
                "Azure Voice Live session configured",
                extra={
                    "session_id": new_session_id,
                    "voice": session.get("voice"),
                    "input_audio_transcription": session.get("input_audio_transcription"),
                    "turn_detection": session.get("turn_detection"),
                    "modalities": session.get("modalities"),
                },
            )

        if old_session_id is not None and new_session_id != old_session_id:
            for fut in self._response_created_futures.values():
                if not fut.done():
                    fut.set_exception(
                        llm.RealtimeError("pending response discarded due to session reconnection")
                    )
            self._response_created_futures.clear()

        self._connection_ready.set()
        self.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())

    def _close_generation(self, gen: _ResponseGeneration) -> None:
        for msg_gen in gen.messages.values():
            if not msg_gen.modalities.done():
                msg_gen.modalities.set_result([])
            msg_gen.text_ch.close()
            msg_gen.audio_ch.close()

        gen.message_ch.close()
        gen.function_ch.close()

        if not gen._done_fut.done():
            gen._done_fut.set_result(None)

    async def _handle_response_created(self, event: dict[str, Any]) -> None:
        response = event.get("response") or {}
        response_id = response.get("id") if isinstance(response, dict) else None
        self._response_id = response_id

        if self._current_generation is not None:
            logger.warning(
                f"New response {response_id} started while previous generation was still active, "
                "closing previous generation channels"
            )
            self._close_generation(self._current_generation)
            self._current_generation = None

        gen = _ResponseGeneration(
            message_ch=utils.aio.Chan[llm.MessageGeneration](),
            function_ch=utils.aio.Chan[llm.FunctionCall](),
            messages={},
            function_calls={},
            _done_fut=asyncio.Future(),
            _created_timestamp=time.time(),
        )
        self._current_generation = gen

        generation_ev = llm.GenerationCreatedEvent(
            message_stream=gen.message_ch,
            function_stream=gen.function_ch,
            user_initiated=False,
            response_id=response_id,
        )

        # Azure doesn't echo back our event_id, so we can't strictly match.
        # Pop the oldest pending generate_reply future (FIFO).
        if self._response_created_futures:
            event_id, fut = next(iter(self._response_created_futures.items()))
            self._response_created_futures.pop(event_id)

            if not fut.done():
                generation_ev.user_initiated = True
                fut.set_result(generation_ev)

        self.emit("generation_created", generation_ev)

    async def _handle_output_item_added(self, event: dict[str, Any]) -> None:
        if not self._current_generation:
            return

        item = event.get("item") or {}
        if not isinstance(item, dict):
            return

        item_id = item.get("id") or utils.shortuuid("msg_")
        item_type = item.get("type")

        if item_type == "message":
            msg_gen = _MessageGeneration(
                message_id=item_id,
                text_ch=utils.aio.Chan[str](),
                audio_ch=utils.aio.Chan[rtc.AudioFrame](maxsize=25),  # ~500ms buffer
                modalities=asyncio.Future(),
            )
            self._current_generation.messages[item_id] = msg_gen
            self._current_generation.message_ch.send_nowait(
                llm.MessageGeneration(
                    message_id=item_id,
                    text_stream=msg_gen.text_ch,
                    audio_stream=msg_gen.audio_ch,
                    modalities=msg_gen.modalities,
                )
            )

        elif item_type == "function_call":
            self._current_generation.function_calls[item_id] = _FunctionCallGeneration(
                item_id=item_id,
                call_id=item.get("call_id", "") or "",
                name=item.get("name", "") or "",
                arguments="",
            )

    async def _handle_content_part_added(self, event: dict[str, Any]) -> None:
        if not self._current_generation:
            return

        item_id = event.get("item_id")
        part = event.get("part")
        if (
            not item_id
            or not isinstance(part, dict)
            or item_id not in self._current_generation.messages
        ):
            return

        msg_gen = self._current_generation.messages[item_id]
        part_type = part.get("type")

        result_modalities: list[Literal["text", "audio"]]
        if part_type == "audio":
            result_modalities = ["audio", "text"]
        elif part_type == "text":
            result_modalities = ["text"]
        else:
            return

        with contextlib.suppress(asyncio.InvalidStateError):
            msg_gen.modalities.set_result(result_modalities)

    async def _handle_audio_delta(self, event: dict[str, Any]) -> None:
        if not self._current_generation:
            return

        item_id = event.get("item_id")
        delta_b64 = event.get("delta")
        if not delta_b64 or not item_id:
            return

        msg_gen = self._current_generation.messages.get(item_id)
        if not msg_gen:
            return

        if self._current_generation._first_token_timestamp is None:
            self._current_generation._first_token_timestamp = time.time()

        try:
            audio_bytes = base64.b64decode(delta_b64)
            frame = rtc.AudioFrame(
                data=audio_bytes,
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=len(audio_bytes) // BYTES_PER_SAMPLE,
            )
            msg_gen.audio_ch.send_nowait(frame)
        except Exception:
            logger.exception("Failed to process audio delta")

    async def _handle_text_delta(self, event: dict[str, Any]) -> None:
        if not self._current_generation:
            return

        item_id = event.get("item_id")
        delta = event.get("delta")
        if not delta or not item_id or item_id not in self._current_generation.messages:
            return

        if self._current_generation._first_token_timestamp is None:
            self._current_generation._first_token_timestamp = time.time()

        msg_gen = self._current_generation.messages[item_id]
        msg_gen.text_ch.send_nowait(delta)
        msg_gen.audio_transcript += delta

    async def _handle_function_call_arguments_delta(self, event: dict[str, Any]) -> None:
        if not self._current_generation:
            return

        item_id = event.get("item_id")
        delta = event.get("delta") or ""
        if not item_id or item_id not in self._current_generation.function_calls:
            return

        if self._current_generation._first_token_timestamp is None:
            self._current_generation._first_token_timestamp = time.time()

        self._current_generation.function_calls[item_id].arguments += delta

    async def _handle_function_call_arguments_done(self, event: dict[str, Any]) -> None:
        if not self._current_generation:
            return

        item_id = event.get("item_id")
        if not item_id or item_id not in self._current_generation.function_calls:
            return

        fnc = self._current_generation.function_calls[item_id]
        self._current_generation.function_ch.send_nowait(
            llm.FunctionCall(
                id=fnc.item_id,
                call_id=fnc.call_id,
                name=fnc.name,
                arguments=fnc.arguments,
            )
        )

    async def _handle_response_done(self, event: dict[str, Any]) -> None:
        if not self._current_generation:
            return

        self._close_generation(self._current_generation)

        if self._response_id:
            ttft = -1.0
            if self._current_generation._first_token_timestamp:
                ttft = (
                    self._current_generation._first_token_timestamp
                    - self._current_generation._created_timestamp
                )

            duration = time.time() - self._current_generation._created_timestamp

            response = event.get("response") or {}
            usage = response.get("usage") if isinstance(response, dict) else None
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            input_audio_tokens = 0
            input_text_tokens = 0
            output_audio_tokens = 0
            output_text_tokens = 0

            if isinstance(usage, dict):
                input_tokens = usage.get("input_tokens", 0) or 0
                output_tokens = usage.get("output_tokens", 0) or 0
                total_tokens = usage.get("total_tokens", 0) or 0

                input_token_details = usage.get("input_token_details") or {}
                if isinstance(input_token_details, dict):
                    input_audio_tokens = input_token_details.get("audio_tokens", 0) or 0
                    input_text_tokens = input_token_details.get("text_tokens", 0) or 0

                output_token_details = usage.get("output_token_details") or {}
                if isinstance(output_token_details, dict):
                    output_audio_tokens = output_token_details.get("audio_tokens", 0) or 0
                    output_text_tokens = output_token_details.get("text_tokens", 0) or 0

            tokens_per_second = output_tokens / duration if duration > 0 else 0

            self.emit(
                "metrics_collected",
                RealtimeModelMetrics(
                    timestamp=time.time(),
                    request_id=self._response_id,
                    ttft=ttft,
                    duration=duration,
                    cancelled=False,
                    label=self.realtime_model.label,
                    error=None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    tokens_per_second=tokens_per_second,
                    input_token_details=RealtimeModelMetrics.InputTokenDetails(
                        audio_tokens=input_audio_tokens,
                        text_tokens=input_text_tokens,
                        image_tokens=0,
                        cached_tokens=0,
                        cached_tokens_details=None,
                    ),
                    output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                        text_tokens=output_text_tokens,
                        audio_tokens=output_audio_tokens,
                        image_tokens=0,
                    ),
                ),
            )

        self._current_generation = None
        self._response_id = None

    # ------------------------------------------------------------------ #
    # Audio input
    # ------------------------------------------------------------------ #

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if not self._ws or not self._connection_ready.is_set():
            return

        for resampled_frame in self._resample_audio(frame):
            for audio_frame in self._bstream.push(resampled_frame.data):
                audio_b64 = base64.b64encode(audio_frame.data).decode("utf-8")
                self._send_event({"type": "input_audio_buffer.append", "audio": audio_b64})

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler:
            if frame.sample_rate != self._input_resampler._input_rate:
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
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    def push_video(self, frame: rtc.VideoFrame) -> None:
        logger.warning("push_video() is not supported by Azure Voice Live")

    def commit_audio(self) -> None:
        self._send_event({"type": "input_audio_buffer.commit"})

    def clear_audio(self) -> None:
        self._send_event({"type": "input_audio_buffer.clear"})

    def commit_user_turn(self) -> None:
        logger.warning("commit_user_turn is not supported by Azure Realtime API.")

    # ------------------------------------------------------------------ #
    # Options / instructions / chat ctx / tools updates
    # ------------------------------------------------------------------ #

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        if is_given(tool_choice):
            if self._realtime_model._opts.tool_choice != tool_choice:
                self._realtime_model._opts.tool_choice = tool_choice
                self._send_event(
                    {
                        "type": "session.update",
                        "session": {"tool_choice": to_azure_tool_choice(tool_choice)},
                    }
                )

    async def update_instructions(self, instructions: str) -> None:
        self._instructions = instructions
        self._send_event({"type": "session.update", "session": {"instructions": instructions}})

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        async with self._update_chat_ctx_lock:
            chat_ctx = chat_ctx.copy(
                exclude_handoff=True,
                exclude_config_update=True,
            )
            from livekit.agents.voice.generation import remove_instructions

            remove_instructions(chat_ctx)

            try:
                await asyncio.wait_for(self._connection_ready.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for connection, cannot update chat context")
                return

            remote_ctx = self._remote_chat_ctx.to_chat_ctx()
            diff_ops = llm.utils.compute_chat_ctx_diff(remote_ctx, chat_ctx)

            for previous_msg_id, msg_id in diff_ops.to_create:
                chat_item = chat_ctx.get_by_id(msg_id)
                if not chat_item:
                    continue
                try:
                    azure_item = livekit_item_to_azure_item(chat_item)
                    self._send_event({"type": "conversation.item.create", "item": azure_item})
                    self._remote_chat_ctx.insert(previous_msg_id, chat_item)
                except Exception:
                    logger.exception(f"Failed to create conversation item {msg_id}")

            if diff_ops.to_remove:
                logger.debug(f"Ignoring {len(diff_ops.to_remove)} items to remove (not supported)")
            if diff_ops.to_update:
                logger.debug(f"Ignoring {len(diff_ops.to_update)} items to update (not supported)")

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        async with self._update_fnc_ctx_lock:
            self._tools = llm.ToolContext(tools)

            tools_list: list[dict[str, Any]] = []
            for t in tools:
                converted = livekit_tool_to_azure_tool(t)
                if converted is not None:
                    tools_list.append(converted)

            self._send_event(
                {
                    "type": "session.update",
                    "session": {"tools": tools_list if tools_list else None},
                }
            )

    # ------------------------------------------------------------------ #
    # Generate / cancel / truncate
    # ------------------------------------------------------------------ #

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        event_id = utils.shortuuid("response_create_")
        fut = asyncio.Future[llm.GenerationCreatedEvent]()
        self._response_created_futures[event_id] = fut

        asyncio.create_task(self._run_generate_reply(event_id, instructions))

        def _on_timeout() -> None:
            if fut and not fut.done():
                fut.set_exception(llm.RealtimeError("generate_reply timed out."))
                self._response_created_futures.pop(event_id, None)

        handle = asyncio.get_running_loop().call_later(5.0, _on_timeout)
        fut.add_done_callback(lambda _: handle.cancel())

        return fut

    async def _run_generate_reply(self, event_id: str, instructions: NotGivenOr[str]) -> None:
        try:
            await asyncio.wait_for(self._connection_ready.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            if fut := self._response_created_futures.pop(event_id, None):
                fut.set_exception(APIError("Timeout waiting for connection"))
            return

        if not self._ws:
            if fut := self._response_created_futures.pop(event_id, None):
                fut.set_exception(APIError("No active connection"))
            return

        event: dict[str, Any] = {"type": "response.create", "event_id": event_id}
        if is_given(instructions):
            event["additional_instructions"] = instructions
        self._send_event(event)

    def interrupt(self) -> None:
        asyncio.create_task(self._cancel_response())

    async def _cancel_response(self) -> None:
        if not self._ws or not self._current_generation:
            return
        self._send_event({"type": "response.cancel"})

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        logger.warning("truncate() is not supported by Azure Voice Live")

    # ------------------------------------------------------------------ #
    # Shutdown
    # ------------------------------------------------------------------ #

    def _emit_error(self, error: Exception, recoverable: bool) -> None:
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self.realtime_model.label,
                error=error,
                recoverable=recoverable,
            ),
        )

    async def aclose(self) -> None:
        self._closing = True
        self._msg_ch.close()
        if self._ws:
            with contextlib.suppress(Exception):
                await self._ws.close()
        self._main_atask.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_atask
