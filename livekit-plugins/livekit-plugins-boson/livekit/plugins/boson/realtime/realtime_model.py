from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass, replace
from typing import Any, Literal, cast
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import aiohttp
from openai.types.realtime import (
    ConversationItemAdded,
    ConversationItemDeletedEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionFailedEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseContentPartAddedEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from pydantic import BaseModel

from livekit import rtc
from livekit.agents import APIConnectionError, llm, utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.plugins.openai.realtime import realtime_model as openai_rt
from livekit.plugins.openai.realtime.utils import calculate_confidence_from_logprobs

from ..log import logger

SAMPLE_RATE = openai_rt.SAMPLE_RATE
NUM_CHANNELS = openai_rt.NUM_CHANNELS

_DEFAULT_TURN_DETECTION = {
    "type": "server_vad",
    "create_response": True,
    "interrupt_response": True,
    "prefix_padding_ms": 300,
    "silence_duration_ms": 500,
    "threshold": 0.55,
}

_DEFAULT_OUTPUT_MODALITIES: list[Literal["audio"]] = ["audio"]
_VALID_OUTPUT_MODALITIES = ("text", "audio")


@dataclass
class _BosonOptions:
    url: str
    api_key: str | None
    model: str
    voice: str
    instructions: str
    temperature: float
    max_output_tokens: int | Literal["inf"]
    tool_choice: llm.ToolChoice | None
    speed: float
    turn_detection: dict[str, Any] | None
    input_audio_transcription: NotGivenOr[dict[str, Any]]
    noise_reduction: dict[str, Any] | None
    output_modalities: list[Literal["text", "audio"]]


class RealtimeModel(openai_rt.RealtimeModel):
    def __init__(
        self,
        *,
        url: str,
        api_key: str | None = None,
        model: str = "boson-realtime",
        voice: str = "default",
        instructions: str = "You are a helpful AI assistant",
        modalities: list[Literal["text", "audio"]] | None = None,
        temperature: float = 0.7,
        max_output_tokens: int | Literal["inf"] = "inf",
        tool_choice: llm.ToolChoice | None = "auto",
        speed: float = 1.0,
        turn_detection: NotGivenOr[dict[str, Any] | None] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[dict[str, Any] | None] = NOT_GIVEN,
        input_audio_transcription_model: str = "",
        input_audio_transcription_language: str | None = None,
        input_audio_transcription_prompt: str | None = None,
        input_audio_noise_reduction: NotGivenOr[str | dict[str, Any] | None] = NOT_GIVEN,
        query_params: dict[str, str] | None = None,
        http_session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        output_modalities = _resolve_output_modalities(modalities)
        turn_detection_config = (
            dict(_DEFAULT_TURN_DETECTION)
            if not is_given(turn_detection)
            else _copy_dict_or_none(turn_detection)
        )
        input_audio_transcription_config = _build_input_audio_transcription(
            input_audio_transcription=input_audio_transcription,
            model=input_audio_transcription_model,
            language=input_audio_transcription_language,
            prompt=input_audio_transcription_prompt,
        )
        noise_reduction_config = _build_noise_reduction(input_audio_noise_reduction)

        # Initialize the LiveKit/OpenAI realtime runtime, then override the pieces
        # that are Boson-specific. The OpenAI base opts are still used by inherited
        # audio, response, metrics, and function-call plumbing.
        super().__init__(
            base_url=url,
            model=model,
            voice=voice,
            modalities=list(output_modalities),
            tool_choice=tool_choice,
            input_audio_transcription=None,
            turn_detection=None,
            api_key=api_key or "boson",
            http_session=http_session,
            max_session_duration=None,
            conn_options=conn_options,
            speed=speed,
        )

        self._provider_label = "Boson Realtime API"
        self._boson_opts = _BosonOptions(
            url=_normalize_ws_url(url, query_params or {}),
            api_key=api_key,
            model=model,
            voice=voice,
            instructions=instructions,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tool_choice=tool_choice,
            speed=speed,
            turn_detection=turn_detection_config,
            input_audio_transcription=input_audio_transcription_config,
            noise_reduction=noise_reduction_config,
            output_modalities=output_modalities,
        )
        self._capabilities.turn_detection = turn_detection_config is not None
        self._capabilities.user_transcription = _input_audio_transcription_enabled(
            input_audio_transcription_config
        )
        self._capabilities.auto_tool_reply_generation = True
        self._capabilities.audio_output = "audio" in output_modalities
        self._capabilities.mutable_chat_context = False

    @property
    def model(self) -> str:
        return self._boson_opts.model

    @property
    def provider(self) -> str:
        return urlparse(self._boson_opts.url).netloc

    def session(self) -> RealtimeSession:
        session = RealtimeSession(self)
        self._sessions.add(session)
        return session

    async def aclose(self) -> None:
        for session in list(self._sessions):
            await session.aclose()
        await super().aclose()

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: NotGivenOr[Any | None] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[Any | None] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[Any | None] = NOT_GIVEN,
        max_response_output_tokens: NotGivenOr[int | Literal["inf"] | None] = NOT_GIVEN,
        tracing: NotGivenOr[Any | None] = NOT_GIVEN,
        truncation: NotGivenOr[Any | None] = NOT_GIVEN,
        reasoning: NotGivenOr[Any | None] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_output_tokens: NotGivenOr[int | Literal["inf"] | None] = NOT_GIVEN,
    ) -> None:
        _ = (tracing, truncation, reasoning)
        next_max_output_tokens = (
            max_output_tokens if is_given(max_output_tokens) else max_response_output_tokens
        )

        if is_given(voice):
            self._boson_opts.voice = voice
            self._opts.voice = voice
        if is_given(tool_choice):
            self._boson_opts.tool_choice = tool_choice
            self._opts.tool_choice = tool_choice
        if is_given(temperature):
            self._boson_opts.temperature = temperature
        if is_given(next_max_output_tokens) and next_max_output_tokens is not None:
            self._boson_opts.max_output_tokens = next_max_output_tokens
            self._opts.max_response_output_tokens = next_max_output_tokens
        if is_given(speed):
            self._boson_opts.speed = speed
            self._opts.speed = speed
        if is_given(turn_detection):
            self._boson_opts.turn_detection = _copy_dict_or_none(turn_detection)
            self._capabilities.turn_detection = self._boson_opts.turn_detection is not None
        if is_given(input_audio_transcription):
            self._boson_opts.input_audio_transcription = _normalize_input_audio_transcription(
                input_audio_transcription
            )
            self._capabilities.user_transcription = _input_audio_transcription_enabled(
                self._boson_opts.input_audio_transcription
            )
        if is_given(input_audio_noise_reduction):
            self._boson_opts.noise_reduction = _build_noise_reduction(input_audio_noise_reduction)

        for session in self._sessions:
            boson_session = cast(RealtimeSession, session)
            boson_session.update_options(
                tool_choice=tool_choice,
                voice=voice,
                temperature=temperature,
                max_response_output_tokens=next_max_output_tokens,
                max_output_tokens=max_output_tokens,
                speed=speed,
                turn_detection=turn_detection,
                input_audio_transcription=input_audio_transcription,
                input_audio_noise_reduction=input_audio_noise_reduction,
            )


class RealtimeSession(openai_rt.RealtimeSession):
    def __init__(self, realtime_model: RealtimeModel) -> None:
        self._boson_model = realtime_model
        self._boson_opts = replace(realtime_model._boson_opts)
        self._boson_closing = False
        self._closing = False
        self._closed = False
        self._suppress_next_response_cancel = False
        self._current_response_id: str | None = None
        self._boson_remote_item_ids: set[str] = set()
        super().__init__(realtime_model)
        # Compatibility alias for the previous Boson implementation and tests.
        self._pending_response_futures = self._response_created_futures

    def send_event(self, event: Any) -> None:
        if self._closed or self._msg_ch.closed:
            return
        if isinstance(event, BaseModel):
            event = event.model_dump(by_alias=True, exclude_unset=True, exclude_defaults=False)
        with contextlib.suppress(utils.aio.ChanClosed):
            self._msg_ch.send_nowait(event)

    async def _main_task(self) -> None:
        await self._run()

    async def _run(self) -> None:
        ws: aiohttp.ClientWebSocketResponse | None = None
        try:
            ws = await self._create_ws_conn()
            await self._run_ws(ws)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._boson_closing:
                self._emit_error(exc, recoverable=False)
                self._fail_response_created_futures(_as_realtime_error(exc))
                self._close_current_generation("Boson realtime session failed")
                self._closed = True
                self._msg_ch.close()
        finally:
            if ws is not None and not ws.closed:
                await ws.close()

    async def _create_ws_conn(self) -> aiohttp.ClientWebSocketResponse:
        headers = {"User-Agent": "LiveKit Agents Boson plugin"}
        if self._boson_opts.api_key:
            headers["Authorization"] = f"Bearer {self._boson_opts.api_key}"

        t0 = time.perf_counter()
        try:
            ws = await asyncio.wait_for(
                self._boson_model._ensure_http_session().ws_connect(
                    url=self._boson_opts.url,
                    headers=headers,
                ),
                self._opts.conn_options.timeout,
            )
            self._report_connection_acquired(time.perf_counter() - t0)
            return ws
        except aiohttp.ClientError as exc:
            raise APIConnectionError("Boson realtime client connection error") from exc
        except asyncio.TimeoutError as exc:
            raise APIConnectionError("Boson realtime connection timed out") from exc

    async def _run_ws(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        async def _send_task() -> None:
            async for event in self._msg_ch:
                try:
                    if isinstance(event, BaseModel):
                        event = event.model_dump(
                            by_alias=True,
                            exclude_unset=True,
                            exclude_defaults=False,
                        )
                    self.emit("openai_client_event_queued", event)
                    await ws_conn.send_str(json.dumps(event))
                except Exception:
                    logger.exception("failed to send Boson realtime event")

        recv_task = asyncio.create_task(
            self._recv_loop(ws_conn), name="BosonRealtimeSession._recv_loop"
        )
        send_task = asyncio.create_task(_send_task(), name="BosonRealtimeSession._send_loop")
        pending: set[asyncio.Task[None]] = {recv_task, send_task}
        try:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                task.result()
            if not self._boson_closing:
                raise ConnectionError("Boson realtime WebSocket closed unexpectedly.")
        finally:
            await utils.aio.cancel_and_wait(*pending)

    async def _recv_loop(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        while True:
            msg = await ws_conn.receive()
            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                if self._boson_closing or self._closing:
                    return
                raise ConnectionError(_format_ws_close_message(ws_conn, msg))
            if msg.type == aiohttp.WSMsgType.ERROR:
                raise ConnectionError(_format_ws_close_message(ws_conn, msg))
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                event = json.loads(msg.data)
            except json.JSONDecodeError as exc:
                logger.warning("Ignoring invalid Boson realtime JSON message: %s", msg.data)
                self._emit_error(
                    llm.RealtimeError(f"Invalid Boson realtime JSON message: {exc}"),
                    recoverable=True,
                )
                continue
            if not isinstance(event, dict):
                self._emit_error(
                    llm.RealtimeError("Invalid Boson realtime message: expected a JSON object."),
                    recoverable=True,
                )
                continue
            self.emit("openai_server_event_received", event)
            self._handle_server_event(event)

    def _handle_server_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        try:
            if event_type == "input_audio_buffer.speech_started":
                self._handle_input_audio_buffer_speech_started(
                    InputAudioBufferSpeechStartedEvent.construct(**event)
                )
            elif event_type == "input_audio_buffer.speech_stopped":
                self._handle_input_audio_buffer_speech_stopped(
                    InputAudioBufferSpeechStoppedEvent.construct(**event)
                )
            elif event_type == "conversation.item.input_audio_transcription.completed":
                self._handle_conversion_item_input_audio_transcription_completed(
                    ConversationItemInputAudioTranscriptionCompletedEvent.construct(**event)
                )
            elif event_type == "conversation.item.input_audio_transcription.failed":
                self._handle_conversion_item_input_audio_transcription_failed(
                    ConversationItemInputAudioTranscriptionFailedEvent.construct(**event)
                )
            elif event_type == "conversation.item.added":
                self._handle_conversion_item_added(ConversationItemAdded.construct(**event))
            elif event_type == "conversation.item.deleted":
                self._handle_conversion_item_deleted(
                    ConversationItemDeletedEvent.construct(**event)
                )
            elif event_type == "response.created":
                self._handle_response_created(ResponseCreatedEvent.construct(**event))
            elif event_type == "response.output_item.added":
                self._handle_response_output_item_added(
                    ResponseOutputItemAddedEvent.construct(**event)
                )
            elif event_type == "response.content_part.added":
                self._handle_response_content_part_added(
                    ResponseContentPartAddedEvent.construct(**event)
                )
            elif event_type == "response.output_text.delta":
                self._handle_response_text_delta(ResponseTextDeltaEvent.construct(**event))
            elif event_type == "response.output_text.done":
                self._handle_response_text_done(ResponseTextDoneEvent.construct(**event))
            elif event_type == "response.output_audio_transcript.delta":
                self._handle_response_audio_transcript_delta(event)
            elif event_type == "response.output_audio.delta":
                self._handle_response_audio_delta(ResponseAudioDeltaEvent.construct(**event))
            elif event_type == "response.output_audio.done":
                self._handle_response_audio_done(ResponseAudioDoneEvent.construct(**event))
            elif event_type in (
                "response.output_audio_transcript.length",
                "response.output_audio_transcript.done",
                "response.content_part.done",
            ):
                pass
            elif event_type == "response.output_item.done":
                self._handle_response_output_item_done(
                    ResponseOutputItemDoneEvent.construct(**event)
                )
            elif event_type == "response.done":
                self._handle_response_done(ResponseDoneEvent.construct(**event))
            elif event_type == "error":
                self._handle_boson_error_event(event)
        except Exception as exc:
            logger.exception("failed to handle Boson realtime event: %s", event_type)
            self._emit_error(exc, recoverable=True)

    def _handle_conversion_item_added(self, event: ConversationItemAdded) -> None:
        super()._handle_conversion_item_added(event)
        if event.item.id is not None:
            self._boson_remote_item_ids.add(event.item.id)

    def _handle_conversion_item_input_audio_transcription_completed(
        self, event: ConversationItemInputAudioTranscriptionCompletedEvent
    ) -> None:
        self._clear_transcript_accumulator(event.item_id, event.content_index or 0)
        confidence = calculate_confidence_from_logprobs(event.logprobs)

        if remote_item := self._remote_chat_ctx.get(event.item_id):
            assert isinstance(remote_item.item, llm.ChatMessage)
            if event.transcript:
                text_index = next(
                    (
                        idx
                        for idx, content in enumerate(remote_item.item.content)
                        if isinstance(content, str)
                    ),
                    None,
                )
                if text_index is None:
                    remote_item.item.content.append(event.transcript)
                else:
                    remote_item.item.content[text_index] = event.transcript
            remote_item.item.transcript_confidence = confidence

        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=event.item_id,
                transcript=event.transcript,
                is_final=True,
                confidence=confidence,
            ),
        )

    def _create_session_update_event(self) -> dict[str, Any]:
        return self._build_session_update_event("session_update_")

    def _build_session_update_event(self, event_prefix: str) -> dict[str, Any]:
        audio_input: dict[str, Any] = {
            "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
            "turn_detection": self._boson_opts.turn_detection,
        }
        if is_given(self._boson_opts.input_audio_transcription):
            audio_input["transcription"] = self._boson_opts.input_audio_transcription
        if self._boson_opts.noise_reduction is not None:
            audio_input["noise_reduction"] = self._boson_opts.noise_reduction

        audio_output: dict[str, Any] = {
            "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
            "voice": self._boson_opts.voice,
            "speed": self._boson_opts.speed,
        }

        payload: dict[str, Any] = {
            "type": "realtime",
            "model": self._boson_opts.model,
            "instructions": self._instructions or self._boson_opts.instructions,
            "output_modalities": list(self._boson_opts.output_modalities),
            "audio": {
                "input": audio_input,
                "output": audio_output,
            },
            "tools": _tools_to_boson(self._tools.flatten()),
            "tool_choice": _tool_choice_to_boson(self._boson_opts.tool_choice),
            "temperature": self._boson_opts.temperature,
            "max_output_tokens": self._boson_opts.max_output_tokens,
        }
        return {
            "type": "session.update",
            "event_id": utils.shortuuid(event_prefix),
            "session": payload,
        }

    async def update_instructions(self, instructions: str) -> None:
        self._instructions = instructions
        self.send_event(self._build_session_update_event("instructions_update_"))

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        previous_item_id: str | None = None
        for item in chat_ctx.copy(exclude_handoff=True, exclude_config_update=True).items:
            payload = _livekit_item_to_boson_item(item)
            if payload is None:
                continue
            item_id = payload.get("id")
            if not isinstance(item_id, str):
                continue
            if item_id in self._boson_remote_item_ids:
                previous_item_id = item_id
                continue
            self._boson_remote_item_ids.add(item_id)
            self.send_event(
                {
                    "type": "conversation.item.create",
                    "event_id": utils.shortuuid("chat_ctx_create_"),
                    "previous_item_id": previous_item_id,
                    "item": payload,
                }
            )
            previous_item_id = item_id

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        self._tools = llm.ToolContext(tools)
        self.send_event(self._build_session_update_event("tools_update_"))

    def update_options(
        self,
        *,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        turn_detection: NotGivenOr[Any | None] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[Any | None] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[Any | None] = NOT_GIVEN,
        max_response_output_tokens: NotGivenOr[int | Literal["inf"] | None] = NOT_GIVEN,
        tracing: NotGivenOr[Any | None] = NOT_GIVEN,
        truncation: NotGivenOr[Any | None] = NOT_GIVEN,
        reasoning: NotGivenOr[Any | None] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_output_tokens: NotGivenOr[int | Literal["inf"] | None] = NOT_GIVEN,
    ) -> None:
        _ = (tracing, truncation, reasoning)
        next_max_output_tokens = (
            max_output_tokens if is_given(max_output_tokens) else max_response_output_tokens
        )

        if is_given(tool_choice):
            self._boson_opts.tool_choice = tool_choice
            self._opts.tool_choice = tool_choice
        if is_given(voice):
            self._boson_opts.voice = voice
            self._opts.voice = voice
        if is_given(temperature):
            self._boson_opts.temperature = temperature
        if is_given(next_max_output_tokens) and next_max_output_tokens is not None:
            self._boson_opts.max_output_tokens = next_max_output_tokens
            self._opts.max_response_output_tokens = next_max_output_tokens
        if is_given(speed):
            self._boson_opts.speed = speed
            self._opts.speed = speed
        if is_given(turn_detection):
            self._boson_opts.turn_detection = _copy_dict_or_none(turn_detection)
        if is_given(input_audio_transcription):
            self._boson_opts.input_audio_transcription = _normalize_input_audio_transcription(
                input_audio_transcription
            )
        if is_given(input_audio_noise_reduction):
            self._boson_opts.noise_reduction = _build_noise_reduction(input_audio_noise_reduction)
        self.send_event(self._build_session_update_event("options_update_"))

    def push_video(self, frame: rtc.VideoFrame) -> None:
        logger.debug("Boson RealtimeModel does not support video input yet; frame ignored.")

    def commit_audio(self) -> None:
        if self._pushed_duration_s <= 0.1:
            return
        self.send_event(
            {"type": "input_audio_buffer.commit", "event_id": utils.shortuuid("audio_commit_")}
        )
        self._pushed_duration_s = 0.0

    def clear_audio(self) -> None:
        self.send_event(
            {"type": "input_audio_buffer.clear", "event_id": utils.shortuuid("audio_clear_")}
        )
        self._pushed_duration_s = 0.0

    def interrupt(self) -> None:
        if not self.has_active_generation:
            return
        if self._suppress_next_response_cancel:
            self._suppress_next_response_cancel = False
            logger.debug("Skipping duplicate response.cancel after server-side VAD interruption.")
            return
        event: dict[str, Any] = {
            "type": "response.cancel",
            "event_id": utils.shortuuid("response_cancel_"),
        }
        if self._current_response_id:
            event["response_id"] = self._current_response_id
        self.send_event(event)

    async def aclose(self) -> None:
        self._boson_closing = True
        self._closing = True
        self._closed = True
        self._close_current_generation("session closed")
        self._msg_ch.close()
        await utils.aio.cancel_and_wait(self._main_atask)

    def _handle_input_audio_buffer_speech_started(
        self, _: InputAudioBufferSpeechStartedEvent
    ) -> None:
        self._suppress_next_response_cancel = _turn_detection_interrupts_response(
            self._boson_opts.turn_detection
        )
        try:
            self.emit("input_speech_started", llm.InputSpeechStartedEvent())
        finally:
            self._suppress_next_response_cancel = False

    def _handle_input_audio_buffer_speech_stopped(
        self, _: InputAudioBufferSpeechStoppedEvent
    ) -> None:
        self._pushed_duration_s = 0.0
        self.emit(
            "input_speech_stopped",
            llm.InputSpeechStoppedEvent(
                user_transcription_enabled=_input_audio_transcription_enabled(
                    self._boson_opts.input_audio_transcription
                )
            ),
        )

    def _handle_response_created(self, event: ResponseCreatedEvent) -> None:
        if self._current_generation is not None:
            self._close_current_generation("new response created before previous response.done")
        self._current_response_id = event.response.id
        super()._handle_response_created(event)

    def _handle_response_done(self, event: ResponseDoneEvent) -> None:
        super()._handle_response_done(event)
        self._current_response_id = None

    def _handle_boson_error_event(self, event: dict[str, Any]) -> None:
        error = event.get("error") or {}
        realtime_error = llm.RealtimeError(_format_error_message(error, event))
        self._emit_error(realtime_error, recoverable=True)
        self._fail_response_created_futures(
            realtime_error,
            event_id=error.get("event_id") or event.get("event_id"),
        )

    def _fail_response_created_futures(
        self, error: Exception, *, event_id: str | None = None
    ) -> None:
        if event_id and event_id in self._response_created_futures:
            futures = [self._response_created_futures.pop(event_id)]
        else:
            futures = list(self._response_created_futures.values())
            self._response_created_futures.clear()
        for fut in futures:
            if not fut.done():
                fut.set_exception(error)


def _normalize_ws_url(url: str, query_params: dict[str, str]) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme
    if scheme == "http":
        scheme = "ws"
    elif scheme == "https":
        scheme = "wss"
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update(query_params)
    return urlunparse(
        (scheme, parsed.netloc, parsed.path, parsed.params, urlencode(query), parsed.fragment)
    )


def _copy_dict_or_none(value: Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return dict(value)


def _normalize_input_audio_transcription(
    value: Any | None,
) -> NotGivenOr[dict[str, Any]]:
    if value is None:
        return NOT_GIVEN
    return dict(value)


def _build_input_audio_transcription(
    *,
    input_audio_transcription: NotGivenOr[dict[str, Any] | None],
    model: str,
    language: str | None,
    prompt: str | None,
) -> NotGivenOr[dict[str, Any]]:
    has_convenience_options = any(
        value is not None and value != "" for value in (model, language, prompt)
    )
    if not is_given(input_audio_transcription) and not has_convenience_options:
        return NOT_GIVEN
    if input_audio_transcription is None and not has_convenience_options:
        return NOT_GIVEN

    transcription = (
        dict(input_audio_transcription)
        if is_given(input_audio_transcription) and input_audio_transcription is not None
        else {}
    )
    if model:
        transcription["model"] = model
    if language is not None:
        transcription["language"] = language
    if prompt is not None:
        transcription["prompt"] = prompt
    return transcription


def _input_audio_transcription_enabled(transcription: NotGivenOr[dict[str, Any]]) -> bool:
    """Whether the server will emit user-transcription events for this config.

    Mirrors the server gate (``SessState.emit_input_transcription``): transcript
    events are returned only when the client sets a non-empty ``model``. Omitting
    the transcription block, sending ``null``, or sending a block without a model
    all run ASR internally (for the LLM/logging) but emit no client-facing
    ``conversation.item.input_audio_transcription.completed`` events.
    """
    return is_given(transcription) and bool(transcription.get("model"))


def _resolve_output_modalities(
    modalities: list[Literal["text", "audio"]] | None,
) -> list[Literal["text", "audio"]]:
    """Validate ``modalities`` to exactly ``["text"]`` or ``["audio"]``.

    The server rejects mixed (``["text", "audio"]``) and empty lists — output is
    single-modality. ``None`` defaults to ``["audio"]``.
    """
    if modalities is None:
        return list(_DEFAULT_OUTPUT_MODALITIES)
    if len(modalities) != 1 or modalities[0] not in _VALID_OUTPUT_MODALITIES:
        raise ValueError(
            "modalities must be exactly one of ['text'] or ['audio'] "
            f"(got {modalities!r}); mixed and empty lists are not supported."
        )
    return list(modalities)


def _build_noise_reduction(value: NotGivenOr[Any | None]) -> dict[str, Any] | None:
    """Normalize ``input_audio_noise_reduction`` to the OpenAI object form.

    Accepts a bare type string (``"near_field"`` / ``"far_field"``) or a dict
    (``{"type": ...}``). ``NOT_GIVEN`` and ``None`` both disable it (nothing is
    sent, which the server treats as disabled).
    """
    if not is_given(value) or value is None:
        return None
    if isinstance(value, str):
        return {"type": value}
    return dict(value)


def _turn_detection_interrupts_response(turn_detection: dict[str, Any] | None) -> bool:
    if not turn_detection:
        return False
    return turn_detection.get("interrupt_response") is not False


def _tools_to_boson(tools: list[llm.Tool]) -> list[dict[str, Any]]:
    boson_tools: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, llm.FunctionTool):
            boson_tools.append(llm.utils.build_legacy_openai_schema(tool, internally_tagged=True))
        elif isinstance(tool, llm.RawFunctionTool):
            raw_schema = dict(tool.info.raw_schema)
            raw_schema.pop("meta", None)
            raw_schema["type"] = "function"
            boson_tools.append(raw_schema)
    return boson_tools


def _tool_choice_to_boson(tool_choice: llm.ToolChoice | None) -> Any:
    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):
        return tool_choice
    function = tool_choice.get("function", {})
    name = function.get("name")
    if name:
        return {"type": "function", "name": name}
    return "auto"


def _livekit_item_to_boson_item(item: llm.ChatItem) -> dict[str, Any] | None:
    if item.type == "message":
        role = "system" if item.role == "developer" else item.role
        content_text = _text_from_content(item.content)
        if not content_text:
            return None
        content_type = "text" if role == "assistant" else "input_text"
        return {
            "id": item.id,
            "object": "realtime.item",
            "type": "message",
            "role": role,
            "content": [{"type": content_type, "text": content_text}],
        }
    if item.type == "function_call":
        return {
            "id": item.id,
            "object": "realtime.item",
            "type": "function_call",
            "call_id": item.call_id,
            "name": item.name,
            "arguments": item.arguments,
            "status": "completed",
        }
    if item.type == "function_call_output":
        return {
            "id": item.id,
            "object": "realtime.item",
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": item.output,
            "status": "completed",
        }
    return None


def _text_from_content(content: list[llm.ChatContent]) -> str:
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, llm.AudioContent) and part.transcript:
            parts.append(part.transcript)
    return "\n".join(parts)


def _format_error_message(error: dict[str, Any], event: dict[str, Any]) -> str:
    message = error.get("message") or "Boson realtime API error"
    details = {
        "type": error.get("type"),
        "code": error.get("code"),
        "event_id": error.get("event_id") or event.get("event_id"),
    }
    details = {key: value for key, value in details.items() if value is not None}
    if details:
        return f"{message} ({', '.join(f'{key}={value}' for key, value in details.items())})"
    return message


def _format_ws_close_message(ws: aiohttp.ClientWebSocketResponse, msg: aiohttp.WSMessage) -> str:
    close_code = getattr(ws, "close_code", None)
    if close_code is None and msg.data is not None:
        close_code = msg.data
    reason = msg.extra
    details = []
    if close_code is not None:
        details.append(f"close_code={close_code}")
    if reason:
        details.append(f"reason={reason}")
    if details:
        return f"Boson realtime WebSocket closed unexpectedly ({', '.join(details)})."
    return "Boson realtime WebSocket closed unexpectedly."


def _as_realtime_error(error: Exception) -> llm.RealtimeError:
    if isinstance(error, llm.RealtimeError):
        return error
    return llm.RealtimeError(str(error))
