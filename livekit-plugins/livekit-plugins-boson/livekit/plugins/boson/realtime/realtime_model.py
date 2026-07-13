from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, replace
from typing import Any, Literal, cast
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import aiohttp
from openai.types.realtime import (
    ConversationItemAdded,
    ConversationItemCreateEvent,
    ConversationItemDeleteEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    RealtimeErrorEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
)
from pydantic import BaseModel, ConfigDict

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
from livekit.plugins.openai.realtime.utils import (
    calculate_confidence_from_logprobs,
    openai_item_to_livekit_item,
)

from ..log import logger

SAMPLE_RATE = openai_rt.SAMPLE_RATE

_DEFAULT_TURN_DETECTION = {
    "type": "server_vad",
    "create_response": True,
    "interrupt_response": True,
    "prefix_padding_ms": 300,
    "silence_duration_ms": 500,
    "threshold": 0.55,
}

# Server close codes that a reconnect cannot fix:
# 3000 = invalid API key / ephemeral key.
_NON_RETRYABLE_CLOSE_CODES = frozenset({3000})


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
        # The server treats conversation.item.create as a pure insert and never
        # auto-generates a response for it, so the framework must send
        # response.create after posting tool outputs, like OpenAI.
        self._capabilities.auto_tool_reply_generation = False
        self._capabilities.audio_output = "audio" in output_modalities
        # mutable_chat_context stays True (base default): the server preserves
        # client-supplied item ids, so items are addressable for the base
        # diff/create/delete chat-context synchronization.

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
        self._closed = False
        self._suppress_next_response_cancel = False
        self._video_unsupported_warned = False
        self._current_response_id: str | None = None
        # Responses discarded while another generation was streaming; their
        # terminal events must not close the active generation (the base
        # generation slot and handlers are not response-id aware).
        self._boson_discarded_response_ids: set[str] = set()
        # Server-assigned session id (from session.created), kept for logging.
        # The server does not persist sessions: every connection is a fresh
        # session, so reconnection replays the local chat context instead.
        self._session_id: str | None = None
        # Set when the server announces it is ending the session on purpose
        # (session.idle_timeout / session.max_duration_reached); the following
        # close must not trigger a reconnect, which would restart the session
        # the server just ended.
        self._server_terminal_reason: str | None = None
        super().__init__(realtime_model)
        # The base recv loop dispatches OpenAI event types to _handle_* methods
        # and re-emits every raw event on this hook; Boson-specific events are
        # handled off it instead of forking the whole dispatch.
        self.on("openai_server_event_received", self._handle_boson_server_event)
        self.on("session_reconnected", self._on_session_reconnected)

    def send_event(self, event: Any) -> None:
        if self._closed or self._msg_ch.closed:
            return
        if isinstance(event, BaseModel):
            event = event.model_dump(by_alias=True, exclude_unset=True, exclude_defaults=False)
        with contextlib.suppress(utils.aio.ChanClosed):
            self._msg_ch.send_nowait(event)

    @property
    def session_id(self) -> str | None:
        return self._session_id

    async def _main_task(self) -> None:
        # The base task owns connect/retry/reconnect (including the chat-context
        # replay via _create_update_chat_ctx_events). On terminal failure it
        # leaves pending response futures to their own timeouts; fail them and
        # stop accepting events right away instead.
        try:
            await super()._main_task()
        except Exception as exc:
            error = exc if isinstance(exc, llm.RealtimeError) else llm.RealtimeError(str(exc))
            self._fail_response_created_futures(error)
            self._close_current_generation("Boson realtime session failed")
            self._closed = True
            self._msg_ch.close()
            raise

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
        try:
            await super()._run_ws(ws_conn)
        except APIConnectionError as exc:
            # The base recv loop raises every unexpected close as retryable;
            # reclassify the ones a reconnect cannot fix.
            if self._server_terminal_reason is not None:
                raise APIConnectionError(
                    f"Boson realtime session ended by server: {self._server_terminal_reason}",
                    retryable=False,
                ) from exc
            close_code = ws_conn.close_code
            if close_code is not None:
                raise APIConnectionError(
                    f"Boson realtime WebSocket closed unexpectedly (close_code={close_code}).",
                    retryable=close_code not in _NON_RETRYABLE_CLOSE_CODES,
                ) from exc
            raise

    def _handle_boson_server_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type in ("session.created", "session.updated"):
            # Track the server-assigned id for logging/diagnostics.
            session_obj = event.get("session") or {}
            session_id = session_obj.get("id")
            if isinstance(session_id, str) and session_id:
                self._session_id = session_id
        elif event_type == "session.idle_timeout":
            seconds_idle = event.get("seconds_idle")
            self._server_terminal_reason = f"idle timeout ({seconds_idle}s)"
            logger.info(
                "Boson realtime session idle timeout announced by server",
                extra={"session_id": self._session_id, "seconds_idle": seconds_idle},
            )
        elif event_type == "session.max_duration_reached":
            max_duration_sec = event.get("max_duration_sec")
            self._server_terminal_reason = f"max session duration reached ({max_duration_sec}s)"
            logger.info(
                "Boson realtime session max duration announced by server",
                extra={"session_id": self._session_id, "max_duration_sec": max_duration_sec},
            )

    def _on_session_reconnected(self, _event: llm.RealtimeSessionReconnectedEvent) -> None:
        # Per-connection state the base _reconnect doesn't know about.
        self._pushed_duration_s = 0.0
        self._current_response_id = None
        self._suppress_next_response_cancel = False
        logger.info("reconnected to Boson realtime session", extra={"session_id": self._session_id})

    def _handle_conversion_item_added(self, event: ConversationItemAdded) -> None:
        item_id = event.item.id
        if item_id is not None and (remote_item := self._remote_chat_ctx.get(item_id)) is not None:
            # The server merges consecutive same-role speech turns into a single
            # item and re-emits conversation.item.added with the same id and
            # cumulative content. Update the mirrored text in place instead of
            # letting the base insert fail with a warning.
            # Audio-input configs re-add with an empty input_audio part and no
            # transcript — keep whatever transcription has already arrived.
            lk_item = openai_item_to_livekit_item(event.item)
            if (
                isinstance(lk_item, llm.ChatMessage)
                and isinstance(remote_item.item, llm.ChatMessage)
                and lk_item.text_content
            ):
                _set_message_text(remote_item.item, lk_item.text_content)
            if fut := self._item_create_future.pop(item_id, None):
                if not fut.cancelled():
                    fut.set_result(None)
            return

        super()._handle_conversion_item_added(event)

    def _handle_conversion_item_input_audio_transcription_completed(
        self, event: ConversationItemInputAudioTranscriptionCompletedEvent
    ) -> None:
        self._clear_transcript_accumulator(event.item_id, event.content_index or 0)
        confidence = calculate_confidence_from_logprobs(event.logprobs)

        if remote_item := self._remote_chat_ctx.get(event.item_id):
            assert isinstance(remote_item.item, llm.ChatMessage)
            if event.transcript:
                _set_message_text(remote_item.item, event.transcript)
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

    def _create_tools_update_event(self, tools: list[llm.Tool]) -> dict[str, Any]:
        # The server treats session.update as a full replace (not OpenAI's
        # partial merge), so a tools-only update must carry the whole config.
        return self._build_session_update_event("tools_update_", tools=tools)

    def _create_update_chat_ctx_events(
        self, chat_ctx: llm.ChatContext
    ) -> list[ConversationItemCreateEvent | ConversationItemDeleteEvent]:
        # The base diff (used both by update_chat_ctx and by _reconnect's
        # chat-context replay) produces GA-shaped creates; the server stores
        # items in the Boson shape (a single text content part, type "text" for
        # assistant) and has no "root" previous_item_id sentinel. Rebuild each
        # create with the Boson payload. The server preserves client-supplied
        # item ids, so the rest of the base machinery — echo correlation via
        # _item_create_future/_item_delete_future and the _remote_chat_ctx
        # diff — works unchanged.
        #
        # Diff against a text-only mirror of the context: the server stores a
        # single text part per message (audio is represented by its transcript,
        # images are unsupported), and the base GA converter must never see
        # audio frames (rtc.combine_audio_frames raises on empty ones). A
        # message with no text keeps an empty mirror so the base diff retains
        # it when it already exists remotely (e.g. an audio item whose
        # transcription is still pending) instead of deleting it, and filters
        # it out otherwise.
        sanitized: list[llm.ChatItem] = []
        for item in chat_ctx.items:
            if item.type == "message":
                text = _text_from_content(item.content)
                sanitized.append(
                    llm.ChatMessage(id=item.id, role=item.role, content=[text] if text else [])
                )
            else:
                sanitized.append(item)
        boson_ctx = llm.ChatContext(sanitized)

        events: list[ConversationItemCreateEvent | ConversationItemDeleteEvent] = []
        # Safety net: a create the conversion still cannot express is not sent;
        # remap a previous_item_id pointing at it to its own predecessor.
        dropped: dict[str, str | None] = {}
        for ev in super()._create_update_chat_ctx_events(boson_ctx):
            if not isinstance(ev, ConversationItemCreateEvent):
                events.append(ev)
                continue
            assert ev.item.id is not None
            previous_item_id = ev.previous_item_id
            if previous_item_id == "root":
                # The server cannot express insert-at-head; None appends at the
                # tail, which is correct for the replay/append cases that
                # produce it (the remote context is empty or being extended).
                previous_item_id = None
            elif previous_item_id is not None and previous_item_id in dropped:
                previous_item_id = dropped[previous_item_id]
            chat_item = boson_ctx.get_by_id(ev.item.id)
            payload = _livekit_item_to_boson_item(chat_item) if chat_item is not None else None
            if payload is None:
                # The server skips items without text content instead of
                # storing them, so their conversation.item.added echo would
                # never resolve the create future.
                dropped[ev.item.id] = previous_item_id
                continue
            events.append(
                _BosonConversationItemCreateEvent(
                    type="conversation.item.create",
                    event_id=ev.event_id or utils.shortuuid("chat_ctx_create_"),
                    previous_item_id=previous_item_id,
                    item=_BosonConversationItem(**payload),
                )
            )
        return events

    def _build_session_update_event(
        self, event_prefix: str, tools: list[llm.Tool] | None = None
    ) -> dict[str, Any]:
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
            "tools": _tools_to_boson(tools if tools is not None else self._tools.flatten()),
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
        if not self._video_unsupported_warned:
            self._video_unsupported_warned = True
            logger.warning("Boson RealtimeModel does not support video input; frames are ignored.")

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
        self._closing = True
        self._closed = True
        self._close_current_generation("session closed")
        self._msg_ch.close()
        # Cancel instead of the base's await: the main task may be sleeping in a
        # retry backoff or mid-connect, which close should not wait out.
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
        client_event_id: str | None = None
        if isinstance(event.response.metadata, dict):
            client_event_id = event.response.metadata.get("client_event_id")
        if client_event_id and client_event_id in self._discarded_event_ids:
            # A response that timed out or was interrupted before the server
            # created it. The base handler cancels it and parks a discard
            # marker in the generation slot so its trailing events are skipped.
            # When a legitimate generation is already streaming, keep it in the
            # slot instead, and remember the stale response id so its
            # response.done doesn't close the active generation.
            active = self._current_generation
            super()._handle_response_created(event)
            if isinstance(active, openai_rt._ResponseGeneration):
                if event.response.id is not None:
                    self._boson_discarded_response_ids.add(event.response.id)
                self._current_generation = active
            return

        if self._current_generation is not None:
            self._close_current_generation("new response created before previous response.done")
        self._current_response_id = event.response.id
        super()._handle_response_created(event)

    def _handle_response_done(self, event: ResponseDoneEvent) -> None:
        response_id = event.response.id
        if response_id is not None and response_id in self._boson_discarded_response_ids:
            # Terminal event of a response discarded while another generation
            # was streaming; it must not close the active generation or clear
            # the response id that interrupt() targets.
            self._boson_discarded_response_ids.discard(response_id)
            return
        super()._handle_response_done(event)
        self._current_response_id = None

    def _handle_error(self, event: RealtimeErrorEvent) -> None:
        # Unlike the base handler, fail the pending generate_reply future the
        # error refers to (the server reports the offending client event_id and
        # may not follow up with a response.done).
        error: dict[str, Any] = (
            event.error if isinstance(event.error, dict) else event.error.model_dump()
        )
        event_id = error.get("event_id") or event.event_id
        realtime_error = llm.RealtimeError(_format_error_message(error, event_id))
        self._emit_error(realtime_error, recoverable=True)
        self._fail_response_created_futures(realtime_error, event_id=event_id)

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


def _set_message_text(message: llm.ChatMessage, text: str) -> None:
    """Replace the first text part of ``message`` (or append one) with ``text``."""
    text_index = next(
        (idx for idx, content in enumerate(message.content) if isinstance(content, str)),
        None,
    )
    if text_index is None:
        message.content.append(text)
    else:
        message.content[text_index] = text


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
    has_convenience_options = bool(model) or language is not None or prompt is not None
    if not has_convenience_options and (
        not is_given(input_audio_transcription) or input_audio_transcription is None
    ):
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

    The server returns transcript events only when the client sets a non-empty
    ``model``. Omitting the transcription block, sending ``null``, or sending a
    block without a model all run ASR server-side (for the LLM) but emit no
    client-facing ``conversation.item.input_audio_transcription.completed``
    events.
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
        return ["audio"]
    if len(modalities) != 1 or modalities[0] not in ("text", "audio"):
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


class _BosonConversationItem(BaseModel):
    """A conversation item in the Boson wire shape.

    Typed only as far as the base ``update_chat_ctx`` machinery needs
    (``item.id`` for echo correlation); the payload rides in extra fields
    because it deviates from the GA models (assistant content uses type
    ``"text"``, which their literals reject).
    """

    model_config = ConfigDict(extra="allow")
    id: str


class _BosonConversationItemCreateEvent(ConversationItemCreateEvent):
    item: _BosonConversationItem  # type: ignore[assignment]


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


def _format_error_message(error: dict[str, Any], event_id: str | None) -> str:
    message = error.get("message") or "Boson realtime API error"
    details = {
        "type": error.get("type"),
        "code": error.get("code"),
        "event_id": event_id,
    }
    details = {key: value for key, value in details.items() if value is not None}
    if details:
        return f"{message} ({', '.join(f'{key}={value}' for key, value in details.items())})"
    return message
