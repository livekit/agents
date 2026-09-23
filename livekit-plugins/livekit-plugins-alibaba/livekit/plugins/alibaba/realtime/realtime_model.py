# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

import aiohttp
from openai.types.beta.realtime.session import TurnDetection
from openai.types.realtime import (
    AudioTranscription,
    ConversationItemCreateEvent,
    ConversationItemDeleteEvent,
    RealtimeErrorEvent,
    RealtimeReasoning,
    RealtimeResponseCreateParams,
    RealtimeSessionCreateRequest,
    ResponseAudioDeltaEvent,
    ResponseCancelEvent,
    ResponseCreatedEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
)
from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad
from openai.types.realtime.session_update_event import SessionUpdateEvent

from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.plugins import openai
from livekit.plugins.openai.realtime.realtime_model import _DiscardedGeneration

from ..log import logger
from ..models import (
    DEFAULT_MODEL,
    DEFAULT_REGION,
    DEFAULT_VOICE,
    INPUT_SAMPLE_RATE,
    AlibabaRealtimeModels,
    AlibabaRegion,
    AlibabaVoices,
    get_realtime_url,
)

NUM_CHANNELS = 1


def _validate_options(options: dict[str, Any]) -> None:
    for name in (
        "reasoning",
        "speed",
        "tracing",
        "truncation",
        "input_audio_noise_reduction",
        "max_response_output_tokens",
    ):
        if name in options and options[name] is not None and is_given(options[name]):
            raise ValueError(f"{name} is not supported by the Alibaba realtime plugin")


# DashScope Realtime API uses beta event names that map to OpenAI GA event names
_DASHSCOPE_EVENT_MAPPING: dict[str, str] = {
    "response.text.delta": "response.output_text.delta",
    "response.text.done": "response.output_text.done",
    "response.audio_transcript.delta": "response.output_audio_transcript.delta",
    "response.audio_transcript.done": "response.output_audio_transcript.done",
    "response.audio.delta": "response.output_audio.delta",
    "response.audio.done": "response.output_audio.done",
    "conversation.item.created": "conversation.item.added",
}

# Leave enough silence for natural pauses without excessive response latency.
DEFAULT_TURN_DETECTION = ServerVad(
    type="server_vad",
    threshold=0.6,
    prefix_padding_ms=300,
    silence_duration_ms=650,
    create_response=True,
    interrupt_response=True,
)


class _DashScopeWSAdapter:
    """Adapts incoming DashScope WebSocket frames to OpenAI GA event names."""

    def __init__(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        send: Callable[[str], Awaitable[None]],
    ) -> None:
        self._ws = ws
        self._send = send

    async def send_str(self, data: str) -> None:
        await self._send(data)

    def _transform(self, msg: aiohttp.WSMessage) -> aiohttp.WSMessage:
        if msg.type == aiohttp.WSMsgType.TEXT:
            try:
                data = json.loads(msg.data)
                ev_type = data.get("type", "")
                if ev_type in _DASHSCOPE_EVENT_MAPPING:
                    data["type"] = _DASHSCOPE_EVENT_MAPPING[ev_type]
                    return aiohttp.WSMessage(
                        type=msg.type,
                        data=json.dumps(data),
                        extra=msg.extra,
                    )
            except Exception:
                pass
        return msg

    async def receive(self, timeout: float | None = None) -> aiohttp.WSMessage:
        msg = await self._ws.receive(timeout=timeout)
        return self._transform(msg)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ws, name)


class RealtimeModel(openai.realtime.RealtimeModel):
    """Alibaba DashScope speech-to-speech model with 16 kHz input and 24 kHz output."""

    def __init__(
        self,
        *,
        model: NotGivenOr[AlibabaRealtimeModels | str] = NOT_GIVEN,
        voice: NotGivenOr[AlibabaVoices | str | None] = DEFAULT_VOICE,
        api_key: str | None = None,
        region: NotGivenOr[AlibabaRegion] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        workspace_id: str | None = None,
        turn_detection: NotGivenOr[ServerVad | TurnDetection | None] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[AudioTranscription | None] = NOT_GIVEN,
        reasoning: NotGivenOr[RealtimeReasoning | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """Create an Alibaba realtime model.

        Args:
            model: Model ID. Defaults to ``qwen-audio-3.1-realtime-plus``.
            voice: Output voice. Defaults to ``longanqian``.
            api_key: DashScope key, or ``DASHSCOPE_API_KEY`` when omitted.
            region: ``cn`` or ``intl``. Defaults to ``DASHSCOPE_REGION``, then ``cn``.
            base_url: Custom endpoint, taking precedence over workspace and region.
            workspace_id: Dedicated MaaS workspace, or ``DASHSCOPE_WORKSPACE_ID``.
            turn_detection: Server VAD options. Omit for the default server VAD;
                pass ``None`` for client turn-taking instead of server VAD.
                Only threshold, prefix padding and silence duration are configurable.
                VAD configuration is fixed for the lifetime of the model/session.
            input_audio_transcription: Input transcription options. Defaults to
                ``gummy-realtime-v1``; pass ``None`` to disable transcription.
            reasoning: Unsupported OpenAI option; a non-null value raises ``ValueError``.
            speed: Unsupported OpenAI option; a non-null value raises ``ValueError``.
            http_session: Optional caller-owned HTTP session.
            max_session_duration: Recycle connections after this many seconds.
                Defaults to no scheduled recycling.
            conn_options: Connection timeout and retry policy.

        Raises:
            ValueError: Credentials are missing, the region is invalid, or an
                unsupported option is specified.
        """
        _validate_options({"reasoning": reasoning, "speed": speed})
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key "
                "to the client or by setting the DASHSCOPE_API_KEY environment variable"
            )

        resolved_model = model if is_given(model) else DEFAULT_MODEL
        resolved_voice = voice if is_given(voice) else DEFAULT_VOICE
        resolved_workspace = workspace_id or os.environ.get("DASHSCOPE_WORKSPACE_ID")
        region_value = (
            region if is_given(region) else os.environ.get("DASHSCOPE_REGION", DEFAULT_REGION)
        )
        if region_value not in ("cn", "intl"):
            raise ValueError("region must be 'cn' or 'intl'")
        resolved_region: AlibabaRegion = "intl" if region_value == "intl" else "cn"

        custom_base_url = base_url if is_given(base_url) else None
        resolved_url = get_realtime_url(
            region=resolved_region,
            base_url=custom_base_url,
            workspace_id=resolved_workspace,
        )

        resolved_turn_detection = (
            turn_detection if is_given(turn_detection) else DEFAULT_TURN_DETECTION
        )
        if resolved_turn_detection is not None:
            vad_options = resolved_turn_detection.model_dump(exclude_none=True)
            if (
                vad_options.get("type") != "server_vad"
                or vad_options.get("create_response") is False
                or vad_options.get("interrupt_response") is False
                or vad_options.keys()
                - {
                    "type",
                    "threshold",
                    "prefix_padding_ms",
                    "silence_duration_ms",
                    "create_response",
                    "interrupt_response",
                }
            ):
                raise ValueError(
                    "turn_detection only supports server_vad threshold, prefix_padding_ms and "
                    "silence_duration_ms; use None for client turn-taking"
                )
        resolved_max_session_duration = (
            max_session_duration if is_given(max_session_duration) else None
        )

        init_kwargs: dict[str, Any] = {
            "base_url": resolved_url,
            "model": resolved_model,
            "voice": resolved_voice,
            "api_key": api_key,
            "modalities": ["audio", "text"],
            "turn_detection": resolved_turn_detection,
            "http_session": http_session,
            "max_session_duration": resolved_max_session_duration,
            "conn_options": conn_options,
            "input_audio_transcription": AudioTranscription(model="gummy-realtime-v1"),
        }
        if is_given(input_audio_transcription):
            init_kwargs["input_audio_transcription"] = input_audio_transcription
        super().__init__(**init_kwargs)
        self._capabilities.per_response_tool_choice = False
        self._capabilities.can_disable_turn_detection = True
        self._provider_label = "Alibaba Realtime API"

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        """Create a session, optionally disabling server VAD for client turn-taking."""
        sess = RealtimeSession(self, turn_detection_disabled=turn_detection_disabled)
        self._sessions.add(sess)
        return sess

    def update_options(self, **kwargs: Any) -> None:
        """Update supported model defaults and active sessions.

        Non-null OpenAI-only options are rejected. VAD options are fixed at
        construction because DashScope cannot change them after audio begins.
        """
        _validate_options(kwargs)
        if is_given(kwargs.get("turn_detection", NOT_GIVEN)):
            raise ValueError("turn_detection is fixed at construction; create a new Alibaba model")
        # Validate every session before the parent mutates model defaults.
        for session in self._sessions:
            if isinstance(session, RealtimeSession):
                session._validate_update(kwargs)
        super().update_options(**kwargs)


class RealtimeSession(openai.realtime.RealtimeSession):
    """DashScope protocol adapter sharing OpenAI's chat, tools and output streams.

    Uses one pending generation slot, like the Google realtime adapter, alongside
    the active server response. Conversation item IDs are never rewritten.
    """

    def __init__(
        self,
        realtime_model: RealtimeModel,
        *,
        turn_detection_disabled: bool = False,
    ) -> None:
        """Create a session for a model; prefer :meth:`RealtimeModel.session`."""
        # Qwen Audio 3.1 does not echo response metadata in our live verification;
        # its documented response.created schema provides no request correlation.
        # It also allows only ONE generating response: another response.create
        # is rejected, not queued. See fun-audiochat-client-events on help.aliyun.com.
        # A single slot associates the next generation with the waiting caller
        # (Google adapter semantics), NOT a provider-guaranteed causal/FIFO match.
        self._pending_response_id: str | None = None
        self._pending_response_sent = False
        self._response_timeout: asyncio.TimerHandle | None = None
        self._cancel_timeout: asyncio.TimerHandle | None = None
        self._active_response_id: str | None = None
        self._response_idle = asyncio.Event()
        self._response_idle.set()
        self._cancelling = False
        self._audio_started = False
        self._reset_connection = asyncio.Event()
        self._correlation_lost = False
        super().__init__(realtime_model, turn_detection_disabled=turn_detection_disabled)
        # DashScope requires 16kHz mono audio input
        self._bstream = utils.audio.AudioByteStream(
            INPUT_SAMPLE_RATE, NUM_CHANNELS, samples_per_channel=INPUT_SAMPLE_RATE // 10
        )

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler:
            if frame.sample_rate != self._input_resampler._input_rate:
                self._input_resampler = None

        if self._input_resampler is None and (
            frame.sample_rate != INPUT_SAMPLE_RATE or frame.num_channels != NUM_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=INPUT_SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )

        if self._input_resampler:
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    def update_options(self, **kwargs: Any) -> None:
        """Update voice, tool choice or transcription; VAD is fixed at construction."""
        _validate_options(kwargs)
        self._validate_update(kwargs)
        if is_given(kwargs.get("turn_detection", NOT_GIVEN)):
            raise ValueError(
                "turn_detection is fixed at construction; create a new Alibaba session"
            )
        super().update_options(**kwargs)

    def _validate_update(self, options: dict[str, Any]) -> None:
        voice = options.get("voice", NOT_GIVEN)
        if self._audio_started and is_given(voice) and voice != self._opts.voice:
            raise ValueError("voice cannot change after audio has started; create a new session")

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        """Push audio, freezing voice configuration as required by DashScope."""
        self._audio_started = True
        super().push_audio(frame)

    def _handle_response_audio_delta(self, event: ResponseAudioDeltaEvent) -> None:
        self._audio_started = True
        super()._handle_response_audio_delta(event)

    def _create_update_chat_ctx_events(
        self, chat_ctx: llm.ChatContext
    ) -> list[ConversationItemCreateEvent | ConversationItemDeleteEvent]:
        events = super()._create_update_chat_ctx_events(chat_ctx)
        for event in events:
            if isinstance(event, ConversationItemCreateEvent) and event.previous_item_id == "root":
                # Qwen rejects OpenAI's literal "root" sentinel. An empty
                # predecessor denotes the first item; actual item IDs are intact.
                event.previous_item_id = ""
        return events

    async def _run_ws(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        self._reset_connection.clear()

        async def send(data: str) -> None:
            event = json.loads(data)
            if event["type"] == "response.create":
                event_id = event["event_id"]
                if self._cancelling:
                    await self._response_idle.wait()
                future = self._response_created_futures.get(event_id)
                # The inherited outgoing queue survives reconnects. Never replay
                # a discarded request, or one cancelled before transmission.
                if future is None or future.done() or self._correlation_lost:
                    self._forget_response(event_id)
                    return
                if self._active_response_id is not None:
                    future.set_exception(
                        llm.RealtimeError("a response became active before request transmission")
                    )
                    self._forget_response(event_id)
                    return
                self._pending_response_sent = True
            await ws_conn.send_str(data)

        run = asyncio.create_task(super()._run_ws(_DashScopeWSAdapter(ws_conn, send)))  # type: ignore[arg-type]
        reset = asyncio.create_task(self._reset_connection.wait())
        try:
            done, _ = await asyncio.wait((run, reset), return_when=asyncio.FIRST_COMPLETED)
            if run in done:
                await run
            # Returning normally asks the inherited connection loop to reconnect.
        finally:
            self._correlation_lost = True
            await utils.aio.cancel_and_wait(run, reset)
            self._forget_response(self._pending_response_id)
            if self._cancel_timeout:
                self._cancel_timeout.cancel()
                self._cancel_timeout = None
            self._active_response_id = None
            self._cancelling = False
            self._response_idle.set()

    def emit(self, event: Any, *args: Any) -> None:
        # Parent dispatches reconnect before _run_ws. EventEmitter listeners are
        # unordered, so readiness must be set before dispatch, not by a listener.
        if event == "session_reconnected":
            self._correlation_lost = False
        super().emit(event, *args)

    def _forget_response(self, event_id: str | None) -> None:
        if event_id == self._pending_response_id:
            if self._response_timeout:
                self._response_timeout.cancel()
            self._response_timeout = None
            self._pending_response_id = None
            self._pending_response_sent = False

    def _arm_reset_timeout(self, *, cancellation: bool = False) -> None:
        handle = self._cancel_timeout if cancellation else self._response_timeout
        if handle:
            handle.cancel()

        def reset() -> None:
            if self._pending_response_id:
                future = self._response_created_futures.pop(self._pending_response_id, None)
                if future and not future.done():
                    future.set_exception(llm.RealtimeError("generate_reply timed out."))
            self._correlation_lost = True
            self._reset_connection.set()

        handle = asyncio.get_running_loop().call_later(10.0, reset)
        if cancellation:
            self._cancel_timeout = handle
        else:
            self._response_timeout = handle

    def interrupt(self) -> None:
        """Cancel generation, retaining the active slot until response.done.

        A sent cancel is not an acknowledgement. A pending create also keeps its
        slot so a late response.created is discarded rather than spoken.
        """
        if self._pending_response_id:
            future = self._response_created_futures.get(self._pending_response_id)
            if future and not future.done():
                future.cancel()
        else:
            super().interrupt()
        if self._active_response_id:
            self._cancelling = True
            # Stop local consumers immediately, while keeping the remote slot
            # occupied until its terminal event. Ignore trailing audio deltas.
            self._close_current_generation()
            self._current_generation = _DiscardedGeneration()
        if self._active_response_id or self._pending_response_sent:
            self._arm_reset_timeout(cancellation=True)

    def _wrap_session_update(
        self, event_id: str, session: RealtimeSessionCreateRequest
    ) -> SessionUpdateEvent | dict[str, Any]:
        """Convert session update request to DashScope format."""
        session_dict: dict[str, Any] = {}

        # Initialization/reconnection includes model AND audio. Tool updates
        # also include model; voice/transcription updates also include audio.
        # Neither alone is a full update (VAD/voice become immutable after audio).
        is_full_update = "model" in session.model_fields_set and session.audio is not None

        if is_full_update:
            session_dict["modalities"] = ["text", "audio"]
            session_dict["input_audio_format"] = "pcm"
            session_dict["output_audio_format"] = "pcm"

        # The parent produces GA nested audio config, including explicit null VAD
        # for client turn-taking. Preserve omission versus null in partial updates.
        audio_input = session.audio.input if session.audio is not None else None
        audio_output = session.audio.output if session.audio is not None else None
        voice = audio_output.voice if audio_output is not None else None
        if voice is not None:
            session_dict["voice"] = voice
        elif is_full_update:
            session_dict["voice"] = self._opts.voice or DEFAULT_VOICE

        if audio_input is not None:
            if "turn_detection" in audio_input.model_fields_set:
                td = audio_input.turn_detection
                session_dict["turn_detection"] = (
                    td.model_dump(
                        include={"type", "threshold", "prefix_padding_ms", "silence_duration_ms"}
                    )
                    if td is not None
                    else None
                )
            if "transcription" in audio_input.model_fields_set:
                transcription = audio_input.transcription
                session_dict["input_audio_transcription"] = (
                    transcription.model_dump(exclude_none=True)
                    if transcription is not None
                    else None
                )

        if "instructions" in session.model_fields_set and session.instructions is not None:
            session_dict["instructions"] = session.instructions

        if "tools" in session.model_fields_set and session.tools is not None:
            session_dict["tools"] = [t.model_dump(exclude_none=True) for t in session.tools]

        if "tool_choice" in session.model_fields_set:
            session_dict["tool_choice"] = session.model_dump(exclude_unset=True)["tool_choice"]

        return {
            "event_id": event_id,
            "type": "session.update",
            "session": session_dict,
        }

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        """Wait for the next generation using a single pending slot.

        Raises:
            llm.RealtimeError: A request or response is already active, or the
                connection is being reset. Interrupt and await completion first.
        """
        if self._correlation_lost:
            raise llm.RealtimeError("response correlation lost; wait for session reconnection")
        if self._active_response_id is not None and not self._cancelling:
            raise llm.RealtimeError("an Alibaba response is still active; wait for response.done")
        if self._pending_response_id or self._response_created_futures:
            raise llm.RealtimeError("an Alibaba response request is already pending")
        # Keep the parent's generation/stream handlers, but own cancellation:
        # its generic future callback sends cancel even for an unsent create.
        if is_given(tool_choice):
            self.update_options(tool_choice=tool_choice)
        event_id = utils.shortuuid("response_create_")
        fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.get_running_loop().create_future()
        self._response_created_futures[event_id] = fut
        self._pending_response_id = event_id
        if is_given(instructions) and self._instructions:
            instructions = f"{self._instructions}\n{instructions}"
        params = RealtimeResponseCreateParams(
            instructions=instructions or None,
            metadata={"client_event_id": event_id},
        )
        if is_given(tools):
            params.tools = list(self._convert_tools_to_oai(tools))
        self.send_event(
            ResponseCreateEvent(type="response.create", event_id=event_id, response=params)
        )
        self._arm_reset_timeout()

        def on_done(future: asyncio.Future[llm.GenerationCreatedEvent]) -> None:
            self._response_created_futures.pop(event_id, None)
            if self._pending_response_id != event_id:
                return
            if future.cancelled():
                if self._pending_response_sent:
                    self._discarded_event_ids.add(event_id)
                    self.send_event(ResponseCancelEvent(type="response.cancel"))
                else:
                    self._forget_response(event_id)

        fut.add_done_callback(on_done)

        return fut

    def _handle_response_created(self, event: ResponseCreatedEvent) -> None:
        if self._correlation_lost:
            return
        self._active_response_id = event.response.id
        self._response_idle.clear()
        if (
            self._pending_response_sent
            and self._pending_response_id
            and not (
                isinstance(event.response.metadata, dict)
                and event.response.metadata.get("client_event_id")
            )
        ):
            if not isinstance(event.response.metadata, dict):
                event.response.metadata = {}
            event.response.metadata["client_event_id"] = self._pending_response_id

        if isinstance(event.response.metadata, dict):
            if event_id := event.response.metadata.get("client_event_id"):
                pending = self._response_created_futures.get(event_id)
                cancelled = event_id in self._discarded_event_ids or (
                    pending is not None and pending.cancelled()
                )
                if cancelled:
                    self._discarded_event_ids.add(event_id)
                self._forget_response(event_id)
                if cancelled:
                    self._arm_reset_timeout(cancellation=True)

        super()._handle_response_created(event)

    def _handle_response_done(self, event: ResponseDoneEvent) -> None:
        if event.response.id != self._active_response_id:
            return  # A delayed terminal event must not close a newer generation.
        self._active_response_id = None
        self._cancelling = False
        self._response_idle.set()
        if self._cancel_timeout:
            self._cancel_timeout.cancel()
            self._cancel_timeout = None
        super()._handle_response_done(event)

    def _handle_error(self, event: RealtimeErrorEvent) -> None:
        # Native errors may only identify an operation. A delayed rejection can
        # belong to a request already satisfied by an automatic VAD generation;
        # never fabricate the newer waiter's ID. Reset ambiguous connection state.
        if not event.error.event_id and event.error.param == "response.create":
            if self._pending_response_id:
                pending = self._response_created_futures.pop(self._pending_response_id, None)
                if pending and not pending.done():
                    pending.set_exception(
                        llm.RealtimeError("uncorrelated response.create error; reconnecting")
                    )
            self._correlation_lost = True
            self._reset_connection.set()
        if event.error.event_id:
            if (
                event.error.event_id == self._pending_response_id
                and self._active_response_id is None
                and self._cancel_timeout
            ):
                # A rejected create cannot produce the response whose cancellation
                # we were awaiting. Do not let its watchdog reset a later request.
                self._cancel_timeout.cancel()
                self._cancel_timeout = None
            self._forget_response(event.error.event_id)
        if event.error.event_id and "already exists" in event.error.message.lower():
            if fut := self._chat_ctx_event_futures.pop(event.error.event_id, None):
                if not fut.done():
                    fut.set_result(None)
                    return

        if event.error.type == "invalid_request_error":
            msg = (event.error.message or "").lower()
            if "no ongoing response" in msg or "no active response" in msg:
                logger.debug(
                    "Ignored benign cancel/update race error from DashScope",
                )
                return
        super()._handle_error(event)

    async def aclose(self) -> None:
        """Cancel correlation timers and close the inherited realtime session."""
        self._correlation_lost = True
        self._response_idle.set()
        if self._cancel_timeout:
            self._cancel_timeout.cancel()
            self._cancel_timeout = None
        self._forget_response(self._pending_response_id)
        await super().aclose()
