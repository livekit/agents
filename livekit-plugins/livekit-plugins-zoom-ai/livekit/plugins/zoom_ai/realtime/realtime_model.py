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

"""Zoom Voice Agent API as a LiveKit realtime model.

Voice Agent is a speech-to-speech service: one WebSocket carries the caller's audio in and
the agent's speech, transcripts and function calls out, with Zoom running recognition,
the model and synthesis. Protocol:
https://developers.zoom.us/docs/ai-services/voice-agent/live-mode/

What the protocol allows, and how the plugin maps it onto the framework:

* The first client event must be ``session.update`` and carries the whole configuration
  (``listen`` / ``think`` / ``speak``). The plugin connects only once the agent has supplied
  its instructions and tools, so that first update is complete.
* Turn detection and barge-in are Zoom's and always on. The server starts a response at the
  end of each user turn; ``generate_reply()`` sends ``response.create``.
* Only one response may be active at a time, so reply requests are queued until the active
  response's ``response.done``.
* ``response.create`` takes no per-reply instructions or tools, and conversation items other
  than function results can't be added. A chat history that exists before the session starts
  (an agent handoff, a reconnection) is sent as part of the instructions instead.
* A session lasts at most 60 minutes; when Zoom closes it, the plugin reconnects and carries
  the conversation over the same way.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import time
import uuid
import weakref
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import aiohttp

from livekit import rtc
from livekit.agents import APIConnectionError, APIError, APIStatusError, llm, utils
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from ..log import logger
from ..version import __version__

DEFAULT_URL = "wss://api.zoom.us/v2/ws/aiservices/voice-agent/live"
SUBPROTOCOL = "voice-agent.zoom.v2"
DEFAULT_MODEL = "gemma-4"
DEFAULT_VOICE = "jenna"
SAMPLE_RATE = 16000  # speak.format.rate supports 16 kHz only; listen is set to match
NUM_CHANNELS = 1
USER_AGENT = f"livekit-plugins-zoom-ai/{__version__}"

Voice = Literal["alex", "ariel", "ava", "brooke", "hazel", "jenna", "mark", "reid", "scott", "tony"]
LLMProvider = Literal["zoom", "openai", "anthropic"]
ZoomToolChoice = Literal["auto", "none", "required"]

_REPLY_TIMEOUT_S = 10.0
_HISTORY_HEADER = (
    "The conversation so far, before this session started. Continue it; don't repeat it:"
)


@dataclass
class _RealtimeOptions:
    api_key: str
    url: str
    model: str
    llm_provider: LLMProvider
    provider_headers: dict[str, str] | None
    provider_metadata: dict[str, Any] | None
    voice: str
    vocabulary: list[str] | None
    reasoning_effort: str | None
    response_token_limit: int | Literal["inf"] | None
    tool_choice: llm.ToolChoice | None
    conn_options: APIConnectOptions


class RealtimeModel(llm.RealtimeModel):
    """Zoom Voice Agent, ready to pass to ``AgentSession(llm=...)`` or ``Agent(llm=...)``.

    Zoom detects turns and handles barge-in itself, so no VAD or turn detector is needed.
    """

    def __init__(
        self,
        *,
        voice: Voice | str = DEFAULT_VOICE,
        model: str = DEFAULT_MODEL,
        llm_provider: LLMProvider = "zoom",
        provider_headers: dict[str, str] | None = None,
        provider_metadata: dict[str, Any] | None = None,
        vocabulary: list[str] | None = None,
        reasoning_effort: str | None = None,
        response_token_limit: int | Literal["inf"] | None = None,
        tool_choice: llm.ToolChoice | None = None,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        url: str = DEFAULT_URL,
        http_session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """Create a Zoom Voice Agent realtime model.

        Args:
            voice: Voice ID (``speak.voice``), e.g. ``"jenna"`` (default) or ``"tony"``.
            model: Conversational model (``think.provider.model``). ``"gemma-4"`` for the
                built-in ``zoom`` provider, or a model ID of your own provider.
            llm_provider: ``"zoom"`` (built-in model), or ``"openai"`` / ``"anthropic"`` to bring
                your own key, passed in ``provider_headers``.
            provider_headers: Headers for your own provider, e.g. ``{"X-Api-Key": "sk-..."}``.
                Required unless ``llm_provider`` is ``"zoom"``. Never your Zoom credentials.
            provider_metadata: Your model's limits, e.g.
                ``{"context_window": 1000000, "max_output_tokens": 128000}``.
            vocabulary: Names and terms that improve transcription accuracy.
            reasoning_effort: Reasoning level, such as ``"low"``. Set once per connection.
            response_token_limit: Maximum tokens per response; ``None`` uses Zoom's default
                (``"inf"``).
            tool_choice: ``"auto"``, ``"none"`` or ``"required"``; ``None`` uses Zoom's
                default (``"auto"``). A named function is sent as ``"required"``.
            api_key: Zoom AI Services API key, created at https://www.zoom.ai/. Defaults to
                ``ZOOM_AI_API_KEY``.
            url: Voice Agent WebSocket endpoint.
            http_session: aiohttp session to use; defaults to the job's shared session.
            conn_options: Connection timeout and retries.
        """
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=True,
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=False,
                audio_output=True,
                manual_function_calls=False,
                can_disable_turn_detection=False,
                mutable_chat_context=False,
                mutable_instructions=True,
                mutable_tools=True,
                per_response_tool_choice=False,
                supports_say=False,
            )
        )
        api_key = api_key if is_given(api_key) else os.environ.get("ZOOM_AI_API_KEY", "")
        if not api_key:
            raise ValueError("Zoom AI API key is required: pass api_key or set ZOOM_AI_API_KEY")
        if llm_provider != "zoom" and not provider_headers:
            raise ValueError(
                f"llm_provider={llm_provider!r} needs provider_headers with your "
                f"{llm_provider} API key"
            )
        if not url.startswith(("wss://", "ws://")):
            raise ValueError(f"url must be a ws(s):// URL, got {url!r}")

        self._opts = _RealtimeOptions(
            api_key=api_key,
            url=url,
            model=model,
            llm_provider=llm_provider,
            provider_headers=dict(provider_headers) if provider_headers else None,
            provider_metadata=dict(provider_metadata) if provider_metadata else None,
            voice=voice,
            vocabulary=list(vocabulary) if vocabulary else None,
            reasoning_effort=reasoning_effort,
            response_token_limit=response_token_limit,
            tool_choice=tool_choice,
            conn_options=conn_options,
        )
        self._http_session = http_session
        self._sessions = weakref.WeakSet[RealtimeSession]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Zoom"

    def update_options(
        self,
        *,
        voice: NotGivenOr[Voice | str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        vocabulary: NotGivenOr[list[str] | None] = NOT_GIVEN,
    ) -> None:
        """Change options for this model and its live sessions."""
        if is_given(voice):
            self._opts.voice = voice
        if is_given(tool_choice):
            self._opts.tool_choice = tool_choice
        if is_given(vocabulary):
            self._opts.vocabulary = list(vocabulary) if vocabulary else None
        for sess in self._sessions:
            sess._update_model_options(voice=voice, tool_choice=tool_choice, vocabulary=vocabulary)

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if not self._http_session:
            self._http_session = utils.http_context.http_session()
        return self._http_session

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        if turn_detection_disabled:
            logger.warning(
                "Zoom Voice Agent turn detection is server-side and always on; "
                "turn_detection_disabled is ignored"
            )
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None:
        pass


@dataclass
class _MessageGeneration:
    message_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    modalities: asyncio.Future[list[Literal["text", "audio"]]]
    transcript: str = ""


@dataclass
class _ResponseGeneration:
    response_id: str
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]
    created_at: float
    messages: dict[str, _MessageGeneration] = field(default_factory=dict)
    first_token_at: float | None = None

    def close(self) -> None:
        for msg in self.messages.values():
            msg.text_ch.close()
            msg.audio_ch.close()
            if not msg.modalities.done():
                msg.modalities.set_result(["audio", "text"])
        self.function_ch.close()
        self.message_ch.close()


class _DiscardedGeneration:
    """A response cancelled before it surfaced; its trailing events are skipped."""


@dataclass
class _PendingReply:
    fut: asyncio.Future[llm.GenerationCreatedEvent]
    timer: asyncio.TimerHandle | None = None

    def cancel_timer(self) -> None:
        if self.timer:
            self.timer.cancel()
            self.timer = None


class RealtimeSession(llm.RealtimeSession[Literal["zoom_server_event_received"]]):
    """One Zoom Voice Agent conversation.

    Also emits ``zoom_server_event_received`` with every server event (unwrapped from its
    envelope). ``session_id`` holds Zoom's session ID once the session is configured.
    """

    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model: RealtimeModel = realtime_model
        self._opts = replace(realtime_model._opts)
        self.session_id: str | None = None

        self._instructions: str | None = None
        self._history: str | None = None
        self._tools = llm.ToolContext.empty()
        self._zoom_tools: list[dict[str, Any]] = []
        # what the server has: transcripts, replies, function calls and their outputs
        self._chat_ctx = llm.ChatContext.empty()
        self._conversation_started = False

        self._msg_ch = utils.aio.Chan[dict[str, Any]]()
        self._configured = asyncio.Event()
        self._closing = False
        # an error before the connection's first session.updated is a configuration error
        self._awaiting_config_ack = False

        self._input_resampler: rtc.AudioResampler | None = None
        self._bstream = utils.audio.AudioByteStream(
            SAMPLE_RATE, NUM_CHANNELS, samples_per_channel=SAMPLE_RATE // 10
        )

        self._current_generation: _ResponseGeneration | _DiscardedGeneration | None = None
        self._active_response_id: str | None = None
        self._cancel_on_create = False
        self._reply_queue: deque[_PendingReply] = deque()
        self._reply_inflight: deque[_PendingReply] = deque()
        self._speech_started_at: float | None = None
        self._warned: set[str] = set()

        self._main_atask = asyncio.create_task(self._main_task(), name="ZoomRealtime._main_task")

    # ------------------------------------------------------------------ framework API

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools.copy()

    async def _update_session(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> None:
        # The framework's default sends instructions, history and tools one by one. Zoom's
        # first session.update must be complete, so collect all three and send one update.
        if is_given(instructions):
            self._instructions = instructions
        if is_given(tools):
            self._set_tools(tools)
        if is_given(chat_ctx):
            self._apply_chat_ctx(chat_ctx, send_update=False)
        if self._configured.is_set():
            self._send_session_update()
        else:
            self._configured.set()

    async def update_instructions(self, instructions: str) -> None:
        self._instructions = instructions
        if self._configured.is_set():
            self._send_session_update()

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        self._set_tools(tools)
        if self._configured.is_set():
            self._send_session_update()

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        self._apply_chat_ctx(chat_ctx, send_update=self._configured.is_set())

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        self._update_model_options(tool_choice=tool_choice)

    def _update_model_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        vocabulary: NotGivenOr[list[str] | None] = NOT_GIVEN,
    ) -> None:
        changed = False
        if is_given(voice) and voice != self._opts.voice:
            self._opts.voice = voice
            changed = True
        if is_given(tool_choice) and tool_choice != self._opts.tool_choice:
            self._opts.tool_choice = tool_choice
            changed = True
        if is_given(vocabulary):
            vocabulary = list(vocabulary) if vocabulary else None
            if vocabulary != self._opts.vocabulary:
                self._opts.vocabulary = vocabulary
                changed = True
        if changed and self._configured.is_set():
            self._send_session_update()

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        self._configured.set()  # audio without an agent (standalone use) starts the session
        for f in self._resample_audio(frame):
            self._append_audio(self._bstream.write(f.data.tobytes()))

    def push_video(self, frame: rtc.VideoFrame) -> None:
        self._warn_once("video", "Zoom Voice Agent doesn't accept video; frames are dropped")

    def commit_audio(self) -> None:
        # Zoom ends each turn itself; there is no commit event. Send the partial chunk.
        self._append_audio(self._bstream.flush())

    def clear_audio(self) -> None:
        # Zoom has no clear event; drop only what hasn't been sent yet.
        self._bstream.clear()

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        if is_given(instructions):
            self._warn_once(
                "reply_instructions",
                "Zoom Voice Agent doesn't take per-reply instructions; the reply uses the "
                "session instructions. Put greetings and the like in the agent's instructions.",
            )
        # tool_choice and tools per reply aren't supported either (per_response_tool_choice
        # is False); the framework only passes them for tool replies, where the session's
        # settings are what Zoom applies anyway.
        self._configured.set()
        fut = asyncio.get_running_loop().create_future()
        reply = _PendingReply(fut=fut)

        def _on_done(f: asyncio.Future[llm.GenerationCreatedEvent]) -> None:
            reply.cancel_timer()
            if not f.cancelled():
                return
            if reply in self._reply_queue:
                self._reply_queue.remove(reply)
            elif reply in self._reply_inflight:
                # response.create is out; cancel the response once the server creates it
                self._reply_inflight.remove(reply)
                self._cancel_on_create = True

        fut.add_done_callback(_on_done)
        self._reply_queue.append(reply)
        self._pump_replies()
        return fut

    def interrupt(self) -> None:
        if self._active_response_id:
            self._send({"type": "response.cancel", "response_id": self._active_response_id})
        elif self._reply_inflight:
            # requested but not created yet: cancel it when it arrives
            self._cancel_on_create = True

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if "audio" in modalities:
            self._send(
                {
                    "type": "conversation.item.truncate",
                    "item_id": message_id,
                    "content_index": 0,
                    "audio_end_ms": max(0, int(audio_end_ms)),
                }
            )
        if is_given(audio_transcript):
            item = self._chat_ctx.get_by_id(message_id)
            if isinstance(item, llm.ChatMessage):
                item.content = [audio_transcript] if audio_transcript else []

    async def aclose(self) -> None:
        self._closing = True
        self._close_current_generation()
        self._msg_ch.close()
        self._configured.set()  # let a session that never started exit
        # a failure that ended the session was already reported as an "error" event
        with contextlib.suppress(Exception):
            await self._main_atask

    # ------------------------------------------------------------------ outgoing

    def _send(self, event: dict[str, Any]) -> None:
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(event)

    def _send_session_update(self) -> None:
        self._send(self._session_update_event(first=False))

    def _session_update_event(self, *, first: bool) -> dict[str, Any]:
        opts = self._opts
        listen: dict[str, Any] = {"format": {"type": "pcm16", "rate": SAMPLE_RATE}}
        if opts.vocabulary:
            listen["transcription"] = {"vocabulary": {"phrases": opts.vocabulary}}

        provider: dict[str, Any] = {"type": opts.llm_provider, "model": opts.model}
        if opts.provider_headers:
            provider["headers"] = opts.provider_headers
        if opts.provider_metadata:
            provider["metadata"] = opts.provider_metadata
        think: dict[str, Any] = {"provider": provider, "tools": self._zoom_tools}
        if instructions := self._full_instructions():
            think["instructions"] = instructions
        if opts.tool_choice is not None:
            think["tool_choice"] = _to_zoom_tool_choice(opts.tool_choice)
        if opts.response_token_limit is not None:
            think["response_token_limit"] = opts.response_token_limit
        if first and opts.reasoning_effort:
            # "Set the reasoning level in your first session.update"
            think["reasoning"] = {"effort": opts.reasoning_effort}

        speak: dict[str, Any] = {"format": {"type": "pcm16"}, "voice": opts.voice}
        return {
            "type": "session.update",
            "session": {"listen": listen, "think": think, "speak": speak},
        }

    def _full_instructions(self) -> str | None:
        parts = [p for p in (self._instructions, self._history) if p]
        return "\n\n".join(parts) if parts else None

    def _set_tools(self, tools: list[llm.Tool]) -> None:
        zoom_tools: list[dict[str, Any]] = []
        retained: list[llm.Tool] = []
        for tool in tools:
            if isinstance(tool, llm.FunctionTool):
                desc = llm.utils.build_legacy_openai_schema(tool, internally_tagged=True)
            elif isinstance(tool, llm.RawFunctionTool):
                desc = dict(tool.info.raw_schema)
                desc.pop("meta", None)
                desc["type"] = "function"
            else:
                logger.warning(
                    "Zoom Voice Agent supports function tools only; skipping a tool",
                    extra={"tool": type(tool).__name__},
                )
                continue
            zoom_tools.append(
                {
                    "type": "function",
                    "name": desc["name"],
                    "description": desc.get("description") or "",
                    "parameters": desc.get("parameters") or {"type": "object", "properties": {}},
                }
            )
            retained.append(tool)
        self._zoom_tools = zoom_tools
        self._tools = llm.ToolContext(retained)

    def _apply_chat_ctx(self, chat_ctx: llm.ChatContext, *, send_update: bool) -> None:
        new_items = [item for item in chat_ctx.items if self._chat_ctx.get_by_id(item.id) is None]
        if not new_items:
            return

        if not self._conversation_started:
            # Nothing has been said in this session yet, so the history can still be given
            # as part of the instructions.
            history: list[llm.ChatItem] = [
                item
                for item in chat_ctx.items
                if item.type in ("message", "function_call", "function_call_output")
                and not (item.type == "message" and item.role in ("system", "developer"))
            ]
            self._chat_ctx = llm.ChatContext(history)
            self._history = _render_history(history)
            if send_update:
                self._send_session_update()
            return

        for item in new_items:
            if item.type == "function_call_output":
                self._send(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": item.output,
                        },
                    }
                )
                self._chat_ctx.items.append(item)
            elif item.type == "message" and item.role in ("system", "developer"):
                continue
            elif item.type in ("message", "function_call"):
                self._warn_once(
                    "chat_ctx",
                    "Zoom Voice Agent can't add messages to a running conversation; only "
                    "function results are sent. Use instructions to give the agent context.",
                )

    def _append_audio(self, frames: list[rtc.AudioFrame]) -> None:
        for f in frames:
            self._send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(f.data).decode("ascii"),
                }
            )

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler and frame.sample_rate != self._input_resampler._input_rate:
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

    def _pump_replies(self) -> None:
        """Send the next queued response.create once no response is active."""
        if self._active_response_id or self._reply_inflight:
            return
        while self._reply_queue:
            reply = self._reply_queue.popleft()
            if reply.fut.done():
                continue
            self._reply_inflight.append(reply)
            self._send({"type": "response.create"})

            def _on_timeout(reply: _PendingReply = reply) -> None:
                reply.timer = None
                if reply in self._reply_inflight:
                    self._reply_inflight.remove(reply)
                if not reply.fut.done():
                    reply.fut.set_exception(llm.RealtimeError("generate_reply timed out."))
                self._pump_replies()

            reply.timer = asyncio.get_running_loop().call_later(_REPLY_TIMEOUT_S, _on_timeout)
            return

    # ------------------------------------------------------------------ connection

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        await self._configured.wait()
        max_retries = self._opts.conn_options.max_retry
        num_retries = 0
        reconnecting = False
        try:
            while not self._msg_ch.closed:
                try:
                    ws = await self._connect()
                    if reconnecting:
                        self._on_reconnected()
                    num_retries = 0
                    reconnecting = True
                    await self._run_ws(ws)
                except APIError as e:
                    if self._closing:
                        break
                    if max_retries == 0 or not e.retryable or num_retries >= max_retries:
                        self._emit_error(e, recoverable=False)
                        raise
                    self._emit_error(e, recoverable=True)
                    interval = self._opts.conn_options._interval_for_retry(num_retries)
                    logger.warning(
                        f"Zoom Voice Agent connection failed, retrying in {interval}s",
                        exc_info=e,
                        extra={"attempt": num_retries, "max_retries": max_retries},
                    )
                    num_retries += 1
                    await asyncio.sleep(interval)
        finally:
            self._close_current_generation()
            for reply in (*self._reply_inflight, *self._reply_queue):
                reply.cancel_timer()
                if not reply.fut.done():
                    reply.fut.set_exception(llm.RealtimeError("realtime session closed"))
            self._reply_inflight.clear()
            self._reply_queue.clear()

    async def _connect(self) -> aiohttp.ClientWebSocketResponse:
        tracking_id = str(uuid.uuid4())
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}",
            "User-Agent": USER_AGENT,
            "x-zm-trackingid": tracking_id,
        }
        t0 = time.perf_counter()
        try:
            ws = await asyncio.wait_for(
                self._realtime_model._ensure_http_session().ws_connect(
                    self._opts.url, headers=headers, protocols=(SUBPROTOCOL,)
                ),
                self._opts.conn_options.timeout,
            )
        except aiohttp.WSServerHandshakeError as e:
            # `from None`: the handshake error's repr carries the request headers (the key)
            raise APIStatusError(
                f"Zoom refused the Voice Agent connection (HTTP {e.status})",
                status_code=e.status,
                request_id=tracking_id,
            ) from None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise APIConnectionError("failed to connect to Zoom Voice Agent") from e
        self._report_connection_acquired(time.perf_counter() - t0)
        return ws

    def _on_reconnected(self) -> None:
        # The new server session knows nothing: carry the conversation over in the
        # instructions, and drop what was in flight on the old one.
        self._history = _render_history(self._chat_ctx.items)
        self._close_current_generation()
        self._active_response_id = None
        self._cancel_on_create = False
        for reply in reversed(self._reply_inflight):
            reply.cancel_timer()
            self._reply_queue.appendleft(reply)
        self._reply_inflight.clear()
        logger.debug("reconnected to Zoom Voice Agent")
        self.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())

    async def _run_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        closing = False

        try:
            # the first client event must be session.update
            self._awaiting_config_ack = True
            await ws.send_str(json.dumps(self._session_update_event(first=True)))
        except Exception as e:
            await ws.close()
            raise APIConnectionError("failed to configure the Zoom Voice Agent session") from e
        self._pump_replies()

        @utils.log_exceptions(logger=logger)
        async def _send_task() -> None:
            nonlocal closing
            async for msg in self._msg_ch:
                try:
                    await ws.send_str(json.dumps(msg))
                except Exception as e:
                    raise APIConnectionError("failed to send to Zoom Voice Agent") from e
            closing = True
            await ws.close()

        @utils.log_exceptions(logger=logger)
        async def _recv_task() -> None:
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.CLOSED,
                ):
                    if closing or self._closing:
                        return
                    # sessions end after 60 minutes; this reconnects
                    raise APIConnectionError("Zoom Voice Agent connection closed unexpectedly")
                if msg.type != aiohttp.WSMsgType.TEXT or self._closing:
                    continue
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.warning("invalid JSON from Zoom Voice Agent")
                    continue
                if isinstance(data, dict):
                    self._handle_server_event(_unwrap(data))

        tasks = [
            asyncio.create_task(_recv_task(), name="ZoomRealtime._recv_task"),
            asyncio.create_task(_send_task(), name="ZoomRealtime._send_task"),
        ]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await ws.close()

    # ------------------------------------------------------------------ incoming

    def _handle_server_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        self.emit("zoom_server_event_received", event)
        try:
            handler = _HANDLERS.get(event_type) if isinstance(event_type, str) else None
            if handler:
                handler(self, event)
        except APIError:
            raise
        except Exception:
            logger.exception(
                "failed to handle Zoom Voice Agent event", extra={"event_type": event_type}
            )

    def _on_session_updated(self, event: dict[str, Any]) -> None:
        self._awaiting_config_ack = False
        session = event.get("session")
        if isinstance(session, dict) and isinstance(session.get("id"), str):
            self.session_id = session["id"]

    def _on_speech_started(self, event: dict[str, Any]) -> None:
        self._speech_started_at = time.time()
        self.emit("input_speech_started", llm.InputSpeechStartedEvent())

    def _on_speech_stopped(self, event: dict[str, Any]) -> None:
        self.emit(
            "input_speech_stopped",
            llm.InputSpeechStoppedEvent(user_transcription_enabled=True),
        )

    def _on_input_transcription(self, event: dict[str, Any]) -> None:
        transcript = event.get("transcript")
        if not isinstance(transcript, str):
            text = event.get("text")
            transcript = text if isinstance(text, str) else ""
        item_id = event.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            item_id = utils.shortuuid("zoom_user_")
        self._conversation_started = True
        if transcript and self._chat_ctx.get_by_id(item_id) is None:
            self._chat_ctx.items.append(
                llm.ChatMessage(id=item_id, role="user", content=[transcript])
            )
        started_at, self._speech_started_at = self._speech_started_at, None
        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=item_id,
                transcript=transcript,
                is_final=True,
                turn_started_at=started_at,
            ),
        )

    def _on_response_created(self, event: dict[str, Any]) -> None:
        response = event.get("response")
        response_id = response.get("id") if isinstance(response, dict) else None
        if not isinstance(response_id, str) or not response_id:
            response_id = event.get("response_id")
        if not isinstance(response_id, str) or not response_id:
            response_id = utils.shortuuid("zoom_resp_")
        self._active_response_id = response_id
        self._conversation_started = True

        # Zoom doesn't echo a client id on responses; the oldest reply request in flight is
        # the one this answers (only one is ever sent while no response is active).
        reply = self._reply_inflight.popleft() if self._reply_inflight else None
        if reply:
            reply.cancel_timer()

        if self._cancel_on_create:
            self._cancel_on_create = False
            self._send({"type": "response.cancel", "response_id": response_id})
            self._current_generation = _DiscardedGeneration()
            if reply and not reply.fut.done():
                reply.fut.set_exception(llm.RealtimeError("generate_reply interrupted"))
            return

        self._close_current_generation()
        gen = _ResponseGeneration(
            response_id=response_id,
            message_ch=utils.aio.Chan(),
            function_ch=utils.aio.Chan(),
            created_at=time.time(),
        )
        self._current_generation = gen
        generation_ev = llm.GenerationCreatedEvent(
            message_stream=gen.message_ch,
            function_stream=gen.function_ch,
            user_initiated=False,
            response_id=response_id,
        )
        if reply and not reply.fut.done():
            generation_ev.user_initiated = True
            reply.fut.set_result(generation_ev)
        self.emit("generation_created", generation_ev)

    def _message(self, item_id: str) -> _MessageGeneration | None:
        gen = self._current_generation
        if not isinstance(gen, _ResponseGeneration):
            return None
        if msg := gen.messages.get(item_id):
            return msg
        msg = _MessageGeneration(
            message_id=item_id,
            text_ch=utils.aio.Chan(),
            audio_ch=utils.aio.Chan(),
            modalities=asyncio.get_running_loop().create_future(),
        )
        gen.messages[item_id] = msg
        gen.message_ch.send_nowait(
            llm.MessageGeneration(
                message_id=item_id,
                text_stream=msg.text_ch,
                audio_stream=msg.audio_ch,
                modalities=msg.modalities,
            )
        )
        return msg

    def _message_for_event(self, item_id: object) -> _MessageGeneration | None:
        if isinstance(item_id, str) and item_id:
            return self._message(item_id)
        gen = self._current_generation
        if not isinstance(gen, _ResponseGeneration):
            return None
        return self._message(f"{gen.response_id}_message")

    def _on_output_item_added(self, event: dict[str, Any]) -> None:
        item = event.get("item")
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            return
        if item.get("type") == "message":
            self._message(item["id"])
        # other items (function calls) surface when their arguments are done; untyped ones
        # when their first audio or transcript arrives

    def _on_output_item_done(self, event: dict[str, Any]) -> None:
        item = event.get("item")
        gen = self._current_generation
        if not isinstance(item, dict) or not isinstance(gen, _ResponseGeneration):
            return
        if msg := gen.messages.get(item.get("id", "")):
            msg.text_ch.close()
            msg.audio_ch.close()

    def _on_audio_delta(self, event: dict[str, Any]) -> None:
        item_id = event.get("item_id")
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta:
            return
        msg = self._message_for_event(item_id)
        gen = self._current_generation
        if msg is None or not isinstance(gen, _ResponseGeneration):
            return
        if gen.first_token_at is None:
            gen.first_token_at = time.time()
        if not msg.modalities.done():
            msg.modalities.set_result(["audio", "text"])
        data = base64.b64decode(delta)
        if len(data) < 2:
            return
        data = data[: len(data) - len(data) % 2]
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            msg.audio_ch.send_nowait(
                rtc.AudioFrame(
                    data=data,
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    samples_per_channel=len(data) // 2,
                )
            )

    def _on_transcript_delta(self, event: dict[str, Any]) -> None:
        item_id = event.get("item_id")
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta:
            return
        msg = self._message_for_event(item_id)
        if msg is None:
            return
        msg.transcript += delta
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            msg.text_ch.send_nowait(delta)

    def _on_function_call(self, event: dict[str, Any]) -> None:
        call_id = event.get("call_id")
        name = event.get("name")
        if not isinstance(call_id, str) or not isinstance(name, str):
            logger.warning("Zoom Voice Agent function call without call_id or name")
            return
        arguments = event.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments) if arguments is not None else "{}"
        item_id = event.get("item_id")
        call = llm.FunctionCall(
            id=item_id if isinstance(item_id, str) and item_id else call_id,
            call_id=call_id,
            name=name,
            arguments=arguments,
        )
        previous_item_id = self._chat_ctx.items[-1].id if self._chat_ctx.items else None
        if self._chat_ctx.get_by_id(call.id) is None:
            self._chat_ctx.items.append(call)
            self.emit(
                "remote_item_added",
                llm.RemoteItemAddedEvent(previous_item_id=previous_item_id, item=call),
            )
        gen = self._current_generation
        if isinstance(gen, _ResponseGeneration):
            with contextlib.suppress(utils.aio.channel.ChanClosed):
                gen.function_ch.send_nowait(call)

    def _on_response_done(self, event: dict[str, Any]) -> None:
        self._active_response_id = None
        gen = self._current_generation
        self._current_generation = None
        response = event.get("response")
        response = response if isinstance(response, dict) else {}
        status = response.get("status") or event.get("status")

        if isinstance(gen, _ResponseGeneration):
            for msg in gen.messages.values():
                if msg.transcript and self._chat_ctx.get_by_id(msg.message_id) is None:
                    self._chat_ctx.items.append(
                        llm.ChatMessage(
                            id=msg.message_id, role="assistant", content=[msg.transcript]
                        )
                    )
            gen.close()
            self._emit_metrics(gen, response, cancelled=status == "cancelled")

        if status == "failed":
            details = response.get("status_details")
            self._emit_error(
                APIError("Zoom Voice Agent response failed", body=details, retryable=True),
                recoverable=True,
            )
        elif status == "incomplete":
            logger.debug("Zoom Voice Agent response incomplete (response_token_limit reached)")

        self._pump_replies()

    def _on_error(self, event: dict[str, Any]) -> None:
        error = event.get("error")
        error = error if isinstance(error, dict) else {}
        message = str(error.get("message") or "unknown error")
        error_type = error.get("type")

        if "active response" in message.lower():
            # Our response.create reached the server while a response it started itself was
            # active. Zoom doesn't label responses, so that response was already credited to
            # the request if it was created after the request went out; otherwise ask again
            # after its response.done.
            if self._reply_inflight:
                reply = self._reply_inflight.popleft()
                reply.cancel_timer()
                self._reply_queue.appendleft(reply)
            else:
                logger.debug("reply request answered by the already active response")
            return

        if self._awaiting_config_ack:
            # Zoom keeps the socket open after rejecting a session.update, but the agent
            # can't work with a configuration it refused (unknown voice, bad provider key...)
            raise APIError(
                f"Zoom Voice Agent rejected the session configuration: {message}",
                body=error,
                retryable=False,
            )

        logger.error(f"Zoom Voice Agent returned an error: {message}", extra={"error": error})
        self._emit_error(
            APIError(
                f"Zoom Voice Agent returned an error: {message}",
                body=error,
                retryable=error_type == "server_error",
            ),
            recoverable=True,
        )

    # ------------------------------------------------------------------ helpers

    def _close_current_generation(self) -> None:
        gen = self._current_generation
        self._current_generation = None
        if isinstance(gen, _ResponseGeneration):
            gen.close()

    def _emit_metrics(
        self, gen: _ResponseGeneration, response: dict[str, Any], *, cancelled: bool
    ) -> None:
        usage = response.get("usage")
        usage = usage if isinstance(usage, dict) else {}

        def _int(key: str) -> int:
            value = usage.get(key)
            return value if isinstance(value, int) else 0

        duration = time.time() - gen.created_at
        output_tokens = _int("output_tokens")
        self.emit(
            "metrics_collected",
            RealtimeModelMetrics(
                timestamp=gen.created_at,
                request_id=gen.response_id,
                ttft=gen.first_token_at - gen.created_at if gen.first_token_at else -1,
                duration=duration,
                cancelled=cancelled,
                label=self._realtime_model.label,
                input_tokens=_int("input_tokens"),
                output_tokens=output_tokens,
                total_tokens=_int("total_tokens"),
                tokens_per_second=output_tokens / duration if duration > 0 else 0,
                input_token_details=RealtimeModelMetrics.InputTokenDetails(),
                output_token_details=RealtimeModelMetrics.OutputTokenDetails(),
                metadata=Metadata(
                    model_name=self._realtime_model.model,
                    model_provider=self._realtime_model.provider,
                ),
            ),
        )

    def _emit_error(self, error: Exception, recoverable: bool) -> None:
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self._realtime_model.label,
                error=error,
                recoverable=recoverable,
            ),
        )

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message)


_HANDLERS: dict[str, Any] = {
    "session.updated": RealtimeSession._on_session_updated,
    "input_audio_buffer.speech_started": RealtimeSession._on_speech_started,
    "input_audio_buffer.speech_stopped": RealtimeSession._on_speech_stopped,
    "conversation.item.input_audio_transcription.completed": (
        RealtimeSession._on_input_transcription
    ),
    "response.created": RealtimeSession._on_response_created,
    "response.output_item.added": RealtimeSession._on_output_item_added,
    "response.output_item.done": RealtimeSession._on_output_item_done,
    "response.output_audio.delta": RealtimeSession._on_audio_delta,
    "response.output_audio_transcript.delta": RealtimeSession._on_transcript_delta,
    "response.function_call_arguments.done": RealtimeSession._on_function_call,
    "response.done": RealtimeSession._on_response_done,
    "error": RealtimeSession._on_error,
}


def _unwrap(data: dict[str, Any]) -> dict[str, Any]:
    """Server events come as ``{"type", "payload"}``; the payload is the full event."""
    payload = data.get("payload")
    if isinstance(payload, dict):
        if "type" not in payload and isinstance(data.get("type"), str):
            payload = {**payload, "type": data["type"]}
        return payload
    return data


def _to_zoom_tool_choice(tool_choice: llm.ToolChoice) -> ZoomToolChoice:
    if tool_choice in ("auto", "none", "required"):
        return tool_choice
    # Zoom can't force a specific function; requiring a tool call is the closest.
    return "required"


def _render_history(items: list[llm.ChatItem]) -> str | None:
    lines: list[str] = []
    for item in items:
        if item.type == "message" and item.role in ("user", "assistant"):
            if text := (item.text_content or "").strip():
                lines.append(f"{item.role}: {text}")
        elif item.type == "function_call":
            lines.append(f"assistant called {item.name}({item.arguments})")
        elif item.type == "function_call_output":
            lines.append(f"{item.name or 'function'} returned: {item.output}")
    if not lines:
        return None
    return _HISTORY_HEADER + "\n" + "\n".join(lines)
