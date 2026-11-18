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

Its events are those of the OpenAI Realtime API, so this model runs on the shared OpenAI
Realtime engine (``livekit.agents.llm._realtime.openai``) and translates what differs:

* Server events come in ``{"type", "payload"}`` envelopes, and a function result is
  acknowledged with ``conversation.item.created`` (the engine expects ``.added``).
* ``session.update`` has Zoom's own shape (``listen`` / ``think`` / ``speak``), and the first
  client event must be a complete one. The plugin connects once the agent has supplied its
  instructions and tools, and sends one full update whenever any setting changes.
* Audio is 16 kHz PCM16 both ways.
* ``response.create`` takes no response object and Zoom echoes no client metadata, so each
  response is credited to the oldest reply request in flight. Only one response may be
  active at a time, so reply requests wait for the active response's ``response.done``.
* Turn detection and barge-in are Zoom's and always on: there is no
  ``input_audio_buffer.commit`` or ``clear``, and Zoom cancels the active response itself
  when the caller starts speaking.
* Conversation items other than function results can't be added. A chat history that
  exists before the session starts (an agent handoff, a reconnection after Zoom's 60-minute
  session limit) is sent as part of the instructions instead.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import json
import os
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import aiohttp
from pydantic import BaseModel

from livekit import rtc
from livekit.agents import APIConnectionError, APIError, APIStatusError, llm, utils
from livekit.agents.llm._realtime import openai as _engine
from livekit.agents.llm.remote_chat_context import RemoteChatContext
from livekit.agents.metrics import STTMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGiven,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from ..log import logger
from ..stt import _tracking_id
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

# Every server event, unwrapped from its envelope. Typed Any because the base session fixes
# its event-name Literal.
SERVER_EVENT: Any = "zoom_server_event_received"

# how long a sent reply request may wait for its response.created
_REPLY_TIMEOUT = 10.0

_HISTORY_HEADER = (
    "The conversation so far, before this session started. Continue it; don't repeat it:"
)
# client events Zoom doesn't accept: turn detection is Zoom's, and items can't be deleted
_DROPPED_EVENTS = frozenset(
    {"input_audio_buffer.commit", "input_audio_buffer.clear", "conversation.item.delete"}
)


@dataclass
class _ZoomOptions:
    api_key: str
    url: str
    llm_provider: LLMProvider
    provider_headers: dict[str, str] | None
    provider_metadata: dict[str, Any] | None
    vocabulary: list[str] | None
    reasoning_effort: str | None
    response_token_limit: int | Literal["inf"] | None


class RealtimeModel(_engine.RealtimeModel):
    """Zoom Voice Agent, ready to pass to ``AgentSession(llm=...)`` or ``Agent(llm=...)``.

    Zoom detects turns and handles barge-in itself, so no VAD or turn detector is needed.
    """

    def __init__(
        self,
        *,
        voice: Voice | str = DEFAULT_VOICE,
        model: str = DEFAULT_MODEL,
        llm_provider: LLMProvider = "zoom",
        provider_headers: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        provider_metadata: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
        vocabulary: NotGivenOr[list[str]] = NOT_GIVEN,
        reasoning_effort: NotGivenOr[str] = NOT_GIVEN,
        response_token_limit: NotGivenOr[int | Literal["inf"]] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
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
            response_token_limit: Maximum tokens per response; defaults to Zoom's default
                (``"inf"``).
            tool_choice: ``"auto"``, ``"none"`` or ``"required"``; defaults to Zoom's default
                (``"auto"``). A named function is sent as ``"required"``.
            api_key: Zoom AI Services API key, created at https://www.zoom.ai/. Defaults to
                ``ZOOM_AI_API_KEY``.
            url: Voice Agent WebSocket endpoint.
            http_session: aiohttp session to use; defaults to the job's shared session.
            conn_options: Connection timeout and retries.
        """
        api_key = api_key if is_given(api_key) else os.environ.get("ZOOM_AI_API_KEY", "")
        if not api_key:
            raise ValueError("Zoom AI API key is required: pass api_key or set ZOOM_AI_API_KEY")
        if llm_provider != "zoom" and not (is_given(provider_headers) and provider_headers):
            raise ValueError(
                f"llm_provider={llm_provider!r} needs provider_headers with your "
                f"{llm_provider} API key"
            )
        if not url.startswith(("wss://", "ws://")):
            raise ValueError(f"url must be a ws(s):// URL, got {url!r}")

        # is_given doesn't narrow a union with a Literal for mypy
        token_limit = None if isinstance(response_token_limit, NotGiven) else response_token_limit
        self._zoom_opts = _ZoomOptions(
            api_key=api_key,
            url=url,
            llm_provider=llm_provider,
            provider_headers=dict(provider_headers) if is_given(provider_headers) else None,
            provider_metadata=dict(provider_metadata) if is_given(provider_metadata) else None,
            vocabulary=list(vocabulary) if is_given(vocabulary) and vocabulary else None,
            reasoning_effort=reasoning_effort if is_given(reasoning_effort) else None,
            response_token_limit=token_limit,
        )
        super().__init__(
            model=model,
            voice=voice,
            modalities=["audio"],
            tool_choice=tool_choice,
            api_key=api_key,
            base_url=url,
            http_session=http_session,
            # Zoom ends a session after 60 minutes itself; the engine then reconnects
            max_session_duration=None,
            conn_options=conn_options,
        )
        # not Azure, whatever OPENAI_API_VERSION says
        self._opts.is_azure = False
        self._opts.api_version = None
        self._provider_label = "Zoom Voice Agent"
        self._capabilities = dataclasses.replace(
            self._capabilities,
            turn_detection=True,
            can_disable_turn_detection=False,
            user_transcription=True,
            audio_output=True,
            manual_function_calls=False,
            mutable_chat_context=False,
            per_response_tool_choice=False,
            supports_say=False,
        )

    @property
    def provider(self) -> str:
        return "Zoom"

    def update_options(  # type: ignore[override]
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
            self._zoom_opts.vocabulary = list(vocabulary) if vocabulary else None
        for sess in self._sessions:
            if isinstance(sess, RealtimeSession):
                sess.update_options(voice=voice, tool_choice=tool_choice, vocabulary=vocabulary)

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        if turn_detection_disabled:
            logger.warning(
                "Zoom Voice Agent turn detection is server-side and always on; "
                "turn_detection_disabled is ignored"
            )
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess


class RealtimeSession(_engine.RealtimeSession):
    """One Zoom Voice Agent conversation.

    Also emits ``zoom_server_event_received`` with every server event (unwrapped from its
    envelope). ``session_id`` holds Zoom's session ID once the session is configured.
    """

    def __init__(self, realtime_model: RealtimeModel) -> None:
        # set before the engine's __init__, which already sends a session update
        self._zoom_model = realtime_model
        self._zoom_opts = replace(
            realtime_model._zoom_opts,
            vocabulary=list(realtime_model._zoom_opts.vocabulary or []) or None,
        )
        self.session_id: str | None = None
        # set once the agent has supplied its configuration (or audio or a reply request
        # arrives without one); the session connects then
        self._zoom_configured = asyncio.Event()
        self._zoom_tools: list[dict[str, Any]] = []
        # the history carried over in the instructions (handoff, reconnection)
        self._zoom_history: str | None = None
        self._zoom_conversation_started = False
        # optional settings sent on the current connection, which need a reset to clear
        self._zoom_sent_optional: set[str] = set()
        # an error before the connection's first session.updated is a configuration error
        self._zoom_awaiting_config_ack = False
        # connections in a row that ended before Zoom accepted the configuration
        self._zoom_unconfigured_conns = 0
        # reply requests (response.create event ids) not yet sent, and sent but not created
        self._zoom_queue: deque[str] = deque()
        self._zoom_inflight: deque[str] = deque()
        self._zoom_active_response: str | None = None
        # response-created timeouts of the requests in flight, started when each is sent
        self._zoom_reply_timers: dict[str, asyncio.TimerHandle] = {}
        # the current connection, closed to reconnect when Zoom stops answering requests
        self._zoom_ws: _ZoomSocket | None = None
        self._zoom_abort_task: asyncio.Task[None] | None = None
        # the active response is already being cancelled, by us or by Zoom's barge-in
        self._zoom_active_cancelled = False
        self._zoom_sent_outputs: set[str] = set()
        self._zoom_warned: set[str] = set()

        super().__init__(realtime_model)
        self._bstream = utils.audio.AudioByteStream(
            SAMPLE_RATE, NUM_CHANNELS, samples_per_channel=SAMPLE_RATE // 10
        )
        self.on("openai_server_event_received", self._zoom_on_server_event)
        self.on("session_reconnected", self._zoom_on_reconnected)

    # ------------------------------------------------------------------ framework API

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
            self._zoom_set_tools(tools)
        if is_given(chat_ctx):
            self._zoom_apply_chat_ctx(chat_ctx, send_update=False)
        if self._zoom_configured.is_set():
            self._zoom_send_session_update()
        else:
            self._zoom_start()

    async def update_instructions(self, instructions: str) -> None:
        self._instructions = instructions
        self._zoom_send_session_update()

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        self._zoom_set_tools(tools)
        self._zoom_send_session_update()

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        self._zoom_apply_chat_ctx(chat_ctx, send_update=True)

    def update_options(  # type: ignore[override]
        self,
        *,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        vocabulary: NotGivenOr[list[str] | None] = NOT_GIVEN,
    ) -> None:
        changed = False
        if is_given(voice) and voice != self._opts.voice:
            self._opts.voice = voice
            changed = True
        if is_given(tool_choice):
            # the framework sets "auto" for a tool reply and None after it; both are "auto"
            # for Zoom, so only a change in the effective value is sent
            if _effective_tool_choice(tool_choice) != _effective_tool_choice(
                self._opts.tool_choice
            ):
                changed = True
            self._opts.tool_choice = tool_choice
        if is_given(vocabulary):
            vocabulary = list(vocabulary) if vocabulary else None
            if vocabulary != self._zoom_opts.vocabulary:
                self._zoom_opts.vocabulary = vocabulary
                changed = True
        if changed:
            self._zoom_send_session_update()

    def push_video(self, frame: rtc.VideoFrame) -> None:
        self._zoom_warn_once("video", "Zoom Voice Agent doesn't accept video; frames are dropped")

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
            self._zoom_warn_once(
                "reply_instructions",
                "Zoom Voice Agent doesn't take per-reply instructions; the reply uses the "
                "session instructions. Put greetings and the like in the agent's instructions.",
            )
        # tool_choice and tools per reply aren't supported either (per_response_tool_choice
        # is False); response.create goes out without a response object.
        #
        # Not the engine's generate_reply: its timeout starts right away, but a request here
        # can wait locally behind an active response for as long as that response lasts. The
        # timeout starts when _zoom_pump_replies sends the request instead.
        event_id = utils.shortuuid("response_create_")
        fut = asyncio.Future[llm.GenerationCreatedEvent]()
        self._response_created_futures[event_id] = fut

        def _on_fut_done(f: asyncio.Future[llm.GenerationCreatedEvent]) -> None:
            self._zoom_stop_reply_timer(event_id)
            self._response_created_futures.pop(event_id, None)
            if f.cancelled():
                # skipped if still queued; its response is cancelled when it arrives if sent
                self._discarded_event_ids.add(event_id)

        fut.add_done_callback(_on_fut_done)
        self.send_event({"type": "response.create", "event_id": event_id})
        return fut

    def interrupt(self) -> None:
        if self._zoom_active_response:
            self.send_event({"type": "response.cancel", "response_id": self._zoom_active_response})
            return
        # requested but not created yet: the engine cancels each one by id when it arrives
        for event_id in self._zoom_inflight:
            self._discarded_event_ids.add(event_id)
            if (fut := self._response_created_futures.pop(event_id, None)) and not fut.done():
                fut.set_exception(llm.RealtimeError("generate_reply interrupted"))

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
                {
                    "type": "conversation.item.truncate",
                    "item_id": message_id,
                    "content_index": 0,
                    "audio_end_ms": max(0, int(audio_end_ms)),
                }
            )
        if is_given(audio_transcript):
            remote = self._remote_chat_ctx.get(message_id)
            if remote and isinstance(remote.item, llm.ChatMessage):
                remote.item.content = [audio_transcript] if audio_transcript else []

    async def aclose(self) -> None:
        self._closing = True
        self._zoom_configured.set()  # let a session that never started exit
        # a failure that ended the session was already reported as an "error" event
        with contextlib.suppress(Exception):
            await super().aclose()
        if self._zoom_abort_task:
            await utils.aio.cancel_and_wait(self._zoom_abort_task)

    # ------------------------------------------------------------------ outgoing

    def send_event(self, event: Any) -> None:
        payload: dict[str, Any] = (
            event.model_dump(by_alias=True, exclude_unset=True, exclude_defaults=False)
            if isinstance(event, BaseModel)
            else dict(event)
        )
        event_type = payload.get("type")
        if event_type == "session.update" or event_type in _DROPPED_EVENTS:
            # session updates are built whole by _zoom_session_update and sent directly
            return
        if event_type == "conversation.item.create":
            item = payload.get("item")
            if not isinstance(item, dict) or item.get("type") != "function_call_output":
                return
        if event_type == "response.cancel":
            response_id = payload.get("response_id")
            if not response_id:
                # The engine sends this when a reply is cancelled before its response exists;
                # a bare cancel would hit whatever response is active. The response is
                # cancelled by id when it arrives instead.
                return
            if response_id == self._zoom_active_response:
                if self._zoom_active_cancelled:
                    return  # already being cancelled, by us or by Zoom's barge-in
                self._zoom_active_cancelled = True

        # anything else needs the session: start it with its configuration first
        self._zoom_start()
        if event_type == "response.create":
            self._zoom_queue.append(payload.get("event_id") or utils.shortuuid("response_create_"))
            self._zoom_pump_replies()
            return
        super().send_event(payload)

    def _append_audio(self, frames: list[rtc.AudioFrame]) -> None:
        for f in frames:
            self.send_event(
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

    def _zoom_start(self) -> None:
        if self._zoom_configured.is_set():
            return
        self._zoom_configured.set()
        # the connection's first client event
        _engine.RealtimeSession.send_event(self, self._zoom_session_update(first=True))

    def _zoom_send_session_update(self) -> None:
        if self._zoom_configured.is_set():
            _engine.RealtimeSession.send_event(self, self._zoom_session_update(first=False))

    def _create_session_update_event(self) -> dict[str, Any]:
        # A connection's first update: unused at start-up (the session starts in _zoom_start)
        # and sent first on a reconnection.
        if self._zoom_conversation_started:
            # the new server session knows nothing: carry the conversation over
            self._zoom_history = _render_history(self.chat_ctx.items)
        return self._zoom_session_update(first=True)

    def _create_tools_update_event(self, tools: list[llm.Tool]) -> dict[str, Any]:
        # A reconnection sends this right after the session update, which already carries
        # the tools; the socket drops it as unchanged.
        return self._zoom_session_update(first=False)

    def _create_update_chat_ctx_events(self, chat_ctx: llm.ChatContext) -> list[Any]:
        # Only a reconnection gets here (update_chat_ctx and truncate are overridden). The new
        # server session has the conversation in its instructions; keep it as the history.
        previous: str | None = None
        for item in chat_ctx.items:
            with contextlib.suppress(ValueError):
                self._remote_chat_ctx.insert(previous, item)
                previous = item.id
        return []

    def _zoom_session_update(self, *, first: bool) -> dict[str, Any]:
        zopts, opts = self._zoom_opts, self._opts
        if first:
            self._zoom_sent_optional.clear()

        # Optional settings are left out of a connection's first update so Zoom's defaults
        # apply. A session.update only changes the fields it carries, so once a setting has
        # been sent on this connection, clearing it has to send its reset value.
        def _optional(key: str, value: Any, reset: Any) -> Any:
            if value is not None:
                self._zoom_sent_optional.add(key)
                return value
            return reset if key in self._zoom_sent_optional else None

        listen: dict[str, Any] = {"format": {"type": "pcm16", "rate": SAMPLE_RATE}}
        phrases = _optional("vocabulary", zopts.vocabulary or None, [])
        if phrases is not None:
            listen["transcription"] = {"vocabulary": {"phrases": phrases}}

        provider: dict[str, Any] = {"type": zopts.llm_provider, "model": opts.model}
        if zopts.provider_headers:
            provider["headers"] = zopts.provider_headers
        if zopts.provider_metadata:
            provider["metadata"] = zopts.provider_metadata
        think: dict[str, Any] = {"provider": provider, "tools": self._zoom_tools}
        instructions = "\n\n".join(p for p in (self._instructions, self._zoom_history) if p)
        optional_think = {
            "instructions": _optional("instructions", instructions or None, ""),
            "tool_choice": _optional(
                "tool_choice",
                _to_zoom_tool_choice(opts.tool_choice) if opts.tool_choice is not None else None,
                "auto",
            ),
            "response_token_limit": _optional(
                "response_token_limit", zopts.response_token_limit, "inf"
            ),
        }
        think.update({k: v for k, v in optional_think.items() if v is not None})
        if first and zopts.reasoning_effort:
            # "Set the reasoning level in your first session.update"
            think["reasoning"] = {"effort": zopts.reasoning_effort}

        speak: dict[str, Any] = {"format": {"type": "pcm16"}, "voice": opts.voice}
        return {
            "type": "session.update",
            "session": {"listen": listen, "think": think, "speak": speak},
        }

    def _zoom_set_tools(self, tools: list[llm.Tool]) -> None:
        zoom_tools: list[dict[str, Any]] = []
        retained: list[llm.Tool] = []
        for tool in tools:
            if isinstance(tool, llm.FunctionTool):
                desc = llm.utils.build_legacy_openai_schema(tool, internally_tagged=True)
            elif isinstance(tool, llm.RawFunctionTool):
                desc = dict(tool.info.raw_schema)
                desc.pop("meta", None)
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

    def _zoom_apply_chat_ctx(self, chat_ctx: llm.ChatContext, *, send_update: bool) -> None:
        if not self._zoom_conversation_started:
            # Nothing has been said in this session yet, so the history can still be given
            # as part of the instructions. The given context replaces the history whole,
            # so removed and edited items are dropped too.
            history: list[llm.ChatItem] = [
                item
                for item in chat_ctx.items
                if item.type in ("message", "function_call", "function_call_output")
                and not (item.type == "message" and item.role in ("system", "developer"))
            ]
            self._remote_chat_ctx = RemoteChatContext()
            previous: str | None = None
            for item in history:
                self._remote_chat_ctx.insert(previous, item)
                previous = item.id
            rendered = _render_history(history)
            if rendered != self._zoom_history:
                self._zoom_history = rendered
                if send_update:
                    self._zoom_send_session_update()
            return

        # Mid-conversation only function results can be added. Zoom gives the item its own
        # id, so a result is known by its call_id.
        known = self._zoom_sent_outputs | {
            item.call_id for item in self.chat_ctx.items if item.type == "function_call_output"
        }
        for item in chat_ctx.items:
            if self._remote_chat_ctx.get(item.id) is not None:
                continue
            if item.type == "function_call_output":
                if item.call_id in known:
                    continue
                self._zoom_sent_outputs.add(item.call_id)
                self.send_event(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": item.output,
                        },
                    }
                )
            elif item.type == "message" and item.role in ("system", "developer"):
                continue
            elif item.type in ("message", "function_call"):
                self._zoom_warn_once(
                    "chat_ctx",
                    "Zoom Voice Agent can't add messages to a running conversation; only "
                    "function results are sent. Use instructions to give the agent context.",
                )

    def _zoom_pump_replies(self) -> None:
        """Send the next queued response.create once no response is active."""
        if self._zoom_active_response or self._zoom_inflight:
            return
        while self._zoom_queue:
            event_id = self._zoom_queue.popleft()
            if event_id in self._discarded_event_ids:
                # cancelled or timed out before it went out
                self._discarded_event_ids.discard(event_id)
                continue
            if event_id not in self._response_created_futures:
                continue
            self._zoom_inflight.append(event_id)
            self._zoom_start_reply_timer(event_id)
            _engine.RealtimeSession.send_event(
                self, {"type": "response.create", "event_id": event_id}
            )
            return

    def _zoom_start_reply_timer(self, event_id: str) -> None:
        def _on_timeout() -> None:
            self._zoom_reply_timers.pop(event_id, None)
            fut = self._response_created_futures.pop(event_id, None)
            if fut and not fut.done():
                # discard the response if Zoom still creates it after the timeout
                self._discarded_event_ids.add(event_id)
                fut.set_exception(llm.RealtimeError("generate_reply timed out."))
            if event_id in self._zoom_inflight:
                # Zoom answered neither with a response nor an error. Nothing tells a late
                # response to this request from one to the next (Zoom echoes no client id),
                # so the next request can't go out on this connection: start a new one. The
                # reconnect fails the queued requests and replays the conversation.
                self._zoom_reconnect(
                    f"Zoom Voice Agent didn't answer a reply request within {_REPLY_TIMEOUT}s"
                )

        self._zoom_stop_reply_timer(event_id)
        self._zoom_reply_timers[event_id] = asyncio.get_running_loop().call_later(
            _REPLY_TIMEOUT, _on_timeout
        )

    def _zoom_stop_reply_timer(self, event_id: str) -> None:
        if handle := self._zoom_reply_timers.pop(event_id, None):
            handle.cancel()

    def _zoom_reconnect(self, reason: str) -> None:
        ws = self._zoom_ws
        if (
            self._closing
            or ws is None
            or (self._zoom_abort_task and not self._zoom_abort_task.done())
        ):
            return
        logger.warning(f"{reason}; reconnecting")
        self._zoom_abort_task = asyncio.create_task(ws.abort(reason))

    # ------------------------------------------------------------------ connection

    async def _main_task(self) -> None:
        await self._zoom_configured.wait()
        if self._closing:
            return
        await super()._main_task()

    async def _create_ws_conn(self) -> aiohttp.ClientWebSocketResponse:
        # The engine starts counting retries again after every handshake, so a server that
        # accepts and closes straight away would be retried forever. Count the connections
        # that never got as far as session.updated instead.
        if self._zoom_unconfigured_conns > self._opts.conn_options.max_retry:
            raise APIConnectionError(
                f"Zoom Voice Agent closed {self._zoom_unconfigured_conns} connections in a row "
                "before accepting the session",
                retryable=False,
            )
        tracking_id = _tracking_id()
        headers = {
            "Authorization": f"Bearer {self._zoom_opts.api_key}",
            "User-Agent": USER_AGENT,
            "x-zm-trackingid": tracking_id,
        }
        t0 = time.perf_counter()
        try:
            ws = await asyncio.wait_for(
                self._realtime_model._ensure_http_session().ws_connect(
                    self._zoom_opts.url, headers=headers, protocols=(SUBPROTOCOL,)
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
        self._zoom_awaiting_config_ack = True
        self._zoom_unconfigured_conns += 1
        self._zoom_ws = _ZoomSocket(ws)
        return cast(aiohttp.ClientWebSocketResponse, self._zoom_ws)

    def _zoom_on_reconnected(self, _: llm.RealtimeSessionReconnectedEvent) -> None:
        # The engine failed the pending reply requests; what was in flight on the old
        # connection is gone.
        self._zoom_queue.clear()
        self._zoom_inflight.clear()
        self._zoom_active_response = None
        self._zoom_active_cancelled = False
        # Events queued for the old connection refer to its responses and items (cancels,
        # truncations, function results, which the instructions now carry) and would be
        # refused by the new session. Keep only the caller's audio.
        queued: list[Any] = []
        with contextlib.suppress(utils.aio.channel.ChanEmpty, utils.aio.channel.ChanClosed):
            while True:
                queued.append(self._msg_ch.recv_nowait())
        for event in queued:
            if isinstance(event, dict) and event.get("type") == "input_audio_buffer.append":
                _engine.RealtimeSession.send_event(self, event)

    # ------------------------------------------------------------------ incoming

    def _zoom_on_server_event(self, event: dict[str, Any]) -> None:
        # Runs on the raw event before the engine parses it, so fixes made here are what the
        # engine sees.
        self.emit(SERVER_EVENT, event)
        event_type = event.get("type")
        if event_type == "session.updated":
            self._zoom_awaiting_config_ack = False
            self._zoom_unconfigured_conns = 0
            session = event.get("session")
            if isinstance(session, dict) and isinstance(session.get("id"), str):
                self.session_id = session["id"]
        elif event_type == "input_audio_buffer.speech_started":
            if self._zoom_active_response:
                # Zoom's barge-in cancels the active response itself
                self._zoom_active_cancelled = True
        elif event_type == "conversation.item.added":
            self._zoom_conversation_started = True
            self._zoom_fix_item_added(event)
        elif event_type == "conversation.item.input_audio_transcription.completed":
            self._zoom_conversation_started = True
        elif event_type == "response.created":
            self._zoom_conversation_started = True
            self._zoom_claim_response(event)
        elif event_type == "response.done":
            self._zoom_active_response = None
            self._zoom_active_cancelled = False
            self._zoom_pump_replies()

    def _zoom_fix_item_added(self, event: dict[str, Any]) -> None:
        # The first item of a connection has no previous_item_id, which the engine reads as
        # "insert at the head"; it follows a history carried over from before.
        if not event.get("previous_item_id") and (tail := self._remote_chat_ctx.tail_id):
            event["previous_item_id"] = tail
        # Zoom puts the user's transcript in the item and again in the transcription event,
        # which the engine appends; keep the second.
        item = event.get("item")
        if isinstance(item, dict) and item.get("role") == "user":
            for part in item.get("content") or ():
                if isinstance(part, dict) and part.get("type") == "input_audio":
                    part["transcript"] = None

    def _zoom_claim_response(self, event: dict[str, Any]) -> None:
        response = event.get("response")
        if not isinstance(response, dict):
            return
        self._zoom_active_response = response.get("id")
        self._zoom_active_cancelled = False
        # Zoom doesn't echo a client id on responses; the oldest reply request in flight is
        # the one this answers (only one is ever sent while no response is active). The
        # engine resolves the request's future by this id, or cancels the response if the
        # request was cancelled meanwhile.
        if self._zoom_inflight:
            event_id = self._zoom_inflight.popleft()
            metadata = response.get("metadata")
            response["metadata"] = {
                **(metadata if isinstance(metadata, dict) else {}),
                "client_event_id": event_id,
            }

    def _handle_response_audio_delta(self, event: Any) -> None:
        gen = self._current_generation
        if not isinstance(gen, _engine._ResponseGeneration):
            return
        msg = gen.messages.get(event.item_id)
        if msg is None:
            return
        if gen._first_token_timestamp is None:
            gen._first_token_timestamp = time.time()
        if not msg.modalities.done():
            msg.modalities.set_result(["audio", "text"])
        data = base64.b64decode(event.delta)
        data = data[: len(data) - len(data) % 2]
        if not data:
            return
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            msg.audio_ch.send_nowait(
                rtc.AudioFrame(
                    data=data,
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    samples_per_channel=len(data) // 2,
                )
            )

    def _emit_transcription_metrics(self, event: Any) -> None:
        # Zoom reports transcription as seconds of caller audio
        usage = _engine._coerce_transcription_usage(event.usage)
        seconds = getattr(usage, "seconds", None)
        if not isinstance(seconds, (int, float)):
            return
        self.emit(
            "metrics_collected",
            STTMetrics(
                request_id=event.event_id,
                timestamp=time.time(),
                duration=0.0,
                label=self._zoom_model.label,
                audio_duration=float(seconds),
                streamed=True,
                metadata=Metadata(model_name=None, model_provider=self._zoom_model.provider),
            ),
        )

    def _handle_error(self, event: Any) -> None:
        error = event.error
        message = str(
            (error.get("message") if isinstance(error, dict) else getattr(error, "message", None))
            or "unknown error"
        )
        lowered = message.lower()
        if "no active response" in lowered:
            # a response.cancel that crossed the response's own end
            logger.debug(f"Zoom Voice Agent: {message}")
            return
        if "active response" in lowered:
            # Our response.create reached the server while a response it started itself was
            # active. That response was credited to the request if it was created after the
            # request went out; otherwise ask again after its response.done.
            if self._zoom_inflight:
                event_id = self._zoom_inflight.popleft()
                if event_id in self._discarded_event_ids:
                    # cancelled meanwhile, and nothing was created for it
                    self._discarded_event_ids.discard(event_id)
                else:
                    # waiting locally again: the timeout restarts when it is resent
                    self._zoom_stop_reply_timer(event_id)
                    self._zoom_queue.appendleft(event_id)
            else:
                logger.debug("reply request answered by the already active response")
            return
        if self._zoom_awaiting_config_ack:
            # Zoom keeps the socket open after rejecting a session.update, but the agent
            # can't work with a configuration it refused (unknown voice, bad provider key...)
            raise APIError(
                f"Zoom Voice Agent rejected the session configuration: {message}",
                body=error,
                retryable=False,
            )
        # A reply request refused for another reason gets no response.created. The engine
        # fails its future by the error's event_id; it has to leave the in-flight queue too,
        # or no later request would be sent.
        rejected_id = (
            error.get("event_id") if isinstance(error, dict) else getattr(error, "event_id", None)
        )
        rejected = False
        if isinstance(rejected_id, str) and rejected_id in self._zoom_inflight:
            rejected = True
            self._zoom_inflight.remove(rejected_id)
            self._zoom_stop_reply_timer(rejected_id)
            self._discarded_event_ids.discard(rejected_id)  # nothing will be created for it
        super()._handle_error(event)
        if rejected:
            self._zoom_pump_replies()

    def _zoom_warn_once(self, key: str, message: str) -> None:
        if key not in self._zoom_warned:
            self._zoom_warned.add(key)
            logger.warning(message)


class _ZoomSocket:
    """The Voice Agent WebSocket as the engine expects it.

    Unwraps server events from their envelopes, names Zoom's function-result
    acknowledgment the way the engine expects it, and drops a session update that would
    change nothing.
    """

    def __init__(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        self._ws = ws
        self._last_session_update: str | None = None
        self._abort_reason: str | None = None

    async def abort(self, reason: str) -> None:
        """Close the connection so the engine reconnects, failing with ``reason``."""
        self._abort_reason = reason
        await self._ws.close()

    async def receive(self, timeout: float | None = None) -> aiohttp.WSMessage:
        msg = await self._ws.receive(timeout)
        if self._abort_reason is not None:
            # retryable: the engine reconnects
            raise APIConnectionError(self._abort_reason)
        if msg.type != aiohttp.WSMsgType.TEXT:
            return msg
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            data = None
        if not isinstance(data, dict):
            logger.warning("invalid JSON from Zoom Voice Agent")
            return aiohttp.WSMessage(aiohttp.WSMsgType.BINARY, b"", None)  # skipped
        event = _unwrap(data)
        if event.get("type") == "conversation.item.created":
            event["type"] = "conversation.item.added"
        return aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(event), msg.extra)

    async def send_str(self, data: str, compress: int | None = None) -> None:
        if '"session.update"' in data:
            if data == self._last_session_update:
                return
            self._last_session_update = data
        await self._ws.send_str(data, compress)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ws, name)


def _unwrap(data: dict[str, Any]) -> dict[str, Any]:
    """Server events come as ``{"type", "payload"}``; the payload is the full event."""
    payload = data.get("payload")
    if isinstance(payload, dict):
        if "type" not in payload and isinstance(data.get("type"), str):
            payload = {**payload, "type": data["type"]}
        return payload
    return data


def _effective_tool_choice(tool_choice: llm.ToolChoice | None) -> ZoomToolChoice:
    """The tool choice Zoom applies; unset is Zoom's default, ``"auto"``."""
    return _to_zoom_tool_choice(tool_choice) if tool_choice is not None else "auto"


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
