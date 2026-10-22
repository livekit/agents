# Copyright 2023 LiveKit, Inc.
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
import dataclasses
import os
import re
import time
from collections.abc import AsyncIterable
from typing import Any, Literal, cast

import aiohttp
from openai.types.realtime import (
    AudioTranscription,
    ConversationItem,
    ConversationItemCreateEvent,
    ConversationItemDeleteEvent,
    RealtimeConversationItemAssistantMessage,
    RealtimeConversationItemFunctionCallOutput,
    RealtimeConversationItemUserMessage,
)
from openai.types.realtime.realtime_audio_input_turn_detection import SemanticVad
from openai.types.realtime.response_text_delta_event import ResponseTextDeltaEvent

from livekit import rtc
from livekit.agents import APIConnectionError, APIError, llm, utils
from livekit.agents.llm._realtime import openai as _openai
from livekit.agents.llm._realtime.openai_utils import _shorten_call_id
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from ..log import logger
from ..models import RealtimeModels, RealtimeVoices

DEFAULT_BASE_URL = "wss://api.elevenlabs.io/v1/convai/realtime"
DEFAULT_MODEL: RealtimeModels = "eleven_realtime_v1_mini"
DEFAULT_VOICE: RealtimeVoices = "brian"
API_KEY_ENV = "ELEVEN_API_KEY"

DEFAULT_TURN_DETECTION = SemanticVad(
    type="semantic_vad",
    eagerness="medium",
    create_response=True,
    interrupt_response=True,
)

KEEPALIVE_INTERVAL = 20.0
"""Seconds of client silence before a keepalive frame; the server closes idle sessions at 60 s."""

TOOL_OUTPUT_DEADLINE = 20.0
"""Seconds the server waits for a tool result; a later one would swallow the next reply."""

_UNSUPPORTED_SESSION_KEYS = ("truncation", "reasoning")
_FATAL_CLOSE_CODES = frozenset({1000, 1008, 3000, 4001, 4003})
_FATAL_IDLE_CLOSE_REASONS = ("policy", "quota", "payment", "voice")
_AUDIO_TAG_RE = re.compile(r"\[[^\s][^\]]*\]")
_REPEATED_SPACES_RE = re.compile(r"[ \t]{2,}")


class RealtimeModel(_openai.RealtimeModel):
    """Eleven Realtime speech-to-speech model, ready to pass to ``AgentSession(llm=)``."""

    def __init__(
        self,
        *,
        model: RealtimeModels | str = DEFAULT_MODEL,
        voice: RealtimeVoices | str = DEFAULT_VOICE,
        language: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: NotGivenOr[SemanticVad] = NOT_GIVEN,
        modalities: NotGivenOr[list[Literal["text", "audio"]]] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        max_duration_seconds: NotGivenOr[int] = NOT_GIVEN,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        resolved_api_key = api_key or os.environ.get(API_KEY_ENV)
        if not resolved_api_key:
            raise ValueError(
                "ElevenLabs API key is required, either as argument or set "
                f"{API_KEY_ENV} environment variable"
            )

        self._el_max_duration_seconds = max_duration_seconds
        init_kwargs: dict[str, Any] = {
            "model": model,
            "voice": voice,
            "modalities": modalities,
            "tool_choice": tool_choice,
            "base_url": base_url,
            "api_key": resolved_api_key,
            "input_audio_transcription": AudioTranscription(language=language)
            if is_given(language)
            else AudioTranscription(),
            "input_audio_noise_reduction": None,
            "turn_detection": turn_detection
            if is_given(turn_detection)
            else DEFAULT_TURN_DETECTION,
            # the server ends the session at max_duration_seconds; recycling would lose history
            "max_session_duration": None,
            "http_session": http_session,
            "conn_options": conn_options,
        }
        super().__init__(**init_kwargs)
        self._capabilities = dataclasses.replace(
            self._capabilities,
            turn_detection=True,
            can_disable_turn_detection=False,
            user_transcription=True,
            auto_tool_reply_generation=True,
            manual_function_calls=False,
            mutable_chat_context=False,
            per_response_tool_choice=False,
            supports_say=False,
        )
        self._provider_label = "ElevenLabs Realtime API"

    @property
    def provider(self) -> str:
        return "ElevenLabs"

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        if turn_detection_disabled:
            logger.warning(
                "ElevenLabs Realtime turn detection is server-side; turn_detection_disabled "
                "is ignored"
            )
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess


class RealtimeSession(_openai.RealtimeSession):
    def __init__(self, realtime_model: RealtimeModel) -> None:
        self._el_model = realtime_model
        self._el_started = False
        self._el_last_sent = time.monotonic()
        self._el_close: tuple[int, str] | None = None
        self._el_call_ids: dict[str, str] = {}
        self._el_tool_calls: dict[str, float] = {}
        self._el_tool_continuation_pending = False
        self._el_warned: set[str] = set()
        super().__init__(realtime_model)
        self.on("openai_server_event_received", self._el_on_server_event)

    def send_event(self, event: Any) -> None:
        if isinstance(event, dict):
            event_type = event.get("type")
            item = event.get("item")
            item_type = item.get("type") if isinstance(item, dict) else None
        else:
            event_type = getattr(event, "type", None)
            item_type = getattr(getattr(event, "item", None), "type", None)
        if event_type != "session.update":
            self._el_started = True
        if event_type == "conversation.item.create" and item_type == "function_call_output":
            self._el_tool_continuation_pending = True
        self._el_last_sent = time.monotonic()
        super().send_event(event)

    def _create_session_update_event(self) -> dict[str, Any]:
        payload = _dump_event(super()._create_session_update_event())
        session = _drop_none(payload["session"])
        for key in _UNSUPPORTED_SESSION_KEYS:
            session.pop(key, None)
        audio = session.get("audio", {})
        audio_input = audio.get("input", {})
        audio_input.pop("noise_reduction", None)
        if not audio_input.get("transcription"):
            audio_input.pop("transcription", None)
        audio.get("output", {}).pop("speed", None)
        if is_given(self._el_model._el_max_duration_seconds):
            session["max_duration_seconds"] = self._el_model._el_max_duration_seconds
        payload["session"] = session
        return payload

    def _create_tools_update_event(self, tools: list[llm.Tool]) -> dict[str, Any]:
        payload = super()._create_tools_update_event(tools)
        payload["session"].pop("model", None)
        return payload

    def update_options(
        self,
        *,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        **kwargs: Any,
    ) -> None:
        if self._el_started and (locked := sorted(k for k, v in kwargs.items() if is_given(v))):
            logger.warning(
                "ElevenLabs Realtime locks these options once the conversation starts; "
                "ignoring the update",
                extra={"options": locked},
            )
            kwargs = {}
        super().update_options(tool_choice=tool_choice, **kwargs)

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        if is_given(instructions):
            self._el_warn_once(
                "reply_instructions",
                "ElevenLabs Realtime ignores per-reply instructions; replies use the session's "
                "instructions",
            )
        return super().generate_reply(
            instructions=instructions, tool_choice=tool_choice, tools=tools
        )

    def say(self, text: str | AsyncIterable[str]) -> asyncio.Future[llm.GenerationCreatedEvent]:
        raise NotImplementedError("ElevenLabs Realtime does not support say(); use a TTS model")

    def push_video(self, frame: rtc.VideoFrame) -> None:
        self._el_warn_once("video", "ElevenLabs Realtime does not accept video; frames are dropped")

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if "audio" in modalities and audio_end_ms <= 0:
            # the base deletes an unplayed item, which the server does not implement
            return
        super().truncate(
            message_id=message_id,
            modalities=modalities,
            audio_end_ms=audio_end_ms,
            audio_transcript=audio_transcript,
        )

    def _create_update_chat_ctx_events(
        self, chat_ctx: llm.ChatContext
    ) -> list[ConversationItemCreateEvent | ConversationItemDeleteEvent]:
        remote_ids = {item.id for item in self._remote_chat_ctx.to_chat_ctx().items}
        return [
            ev
            for ev in super()._create_update_chat_ctx_events(chat_ctx)
            if isinstance(ev, ConversationItemCreateEvent)
            and ev.item.id not in remote_ids
            and self._el_accepts_item(ev.item)
        ]

    def _el_accepts_item(self, item: ConversationItem) -> bool:
        """Whether the server stores the item and confirms it with ``conversation.item.added``."""
        if isinstance(item, RealtimeConversationItemFunctionCallOutput):
            call_id = self._el_call_ids.get(item.call_id, item.call_id)
            emitted_at = self._el_tool_calls.pop(call_id, None)
            if emitted_at is None or time.monotonic() - emitted_at > TOOL_OUTPUT_DEADLINE:
                logger.warning(
                    "dropping a tool result the ElevenLabs Realtime server no longer waits for",
                    extra={"call_id": call_id},
                )
                return False
            item.call_id = call_id
            return True
        if isinstance(item, RealtimeConversationItemUserMessage):
            return any(part.type == "input_text" and part.text for part in item.content)
        if isinstance(item, RealtimeConversationItemAssistantMessage):
            return any(part.type == "output_text" and part.text for part in item.content)
        return False

    def _el_on_server_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "response.created":
            self._el_claim_response(event)
        elif event_type == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "function_call":
                call_id = item.get("call_id")
                if isinstance(call_id, str):
                    self._el_call_ids[_shorten_call_id(call_id)] = call_id
                    self._el_tool_calls[call_id] = time.monotonic()

    def _el_claim_response(self, event: dict[str, Any]) -> None:
        response = event.get("response")
        if not isinstance(response, dict):
            return
        metadata = response.get("metadata")
        if isinstance(metadata, dict) and metadata.get("client_event_id"):
            return
        if self._el_tool_continuation_pending:
            self._el_tool_continuation_pending = False
            return
        pending = next(iter(self._response_created_futures), None)
        if pending is None:
            return
        response["metadata"] = {
            **(metadata if isinstance(metadata, dict) else {}),
            "client_event_id": pending,
        }

    def _handle_response_audio_transcript_delta(self, event: dict[str, Any]) -> None:
        event["delta"] = _strip_audio_tags(event.get("delta") or "")
        super()._handle_response_audio_transcript_delta(event)

    def _handle_response_text_delta(self, event: ResponseTextDeltaEvent) -> None:
        event.delta = _strip_audio_tags(event.delta)
        super()._handle_response_text_delta(event)

    async def _create_ws_conn(self) -> aiohttp.ClientWebSocketResponse:
        self._el_close = None
        ws = await super()._create_ws_conn()
        return cast(aiohttp.ClientWebSocketResponse, _CloseRecordingSocket(ws, self))

    def _create_ws_url_and_headers(self) -> tuple[str, dict[str, str]]:
        headers = {"User-Agent": "LiveKit Agents"}
        if self._opts.api_key:
            headers["xi-api-key"] = self._opts.api_key
        return self._opts.base_url, headers

    async def _run_ws(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        keepalive = asyncio.create_task(self._el_keepalive(), name="_keepalive_task")
        try:
            await super()._run_ws(ws_conn)
        except APIConnectionError as e:
            if self._el_close is None:
                raise
            raise _close_error(*self._el_close) from e
        finally:
            await utils.aio.cancel_and_wait(keepalive)

    async def _el_keepalive(self) -> None:
        while True:
            remaining = KEEPALIVE_INTERVAL - (time.monotonic() - self._el_last_sent)
            if remaining > 0:
                await asyncio.sleep(remaining)
                continue
            self.send_event({"type": "input_audio_buffer.append", "audio": ""})

    def _el_warn_once(self, key: str, message: str) -> None:
        if key not in self._el_warned:
            self._el_warned.add(key)
            logger.warning(message)


class _CloseRecordingSocket:
    """Records the server's close code and reason, which the base receive loop discards."""

    def __init__(self, ws: aiohttp.ClientWebSocketResponse, session: RealtimeSession) -> None:
        self._ws = ws
        self._session = session

    async def receive(self, timeout: float | None = None) -> aiohttp.WSMessage:
        msg = await self._ws.receive(timeout)
        if msg.type == aiohttp.WSMsgType.CLOSE and isinstance(msg.data, int):
            self._session._el_close = (msg.data, msg.extra or "")
        return msg

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ws, name)


def _close_error(code: int, reason: str) -> APIError:
    lowered = reason.lower()
    retryable = code not in _FATAL_CLOSE_CODES
    if code == 1002:
        retryable = not any(word in lowered for word in _FATAL_IDLE_CLOSE_REASONS)
    elif code == 4300:
        retryable = "daily" not in lowered
    return APIError(
        f"ElevenLabs Realtime API closed the connection ({code}): {reason or 'no reason'}",
        body={"code": code, "reason": reason},
        retryable=retryable,
    )


def _dump_event(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        return event
    return cast(
        dict[str, Any],
        event.model_dump(by_alias=True, exclude_unset=True, exclude_defaults=False),
    )


def _drop_none(value: dict[str, Any]) -> dict[str, Any]:
    """Drop ``None`` fields recursively, keeping ``turn_detection: null`` (manual turns)."""
    return {
        k: _drop_none(v) if isinstance(v, dict) else v
        for k, v in value.items()
        if v is not None or k == "turn_detection"
    }


def _strip_audio_tags(text: str) -> str:
    return _REPEATED_SPACES_RE.sub(" ", _AUDIO_TAG_RE.sub("", text))
