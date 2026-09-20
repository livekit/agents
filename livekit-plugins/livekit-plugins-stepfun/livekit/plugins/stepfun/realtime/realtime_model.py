from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

import aiohttp
from openai.types.beta.realtime.session import TurnDetection
from openai.types.realtime import (
    AudioTranscription,
    ConversationItemAdded,
    ConversationItemCreateEvent,
    ConversationItemDeleteEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    RealtimeAudioConfig,
    RealtimeConversationItemFunctionCall,
    RealtimeErrorEvent,
    RealtimeReasoning,
    RealtimeSessionCreateRequest,
    ResponseCreatedEvent,
    ResponseOutputItemDoneEvent,
)
from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad
from openai.types.realtime.session_update_event import SessionUpdateEvent

from livekit.agents import APIConnectionError, llm
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.plugins import openai
from livekit.plugins.openai.realtime.realtime_model import (
    _DiscardedGeneration,
    process_base_url,
)

from ..log import logger
from ..models import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_VOICE,
    VOICE_MAPPING_TO_DOMESTIC,
    VOICE_MAPPING_TO_OVERSEAS,
    StepAudioRealtimeModels,
)
from ..tools import StepFunTool

STEPFUN_DEFAULT_TURN_DETECTION = ServerVad(
    type="server_vad",
    threshold=0.9,
    prefix_padding_ms=300,
    silence_duration_ms=400,
)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _normalize_voice(voice: str, base_url: str) -> str:
    """Map voice aliases to target cluster naming (api.stepfun.com vs api.stepfun.ai)."""
    return (
        VOICE_MAPPING_TO_DOMESTIC.get(voice, voice)
        if "stepfun.com" in base_url
        else VOICE_MAPPING_TO_OVERSEAS.get(voice, voice)
    )


@dataclass
class _PendingClientItem:
    client_item_id: str
    item_type: str | None
    role: str | None = None
    call_id: str | None = None


def _normalize_tools_to_stepfun(tools: list[Any]) -> list[Any]:
    """Convert flat tool declarations to StepFun nested function schemas."""
    step_tools: list[Any] = []
    for t in tools:
        if isinstance(t, StepFunTool):
            step_tools.append(t.to_dict())
            continue
        d = t.model_dump(exclude_unset=True) if hasattr(t, "model_dump") else t
        if (
            isinstance(d, dict)
            and d.get("type") == "function"
            and "name" in d
            and "function" not in d
        ):
            step_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": d.get("name"),
                        "description": d.get("description", ""),
                        "parameters": d.get("parameters", {}),
                    },
                }
            )
        else:
            step_tools.append(d)
    return step_tools


class RealtimeModel(openai.realtime.RealtimeModel):
    """StepFun (阶跃星辰) StepAudio Realtime Speech-to-Speech Model."""

    def __init__(
        self,
        *,
        model: StepAudioRealtimeModels | str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        modalities: NotGivenOr[list[Literal["text", "audio"]]] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[AudioTranscription | None] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetection | ServerVad | None] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        reasoning: NotGivenOr[RealtimeReasoning | None] = NOT_GIVEN,
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> None:
        resolved_api_key = api_key or os.environ.get("STEPFUN_API_KEY")
        if resolved_api_key is None:
            raise ValueError(
                "The api_key option must be set either by passing api_key "
                "or by setting the STEPFUN_API_KEY environment variable"
            )

        resolved_base_url = (
            base_url if is_given(base_url) else os.environ.get("STEPFUN_BASE_URL", DEFAULT_BASE_URL)
        )
        resolved_model = model if is_given(model) else DEFAULT_MODEL
        raw_voice = voice if is_given(voice) else DEFAULT_VOICE
        resolved_voice = _normalize_voice(raw_voice, resolved_base_url)

        can_disable_turn_detection = not is_given(turn_detection)
        resolved_turn_detection = (
            turn_detection if is_given(turn_detection) else STEPFUN_DEFAULT_TURN_DETECTION
        )

        init_kwargs: dict[str, Any] = {
            "base_url": resolved_base_url,
            "model": resolved_model,
            "voice": resolved_voice,
            "api_key": resolved_api_key,
            "modalities": modalities if is_given(modalities) else ["text", "audio"],
            "turn_detection": resolved_turn_detection,
            "http_session": http_session,
            "conn_options": conn_options,
        }
        if is_given(input_audio_transcription):
            init_kwargs["input_audio_transcription"] = input_audio_transcription
        if is_given(tool_choice):
            init_kwargs["tool_choice"] = tool_choice
        if is_given(speed):
            init_kwargs["speed"] = speed
        if is_given(reasoning):
            init_kwargs["reasoning"] = reasoning
        if is_given(max_session_duration):
            init_kwargs["max_session_duration"] = max_session_duration

        super().__init__(**init_kwargs)
        self._opts.is_azure = True
        self._opts.api_version = "2024-10-01"  # Enables superclass flat-session wrapping
        self._capabilities.can_disable_turn_detection = can_disable_turn_detection
        self._capabilities.manual_function_calls = False
        self._capabilities.per_response_tool_choice = False
        self._provider_label = "StepFun StepAudio Realtime API"

    @property
    def voice(self) -> str:
        return self._opts.voice

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        **kwargs: Any,
    ) -> None:
        if is_given(voice):
            voice = _normalize_voice(voice, self._opts.base_url)
        super().update_options(voice=voice, **kwargs)

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        sess = RealtimeSession(self, turn_detection_disabled=turn_detection_disabled)
        self._sessions.add(sess)
        return sess


class RealtimeSession(openai.realtime.RealtimeSession):
    """StepFun Realtime Session with dialect error suppression and thinking event support."""

    def __init__(
        self,
        realtime_model: RealtimeModel,
        *,
        turn_detection_disabled: bool = False,
    ) -> None:
        super().__init__(realtime_model, turn_detection_disabled=turn_detection_disabled)
        self._stepfun_model: RealtimeModel = realtime_model
        self._pending_response_creates: deque[str] = deque()
        self._pending_client_items: list[_PendingClientItem] = []
        self.on("session_reconnected", self._on_session_reconnected)

    def _on_session_reconnected(self, _: Any) -> None:
        self._pending_response_creates.clear()
        self._pending_client_items.clear()

    async def aclose(self) -> None:
        self._pending_response_creates.clear()
        self._pending_client_items.clear()
        await super().aclose()

    async def _create_ws_conn(self) -> aiohttp.ClientWebSocketResponse:
        """Create WebSocket connection using standard Authorization Bearer header for StepFun."""
        headers = {
            "User-Agent": "LiveKit Agents",
            "Authorization": f"Bearer {self._opts.api_key}",
        }
        url = process_base_url(self._opts.base_url, self._opts.model, is_azure=False)
        t0 = time.perf_counter()
        try:
            ws = await asyncio.wait_for(
                self._realtime_model._ensure_http_session().ws_connect(url=url, headers=headers),
                self._opts.conn_options.timeout,
            )
            self._report_connection_acquired(time.perf_counter() - t0)
            return ws
        except (aiohttp.ClientError, asyncio.TimeoutError):
            raise APIConnectionError(
                f"{self._realtime_model._provider_label} connection failed"
            ) from None

    def _handle_conversion_item_added(self, event: ConversationItemAdded) -> None:
        """Filter out StepFun initial empty items and safely anchor untracked items to tail."""
        if getattr(event.item, "content", None) is None and getattr(event.item, "type", None) in (
            "message",
            None,
        ):
            return

        # StepFun emits initial function_call with arguments=None; placeholder prevents superclass assertion failure
        if (
            isinstance(event.item, RealtimeConversationItemFunctionCall)
            and event.item.arguments is None
        ):
            event.item.arguments = ""

        # StepFun ignores client-assigned item IDs and replaces them with random server UUIDs,
        # dropping correlation tracking for ACK futures. Remap by matching item features
        # (see https://github.com/stepfun-ai/Step-Realtime-Console/issues/13).
        if (
            event.item.id
            and event.item.id not in self._item_create_future
            and self._item_create_future
        ):
            ev_type = getattr(event.item, "type", None)
            ev_role = getattr(event.item, "role", None)
            ev_call_id = getattr(event.item, "call_id", None)
            matched = next(
                (
                    p
                    for p in self._pending_client_items
                    if p.client_item_id in self._item_create_future
                    and p.item_type == ev_type
                    and not (ev_type == "message" and p.role and ev_role and p.role != ev_role)
                    and not (
                        ev_type == "function_call_output"
                        and p.call_id
                        and ev_call_id
                        and p.call_id != ev_call_id
                    )
                ),
                None,
            )
            matched_id = (
                matched.client_item_id
                if matched
                else (
                    None
                    if self._pending_client_items
                    else next(iter(self._item_create_future.keys()))
                )
            )
            if matched:
                self._pending_client_items.remove(matched)
            if matched_id and matched_id in self._item_create_future:
                self._item_create_future[event.item.id] = self._item_create_future.pop(matched_id)

        super()._handle_conversion_item_added(event)

    def _create_tools_update_event(self, tools: list[llm.Tool]) -> dict[str, Any]:
        event = super()._create_tools_update_event(tools)
        step_tools = [tool.to_dict() for tool in tools if isinstance(tool, StepFunTool)]
        if step_tools:
            event["session"]["tools"] = event["session"].get("tools", []) + step_tools
        return event

    # Override to inspect nested schema (t["function"]["name"]), preventing local tools from being cleared
    async def update_tools(self, tools: list[llm.Tool]) -> None:
        async with self._update_fnc_ctx_lock:
            ev = self._create_tools_update_event(tools)
            self.send_event(ev)
            names = {
                t.get("name") or t.get("function", {}).get("name")
                for t in ev["session"]["tools"]
                if isinstance(t, dict) and (t.get("name") or t.get("function", {}).get("name"))
            }
            self._tools = llm.ToolContext(
                [
                    t
                    for t in tools
                    if (
                        isinstance(t, (llm.FunctionTool, llm.RawFunctionTool))
                        and t.info.name in names
                    )
                    or isinstance(t, llm.ProviderTool)
                ]
            )

    def _create_update_chat_ctx_events(
        self, chat_ctx: llm.ChatContext
    ) -> list[ConversationItemCreateEvent | ConversationItemDeleteEvent]:
        """Create chat context update events, excluding client-created function calls and normalizing root."""
        remote_ctx = self._remote_chat_ctx.to_chat_ctx()
        remote_ids = {item.id for item in remote_ctx.items}
        remote_call_ids = {
            getattr(i, "call_id", None) for i in remote_ctx.items if i.type == "function_call"
        }

        sanitized_items = [
            i
            for i in chat_ctx.items
            if not (i.type == "function_call" and i.id not in remote_ids)
            and not (
                i.type == "function_call_output"
                and i.id not in remote_ids
                and getattr(i, "call_id", None) not in remote_call_ids
            )
        ]
        events = super()._create_update_chat_ctx_events(llm.ChatContext(sanitized_items))
        for ev in events:
            if isinstance(ev, ConversationItemCreateEvent) and ev.previous_item_id == "root":
                ev.previous_item_id = None
        return events

    def send_event(self, event: Any) -> None:
        ev_type = _get(event, "type")
        if ev_type == "response.create":
            if eid := _get(event, "event_id"):
                self._pending_response_creates.append(eid)
            resp = _get(event, "response")
            if resp and getattr(resp, "tools", None):
                resp.tools = _normalize_tools_to_stepfun(resp.tools)
        elif ev_type == "conversation.item.create":
            item = _get(event, "item") or {}
            if iid := _get(item, "id"):
                self._pending_client_items.append(
                    _PendingClientItem(
                        iid, _get(item, "type"), _get(item, "role"), _get(item, "call_id")
                    )
                )

        super().send_event(event)

    def _wrap_session_update(
        self, event_id: str, session: RealtimeSessionCreateRequest
    ) -> SessionUpdateEvent | dict[str, Any]:
        """Wrap session update by delegating to superclass and applying StepFun customizations."""
        ev = super()._wrap_session_update(event_id=event_id, session=session)
        flat = ev if isinstance(ev, dict) else ev.model_dump(exclude_unset=True)
        s = flat["session"]

        # StepFun customization 1: Cluster-based voice normalization
        if "voice" in s and s["voice"] is not None:
            s["voice"] = _normalize_voice(str(s["voice"]), self._opts.base_url)
            self._opts.voice = s["voice"]

        # StepFun customization 2: Nested function schemas for tools
        if "tools" in s and s["tools"] is not None:
            s["tools"] = _normalize_tools_to_stepfun(s["tools"])

        # StepFun customization 3: Server VAD energy threshold & explicit null mapping
        audio_inp = session.audio.input if isinstance(session.audio, RealtimeAudioConfig) else None
        if (
            audio_inp
            and "turn_detection" in audio_inp.model_fields_set
            and audio_inp.turn_detection is None
        ) or (
            "turn_detection" in session.model_fields_set
            and getattr(session, "turn_detection", None) is None
        ):
            s["turn_detection"] = None
        elif "turn_detection" in s and isinstance(s["turn_detection"], dict):
            td = s["turn_detection"]
            if td.get("type") == "server_vad":
                thresh = td.pop("threshold", None)
                td["energy_awakeness_threshold"] = (
                    int(thresh * 5000) if thresh is not None else 4500
                )

        # StepFun customization 4: Explicit null for transcription and reasoning partial updates
        if (
            audio_inp
            and "transcription" in audio_inp.model_fields_set
            and audio_inp.transcription is None
        ) or (
            "input_audio_transcription" in session.model_fields_set
            and getattr(session, "input_audio_transcription", None) is None
        ):
            s["input_audio_transcription"] = None

        if "reasoning" in session.model_fields_set and session.reasoning is None:
            s["reasoning"] = None

        return flat

    def _handle_conversion_item_input_audio_transcription_delta(
        self, event: ConversationItemInputAudioTranscriptionDeltaEvent
    ) -> None:
        """Handle cumulative input audio transcription delta from StepFun."""
        if not event.delta:
            return
        content_index = event.content_index or 0
        by_index = self._input_transcript_accumulators.setdefault(event.item_id, {})
        if by_index.get(content_index) == event.delta:
            return
        by_index[content_index] = event.delta
        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=event.item_id, transcript=event.delta, is_final=False
            ),
        )

    def _handle_response_created(self, event: ResponseCreatedEvent) -> None:
        """Handle response.created from StepFun, correlating metadata-free responses."""
        if not event.response.metadata:
            eid = (
                self._pending_response_creates.popleft()
                if self._pending_response_creates
                else next(iter(self._response_created_futures.keys()), None)
            )
            if eid:
                event.response.metadata = {"client_event_id": eid}
        super()._handle_response_created(event)

    def _handle_error(self, event: RealtimeErrorEvent) -> None:
        """Filter out benign protocol divergence errors from StepFun."""
        # Purge rejected request ID from pending queues to avoid miscorrelating subsequent replies
        if event_id := event.error.event_id:
            with contextlib.suppress(ValueError):
                self._pending_response_creates.remove(event_id)
            self._pending_client_items = [
                it for it in self._pending_client_items if it.client_item_id != event_id
            ]

        # Prioritize error.type: only inspect benign cancel races for client request errors
        err = event.error
        if err.type == "invalid_request_error":
            msg = (err.message or "").lower()
            if "no ongoing response" in msg or "no active response" in msg:
                for eid in list(self._pending_response_creates):
                    if eid in self._discarded_event_ids:
                        self._pending_response_creates.remove(eid)
                        break
                logger.debug(
                    "Ignored benign StepFun cancel race error", extra={"lk.pii.error": err.message}
                )
                return

        super()._handle_error(event)

    def interrupt(self) -> None:
        """Interrupt active generation and mark it discarded so trailing packets don't throw AssertionError."""
        super().interrupt()
        if self._current_generation is not None:
            self._close_current_generation(reason="interruption")
            self._current_generation = _DiscardedGeneration()

    def _handle_response_output_item_done(self, event: ResponseOutputItemDoneEvent) -> None:
        if getattr(event.item, "type", None) == "function_call":
            call_id, item_id = getattr(event.item, "call_id", None), getattr(event.item, "id", None)
            if item_id and call_id:
                lk_fnc = llm.FunctionCall(
                    id=item_id,
                    call_id=call_id,
                    name=getattr(event.item, "name", "") or "",
                    arguments=getattr(event.item, "arguments", "") or "",
                )
                if node := self._remote_chat_ctx.get(item_id):
                    node.item = lk_fnc
                else:
                    self._remote_chat_ctx.insert(self._remote_chat_ctx.tail_id, lk_fnc)

        if self._current_generation is not None:
            super()._handle_response_output_item_done(event)
