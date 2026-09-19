from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

import aiohttp
import numpy as np
from openai.types.beta.realtime.session import TurnDetection
from openai.types.realtime import (
    AudioTranscription,
    ConversationItemAdded,
    ConversationItemCreateEvent,
    ConversationItemDeleteEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    RealtimeAudioConfig,
    RealtimeAudioConfigInput,
    RealtimeConversationItemFunctionCall,
    RealtimeErrorEvent,
    RealtimeReasoning,
    RealtimeSessionCreateRequest,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseOutputItemDoneEvent,
)
from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad
from openai.types.realtime.session_update_event import SessionUpdateEvent

from livekit import rtc
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


def _normalize_voice(voice: str, base_url: str) -> str:
    """Map voice aliases to target cluster naming (api.stepfun.com vs api.stepfun.ai)."""
    if "stepfun.com" in base_url:
        return VOICE_MAPPING_TO_DOMESTIC.get(voice, voice)
    return VOICE_MAPPING_TO_OVERSEAS.get(voice, voice)


@dataclass
class _PendingResponseCreate:
    event_id: str
    cancelled: bool = False


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
        elif isinstance(t, dict):
            if "function" in t:
                step_tools.append(t)
            elif t.get("type") == "function" and "name" in t:
                step_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t.get("name"),
                            "description": t.get("description", ""),
                            "parameters": t.get("parameters", {}),
                        },
                    }
                )
            else:
                step_tools.append(t)
        elif hasattr(t, "model_dump"):
            dumped = t.model_dump(exclude_unset=True)
            if dumped.get("type") == "function" and "name" in dumped:
                step_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": dumped.get("name"),
                            "description": dumped.get("description", ""),
                            "parameters": dumped.get("parameters", {}),
                        },
                    }
                )
            else:
                step_tools.append(dumped)
        else:
            step_tools.append(t)
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
        echo_gate_threshold: float = 2000.0,
        **kwargs: Any,
    ) -> None:
        self._echo_gate_threshold = echo_gate_threshold
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

        # Capture caller's turn detection intent before applying default ServerVad
        # so AgentActivity can auto-disable server-side VAD when client VAD is requested
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
        # StepFun uses the original OpenAI Beta event protocol names (e.g. response.audio.delta).
        # Enable Azure normalization so LiveKit maps them to response.output_audio.*.
        self._opts.is_azure = True
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
        self._speaking_active: bool = False
        self._speaking_until: float = 0.0
        self._echo_gate_threshold: float = realtime_model._echo_gate_threshold
        self._pending_response_creates: deque[_PendingResponseCreate] = deque()
        self._pending_client_items: list[_PendingClientItem] = []
        self.on("openai_server_event_received", self._on_stepfun_server_event)
        self.on("session_reconnected", self._on_session_reconnected)

    def _on_session_reconnected(self, _: Any) -> None:
        self._pending_response_creates.clear()
        self._pending_client_items.clear()

    async def aclose(self) -> None:
        self._pending_response_creates.clear()
        self._pending_client_items.clear()
        await super().aclose()

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        **kwargs: Any,
    ) -> None:
        if is_given(voice):
            voice = _normalize_voice(voice, self._opts.base_url)
        super().update_options(voice=voice, **kwargs)

    async def _create_ws_conn(self) -> aiohttp.ClientWebSocketResponse:
        """Create WebSocket connection using standard Authorization Bearer header for StepFun."""
        headers = {
            "User-Agent": "LiveKit Agents",
            "Authorization": f"Bearer {self._opts.api_key}",
        }
        url = process_base_url(
            self._opts.base_url,
            self._opts.model,
            is_azure=False,
        )
        t0 = time.perf_counter()
        try:
            ws = await asyncio.wait_for(
                self._realtime_model._ensure_http_session().ws_connect(url=url, headers=headers),
                self._opts.conn_options.timeout,
            )
            self._report_connection_acquired(time.perf_counter() - t0)
            return ws
        except aiohttp.ClientError:
            raise APIConnectionError(
                f"{self._realtime_model._provider_label} client connection error"
            ) from None
        except asyncio.TimeoutError:
            raise APIConnectionError(
                message=f"{self._realtime_model._provider_label} connection timed out",
            ) from None

    def _handle_conversion_item_added(self, event: ConversationItemAdded) -> None:
        """Filter out StepFun initial empty items and safely anchor untracked items to tail."""
        if getattr(event.item, "content", None) is None and getattr(event.item, "type", None) in (
            "message",
            None,
        ):
            return
        # StepFun emits initial function_call without arguments (arguments=None).
        # Provide an empty string placeholder so openai_item_to_livekit_item does not fail assertion
        # and the item enters _remote_chat_ctx so its call_id is tracked for tool output synchronization.
        if (
            isinstance(event.item, RealtimeConversationItemFunctionCall)
            and event.item.arguments is None
        ):
            event.item.arguments = ""
        if event.previous_item_id and not self._remote_chat_ctx.get(event.previous_item_id):
            # Pre-emptively anchor untracked item to tail to avoid noisy warning
            event.previous_item_id = self._remote_chat_ctx.tail_id

        if (
            event.item.id
            and event.item.id not in self._item_create_future
            and self._item_create_future
        ):
            # StepFun assigns server UUIDs instead of preserving client item.id.
            # Match against pending client-created items by type, role, and call_id to avoid
            # mistakenly consuming futures on concurrent server-generated items (speech, assistant, fnc_call).
            ev_type = getattr(event.item, "type", None)
            ev_role = getattr(event.item, "role", None)
            ev_call_id = getattr(event.item, "call_id", None)

            matched_client_id: str | None = None
            if self._pending_client_items:
                matched_idx = -1
                for idx, pending in enumerate(self._pending_client_items):
                    if pending.client_item_id not in self._item_create_future:
                        continue
                    if pending.item_type != ev_type:
                        continue
                    if (
                        ev_type == "message"
                        and pending.role
                        and ev_role
                        and pending.role != ev_role
                    ):
                        continue
                    if (
                        ev_type == "function_call_output"
                        and pending.call_id
                        and ev_call_id
                        and pending.call_id != ev_call_id
                    ):
                        continue
                    matched_idx = idx
                    break

                if matched_idx >= 0:
                    pending = self._pending_client_items.pop(matched_idx)
                    matched_client_id = pending.client_item_id
            else:
                # Fallback when pending client items was not tracked (e.g. direct future insertion in tests)
                matched_client_id = next(iter(self._item_create_future.keys()))

            if matched_client_id and matched_client_id in self._item_create_future:
                self._item_create_future[event.item.id] = self._item_create_future.pop(
                    matched_client_id
                )

        super()._handle_conversion_item_added(event)

    def _on_stepfun_server_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "response.thinking.delta":
            logger.debug("StepAudio thinking delta", extra={"lk.pii.thinking": event.get("delta")})
        elif event_type == "response.thinking.done":
            logger.debug("StepAudio thinking completed")
        elif event_type == "session.created":
            if sess_obj := event.get("session"):
                model_name = sess_obj.get("model")
                logger.info("StepFun session created", extra={"lk.pii.model": model_name})

    def _create_tools_update_event(self, tools: list[llm.Tool]) -> dict[str, Any]:
        event = super()._create_tools_update_event(tools)
        step_provider_tools: list[dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, StepFunTool):
                step_provider_tools.append(tool.to_dict())
        if step_provider_tools:
            event["session"]["tools"] = event["session"].get("tools", []) + step_provider_tools
        return event

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        async with self._update_fnc_ctx_lock:
            ev = self._create_tools_update_event(tools)
            self.send_event(ev)

            retained_tool_names: set[str] = set()
            for t in ev["session"]["tools"]:
                name = t.get("name") or (
                    t.get("function", {}).get("name")
                    if isinstance(t.get("function"), dict)
                    else None
                )
                if name:
                    retained_tool_names.add(name)

            retained_tools = [
                tool
                for tool in tools
                if (
                    isinstance(tool, (llm.FunctionTool, llm.RawFunctionTool))
                    and tool.info.name in retained_tool_names
                )
                or isinstance(tool, llm.ProviderTool)
            ]
            self._tools = llm.ToolContext(retained_tools)

    def _create_update_chat_ctx_events(
        self, chat_ctx: llm.ChatContext
    ) -> list[ConversationItemCreateEvent | ConversationItemDeleteEvent]:
        """Create chat context update events, excluding client-created function calls and normalizing root."""
        remote_ctx = self._remote_chat_ctx.to_chat_ctx()
        remote_ids = {item.id for item in remote_ctx.items}
        remote_call_ids = {
            getattr(item, "call_id", None)
            for item in remote_ctx.items
            if item.type == "function_call"
        }

        # Filter out client-side function_call items not already on the server before computing diff,
        # and exclude orphan function_call_outputs whose function_call is absent from remote_ctx.
        sanitized_items = [
            item
            for item in chat_ctx.items
            if not (item.type == "function_call" and item.id not in remote_ids)
            and not (
                item.type == "function_call_output"
                and item.id not in remote_ids
                and getattr(item, "call_id", None) not in remote_call_ids
            )
        ]
        sanitized_chat_ctx = llm.ChatContext(sanitized_items)

        events = super()._create_update_chat_ctx_events(sanitized_chat_ctx)
        filtered_events: list[ConversationItemCreateEvent | ConversationItemDeleteEvent] = []
        id_remap: dict[str, str | None] = {}

        for ev in events:
            if isinstance(ev, ConversationItemCreateEvent):
                item_type = getattr(ev.item, "type", None)
                item_id = getattr(ev.item, "id", None)
                # StepFun rejects client-created function_call items (item.type must be message or function_call_output).
                if item_type == "function_call":
                    if item_id:
                        id_remap[item_id] = ev.previous_item_id
                    continue

                # Re-anchor if previous_item_id was skipped or remapped
                prev_id = ev.previous_item_id
                visited: set[str] = set()
                while prev_id in id_remap and prev_id not in visited:
                    visited.add(prev_id)
                    prev_id = id_remap[prev_id]
                ev.previous_item_id = prev_id

                # StepFun rejects 'previous_item_id: root'; omit it when item is at root
                if ev.previous_item_id == "root":
                    ev.previous_item_id = None

            filtered_events.append(ev)
        return filtered_events

    def send_event(self, event: Any) -> None:
        # Track response.create event IDs in queue so late response.created without metadata
        # can be correlated back to its client_event_id for proper cancellation / discard handling.
        event_type = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)
        if event_type == "response.create":
            event_id = (
                event.get("event_id")
                if isinstance(event, dict)
                else getattr(event, "event_id", None)
            )
            if event_id:
                self._pending_response_creates.append(_PendingResponseCreate(event_id=event_id))

            # Turn-scoped tools: normalize flat OpenAI definitions to StepFun nested function format
            resp_params = (
                event.get("response")
                if isinstance(event, dict)
                else getattr(event, "response", None)
            )
            if resp_params and getattr(resp_params, "tools", None):
                resp_params.tools = _normalize_tools_to_stepfun(resp_params.tools)
        elif event_type == "response.cancel":
            for pending_create in self._pending_response_creates:
                if (
                    pending_create.event_id in self._discarded_event_ids
                    or not pending_create.cancelled
                ):
                    pending_create.cancelled = True

        # StepFun API rejects client conversation.item.create with type="function_call"
        # (item.type must be message or function_call_output).
        if event_type == "conversation.item.create":
            conv_item = (
                event.get("item", {}) if isinstance(event, dict) else getattr(event, "item", None)
            )
            item_type = (
                conv_item.get("type")
                if isinstance(conv_item, dict)
                else getattr(conv_item, "type", None)
            )
            if item_type == "function_call":
                return
            item_id = (
                conv_item.get("id")
                if isinstance(conv_item, dict)
                else getattr(conv_item, "id", None)
            )
            if item_id:
                item_role = (
                    conv_item.get("role")
                    if isinstance(conv_item, dict)
                    else getattr(conv_item, "role", None)
                )
                call_id = (
                    conv_item.get("call_id")
                    if isinstance(conv_item, dict)
                    else getattr(conv_item, "call_id", None)
                )
                self._pending_client_items.append(
                    _PendingClientItem(
                        client_item_id=item_id,
                        item_type=item_type,
                        role=item_role,
                        call_id=call_id,
                    )
                )

        super().send_event(event)

    def _wrap_session_update(
        self, event_id: str, session: RealtimeSessionCreateRequest
    ) -> SessionUpdateEvent | dict[str, Any]:
        """Flatten session config for StepFun Realtime API, preserving omission semantics for partial updates."""
        flat_session: dict[str, Any] = {}

        # Instructions
        if session.instructions is not None:
            flat_session["instructions"] = session.instructions
        elif event_id.startswith("session_update_"):
            flat_session["instructions"] = self._instructions or ""

        # Voice
        voice = None
        if isinstance(session.audio, RealtimeAudioConfig) and session.audio.output:
            voice = session.audio.output.voice
        elif hasattr(session, "voice"):
            voice = getattr(session, "voice", None)
        if voice is not None:
            flat_session["voice"] = _normalize_voice(str(voice), self._opts.base_url)
        elif event_id.startswith("session_update_"):
            flat_session["voice"] = _normalize_voice(str(self._opts.voice), self._opts.base_url)

        # Modalities
        if session.output_modalities is not None:
            if "audio" in session.output_modalities:
                flat_session["modalities"] = ["text", "audio"]
            else:
                flat_session["modalities"] = list(session.output_modalities)
            flat_session["input_audio_format"] = "pcm16"
            flat_session["output_audio_format"] = "pcm16"
        elif hasattr(session, "modalities") and session.modalities is not None:
            flat_session["modalities"] = session.modalities
        elif event_id.startswith("session_update_") or session.audio is not None:
            flat_session["modalities"] = self._opts.modalities or ["text", "audio"]
            flat_session["input_audio_format"] = "pcm16"
            flat_session["output_audio_format"] = "pcm16"

        # Tools: omit when not provided in a partial update, so configured tools are not erased
        if session.tools is not None:
            flat_session["tools"] = _normalize_tools_to_stepfun(session.tools)
        elif "tools" in session.model_fields_set:
            flat_session["tools"] = []

        # Tool choice
        if session.tool_choice is not None:
            flat_session["tool_choice"] = session.tool_choice

        # Speed
        if (
            isinstance(session.audio, RealtimeAudioConfig)
            and session.audio.output
            and session.audio.output.speed is not None
        ):
            flat_session["speed"] = session.audio.output.speed
        elif hasattr(session, "speed") and session.speed is not None:
            flat_session["speed"] = session.speed

        # Max output tokens
        if session.max_output_tokens is not None:
            flat_session["max_response_output_tokens"] = session.max_output_tokens

        # Turn detection
        has_explicit_td = False
        td = None
        if isinstance(session.audio, RealtimeAudioConfig) and isinstance(
            session.audio.input, RealtimeAudioConfigInput
        ):
            if "turn_detection" in session.audio.input.model_fields_set:
                has_explicit_td = True
                td = session.audio.input.turn_detection
        elif hasattr(session, "turn_detection"):
            if (
                hasattr(session, "model_fields_set")
                and "turn_detection" in session.model_fields_set
            ):
                has_explicit_td = True
                td = session.turn_detection
            elif session.turn_detection is not None:
                has_explicit_td = True
                td = session.turn_detection

        if not has_explicit_td and event_id.startswith("session_update_"):
            has_explicit_td = True
            td = self._opts.turn_detection

        if has_explicit_td:
            if td is not None and getattr(td, "type", None) == "server_vad":
                threshold_val = getattr(td, "threshold", None)
                if threshold_val is not None:
                    energy_threshold = int(threshold_val * 5000)
                elif (
                    hasattr(td, "energy_awakeness_threshold")
                    and getattr(td, "energy_awakeness_threshold", None) is not None
                ):
                    energy_threshold = td.energy_awakeness_threshold
                elif isinstance(td, dict) and "energy_awakeness_threshold" in td:
                    energy_threshold = td["energy_awakeness_threshold"]
                else:
                    energy_threshold = 4500

                flat_session["turn_detection"] = {
                    "type": "server_vad",
                    "silence_duration_ms": getattr(td, "silence_duration_ms", 400) or 400,
                    "prefix_padding_ms": getattr(td, "prefix_padding_ms", 300) or 300,
                    "energy_awakeness_threshold": energy_threshold,
                }
            else:
                flat_session["turn_detection"] = None

        # Input audio transcription
        has_explicit_transcription = False
        transcription = None
        if isinstance(session.audio, RealtimeAudioConfig) and isinstance(
            session.audio.input, RealtimeAudioConfigInput
        ):
            if "transcription" in session.audio.input.model_fields_set:
                has_explicit_transcription = True
                transcription = session.audio.input.transcription
        elif hasattr(session, "input_audio_transcription"):
            if (
                hasattr(session, "model_fields_set")
                and "input_audio_transcription" in session.model_fields_set
            ):
                has_explicit_transcription = True
                transcription = session.input_audio_transcription
            elif session.input_audio_transcription is not None:
                has_explicit_transcription = True
                transcription = session.input_audio_transcription

        if (
            not has_explicit_transcription
            and event_id.startswith("session_update_")
            and self._opts.input_audio_transcription is not None
        ):
            has_explicit_transcription = True
            transcription = self._opts.input_audio_transcription

        if has_explicit_transcription:
            if transcription is not None:
                if hasattr(transcription, "model_dump"):
                    flat_session["input_audio_transcription"] = transcription.model_dump(
                        exclude_unset=True
                    )
                elif isinstance(transcription, dict):
                    flat_session["input_audio_transcription"] = transcription
                else:
                    flat_session["input_audio_transcription"] = {
                        "model": getattr(transcription, "model", "whisper-1")
                    }
            else:
                flat_session["input_audio_transcription"] = None

        # Reasoning
        has_explicit_reasoning = False
        reasoning = None
        if hasattr(session, "reasoning"):
            if hasattr(session, "model_fields_set") and "reasoning" in session.model_fields_set:
                has_explicit_reasoning = True
                reasoning = session.reasoning
            elif session.reasoning is not None:
                has_explicit_reasoning = True
                reasoning = session.reasoning

        if (
            not has_explicit_reasoning
            and event_id.startswith("session_update_")
            and self._opts.reasoning is not None
        ):
            has_explicit_reasoning = True
            reasoning = self._opts.reasoning

        if has_explicit_reasoning:
            if reasoning is not None:
                if hasattr(reasoning, "model_dump"):
                    flat_session["reasoning"] = reasoning.model_dump(exclude_unset=True)
                elif isinstance(reasoning, dict):
                    flat_session["reasoning"] = reasoning
                else:
                    flat_session["reasoning"] = {"effort": getattr(reasoning, "effort", "low")}
            else:
                flat_session["reasoning"] = None

        return {
            "type": "session.update",
            "event_id": event_id,
            "session": flat_session,
        }

    def _handle_conversion_item_input_audio_transcription_delta(
        self, event: ConversationItemInputAudioTranscriptionDeltaEvent
    ) -> None:
        """Handle input audio transcription delta.

        StepFun transmits the cumulative full transcript recognized so far in each `event.delta`,
        NOT incremental token deltas. Overwrite rather than accumulate, and dedup identical frames
        to avoid redundant event flooding.
        """
        if not event.delta:
            return

        content_index = event.content_index or 0
        by_index = self._input_transcript_accumulators.setdefault(event.item_id, {})
        if by_index.get(content_index) == event.delta:
            # Skip identical transcripts repeatedly pushed by StepFun ASR while audio streams
            return
        by_index[content_index] = event.delta

        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=event.item_id,
                transcript=event.delta,
                is_final=False,
            ),
        )

    def _handle_response_created(self, event: ResponseCreatedEvent) -> None:
        """Handle response.created from StepFun.

        StepFun does not echo back `metadata.client_event_id` in response.created.
        If metadata is missing, associate the earliest pending response.create ID from the queue
        (or pending futures), ensuring cancelled or timed out generations match _discarded_event_ids.
        """
        self._speaking_active = True
        if not event.response.metadata:
            client_event_id: str | None = None
            if self._pending_response_creates:
                pending = self._pending_response_creates.popleft()
                client_event_id = pending.event_id
                if pending.cancelled:
                    self._discarded_event_ids.add(pending.event_id)
            elif self._response_created_futures:
                client_event_id = next(iter(self._response_created_futures.keys()))

            if client_event_id:
                event.response.metadata = {"client_event_id": client_event_id}

        super()._handle_response_created(event)

    def _handle_response_done(self, event: ResponseDoneEvent) -> None:
        self._speaking_active = False
        self._speaking_until = time.time() + 0.8
        super()._handle_response_done(event)

    def _handle_error(self, event: RealtimeErrorEvent) -> None:
        """Filter out benign protocol divergence errors from StepFun."""
        # Settle correlated futures first so context updates or replies do not hang until timeout
        if event_id := event.error.event_id:
            for item in list(self._pending_response_creates):
                if item.event_id == event_id:
                    self._pending_response_creates.remove(item)
                    break
            self._pending_client_items = [
                it for it in self._pending_client_items if it.client_item_id != event_id
            ]
            if fut := self._chat_ctx_event_futures.pop(event_id, None):
                if not fut.done():
                    fut.set_result(None)
            elif fut := self._response_created_futures.pop(event_id, None):
                if not fut.done():
                    fut.set_exception(llm.RealtimeError(event.error.message, code=event.error.code))

        error = event.error
        msg = (error.message or "").lower()

        # StepFun returns 'no ongoing response to cancel' when response.cancel arrives
        # while no active response is generating.
        # StepFun confirmed no active response exists; purge any cancelled pending create so
        # it doesn't poison subsequent replies.
        if "no ongoing response to cancel" in msg or "has no active response" in msg:
            for item in list(self._pending_response_creates):
                if item.cancelled or item.event_id in self._discarded_event_ids:
                    self._pending_response_creates.remove(item)
                    break
            logger.debug(
                "Ignored benign StepFun cancel error", extra={"lk.pii.error": error.message}
            )
            return

        # Suppress chat template errors on empty context if any still leak through
        if "continue_final_message" in msg:
            logger.warning(
                "Suppressed StepFun template error on empty history",
                extra={"lk.pii.error": error.message},
            )
            return

        # StepFun may reject 'previous_item_id: root'; ignore non-fatal item errors
        if "cannot find previous item" in msg and "root" in msg:
            logger.debug("Ignored StepFun root item error", extra={"lk.pii.error": error.message})
            return

        super()._handle_error(event)

    def interrupt(self) -> None:
        """Interrupt active generation and mark it discarded so trailing packets don't throw AssertionError."""
        self._speaking_active = False
        self._speaking_until = 0.0
        super().interrupt()
        if self._current_generation is not None:
            self._close_current_generation(reason="interruption")
            self._current_generation = _DiscardedGeneration()

    def _is_speaking_active(self) -> bool:
        return self._speaking_active or (time.time() < self._speaking_until)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._echo_gate_threshold > 0 and self._is_speaking_active():
            in_data = np.frombuffer(frame.data, dtype=np.int16)
            rms = float(np.sqrt(np.mean(in_data.astype(np.float32) ** 2)))
            if rms < self._echo_gate_threshold:
                # Gated: acoustic speaker echo or ambient hiss while AI is speaking
                required_bytes = frame.samples_per_channel * frame.num_channels * 2
                frame = rtc.AudioFrame(
                    data=b"\x00" * required_bytes,
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                    samples_per_channel=frame.samples_per_channel,
                )
        super().push_audio(frame)

    def _handle_response_audio_delta(self, event: ResponseAudioDeltaEvent) -> None:
        if self._current_generation is None or isinstance(
            self._current_generation, _DiscardedGeneration
        ):
            return
        super()._handle_response_audio_delta(event)

    def _handle_response_audio_done(self, event: ResponseAudioDoneEvent) -> None:
        if self._current_generation is None or isinstance(
            self._current_generation, _DiscardedGeneration
        ):
            return
        super()._handle_response_audio_done(event)

    def _handle_response_output_item_done(self, event: ResponseOutputItemDoneEvent) -> None:
        if getattr(event.item, "type", None) == "function_call":
            call_id = getattr(event.item, "call_id", None)
            item_id = getattr(event.item, "id", None)
            if item_id and call_id:
                # Upsert the completed function call in _remote_chat_ctx so its arguments
                # and call_id are registered before tool outputs are synchronized.
                lk_fnc = llm.FunctionCall(
                    id=item_id,
                    call_id=call_id,
                    name=getattr(event.item, "name", "") or "",
                    arguments=getattr(event.item, "arguments", "") or "",
                )
                if remote_node := self._remote_chat_ctx.get(item_id):
                    remote_node.item = lk_fnc
                else:
                    self._remote_chat_ctx.insert(self._remote_chat_ctx.tail_id, lk_fnc)

        if self._current_generation is None or isinstance(
            self._current_generation, _DiscardedGeneration
        ):
            return

        super()._handle_response_output_item_done(event)
