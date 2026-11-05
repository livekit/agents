# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/events.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

import time
from dataclasses import dataclass, field
from inspect import get_annotations
from typing import Annotated, Any, Literal, Self, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from livekit.agents.inference.interruption import (
    AdaptiveInterruptionDetector,
    InterruptionDetectionError,
)
from livekit.agents.llm import AgentHandoff, RealtimeModel, RealtimeModelError
from livekit.agents.stt import STT, STTError
from livekit.agents.tts import TTS, TTSError
from livekit.agents.voice.io import SpeechHandle
from livekit.agents.voice.room_io import AudioProcessingError

from .. import _bindings
from .chat_items import ChatMessage, FunctionCall, FunctionCallOutput
from .events import CloseReason, ToolCallEnded, ToolCallUpdated, ToolReplyUpdated
from .llm import LLM, LLMError


@dataclass(frozen=True, kw_only=True, slots=True)
class SpeechCreatedEvent:
    __pydantic_config__ = ConfigDict(arbitrary_types_allowed=True)

    user_initiated: bool
    """True if the speech was created using public methods like `say` or `generate_reply`"""
    source: Literal['say', 'generate_reply']
    """Source indicating how the speech handle was created"""
    speech_handle: Annotated[SpeechHandle, Field(exclude=True)] = field(repr=False)
    """The speech handle that was created"""
    created_at: float = field(default_factory=time.time)
    type: Literal["speech_created"] = "speech_created"


@dataclass(frozen=True, kw_only=True, slots=True)
class ToolCallStarted:
    """A function tool call was dispatched."""

    function_call: FunctionCall
    type: Literal["tool_call_started"] = "tool_call_started"


ToolUpdate: TypeAlias = ToolCallStarted | ToolCallUpdated | ToolCallEnded | ToolReplyUpdated
"""The public tool updates retain validated calls and their typed lifecycle records."""


@dataclass(frozen=True, kw_only=True, slots=True)
class ToolExecutionUpdatedEvent:
    """One flat tool-lifecycle update. Discriminate on ``update.type``: ``tool_call_started``
        → ``tool_call_updated`` → ``tool_call_ended`` → ``tool_reply_updated``.
    """

    update: ToolUpdate
    created_at: float = field(default_factory=time.time)
    type: Literal["tool_execution_updated"] = "tool_execution_updated"


@dataclass(frozen=True, kw_only=True, slots=True)
class _TypeDiscriminator:
    type: Literal["unknown"] = "unknown"


class ConversationItemAddedEvent(BaseModel):
    type: Literal["conversation_item_added"] = "conversation_item_added"
    item: ChatMessage | AgentHandoff | _TypeDiscriminator
    created_at: float = Field(default_factory=time.time)


class CloseEvent(BaseModel):
    type: Literal["close"] = "close"
    error: LLMError | STTError | TTSError | RealtimeModelError | InterruptionDetectionError | None = None
    reason: CloseReason
    created_at: float = Field(default_factory=time.time)


class FunctionToolsExecutedEvent(BaseModel):
    """Emitted after a batch of function tools finishes executing.

        ``function_calls`` and ``function_call_outputs`` are parallel lists: the
        output at a given index belongs to the call at the same index and carries
        the same ``call_id``. Every call has one output, even one whose tool raised
        ``StopResponse``; such an output asks for no reply with ``reply_required``.
    """

    type: Literal["function_tools_executed"] = "function_tools_executed"
    function_calls: list[FunctionCall]
    function_call_outputs: list[FunctionCallOutput]
    created_at: float = Field(default_factory=time.time)
    _handoff_required: bool = PrivateAttr(default_factory=bool)

    @model_validator(mode="after")
    def verify_lists_length(self) -> Self:
        return _bindings.verify_lists_length(self)

    def zipped(self) -> list[tuple[FunctionCall, FunctionCallOutput]]:
        return _bindings.function_tools_zipped(self)

    def cancel_tool_reply(self) -> None:
        return _bindings.function_tools_cancel_tool_reply(self)

    def cancel_agent_handoff(self) -> None:
        return _bindings.function_tools_cancel_agent_handoff(self)

    @property
    def has_tool_reply(self) -> bool:
        return _bindings.function_tools_has_tool_reply(self)

    @property
    def has_agent_handoff(self) -> bool:
        return _bindings.function_tools_has_agent_handoff(self)


class ErrorEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["error"] = "error"
    error: LLMError | STTError | TTSError | RealtimeModelError | InterruptionDetectionError | AudioProcessingError | Any
    source: LLM | STT | TTS | RealtimeModel | AdaptiveInterruptionDetector | Any
    created_at: float = Field(default_factory=time.time)

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> object:
        from .. import _bindings

        return _bindings.session_error_serialize(self, handler)


SpeechCreatedEvent.__annotations__ = get_annotations(SpeechCreatedEvent, eval_str=True)
SpeechCreatedEvent.__module__ = "livekit.agents.voice.events"
ToolCallStarted.__annotations__ = get_annotations(ToolCallStarted, eval_str=True)
ToolCallStarted.__module__ = "livekit.agents.voice.events"
ToolExecutionUpdatedEvent.__annotations__ = get_annotations(ToolExecutionUpdatedEvent, eval_str=True)
ToolExecutionUpdatedEvent.__module__ = "livekit.agents.voice.events"
_TypeDiscriminator.__annotations__ = get_annotations(_TypeDiscriminator, eval_str=True)
_TypeDiscriminator.__module__ = "livekit.agents.voice.events"
ConversationItemAddedEvent.__module__ = "livekit.agents.voice.events"
CloseEvent.__module__ = "livekit.agents.voice.events"
FunctionToolsExecutedEvent.__module__ = "livekit.agents.voice.events"
ErrorEvent.__module__ = "livekit.agents.voice.events"
