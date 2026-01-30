from __future__ import annotations

import base64
import time
from dataclasses import asdict, is_dataclass
from enum import Enum, unique
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    Literal,
    TypeAlias,
    TypeVar,
    Union,
    get_args,
)

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, PrivateAttr, model_validator
from typing_extensions import Self

from livekit import rtc

from ..llm import (
    LLM,
    ChatChunk,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
    GenerationCreatedEvent,
    InputSpeechStartedEvent,
    InputSpeechStoppedEvent,
    InputTranscriptionCompleted,
    LLMError,
    LLMOutputEvent,
    RealtimeModel,
    RealtimeModelError,
)
from ..log import logger
from ..metrics import AgentMetrics
from ..stt import STT, SpeechEvent, STTError
from ..tts import TTS, SynthesizedAudio, TTSError
from ..types import FlushSentinel, TimedString
from ..vad import VADEvent, VADEventType
from .io import PlaybackFinishedEvent, PlaybackStartedEvent
from .room_io.types import TextInputEvent
from .run_result import AgentHandoffEvent, RunEvent
from .speech_handle import SpeechHandle

if TYPE_CHECKING:
    from .agent_session import AgentSession


Userdata_T = TypeVar("Userdata_T")


class RunContext(Generic[Userdata_T]):
    # private ctor
    def __init__(
        self,
        *,
        session: AgentSession[Userdata_T],
        speech_handle: SpeechHandle,
        function_call: FunctionCall,
    ) -> None:
        self._session = session
        self._speech_handle = speech_handle
        self._function_call = function_call

        self._initial_step_idx = speech_handle.num_steps - 1

    @property
    def session(self) -> AgentSession[Userdata_T]:
        return self._session

    @property
    def speech_handle(self) -> SpeechHandle:
        return self._speech_handle

    @property
    def function_call(self) -> FunctionCall:
        return self._function_call

    @property
    def userdata(self) -> Userdata_T:
        return self.session.userdata

    def disallow_interruptions(self) -> None:
        """Disable interruptions for this FunctionCall.

        Delegates to the SpeechHandle.allow_interruptions setter,
        which will raise a RuntimeError if the handle is already interrupted.

        Raises:
            RuntimeError: If the SpeechHandle is already interrupted.
        """
        self.speech_handle.allow_interruptions = False

    async def wait_for_playout(self) -> None:
        """Waits for the speech playout corresponding to this function call step.

        Unlike `SpeechHandle.wait_for_playout`, which waits for the full
        assistant turn to complete (including all function tools),
        this method only waits for the assistant's spoken response prior running
        this tool to finish playing."""
        await self.speech_handle._wait_for_generation(step_idx=self._initial_step_idx)


EventTypes = Literal[
    "user_state_changed",
    "agent_state_changed",
    "user_input_transcribed",
    "conversation_item_added",
    "agent_false_interruption",
    "function_tools_executed",
    "metrics_collected",
    "speech_created",
    "error",
    "close",
]

UserState = Literal["speaking", "listening", "away"]
AgentState = Literal["initializing", "idle", "listening", "thinking", "speaking"]


class UserStateChangedEvent(BaseModel):
    type: Literal["user_state_changed"] = "user_state_changed"
    old_state: UserState
    new_state: UserState
    created_at: float = Field(default_factory=time.time)


class AgentStateChangedEvent(BaseModel):
    type: Literal["agent_state_changed"] = "agent_state_changed"
    old_state: AgentState
    new_state: AgentState
    created_at: float = Field(default_factory=time.time)


class UserInputTranscribedEvent(BaseModel):
    type: Literal["user_input_transcribed"] = "user_input_transcribed"
    transcript: str
    is_final: bool
    speaker_id: str | None = None
    language: str | None = None
    created_at: float = Field(default_factory=time.time)


class AgentFalseInterruptionEvent(BaseModel):
    type: Literal["agent_false_interruption"] = "agent_false_interruption"
    resumed: bool
    """Whether the false interruption was resumed automatically."""
    created_at: float = Field(default_factory=time.time)

    # deprecated
    message: ChatMessage | None = None
    extra_instructions: str | None = None

    def __getattribute__(self, name: str) -> Any:
        if name in ["message", "extra_instructions"]:
            logger.warning(
                f"AgentFalseInterruptionEvent.{name} is deprecated, automatic resume is now supported"
            )
        return super().__getattribute__(name)


class MetricsCollectedEvent(BaseModel):
    type: Literal["metrics_collected"] = "metrics_collected"
    metrics: AgentMetrics
    created_at: float = Field(default_factory=time.time)


class _TypeDiscriminator(BaseModel):
    type: Literal["unknown"] = "unknown"  # force user to use the type discriminator


class ConversationItemAddedEvent(BaseModel):
    type: Literal["conversation_item_added"] = "conversation_item_added"
    item: ChatMessage | _TypeDiscriminator
    created_at: float = Field(default_factory=time.time)


class FunctionToolsExecutedEvent(BaseModel):
    type: Literal["function_tools_executed"] = "function_tools_executed"
    function_calls: list[FunctionCall]
    function_call_outputs: list[FunctionCallOutput | None]
    created_at: float = Field(default_factory=time.time)
    _reply_required: bool = PrivateAttr(default=False)
    _handoff_required: bool = PrivateAttr(default=False)

    def zipped(self) -> list[tuple[FunctionCall, FunctionCallOutput | None]]:
        return list(zip(self.function_calls, self.function_call_outputs))

    def cancel_tool_reply(self) -> None:
        self._reply_required = False

    def cancel_agent_handoff(self) -> None:
        self._handoff_required = False

    @property
    def has_tool_reply(self) -> bool:
        return self._reply_required

    @property
    def has_agent_handoff(self) -> bool:
        return self._handoff_required

    @model_validator(mode="after")
    def verify_lists_length(self) -> Self:
        if len(self.function_calls) != len(self.function_call_outputs):
            raise ValueError("The number of function_calls and function_call_outputs must match.")

        return self


class SpeechCreatedEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["speech_created"] = "speech_created"
    user_initiated: bool
    """True if the speech was created using public methods like `say` or `generate_reply`"""
    source: Literal["say", "generate_reply"]
    """Source indicating how the speech handle was created"""
    speech_handle: SpeechHandle = Field(..., exclude=True)
    """The speech handle that was created"""
    created_at: float = Field(default_factory=time.time)


class ErrorEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["error"] = "error"
    error: LLMError | STTError | TTSError | RealtimeModelError | Any
    source: LLM | STT | TTS | RealtimeModel | Any = Field(..., exclude=True)
    created_at: float = Field(default_factory=time.time)


@unique
class CloseReason(str, Enum):
    ERROR = "error"
    JOB_SHUTDOWN = "job_shutdown"
    PARTICIPANT_DISCONNECTED = "participant_disconnected"
    USER_INITIATED = "user_initiated"
    TASK_COMPLETED = "task_completed"


class CloseEvent(BaseModel):
    type: Literal["close"] = "close"
    error: LLMError | STTError | TTSError | RealtimeModelError | None = None
    reason: CloseReason
    created_at: float = Field(default_factory=time.time)


AgentEvent = Annotated[
    Union[
        UserInputTranscribedEvent,
        UserStateChangedEvent,
        AgentStateChangedEvent,
        AgentFalseInterruptionEvent,
        MetricsCollectedEvent,
        ConversationItemAddedEvent,
        FunctionToolsExecutedEvent,
        SpeechCreatedEvent,
        ErrorEvent,
        CloseEvent,
    ],
    Field(discriminator="type"),
]


_InternalEvent = Annotated[
    Union[
        AgentEvent,
        InputSpeechStartedEvent,
        InputSpeechStoppedEvent,
        InputTranscriptionCompleted,
        GenerationCreatedEvent,
        PlaybackFinishedEvent,
        PlaybackStartedEvent,
        TextInputEvent,
        SynthesizedAudio,
        FlushSentinel,
        LLMOutputEvent,
    ],
    Field(discriminator="type"),
]

# Allow additional events here without a Literal type discriminator
# which Pydantic doesn't support
InternalEvent: TypeAlias = Union[
    _InternalEvent,
    VADEvent,
    SpeechEvent,
]


def _serialize_audio_frame(frame: rtc.AudioFrame) -> dict:
    return {
        "sample_rate": frame.sample_rate,
        "num_channels": frame.num_channels,
        "samples_per_channel": frame.samples_per_channel,
        "data": base64.b64encode(frame.data).decode("utf-8"),
    }


def _internal_event_serializer(event: InternalEvent | RunEvent) -> dict | None:
    """Serialize an internal event to a dictionary or None.

    Args:
        event: The internal event to serialize.

    Returns:
        A dictionary representing the event or None if the event should be ignored.
    """

    if isinstance(event, AgentHandoffEvent):
        data = asdict(event)
        data["item"] = event.item.model_dump(mode="json")
        # replace agent objects with their ids
        data["old_agent"] = event.old_agent.id if event.old_agent else None
        data["new_agent"] = event.new_agent.id if event.new_agent else None
        return data

    if isinstance(event, get_args(RunEvent)):
        data = asdict(event)
        data["item"] = event.item.model_dump(mode="json")
        return data

    if isinstance(event, SynthesizedAudio):
        data = asdict(event)
        data["frame"] = _serialize_audio_frame(event.frame)
        return data

    if isinstance(event, VADEvent):
        # skip inference done events, they are too frequent and too noisy
        if event.type == VADEventType.INFERENCE_DONE:
            return None
        # remove audio frames from VAD event since we can reproduce them cheaply
        data = asdict(event)
        data["frames"] = []
        return data

    if isinstance(event, GenerationCreatedEvent):
        # skip message_stream and function_stream as they are not serializable
        return {
            "message_stream": None,
            "function_stream": None,
            "user_initiated": event.user_initiated,
            "response_id": event.response_id,
            "type": event.type,
        }

    if isinstance(event, LLMOutputEvent):
        data = asdict(event)
        if isinstance(event.data, rtc.AudioFrame):
            data["data"] = _serialize_audio_frame(event.data)
        elif isinstance(event.data, TimedString):
            data["data"] = event.data.to_dict()
        elif isinstance(event.data, str):
            data["data"] = event.data
        elif isinstance(event.data, ChatChunk):
            data["data"] = event.data.model_dump(mode="json")
        return data

    if isinstance(event, BaseModel):
        return event.model_dump(mode="json")

    if is_dataclass(event) and not isinstance(event, type):
        return asdict(event)

    logger.warning(f"Unknown internal event type: {type(event)}")
    return None


class TimedInternalEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # use a custom alias for the created_at field to avoid conflicts with the event field
    created_at: float = Field(default_factory=time.time, serialization_alias="__created_at")
    # Allow Any here to avoid RunEvent(AgentHandoffEvent/Agent) definition dependency issue
    event: Annotated[
        InternalEvent | Any,
        PlainSerializer(_internal_event_serializer),
    ]
