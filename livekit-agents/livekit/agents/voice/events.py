from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from enum import Enum, unique
from typing import TYPE_CHECKING, Annotated, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_serializer, model_validator
from typing_extensions import Self

from ..inference.interruption import (
    AdaptiveInterruptionDetector,
    InterruptionDetectionError,
    OverlappingSpeechEvent,
)
from ..language import LanguageCode
from ..llm import (
    LLM,
    AgentHandoff,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
    LLMError,
    RealtimeModel,
    RealtimeModelError,
)
from ..log import logger
from ..metrics import AgentMetrics, AgentSessionUsage
from ..stt import STT, STTError
from ..tts import TTS, TTSError
from .filler_scheduler import _FillerScheduler, _FillerSource
from .speech_handle import SpeechHandle

if TYPE_CHECKING:
    from .agent_activity import AgentActivity
    from .agent_session import AgentSession
    from .tool_executor import UpdatePromptArgs, _ToolExecutor


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
        self._filler_schedulers: list[_FillerScheduler] = []

        # synthesized progress-update pairs, populated whether or not an executor is attached
        self._updates: list[tuple[FunctionCall, FunctionCallOutput]] = []

        # set/cleared by the executor around the tool's lifetime
        self._executor: _ToolExecutor | None = None
        self._first_update_fut: asyncio.Future[Any] | None = None

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

    @asynccontextmanager
    async def with_filler(
        self,
        source: _FillerSource,
        *,
        delay: float = 0,
        interval: float | None = None,
        max_steps: int | None = None,
    ) -> AsyncIterator[None]:
        """Schedule filler speech while a long-running step blocks the tool.

        While the context is open, a background scheduler waits for the session to be
        continuously idle for ``delay`` seconds, then plays ``source``. With ``interval``
        set, it then sleeps that many wall-clock seconds before restarting the dwell
        wait. ``interval=None`` (default) fires at most once.

        Args:
            source: Either a string (spoken via ``session.say``), or a callable
                ``(step: int) -> SpeechHandle | str | None`` invoked at fire time with
                the iteration count. Returning ``None`` skips this fire and retries on
                the next interval; the step counter only advances when a handle is
                produced. Use ``max_steps`` to cap the total number of fires.
            delay: Continuous-idle dwell required before each fire. ``0`` = fire as
                soon as the session is next idle.
            interval: Wall-clock cooldown after each fire. ``None`` = fire at most once.
            max_steps: Maximum number of fires across the lifetime of the cm.
                ``None`` = no limit.
        """
        scheduler = _FillerScheduler(
            session=self._session,
            speech_handle=self._speech_handle,
            source=source,
            delay=delay,
            interval=interval,
            max_steps=max_steps,
        )
        self._filler_schedulers.append(scheduler)
        try:
            yield
        finally:
            await scheduler.aclose()
            self._filler_schedulers.remove(scheduler)

    @asynccontextmanager
    async def foreground(self) -> AsyncIterator[AgentActivity]:
        """Wait for idle, then hold the floor while interactive work runs.

        Use cases:

        - wrap an ``await AgentTask()`` so it doesn't race with current speech
          or another tool's queued reply
        - wrap a direct ``generate_reply`` / ``say`` for the same reason
        - group multiple interactive calls so no deferred tool reply lands between them

        On enter, drains this tool's pending deferred reply first so its speech
        plays before the floor is held — keeps chat order matching code order.
        """
        await self._drain_pending_reply()
        async with self._session._wait_for_idle_and_hold() as activity:
            yield activity

    async def update(
        self,
        message: str | Any,
        *,
        template: str | Callable[[UpdatePromptArgs], str] | None = None,
    ) -> None:
        """Push a progress update into the conversation.

        The first update releases control to the LLM with ``message`` as the tool's
        synthetic return; subsequent updates are coalesced into a deferred reply.
        Outside the voice path (e.g. ``execute_function_call``) updates are recorded
        on the result but no reply is fired.

        Args:
            message: Progress message; strings are wrapped by ``template``.
            template: Per-call override — either a ``str.format()`` template or a
                callable receiving ``UpdatePromptArgs``. Defaults to the executor's
                resolved ``update`` template (or the module default when standalone).
        """
        # update() is a deliberate agent action — reset any active filler dwell so a
        # pending filler doesn't race the real update to the speech queue
        for s in self._filler_schedulers:
            s.reset_dwell()

        # events carry the raw message, before the LLM-facing template wraps it
        raw_message = message if isinstance(message, str) else str(message)

        if isinstance(message, str):
            if template is None:
                if self._executor is not None:
                    template = self._executor._tool_options["update_template"]
                else:
                    from .tool_executor import UPDATE_TEMPLATE

                    template = UPDATE_TEMPLATE
            from .tool_executor import _render

            message = _render(
                template,
                {
                    "function_name": self.function_call.name,
                    "call_id": self.function_call.call_id,
                    "message": message,
                },
            )

        # first update keeps the original call_id
        update_step = len(self._updates)
        pair = self._make_update_pair(
            message, call_id_suffix=f"_update_{update_step}" if update_step > 0 else ""
        )
        self._updates.append(pair)

        if self._executor is None:
            return  # standalone — no executor, so no tool lifecycle to report

        self._session.emit(
            "tool_execution_updated",
            ToolExecutionUpdatedEvent(
                update=ToolCallUpdated(
                    id=pair[0].call_id,
                    call_id=self.function_call.call_id,
                    message=raw_message,
                )
            ),
        )

        assert self._first_update_fut is not None
        if not self._first_update_fut.done():
            self._first_update_fut.set_result(message)
            self._function_call.extra["__livekit_agents_tool_non_blocking"] = True
            return

        await self._executor._enqueue_reply(self, [pair[0], pair[1]])

    def _attach_executor(
        self, executor: _ToolExecutor, first_update_fut: asyncio.Future[Any]
    ) -> None:
        if self._first_update_fut is not None:
            raise ValueError("Executor already attached")
        self._executor = executor
        self._first_update_fut = first_update_fut

    def _detach_executor(self) -> None:
        self._executor = None
        self._first_update_fut = None

    async def _drain_pending_reply(self) -> None:
        """Wait for this tool's pending deferred reply to finish delivery, if any."""
        if self._executor is None:
            return
        reply_task = self._executor._reply_task
        if reply_task is None or reply_task.done():
            return
        try:
            await asyncio.shield(reply_task)
        except Exception:
            pass  # reply task's own errors aren't our concern

    def _make_update_pair(
        self, message: Any, *, call_id_suffix: str = ""
    ) -> tuple[FunctionCall, FunctionCallOutput]:
        """Synthesize a (FunctionCall, FunctionCallOutput) pair for a progress update.

        The new FunctionCall carries ``{call_id}{call_id_suffix}``; name/arguments/extra
        are copied. ``make_tool_output`` is reused so error handling matches dispatch.
        """
        from .generation import make_tool_output

        fnc_call = FunctionCall(
            call_id=f"{self.function_call.call_id}{call_id_suffix}",
            name=self.function_call.name,
            arguments=self.function_call.arguments,
            extra=dict(self.function_call.extra),
        )
        tool_output = make_tool_output(fnc_call=fnc_call, output=message, exception=None)
        # fall back to a stub when the message isn't a valid tool output (e.g. raw object)
        if tool_output.fnc_call_out is None:
            fnc_call_out = FunctionCallOutput(
                name=fnc_call.name,
                call_id=fnc_call.call_id,
                output=str(message or ""),
                is_error=False,
            )
        else:
            fnc_call_out = tool_output.fnc_call_out
        return (fnc_call, fnc_call_out)


EventTypes = Literal[
    "user_state_changed",
    "agent_state_changed",
    "user_input_transcribed",
    "conversation_item_added",
    "agent_false_interruption",
    "overlapping_speech",
    "function_tools_executed",
    "metrics_collected",
    "session_usage_updated",
    "speech_created",
    "tool_execution_updated",
    "error",
    "close",
    "debug_message",
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
    item_id: str | None = None
    """Provider-specific ID for the transcribed input item, when available."""
    speaker_id: str | None = None
    language: LanguageCode | None = None
    created_at: float = Field(default_factory=time.time)


class EotPredictionEvent(BaseModel):
    type: Literal["eot_prediction"] = "eot_prediction"
    probability: float
    threshold: float
    inference_duration: float
    """Server-side model inference time."""
    delay: float
    """End of user speech → prediction received latency (s), anchored on the
    VAD-backdated last_speaking_time."""
    created_at: float = Field(default_factory=time.time)


class _AgentBackchannelOpportunityEvent(BaseModel):
    """Internal: a window in which the agent could backchannel (a short
    acknowledgment such as "mm-hmm"), as predicted by the turn detector. Passed to
    ``AgentActivity`` only — not surfaced as a public ``AgentSession`` event yet.

    ``AgentActivity`` owns the decision of what to do with it. The end-of-turn margin
    (``end_of_turn_threshold - end_of_turn_probability``) gives a progressive risk axis:
    a large positive margin means the user is clearly still going, so riskier
    backchannels (yeah/okay/right) are safe; a small margin (or a negative one, where
    ``end_of_turn_probability >= end_of_turn_threshold`` and a reply is imminent) calls
    for safe, less ambiguous ones (hmm/uh-huh) that won't collide with the reply."""

    type: Literal["agent_backchannel_opportunity"] = "agent_backchannel_opportunity"
    probability: float
    threshold: float
    end_of_turn_probability: float
    end_of_turn_threshold: float
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
    """Deprecated: use session_usage_updated for usage tracking.
    Per-turn latency metrics are available on ChatMessage.metrics."""

    type: Literal["metrics_collected"] = "metrics_collected"
    metrics: AgentMetrics
    created_at: float = Field(default_factory=time.time)


class SessionUsageUpdatedEvent(BaseModel):
    type: Literal["session_usage_updated"] = "session_usage_updated"
    usage: AgentSessionUsage
    created_at: float = Field(default_factory=time.time)


class _TypeDiscriminator(BaseModel):
    type: Literal["unknown"] = "unknown"  # force user to use the type discriminator


class ConversationItemAddedEvent(BaseModel):
    type: Literal["conversation_item_added"] = "conversation_item_added"
    item: ChatMessage | AgentHandoff | _TypeDiscriminator
    created_at: float = Field(default_factory=time.time)


class FunctionToolsExecutedEvent(BaseModel):
    """Emitted after a batch of function tools finishes executing.

    ``function_calls`` and ``function_call_outputs`` are parallel lists: the
    output at a given index belongs to the call at the same index. When an
    output is present, its ``call_id`` matches the paired function call's
    ``call_id``. A ``None`` output means the function call did not produce a
    value that should be sent back to the LLM, such as when a tool raises
    ``StopResponse`` or returns an invalid output.
    """

    type: Literal["function_tools_executed"] = "function_tools_executed"
    function_calls: list[FunctionCall]
    function_call_outputs: list[FunctionCallOutput | None]
    created_at: float = Field(default_factory=time.time)
    _reply_required: bool = PrivateAttr(default=False)
    _handoff_required: bool = PrivateAttr(default=False)
    _response_had_audio: bool = PrivateAttr(default=False)

    def zipped(self) -> list[tuple[FunctionCall, FunctionCallOutput | None]]:
        """Return calls paired with outputs by list position."""
        return list(zip(self.function_calls, self.function_call_outputs, strict=False))

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

    @property
    def response_had_audio(self) -> bool:
        """Whether the response that triggered these function calls also produced
        audible audio output. Useful for deciding whether a post-tool reply would
        duplicate speech that was already delivered to the user."""
        return self._response_had_audio

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


class ToolCallStarted(BaseModel):
    """A function tool call was dispatched."""

    type: Literal["tool_call_started"] = "tool_call_started"
    function_call: FunctionCall


class ToolCallUpdated(BaseModel):
    """A progress update emitted via ``ctx.update()`` while a tool call runs."""

    type: Literal["tool_call_updated"] = "tool_call_updated"
    id: str
    """Entry id: ``call_id`` inline, ``{call_id}_update_N`` when deferred."""
    call_id: str
    message: str


class ToolCallEnded(BaseModel):
    """A tool call's single terminal entry."""

    type: Literal["tool_call_ended"] = "tool_call_ended"
    id: str
    """Entry id: ``call_id`` inline, ``{call_id}_final`` when deferred."""
    call_id: str
    message: str | None = None
    """Result or error text; None when there is nothing to voice."""
    status: Literal["done", "error", "cancelled"]


class ToolReplyUpdated(BaseModel):
    """Lifecycle of the deferred reply that voices buffered tool updates: ``scheduled``
    when queued, then ``completed`` / ``interrupted`` / ``skipped``. One reply may cover
    several calls; an inline first update never gets one."""

    type: Literal["tool_reply_updated"] = "tool_reply_updated"
    update_ids: list[str]
    """``ToolCallUpdated.id`` values this reply covers."""
    status: Literal["scheduled", "completed", "interrupted", "skipped"]
    speech_id: str
    """Id of the reply speech; ``speech_created`` carries its handle."""


class ToolExecutionUpdatedEvent(BaseModel):
    """One flat tool-lifecycle update. Discriminate on ``update.type``: ``tool_call_started``
    → ``tool_call_updated`` → ``tool_call_ended`` → ``tool_reply_updated``."""

    type: Literal["tool_execution_updated"] = "tool_execution_updated"
    update: Annotated[
        ToolCallStarted | ToolCallUpdated | ToolCallEnded | ToolReplyUpdated,
        Field(discriminator="type"),
    ]
    created_at: float = Field(default_factory=time.time)


class UserTurnExceededEvent(BaseModel):
    type: Literal["user_turn_exceeded"] = "user_turn_exceeded"
    transcript: str
    """Transcript from the current (uncommitted) user turn only.
    Previous turns in the accumulation window are already in the chat context."""
    accumulated_transcript: str
    """Full transcript since the start of user speaking."""
    accumulated_word_count: int
    """Total word count since the start of user speaking."""
    duration: float
    """Duration of the user turn in seconds."""
    created_at: float = Field(default_factory=time.time)


class ErrorEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["error"] = "error"
    error: LLMError | STTError | TTSError | RealtimeModelError | InterruptionDetectionError | Any
    source: LLM | STT | TTS | RealtimeModel | AdaptiveInterruptionDetector | Any
    created_at: float = Field(default_factory=time.time)

    @field_serializer("source")
    def _serialize_source(self, source: Any) -> Any:
        if isinstance(source, LLM | STT | TTS | RealtimeModel | AdaptiveInterruptionDetector):
            return {"model": source.model, "provider": source.provider}
        if isinstance(source, BaseModel):
            return source.model_dump()
        return repr(source)

    @field_serializer("error")
    def _serialize_error(self, error: Any) -> Any:
        if isinstance(error, BaseModel):
            return error.model_dump()
        return repr(error)


@unique
class CloseReason(str, Enum):
    ERROR = "error"
    JOB_SHUTDOWN = "job_shutdown"
    PARTICIPANT_DISCONNECTED = "participant_disconnected"
    USER_INITIATED = "user_initiated"
    TASK_COMPLETED = "task_completed"


class CloseEvent(BaseModel):
    type: Literal["close"] = "close"
    error: (
        LLMError | STTError | TTSError | RealtimeModelError | InterruptionDetectionError | None
    ) = None
    reason: CloseReason
    created_at: float = Field(default_factory=time.time)


AgentEvent = Annotated[
    UserInputTranscribedEvent
    | UserStateChangedEvent
    | AgentStateChangedEvent
    | AgentFalseInterruptionEvent
    | MetricsCollectedEvent
    | SessionUsageUpdatedEvent
    | ConversationItemAddedEvent
    | FunctionToolsExecutedEvent
    | SpeechCreatedEvent
    | ToolExecutionUpdatedEvent
    | ErrorEvent
    | CloseEvent
    | OverlappingSpeechEvent,
    Field(discriminator="type"),
]
