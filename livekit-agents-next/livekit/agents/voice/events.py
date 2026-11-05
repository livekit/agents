from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated, Any, Generic, Literal

from pydantic import (
    Field,
    TypeAdapter,
)
from typing_extensions import TypeVar

from livekit.agents._generated.session_events import (
    CloseEvent as CloseEvent,
    ConversationItemAddedEvent as ConversationItemAddedEvent,
    ErrorEvent as ErrorEvent,
    FunctionToolsExecutedEvent as FunctionToolsExecutedEvent,
    SpeechCreatedEvent as SpeechCreatedEvent,
    ToolCallStarted as ToolCallStarted,
    ToolExecutionUpdatedEvent as ToolExecutionUpdatedEvent,
    ToolUpdate as ToolUpdate,
    _TypeDiscriminator as _TypeDiscriminator,
)

from .._generated.events import (
    AgentFalseInterruptionEvent,
    AgentState,
    AgentStateChangedEvent,
    CloseReason,
    ToolCallEnded,
    ToolCallUpdated,
    ToolReplyUpdated,
    UserInputTranscribedEvent,
    UserState,
    UserStateChangedEvent,
    UserTranscriptionTimeoutEvent,
    UserTurnExceededEvent,
)
from .._generated.tool_report import ToolOutcomeUpdate, ToolReport
from .._generated.voice_session import VoiceCommandOrigin, VoiceCommandOriginTool
from ..inference.interruption import (
    OverlappingSpeechEvent,
)
from ..llm import (
    FunctionCall,
)
from .speech_handle import SpeechHandle

if TYPE_CHECKING:
    from .._generated.agent_run import Invocation
    from .._generated.usage import Usage
    from ..storage import Thread
    from ..storage.context import ContextStore
    from .agent import Agent
    from .agent_run import _TextRun
    from .agent_session import VoiceSession

__all__ = [
    "AgentEvent",
    "AgentFalseInterruptionEvent",
    "AgentState",
    "AgentStateChangedEvent",
    "CloseEvent",
    "CloseReason",
    "ConversationItemAddedEvent",
    "ErrorEvent",
    "EventTypes",
    "FunctionToolsExecutedEvent",
    "RunContext",
    "SpeechCreatedEvent",
    "ToolCallEnded",
    "ToolCallStarted",
    "ToolCallUpdated",
    "ToolExecutionUpdatedEvent",
    "ToolReplyUpdated",
    "UserInputTranscribedEvent",
    "UserState",
    "UserStateChangedEvent",
    "UserTranscriptionTimeoutEvent",
    "UserTurnExceededEvent",
]

# The public events whose fields hold host objects: written here, not generated, since no core
# record crosses as them.


AgentEvent = Annotated[
    UserInputTranscribedEvent
    | UserTranscriptionTimeoutEvent
    | UserStateChangedEvent
    | AgentStateChangedEvent
    | AgentFalseInterruptionEvent
    | ConversationItemAddedEvent
    | FunctionToolsExecutedEvent
    | SpeechCreatedEvent
    | ToolExecutionUpdatedEvent
    | ErrorEvent
    | CloseEvent
    | OverlappingSpeechEvent,
    Field(discriminator="type"),
]

_AGENT_EVENT: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)


def event_dict(event: AgentEvent, *, json: bool = False) -> dict[str, Any]:
    """``event``'s fields as a dict, whether a dataclass or a model: what the session report
    records."""
    out: dict[str, Any] = _AGENT_EVENT.dump_python(event, mode="json" if json else "python")
    return out


Context_T = TypeVar("Context_T", default=Any)


class RunContext(Generic[Context_T]):
    """What a tool, an instructions function or an output function receives (agents.md
    "RunContext").

    ``ctx.context`` is the application object supplied for this invocation. On a call it also
    carries the live session and the speech; a text run carries the run.
    """

    # private ctor
    def __init__(
        self,
        *,
        function_call: FunctionCall | None,
        session: VoiceSession[Context_T] | None = None,
        speech_handle: SpeechHandle | None = None,
        run: _TextRun[Any] | None = None,
    ) -> None:
        self._session = session
        self._speech_handle = speech_handle
        self._function_call = function_call
        self._run = run
        # Captured at the body's entry, for both direct runs and live-session updates.
        self._invocation: Invocation | None = None
        self._initial_step_idx = speech_handle.num_steps - 1 if speech_handle is not None else 0
        # on a call: the person's reply to the question this delegated call's expert asked
        self._expert_answer: asyncio.Future[str] | None = None

    @property
    def live_session(self) -> VoiceSession[Context_T] | None:
        """The ``VoiceSession`` of the call; ``None`` outside a call."""
        return self._session

    @property
    def speech_handle(self) -> SpeechHandle:
        """The speech this tool runs in, on a call.

        Raises:
            RuntimeError: outside a call.
        """
        if self._speech_handle is None:
            raise RuntimeError("no speech handle: this RunContext is not on a call")
        return self._speech_handle

    @property
    def agent(self) -> Agent[Context_T, object]:
        """The agent this invocation runs; in a session's pipeline callback, the agent holding
        the call."""
        from .agent import _BindingContextVar

        binding = _BindingContextVar.get(None)
        if binding is None:
            raise RuntimeError("no agent runs in this context")
        return binding.agent  # type: ignore[no-any-return]

    @property
    def store(self) -> ContextStore:
        """The managed store for explicit ``load_context`` and ``save_context`` calls."""
        if self._run is not None:
            return self._run.store
        return self._call().store

    @property
    def thread(self) -> Thread:
        """The conversation's thread, supplied by configured storage."""
        thread = self._run.thread if self._run is not None else self._call().thread
        if thread is None:
            raise RuntimeError("thread access requires configured SQLiteStorage or AgentDBStorage")
        return thread

    @property
    def context(self) -> Context_T:
        """The application context supplied to this invocation."""
        if self._run is not None:
            return self._run.context  # type: ignore[no-any-return]
        return self._call().context

    @property
    def usage(self) -> Usage:
        """The latest immutable usage snapshot of the owning scope: the invocation, the
        text-session turn or the call (usage.md "Scope and delegation")."""
        if self._run is not None:
            return self._run.usage
        return self._call().usage

    @property
    def execution_id(self) -> str:
        """The conversational task across human turns: on A2A its task id, on a call one id
        for the whole call."""
        if self._run is None:
            return self._call()._execution_id
        assert self._run.execution_id is not None, "a run's callbacks run under its execution"
        return self._run.execution_id

    @property
    def task_id(self) -> str | None:
        """The A2A task id of the turn; ``None`` outside an A2A turn, such as on a call."""
        hooks = self._run.hooks if self._run is not None else None
        return hooks.task_id if hooks is not None else None

    def _call(self) -> VoiceSession[Context_T]:
        if self._session is None:
            raise RuntimeError("no live session: this RunContext is not on a call")
        return self._session

    @property
    def function_call(self) -> FunctionCall:
        """The call being executed, inside a tool.

        Raises:
            RuntimeError: outside a tool, such as in an instructions function.
        """
        if self._function_call is None:
            raise RuntimeError("no function call: this RunContext is not running a tool")
        return self._function_call

    def disallow_interruptions(self) -> None:
        """Disable interruptions for this FunctionCall.

        Raises:
            RuntimeError: If the SpeechHandle is already interrupted.
        """
        self.speech_handle.allow_interruptions = False

    async def wait_for_playout(self) -> None:
        """Wait for the speech spoken before this tool ran, not the whole turn."""
        await self.speech_handle._wait_for_generation(step_idx=self._initial_step_idx)

    @asynccontextmanager
    async def with_filler(
        self,
        source: str | Callable[[int], SpeechHandle | str | None],
        *,
        delay: float = 0,
        interval: float | None = None,
        max_steps: int | None = None,
    ) -> AsyncIterator[None]:
        """Schedule filler speech while a long-running step blocks the tool.

        After ``delay`` seconds of continuous idle, ``source`` is spoken (a string) or asked
        (a callable taking the step count, returning text, a handle it spoke itself, or
        ``None`` to skip this fire); with ``interval`` set it repeats, up to ``max_steps``.
        """
        text = source if isinstance(source, str) else None
        core = self._call()._core
        filler = core.filler_start(
            self.function_call.call_id,
            self.speech_handle.id,
            delay=delay,
            interval=interval,
            max_steps=max_steps,
            text=text,
            source=None if text is not None else source,
            origin=self._origin,
        )
        try:
            yield
        finally:
            core.filler_stop(filler)

    @asynccontextmanager
    async def foreground(self) -> AsyncIterator[VoiceSession[Context_T]]:
        """Wait for idle, then hold the floor while interactive work runs (
        a ``say``), so no other speech or deferred tool reply lands in between."""
        session = self._call()
        core = session._core
        hold = await core.wait_for_idle_and_hold(origin=self._origin)
        try:
            yield session
        finally:
            core.release_idle_hold(hold)

    @property
    def _origin(self) -> VoiceCommandOrigin | None:
        if self._function_call is None or self._invocation is None:
            return None
        return VoiceCommandOriginTool(
            invocation_id=self._invocation.invocation_id,
            operation=self._invocation.operation,
            call_id=self._function_call.call_id,
        )

    async def update(self, message: str) -> None:
        """A progress message while the tool runs: a ``working`` frame on the text door, speech
        on a call (agents.md "RunContext").

        On a call, the first update releases control to the model with ``message`` as the
        tool's interim return; later ones are coalesced into a deferred reply by the core.
        """
        from .agent import _BindingContextVar, _VoiceBinding
        from .agent_host import _template_tools
        from .tool_executor import UPDATE_TEMPLATE, render_template

        raw = message
        fnc_call = self.function_call
        args = {"function_name": fnc_call.name, "call_id": fnc_call.call_id, "message": message}
        binding = _BindingContextVar.get(None)
        rendered = None
        if isinstance(binding, _VoiceBinding):
            options = binding.agent._tool_handling
            tools = _template_tools(binding.agent, binding)
            rendered = render_template(options, tools, fnc_call.name, "update_template", args)
        if rendered is None:
            rendered = UPDATE_TEMPLATE.format(**args)
        if self._session is None:
            # agent.run(): the run streams it as progress; the door that runs the turn forwards it
            if (
                self._run is not None
                and self._run.runner is not None
                and self._invocation is not None
            ):
                self._run.runner.progress(
                    self._invocation.invocation_id, self._invocation.operation, raw
                )
            hooks = self._run.hooks if self._run is not None else None
            if hooks is not None and hooks.on_update is not None:
                out = hooks.on_update(raw)
                if inspect.isawaitable(out):
                    await out
            return

        update = ToolReport(outcome=ToolOutcomeUpdate(message=rendered, raw=raw))
        self._tool_update(update)

    def _tool_update(self, report: ToolReport) -> None:
        invocation = self._invocation
        if invocation is None:
            raise RuntimeError("tool update outside its captured invocation")
        self._call()._core.tool_update(
            invocation.invocation_id, invocation.operation, self.function_call.call_id, report
        )


EventTypes = Literal[
    "user_state_changed",
    "agent_state_changed",
    "user_input_transcribed",
    "user_transcription_timeout",
    "conversation_item_added",
    "agent_false_interruption",
    "overlapping_speech",
    "function_tools_executed",
    "speech_created",
    "tool_execution_updated",
    "error",
    "close",
    "debug_message",
]
