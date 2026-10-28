from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Generic, Literal

from typing_extensions import TypeVar

from .._generated.events import (
    AgentEvent,
    AgentFalseInterruptionEvent,
    AgentState,
    AgentStateChangedEvent,
    CloseEvent,
    CloseReason,
    ConversationItemAddedEvent,
    EotPredictionEvent,
    ErrorEvent,
    FunctionToolsExecutedEvent,
    SessionUsageUpdatedEvent,
    SpeechCreatedEvent,
    ToolCallEnded,
    ToolCallStarted,
    ToolCallUpdated,
    ToolExecutionUpdatedEvent,
    ToolReplyUpdated,
    UserInputTranscribedEvent,
    UserState,
    UserStateChangedEvent,
    UserTranscriptionTimeoutEvent,
    UserTurnExceededEvent,
)
from .._generated.tool_report import ToolOutcomeUpdate, ToolReport
from ..llm import FunctionCall, FunctionCallOutput
from .speech_handle import SpeechHandle

if TYPE_CHECKING:
    from ..storage.context_store import ContextStore
    from ..storage.resolve import Thread
    from .agent_run import _TextRun
    from .agent_session import VoiceSession
    from .run_result import RunUsage

__all__ = [
    "AgentEvent",
    "AgentFalseInterruptionEvent",
    "AgentState",
    "AgentStateChangedEvent",
    "CloseEvent",
    "CloseReason",
    "ConversationItemAddedEvent",
    "EotPredictionEvent",
    "ErrorEvent",
    "EventTypes",
    "FunctionToolsExecutedEvent",
    "RunContext",
    "SessionUsageUpdatedEvent",
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

Context_T = TypeVar("Context_T", default=Any)
Userdata_T = TypeVar("Userdata_T", default=Any)


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
        session: VoiceSession[Any] | None = None,
        speech_handle: SpeechHandle | None = None,
        run: _TextRun[Any] | None = None,
    ) -> None:
        self._session = session
        self._speech_handle = speech_handle
        self._function_call = function_call
        self._run = run
        # in agent.run(): the host operation running this tool body, which its updates name
        self._operation: int | None = None
        self._initial_step_idx = speech_handle.num_steps - 1 if speech_handle is not None else 0
        # the synthesized progress-update pairs, read back by `execute_function_call`
        self._updates: list[tuple[FunctionCall, FunctionCallOutput]] = []
        # on a call: resolved by the first `update()`, which releases the step to the core
        self._first_update: asyncio.Future[tuple[str, str]] | None = (
            asyncio.get_running_loop().create_future() if session is not None else None
        )

    @property
    def live_session(self) -> VoiceSession[Any] | None:
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
    def store(self) -> ContextStore:
        """The managed store for explicit ``load_context`` and ``save_context`` calls."""
        if self._run is not None:
            return self._run.store
        return self._call().store

    @property
    def thread(self) -> Thread:
        """The conversation's thread; kept in memory when no storage is configured."""
        if self._run is not None:
            thread = self._run.thread
        else:
            thread = self._call().thread
        if thread is None:
            raise RuntimeError("this run belongs to no thread")
        return thread

    @property
    def context(self) -> Context_T:
        """The application context supplied to this invocation."""
        if self._run is not None:
            return self._run.context  # type: ignore[no-any-return]
        return self._call().context  # type: ignore[no-any-return]

    @property
    def usage(self) -> RunUsage:
        """Token usage so far; pass it to a delegated ``run(usage=ctx.usage)``."""
        if self._run is not None:
            return self._run.usage
        from .run_result import RunUsage

        return RunUsage.from_session_usage(self._call().usage)

    @property
    def execution_id(self) -> str:
        """The stored conversational execution; on A2A, its task id."""
        if self._run is not None and self._run.execution_id is not None:
            return self._run.execution_id
        return self.speech_handle.id

    @property
    def task_id(self) -> str | None:
        """The A2A task id of the turn; ``None`` outside an A2A turn, such as on a call."""
        hooks = self._run.hooks if self._run is not None else None
        return hooks.task_id if hooks is not None else None

    def _call(self) -> VoiceSession[Any]:
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
        )
        try:
            yield
        finally:
            core.filler_stop(filler)

    @asynccontextmanager
    async def foreground(self) -> AsyncIterator[VoiceSession[Any]]:
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
    def _origin(self) -> str | None:
        return None if self._function_call is None else f"tool:{self._function_call.call_id}"

    async def update(self, message: str) -> None:
        """A progress message while the tool runs: a ``working`` frame on the text door, speech
        on a call (agents.md "RunContext").

        On a call, the first update releases control to the model with ``message`` as the
        tool's interim return; later ones are coalesced into a deferred reply by the core.
        """
        from .tool_executor import UPDATE_TEMPLATE, _render

        raw = message
        rendered = _render(
            UPDATE_TEMPLATE,
            {
                "function_name": self.function_call.name,
                "call_id": self.function_call.call_id,
                "message": message,
            },
        )
        fnc_call = self.function_call
        suffix = f"_update_{len(self._updates)}" if self._updates else ""
        self._updates.append(
            (
                FunctionCall(
                    call_id=f"{fnc_call.call_id}{suffix}",
                    name=fnc_call.name,
                    arguments=fnc_call.arguments,
                    extra=dict(fnc_call.extra),
                ),
                FunctionCallOutput(
                    name=fnc_call.name,
                    call_id=f"{fnc_call.call_id}{suffix}",
                    output=rendered,
                    is_error=False,
                ),
            )
        )

        if self._first_update is None:
            # agent.run(): the run streams it as progress; the door that runs the turn forwards it
            if (
                self._run is not None
                and self._run.runner is not None
                and self._operation is not None
            ):
                self._run.runner.progress(self._run.id, self._operation, raw)
            hooks = self._run.hooks if self._run is not None else None
            if hooks is not None and hooks.on_update is not None:
                out = hooks.on_update(raw)
                if inspect.isawaitable(out):
                    await out
            return

        if not self._first_update.done():
            self._first_update.set_result((rendered, raw))
            return
        update = ToolReport(outcome=ToolOutcomeUpdate(message=rendered, raw=raw))
        self._call()._core.tool_update(fnc_call.call_id, update)


EventTypes = Literal[
    "user_state_changed",
    "agent_state_changed",
    "user_input_transcribed",
    "user_transcription_timeout",
    "conversation_item_added",
    "agent_false_interruption",
    "overlapping_speech",
    "function_tools_executed",
    "session_usage_updated",
    "speech_created",
    "tool_execution_updated",
    "error",
    "close",
    "debug_message",
]

SpeechCreatedEvent.model_rebuild(_types_namespace={"SpeechHandle": SpeechHandle})
