from __future__ import annotations

import asyncio
import functools
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
    MetricsCollectedEvent,
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
from ..llm import FunctionCall, FunctionCallOutput
from ..utils import shortuuid
from .speech_handle import SpeechHandle

if TYPE_CHECKING:
    from ..storage.context_store import ContextStore
    from .agent_run import AgentRun
    from .agent_session import LiveSession
    from .run_result import RunUsage
    from .tool_executor import UpdatePromptArgs

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
    "MetricsCollectedEvent",
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
    """What a tool, an instructions function or an output validator receives.

    ``ctx.context`` is the application object supplied for this invocation. On a call
    the wrapper also carries the session and speech handle; direct runs carry the run.
    """

    # private ctor
    def __init__(
        self,
        *,
        function_call: FunctionCall | None,
        session: LiveSession[Any] | None = None,
        speech_handle: SpeechHandle | None = None,
        run: AgentRun[Any] | None = None,
    ) -> None:
        self._session = session
        self._speech_handle = speech_handle
        self._function_call = function_call
        self._run = run
        self._initial_step_idx = speech_handle.num_steps - 1 if speech_handle is not None else 0
        # the synthesized progress-update pairs, read back by `execute_function_call`
        self._updates: list[tuple[FunctionCall, FunctionCallOutput]] = []
        # on a call: resolved by the first `update()`, which releases the step to the core
        self._first_update: asyncio.Future[tuple[str, str]] | None = (
            asyncio.get_running_loop().create_future() if session is not None else None
        )

    @property
    def session(self) -> LiveSession[Any]:
        """The live session, on a call.

        Raises:
            RuntimeError: in ``agent.run()``, which has no live session.
        """
        if self._session is None:
            raise RuntimeError("no live session: this RunContext belongs to agent.run()")
        return self._session

    @property
    def speech_handle(self) -> SpeechHandle:
        if self._speech_handle is None:
            raise RuntimeError("no speech handle: this RunContext belongs to agent.run()")
        return self._speech_handle

    @property
    def store(self) -> ContextStore:
        """The host's activation-scoped store for explicit field persistence."""
        if self._run is not None:
            return self._run.store
        return self.session.store

    @property
    def context(self) -> Context_T:
        """The application context supplied to this invocation."""
        if self._run is not None:
            return self._run.context  # type: ignore[no-any-return]
        return self.session.context  # type: ignore[no-any-return]

    @property
    def usage(self) -> RunUsage:
        """Token usage so far; pass it to a delegated ``run(usage=ctx.usage)``."""
        if self._run is not None:
            return self._run.usage
        from .run_result import RunUsage

        return RunUsage.from_session_usage(self.session.usage)

    @property
    def task_id(self) -> str:
        """The id of this run, also the idempotency key for outside calls."""
        if self._run is not None:
            return self._run.id
        if self._speech_handle is not None:
            return self._speech_handle.id
        return self._fallback_task_id

    @functools.cached_property
    def _fallback_task_id(self) -> str:
        return shortuuid("task_")

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
        core = self.session._core
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
    async def foreground(self) -> AsyncIterator[LiveSession[Any]]:
        """Wait for idle, then hold the floor while interactive work runs (
        a ``say``), so no other speech or deferred tool reply lands in between."""
        core = self.session._core
        hold = await core.wait_for_idle_and_hold(origin=self._origin)
        try:
            yield self.session
        finally:
            core.release_idle_hold(hold)

    @property
    def _origin(self) -> str | None:
        return None if self._function_call is None else f"tool:{self._function_call.call_id}"

    async def update(
        self,
        message: str | Any,
        *,
        template: str | Callable[[UpdatePromptArgs], str] | None = None,
    ) -> None:
        """Push a progress update into the conversation.

        The first update releases control to the LLM with ``message`` as the tool's
        synthetic return; later ones are coalesced into a deferred reply by the core.

        Args:
            message: Progress message; strings are wrapped by ``template``.
            template: A ``str.format()`` template or a callable receiving ``UpdatePromptArgs``.
        """
        from .tool_executor import UPDATE_TEMPLATE, _render

        raw = message if isinstance(message, str) else str(message)
        rendered = (
            _render(
                template or UPDATE_TEMPLATE,
                {
                    "function_name": self.function_call.name,
                    "call_id": self.function_call.call_id,
                    "message": message,
                },
            )
            if isinstance(message, str)
            else raw
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
            # agent.run(): the door that runs the turn forwards the text as progress
            from .agent_run import current_run_hooks

            hooks = current_run_hooks()
            if hooks is not None and hooks.on_update is not None:
                out = hooks.on_update(raw)
                if inspect.isawaitable(out):
                    await out
            return

        if not self._first_update.done():
            self._first_update.set_result((rendered, raw))
            return
        self.session._core.tool_update(fnc_call.call_id, "progress", message=rendered, raw=raw)


EventTypes = Literal[
    "user_state_changed",
    "agent_state_changed",
    "user_input_transcribed",
    "user_transcription_timeout",
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

SpeechCreatedEvent.model_rebuild(_types_namespace={"SpeechHandle": SpeechHandle})
