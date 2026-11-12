from __future__ import annotations

import asyncio
import inspect

# the generated event models resolve their string annotations in this module's namespace
from datetime import timedelta as timedelta
from typing import TYPE_CHECKING, Annotated as Annotated, Any, Literal as Literal

from pydantic import Field as Field, TypeAdapter
from typing_extensions import TypeVar

from .._generated import facades_voice_events as _facades, public_voice_events as _public
from .._generated.public_voice_events import *  # noqa: F403
from .._generated.session_dispatch import AgentEvent as AgentEvent, EventTypes as EventTypes
from .._generated.session_events import (
    ToolUpdate as ToolUpdate,
    _TypeDiscriminator as _TypeDiscriminator,
)
from .._generated.tool_report import ToolOutcomeUpdate, ToolReport
from .._generated.voice_session import VoiceCommandOrigin, VoiceCommandOriginTool
from ..llm import (
    FunctionCall,
)

# `SpeechCreatedEvent` annotates its handle under this name.
from .io import SpeechHandle, SpeechHandle as HostSpeechHandle  # noqa: F401

if TYPE_CHECKING:
    from .._generated.agent_run import Invocation
    from .._generated.usage import Usage
    from ..storage import Thread
    from ..storage.context import ContextStore
    from . import Agent
    from .agent_run import _TextRun
    from .agent_session import VoiceSession

__all__ = [
    "AgentEvent",
    "EventTypes",
    "RunContext",
]
__all__ += _public.__all__

_AGENT_EVENT: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)


def event_dict(event: AgentEvent, *, json: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = _AGENT_EVENT.dump_python(event, mode="json" if json else "python")
    return out


Context_T = TypeVar("Context_T", default=Any)


class RunContext(_facades.RunContext[Context_T]):
    """What a tool, an instructions function or an output function receives.

    ``ctx.context`` is the application object supplied for this invocation. On a call it also
    carries the live session and the speech.
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
        self._invocation: Invocation | None = None
        steps = speech_handle.num_steps if speech_handle is not None else 1
        self._initial_step_idx = steps - 1 if steps else None
        # on a call: the person's reply to the delegated expert's question
        self._expert_answer: asyncio.Future[str] | None = None
        # what an agent tool's child, run in process for this call, consumed
        self._delegate_usage: Usage | None = None

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
        from . import _BindingContextVar

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
        """The latest usage snapshot of the owning scope: the invocation, the text-session turn
        or the call."""
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
        await self.speech_handle._wait_for_generation(self._initial_step_idx)

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
        """Report progress while the tool runs: a ``working`` frame on A2A, speech on a call.

        On a call, the first update releases control to the model with ``message`` as the
        tool's interim return; later ones are coalesced into a deferred reply.
        """
        from .._generated.tool_context import UPDATE_TEMPLATE
        from . import _BindingContextVar, _VoiceBinding
        from .agent_run import _template_tools, render_template

        fnc_call = self.function_call
        if self._session is None:
            if (
                self._run is not None
                and self._run.runner is not None
                and self._invocation is not None
            ):
                self._run.runner.progress(
                    self._invocation.invocation_id, self._invocation.operation, message
                )
            hooks = self._run.hooks if self._run is not None else None
            if hooks is not None and hooks.on_update is not None:
                out = hooks.on_update(message)
                if inspect.isawaitable(out):
                    await out
            return

        args = {"function_name": fnc_call.name, "call_id": fnc_call.call_id, "message": message}
        binding = _BindingContextVar.get(None)
        rendered = None
        if isinstance(binding, _VoiceBinding):
            options = binding.agent._tool_handling
            tools = _template_tools(binding.agent, binding)
            rendered = render_template(options, tools, fnc_call.name, "update_template", args)
        if rendered is None:
            rendered = UPDATE_TEMPLATE.format(**args)
        self._tool_update(ToolReport(outcome=ToolOutcomeUpdate(message=rendered, raw=message)))

    def _tool_update(self, report: ToolReport) -> None:
        invocation = self._invocation
        if invocation is None:
            raise RuntimeError("tool update outside its captured invocation")
        self._call()._core.tool_update(
            invocation.invocation_id, invocation.operation, self.function_call.call_id, report
        )
