from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from .. import llm, utils
from ..llm.async_toolset import AsyncToolset
from .events import FunctionToolsExecutedEvent, RunContext
from .generation import make_tool_output

DELEGATE_TOOL_NAME = "lk_agents_delegate"

if TYPE_CHECKING:
    from .agent_activity import AgentActivity
    from .agent_session import AgentSession


@dataclass(frozen=True)
class DelegationRequest:
    """A transport request, not an application operation or a semantic task identity."""

    id: str
    connection_id: str
    chat_ctx: llm.ChatContext
    pending_transcript: str = ""


@dataclass
class _TaskState:
    history: llm.ChatContext = field(default_factory=llm.ChatContext.empty)
    revision: int = 0


class DelegationContext:
    """One revision of backend input. Retain this object to correlate streamed results.

    New input assigned to the same task invalidates this revision's delivery, but does
    not undo or erase its tool outcomes. Use distinct task IDs for independent work.
    """

    def __init__(
        self,
        owner: ClientDelegation,
        request: DelegationRequest,
        task_id: str,
        state: _TaskState,
        send: Callable[[str, bool, Callable[[], bool]], bool],
        connected: Callable[[], bool],
    ) -> None:
        self._owner = owner
        self.request = request
        self.task_id = task_id
        self.revision = state.revision
        self._state = state
        self._send = send
        self._connected = connected
        self._run_ctx: RunContext | None = None

    @property
    def is_current(self) -> bool:
        return (
            not self._owner._closed and self._state.revision == self.revision and self._connected()
        )

    @property
    def run_context(self) -> RunContext:
        """The ordinary tool context: session, userdata, cancellation and foreground work."""
        if self._run_ctx is None:
            raise RuntimeError("delegation has not started")
        return self._run_ctx

    @property
    def chat_ctx(self) -> llm.ChatContext:
        """Snapshot of this task's retained conversation and actual tool outcomes."""
        outputs = {
            item.call_id: item
            for item in self._state.history.items
            if isinstance(item, llm.FunctionCallOutput)
        }
        items: list[llm.ChatItem] = []
        for item in self._state.history.items:
            if isinstance(item, llm.FunctionCall):
                if item.call_id in outputs:
                    items.extend([item, outputs[item.call_id]])
            elif not isinstance(item, llm.FunctionCallOutput):
                items.append(item)
        # Never hand a backend an unmatched tool call, or references it can mutate in place.
        return llm.ChatContext([item.model_copy(deep=True) for item in items])

    async def update(self, message: str, *, silent: bool = False) -> None:
        """Send a bounded progress/result chunk through the tool lifecycle.

        Relevance is checked again by the transport immediately before sending. Delivery
        means context was queued, never that the caller heard it. Silent updates supply
        quiet context; they are not a channel for private reasoning.
        """
        if self._run_ctx is None or self._run_ctx._executor is None:
            raise RuntimeError("delegation is not running")
        await self._run_ctx.update(message, silent=silent)

    async def execute_tool(self, name: str, arguments: dict[str, Any], *, call_id: str) -> Any:
        """Execute a backend tool through the existing session tool lifecycle.

        call_id identifies this model tool call, not a booking/payment operation. The
        application owns operation IDs, confirmation and reconciliation before retry.
        An obsolete revision cannot start more tools. Already-running outcomes remain
        in task history even when their parent is cancelled or disconnected.
        """
        if not self.is_current or self._run_ctx is None or self._run_ctx._executor is None:
            raise llm.ToolError("delegation is no longer current")
        tool = self._owner._backend_tools.function_tools.get(name)
        if tool is None:
            raise llm.ToolError(f"unknown delegated tool: {name}")
        call = llm.FunctionCall(
            call_id=call_id,
            name=name,
            arguments=json.dumps(arguments),
            extra={
                "delegation_id": self.request.id,
                "task_id": self.task_id,
                "revision": self.revision,
                "connection_id": self.request.connection_id,
            },
        )
        if any(
            item.id == call.id or getattr(item, "call_id", None) == call_id
            for item in self._state.history.items
        ):
            raise llm.ToolError("tool call ID already used; inspect the retained outcome")
        self._state.history.insert(call)
        child = RunContext(
            session=self._run_ctx.session,
            activity=self._run_ctx._activity,
            speech_handle=self._run_ctx.speech_handle,
            function_call=call,
        )
        child._hold_result = True
        child._is_relevant = lambda: self.is_current

        async def record(output: Any, final: bool, silent: bool) -> None:
            if final:
                result = make_tool_output(
                    fnc_call=call,
                    output=None if isinstance(output, BaseException) else output,
                    exception=output if isinstance(output, BaseException) else None,
                )
                if result.fnc_call_out is not None:
                    self._state.history.insert(result.fnc_call_out)
                    child.session.emit(
                        "function_tools_executed",
                        FunctionToolsExecutedEvent(
                            function_calls=[call], function_call_outputs=[result.fnc_call_out]
                        ),
                    )
            elif self._run_ctx is not None and self._run_ctx._executor is not None:
                await self.update(str(output), silent=silent)

        child._reply_handler = record
        try:
            output = await self._owner._executor.execute(
                tool=tool, run_ctx=child, raw_arguments=dict(arguments)
            )
        except BaseException as error:
            # Duplicate/admission errors can occur before the executor owns a task.
            # A still-running child owns its eventual outcome even if this await is cancelled.
            if call_id not in self._owner._executor._running_tasks and not any(
                isinstance(item, llm.FunctionCallOutput) and item.call_id == call_id
                for item in self._state.history.items
            ):
                await record(error, True, False)
            raise
        if not any(
            isinstance(item, llm.FunctionCallOutput) and item.call_id == call_id
            for item in self._state.history.items
        ):
            await record(output, True, False)
        return output

    async def _deliver(self, output: Any, final: bool, silent: bool) -> None:
        if isinstance(output, (asyncio.CancelledError, llm.StopResponse)):
            return
        if isinstance(output, BaseException):
            text = (
                output.message
                if isinstance(output, llm.ToolError)
                else "The delegated work failed."
            )
        elif output is None:
            return
        else:
            text = str(output)
        if self.is_current:
            self._send(text, silent, lambda: self.is_current)
        if final and self.is_current and not isinstance(output, BaseException):
            self._state.history.add_message(role="assistant", content=text)


class ClientDelegation(AsyncToolset):
    """Framework-managed client delegation using the existing async-tool executor.

    Supply a persistent backend via ``handler`` (for example its start-or-steer method),
    or an LLM and tools for the default loop. ``select_task`` is application policy:
    equal keys mean updated input, different keys mean independent work. By default
    each transport request is independent. No routing model is inserted.

    Put this toolset on AgentSession to retain context through handoff, or on Agent
    for activity scope. Backend resources supplied by callers remain caller-owned.
    """

    def __init__(
        self,
        *,
        handler: Callable[[DelegationContext], Awaitable[str | None]] | None = None,
        model: llm.LLM | None = None,
        tools: Sequence[llm.FunctionTool | llm.RawFunctionTool] = (),
        instructions: str = "Resolve the current request using verified facts. Keep replies concise.",
        select_task: Callable[[DelegationRequest], str] | None = None,
    ) -> None:
        if handler is None and model is None:
            raise ValueError("client delegation requires a backend handler or model")
        self._handler = handler
        self._model = model
        self._backend_tools = llm.ToolContext(list(tools))
        self._instructions = instructions
        self._select_task = select_task or (lambda request: utils.shortuuid("task_"))
        self._states: dict[str, _TaskState] = {}
        self._requests: dict[str, DelegationContext] = {}
        self._seen: set[tuple[str, str]] = set()
        self._closed = False
        self._session: AgentSession | None = None
        super().__init__(id="client_delegation")

    def _attach_activity(self, *, activity: AgentActivity | None, session: AgentSession) -> None:
        if self._session is not None and self._session is not session:
            raise ValueError("ClientDelegation cannot share backend context between sessions")
        self._session = session
        super()._attach_activity(activity=activity, session=session)

    def _dispatch(
        self,
        request: DelegationRequest,
        *,
        send: Callable[[str, bool, Callable[[], bool]], bool],
        connected: Callable[[], bool],
    ) -> llm.FunctionCall | None:
        key = (request.connection_id, request.id)
        if self._closed or key in self._seen:
            return None
        task_id = self._select_task(request)
        if not task_id:
            raise ValueError("select_task must return a nonempty task ID")
        self._seen.add(key)
        request = replace(
            request,
            chat_ctx=llm.ChatContext(
                [item.model_copy(deep=True) for item in request.chat_ctx.items]
            ),
        )
        state = self._states.setdefault(task_id, _TaskState())
        # Advance before dispatch can yield: a completion already queued in another task
        # must not escape while this revision waits for execution.
        state.revision += 1
        if not state.history.items:
            state.history.add_message(role="system", content=self._instructions)
        for item in request.chat_ctx.copy(
            exclude_function_call=True, exclude_instructions=True
        ).items:
            if (index := state.history.index_by_id(item.id)) is not None:
                state.history.items[index] = item
            else:
                # Startup instructions precede imported history even though their local
                # creation timestamp is newer than the caller's transcript.
                state.history.items.append(item)
        call_id = utils.shortuuid("delegation_")
        ctx = DelegationContext(self, request, task_id, state, send, connected)
        self._requests[call_id] = ctx
        return llm.FunctionCall(
            call_id=call_id,
            name=DELEGATE_TOOL_NAME,
            arguments="{}",
            extra={
                "delegation_id": request.id,
                "task_id": task_id,
                "revision": state.revision,
                "connection_id": request.connection_id,
            },
        )

    @llm.function_tool(name=DELEGATE_TOOL_NAME, flags=llm.ToolFlag.CANCELLABLE)
    async def _delegate(self, ctx: RunContext) -> str | None:
        delegation = self._requests.pop(ctx.function_call.call_id, None)
        if delegation is None:
            raise llm.ToolError("client delegation requires a transport-supplied request")
        if not delegation.is_current:
            return None
        delegation._run_ctx = ctx
        ctx._reply_handler = delegation._deliver
        # Release the voice turn immediately without manufacturing a spoken acknowledgment.
        assert ctx._first_update_fut is not None
        if not ctx._first_update_fut.done():
            ctx._first_update_fut.set_result(None)
        ctx.function_call.extra["__livekit_agents_tool_non_blocking"] = True
        return await self.delegation_node(delegation)

    async def delegation_node(self, ctx: DelegationContext) -> str | None:
        """Override to feed a long-lived backend. The default retains context between steps."""
        if self._handler is not None:
            return await self._handler(ctx)
        assert self._model is not None
        assert ctx._run_ctx is not None
        for _ in range(ctx._run_ctx.session.options.max_tool_steps + 1):
            if not ctx.is_current:
                return None
            # A corrected request must observe any already-running external outcomes before
            # planning its next step. Reuse the executor's task registry; shielding preserves
            # tools whose own cancellation policy disallows cancellation of external work.
            pending = [
                running.exe_task
                for item in ctx._state.history.items
                if isinstance(item, llm.FunctionCall)
                and (running := self._executor._running_tasks.get(item.call_id)) is not None
            ]
            if pending:
                await asyncio.gather(
                    *(asyncio.shield(task) for task in pending), return_exceptions=True
                )
            if not ctx.is_current:
                return None
            text = ""
            calls: list[llm.FunctionToolCall] = []
            async with self._model.chat(
                chat_ctx=ctx.chat_ctx, tools=self._backend_tools.flatten()
            ) as stream:
                async for chunk in stream:
                    if chunk.delta:
                        text += chunk.delta.content or ""
                        calls.extend(chunk.delta.tool_calls)
            if not ctx.is_current:
                return None
            if not calls:
                if not text.strip():
                    raise llm.ToolError("delegation backend returned no answer")
                return text
            if text:
                await ctx.update(text, silent=True)
            for call in calls:
                await ctx.execute_tool(call.name, json.loads(call.arguments), call_id=call.call_id)
        raise llm.ToolError("delegated work exceeded the tool-step limit")

    async def aclose(self) -> None:
        self._closed = True
        self._requests.clear()
        await super().aclose()


def _find_client_delegation(tools: Sequence[llm.Tool]) -> ClientDelegation | None:
    owners = {
        tool._instance
        for tool in tools
        if isinstance(tool, llm.FunctionTool) and isinstance(tool._instance, ClientDelegation)
    }
    if len(owners) > 1:
        raise ValueError("only one client delegation toolset may handle a voice connection")
    return next(iter(owners), None)
