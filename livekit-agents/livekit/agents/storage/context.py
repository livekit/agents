"""What an ``a2a_session`` handler receives (plan 7.3, 7.6.1), and the host the Rust turn calls.

``_core.Turn`` runs the policies and the commit; what it hands over is the ``TurnContext`` model
and a ``Boundary``, and what it reads back is a ``HandlerOutput``. ``TurnHost`` is the one object
it calls into (plan 13.8): ``handle`` builds the ``A2ASessionContext`` the developer's handler
takes, runs it with the run hooks of ``Agent.run()`` installed, and reports the output;
``on_update`` and ``on_token`` are the door's wire hooks, forwarded as they are.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable
from typing import Generic, cast

from pydantic import JsonValue
from typing_extensions import TypeVar

from .._core import Boundary
from .._generated.storage_run import RunUsage, TaskRecord
from .._generated.storage_turn import HandlerOutput, HostEvent, Message, PendingCall, TurnContext
from ..llm import ChatContext, ChatMessage, FunctionCall, FunctionCallOutput
from ..log import logger
from ..voice.agent_run import RunHooks, reset_run_hooks, set_run_hooks
from ..voice.run_result import RunResult
from .protocol import StorageError, dump_state, history_of, item_to_payload, load_state
from .resolve import AgentRegistry, Thread

S = TypeVar("S", default=None)  # a door declared with no state: ``A2ASessionContext``

Handler = Callable[["A2ASessionContext[S]"], "Awaitable[object] | AsyncIterable[str]"]
"""The developer's handler: the context in, the answer out, either returned whole or yielded as
it is written (an async generator), in which case the answer is the last run's result."""

UpdateHook = Callable[[str, str, bool], Awaitable[None]]
"""``(task_id, text, verbatim)``: progress, or with ``verbatim`` exact text, to the door's live
response."""

TokenHook = Callable[[str, str], Awaitable[None]]
"""``(task_id, delta)``: one streamed token to the door's live response."""


class A2ASessionContext(Generic[S]):
    """What an ``@server.a2a_session`` handler receives: the turn as loaded, and the ways to
    report back. ``message`` is the batch as one user message, ``messages`` each message with its
    channel and time; ``message_history`` the lane's history, ``state`` the lane's state, which
    the handler mutates in place; ``caller_history`` a delegating caller's conversation when it
    sent one (plan 7.5), ``caller_lane`` its lane of this thread when it named it, whose history
    holds the person's own words: ``await ctx.thread.chat_history(ctx.caller_lane)``."""

    def __init__(
        self,
        turn: TurnContext,
        boundary: Boundary,
        *,
        state: S,
        thread: Thread,
        on_update: UpdateHook | None,
        on_token: TokenHook | None,
    ) -> None:
        self.message = ChatMessage.model_validate(turn.message)
        self.messages: list[Message] = list(turn.batch)
        self.message_history: ChatContext = history_of(turn.history)
        self.state = state
        self.thread = thread
        self.task_id = turn.task_id
        self.tasks: list[TaskRecord] = list(turn.tasks)
        self.caller_history: ChatContext | None = (
            ChatContext.from_dict(turn.caller_history) if turn.caller_history is not None else None
        )
        from ..a2a.extension import CALLER_LANE

        lane = turn.batch[-1].metadata.get(CALLER_LANE) if turn.batch else None
        self.caller_lane: str | None = lane if isinstance(lane, str) else None
        self.metadata: dict[str, JsonValue] = {}
        """What goes on the task's terminal row and final status event, under short keys."""
        self.events: list[HostEvent] = []
        """What the turn appends to the journal (docs/journal.md): each ``update`` here, the
        tool calls from the runs' results."""
        self.interrupted = False
        self._boundary = boundary
        self._on_update = on_update
        self._on_token = on_token

    async def update(self, text: str) -> None:
        """A ``working`` status frame to the client: progress, for the agent facing the person
        to word."""
        self.events.append(_tool_event("update", text=text))
        if self._on_update is not None:
            await self._on_update(self.task_id, text, False)

    async def say(self, text: str) -> None:
        """A ``working`` status frame with its text flagged: it reaches the person exactly as
        written."""
        self.events.append(_tool_event("say", text=text))
        if self._on_update is not None:
            await self._on_update(self.task_id, text, True)

    async def token(self, delta: str) -> None:
        """One piece of the reply as the model writes it, shown before the turn ends."""
        if self._on_token is not None:
            await self._on_token(self.task_id, delta)

    async def stream(self, source: AsyncIterable[str]) -> str:
        """Forward tokens as they arrive; the boundary is checked once before the first one."""
        if await self._boundary.check():
            self.interrupted = True
            return ""

        parts: list[str] = []
        async for text in source:
            parts.append(text)
            if self._on_token is not None:
                await self._on_token(self.task_id, text)

        return "".join(parts)

    async def task(self, task_id: str) -> TaskRecord | None:
        """One task row of the lane, as it is now."""
        for record in await self.thread.tasks():
            if record.id == task_id:
                return record

        return None


class TurnHost(Generic[S]):
    """What ``_core.Turn`` calls back into for one door: the handler, and the wire hooks."""

    def __init__(
        self,
        handler: Handler[S],
        *,
        thread: Thread,
        state_type: type[S] | None,
        agents: AgentRegistry | None = None,
        on_update: UpdateHook | None = None,
        on_token: TokenHook | None = None,
    ) -> None:
        self._handler = handler
        self._thread = thread
        self._state_type = state_type
        self._agents = agents if agents is not None else AgentRegistry()
        self._on_update = on_update
        self._on_token = on_token
        self.verbatim = False
        """The last turn's answer, or its question, reaches the person exactly as written: the
        door flags its part."""

    async def handle(self, ctx: TurnContext, boundary: Boundary) -> HandlerOutput:
        """The developer's handler on one turn, the run hooks installed around it."""
        context = A2ASessionContext(
            ctx,
            boundary,
            state=self._load_state(ctx.state),
            thread=self._thread,
            on_update=self._on_update,
            on_token=self._on_token,
        )
        results: list[RunResult[object]] = []
        hooks = RunHooks(
            boundary=boundary.check,
            on_result=results.append,
            on_update=context.update,
            on_say=context.say,
            on_token=context.token,
            thread_id=self._thread.id,
            agent_stack=ctx.agent_stack,
            rebuild=self._agents.rebuild,
            answer_tool=ctx.answer_tool,
            delegated=bool(ctx.batch) and all(m.channel == "delegation" for m in ctx.batch),
            lane=ctx.lane,
        )

        token = set_run_hooks(hooks)
        try:
            handled = self._handler(context)
            if isinstance(handled, AsyncIterable):
                # yielded as it is written: each piece reaches the client, the answer is the
                # last run's, or the text itself when no run was made
                parts: list[str] = []
                async for delta in handled:
                    parts.append(delta)
                    await context.token(delta)
                output: object = results[-1] if results else "".join(parts)
            else:
                output = await handled
        except StorageError:
            raise
        except Exception as e:
            logger.exception(
                "the a2a_session handler raised; the task fails",
                extra={"lane": ctx.lane, "taskId": ctx.task_id},
            )
            return HandlerOutput(error=f"{type(e).__name__}: {e}")
        finally:
            reset_run_hooks(token)

        return self._output(context, output, results)

    async def on_update(self, task_id: str, text: str, verbatim: bool = False) -> None:
        if self._on_update is not None:
            await self._on_update(task_id, text, verbatim)

    async def on_token(self, task_id: str, delta: str) -> None:
        if self._on_token is not None:
            await self._on_token(task_id, delta)

    def _load_state(self, stored: JsonValue) -> S:
        if self._state_type is None:
            return cast(S, None)  # a door declared with no state: ``A2ASessionContext[None]``
        return load_state(stored, self._state_type)

    def _output(
        self, ctx: A2ASessionContext[S], output: object, results: list[RunResult[object]]
    ) -> HandlerOutput:
        """What the turn commits: the answer, what the runs added, the state, the ask, the
        agent stack when a run moved it."""
        if isinstance(output, RunResult):
            # a run returned as is answers with its question while it waits, else its output
            output = output.input_required if output.input_required is not None else output.output

        asked = next((r for r in reversed(results) if r.input_required is not None), None)
        last = results[-1] if results else None
        # exact when the handler answered with the last run's exact answer or question
        self.verbatim = last is not None and last.verbatim is not None and output == last.verbatim
        pending = None
        if asked is not None and (call := asked.pending_call) is not None:
            pending = PendingCall(
                name=call.name, call_id=call.call_id, answer_tool=asked._answer_tool
            )
        return HandlerOutput(
            output=output,
            new_items=[item_to_payload(item) for r in results for item in r.new_messages()],
            state=dump_state(ctx.state, self._state_type) if self._state_type is not None else None,
            usage=_usage(results),
            input_required=asked.input_required if asked is not None else None,
            pending_call=pending,
            agent_stack=next(
                (r._agent_stack for r in reversed(results) if r._agent_stack is not None), None
            ),
            interrupted=ctx.interrupted or any(r.interrupted for r in results),
            metadata=dict(ctx.metadata),
            events=[*_tool_events(results), *ctx.events],
        )


def _tool_events(results: list[RunResult[object]]) -> list[HostEvent]:
    """Each call's start and end as the runs' results hold them (docs/journal.md)."""
    events: list[HostEvent] = []
    for result in results:
        for item in result.new_messages():
            if isinstance(item, FunctionCall):
                events.append(_tool_event("started", callId=item.call_id, name=item.name))
            elif isinstance(item, FunctionCallOutput):
                events.append(
                    _tool_event("ended", callId=item.call_id, name=item.name, isError=item.is_error)
                )
    return events


def _tool_event(phase: str, **fields: JsonValue) -> HostEvent:
    return HostEvent(type="tool_execution_updated", payload={"phase": phase, **fields})


def _usage(results: list[RunResult[object]]) -> RunUsage:
    usage = RunUsage()
    for result in results:
        usage.requests += result.usage.requests
        usage.input_tokens += result.usage.input_tokens
        usage.output_tokens += result.usage.output_tokens
        usage.tool_calls += result.usage.tool_calls

    return usage


__all__ = ["A2ASessionContext", "Handler", "TokenHook", "TurnHost", "UpdateHook"]
