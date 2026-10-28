"""``agent.run()``: the slow brain, one input to one result over a given history.

An ``AgentRun`` is the second binding over an ``Agent``, next to the voice ``AgentActivity``:
it sets the same contextvar so hooks, tools and the pipeline nodes resolve the agent the same
way, runs the model and tool loop until the answer or an ``InputRequired``, and holds nothing
after it returns. It knows nothing about rooms, audio or where the message came from.

Who speaks is the lane's agent stack (docs/lanes.md): the root the caller named while it is
empty, else the agent on top. A handoff replaces the top; a tool that awaits an ``AgentTask``
pushes the task with its arguments and leaves the call pending, and the task's ``Completed``
pops it and writes the value as that call's output. The door hands the stack in through the
run hooks and reads it back from the result.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
import time
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Container, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import TypeAdapter, ValidationError

from .. import utils
from .._exceptions import UnexpectedModelBehavior
from .._generated.storage_run import Ref
from ..llm import (
    LLM,
    AgentHandoff,
    ChatChunk,
    ChatContext,
    ChatItem,
    ChatMessage,
    CompletionUsage,
    FunctionCall,
    FunctionCallOutput,
    FunctionTool,
    RawFunctionTool,
    RealtimeModel,
    Tool,
    ToolChoice,
    ToolContext,
    Toolset,
)
from ..llm.chat_context import _ReadOnlyChatContext
from ..llm.tool_context import (
    ASK_IN_OWN_WORDS,
    DELIVERED,
    TASK_KEY,
    Completed,
    Failed,
    InputRequired,
    ModelRetry,
    Rejected,
    StopResponse,
    ToolError,
    ToolResult,
    Transition,
    Working,
    as_model_reads,
    delivered,
    function_tool,
    is_transition,
)
from ..llm.utils import answer_arguments, parse_function_arguments
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import is_given
from .agent import AgentTask, ModelSettings, TaskAwaited
from .events import RunContext
from .generation import (
    ToolExecutionOutput,
    _serialize_output,
    make_tool_output,
    update_instructions,
)
from .run_result import RunResult, RunUsage
from .tool_executor import (
    AsyncToolOptions,
    _reply_instructions,
    _reply_tool_choice,
    _resolve_async_tool_options,
    _RunningTask,
    _ToolExecutor,
    cancel_task,
    get_running_tasks,
    has_cancellable_tool,
)

if TYPE_CHECKING:
    from .agent import Agent

Run_T = TypeVar("Run_T")

DEFAULT_MAX_STEPS = 12
FINAL_RESULT_TOOL = "final_result"
_RETRY_PROMPT = "Validation failed, fix the output and try again: {reason}"
_FINAL_RESULT_PROMPT = (
    f"You have not provided the final output yet. Call `{FINAL_RESULT_TOOL}` with it; "
    "a plain text response alone is not enough."
)


@dataclass
class RunHooks:
    """What a door installs around the ``agent.run()`` calls made inside its turn.

    ``boundary`` is awaited before each model call; ``True`` makes the run yield there with what
    ran so far and no reply (the ``interrupt`` policy of plan 3.3.4). ``on_result`` receives every
    result, so the door can commit what the run added without the handler forwarding it;
    ``on_item`` each item as the run adds it, so a run that raises still commits what it ran.
    ``on_update`` receives a tool's ``ctx.update(text)``: the door forwards it as progress, a
    ``working`` frame on the text door. ``on_say`` receives exact text said during the run, a
    ``ctx.say(text)`` or a verbatim result the model goes on from: the door sends it at once as
    a flagged ``working`` frame. ``on_token`` receives the model's text as it arrives, so the
    text door shows the reply as it is written. ``thread_id`` is the session the run belongs to,
    what ``A2ADelegate`` sends as the expert's ``contextId`` (plan 7.5).
    ``agent_stack`` is the lane's stack as loaded and ``rebuild`` makes an agent of one entry,
    the server's registry; the run resolves them itself since only it knows the root.
    ``answer_tool`` is the tool whose question this message answers, called with it before the
    model runs and with ``answer_ref`` on its ``RunContext``. ``delegated``
    says the turn's message came from another agent: the run does not phrase its questions,
    the agent facing the person does. ``lane`` is the door's lane of the thread, what
    ``A2ADelegate`` names as the caller's. ``task_id`` marks what the run records
    (``TASK_KEY``); ``open_tasks`` are the lane's other open tasks, whose later items it skips.
    ``tools`` is set under the ``live`` inbox: a tool's first ``ctx.update()`` is its output, the
    run goes on, and the tool runs on there; ``delivery`` is what its tools queued that this turn
    delivers, in place of a message.
    """

    boundary: Callable[[], Awaitable[bool]] | None = None
    on_result: Callable[[RunResult[Any]], None] | None = None
    on_item: Callable[[ChatItem], None] | None = None
    on_update: Callable[[str], Any] | None = None
    on_say: Callable[[str], Any] | None = None
    on_token: Callable[[str], Awaitable[None]] | None = None
    thread_id: str | None = None
    agent_stack: Sequence[Ref] = ()
    rebuild: Callable[[Ref], Agent | None] | None = None
    answer_tool: str | None = None
    answer_ref: str | None = None
    delegated: bool = False
    lane: str | None = None
    task_id: str | None = None
    open_tasks: frozenset[str] = frozenset()
    tools: TextTools | None = None
    delivery: Sequence[ToolExecutionOutput] = ()


class TextTools:
    """The sink of a text door's tools: under ``live`` a tool's first update lets its turn go on,
    and what it reports later is sent at once and queued for the next turn."""

    def __init__(
        self, *, live: bool = False, on_update: Callable[[str, str, bool], Awaitable[None]] | None
    ) -> None:
        self.executor = _ToolExecutor(sink=self)
        self._live = live
        self._on_update = on_update
        self._released: dict[str, str] = {}
        """The calls whose turn went on without them, with their task."""
        self._queue: list[tuple[str, ToolExecutionOutput, bool]] = []
        """By task, in call order: each update, said text and end, and whether it is progress."""
        self._ended = asyncio.Event()

    def started(self, task: _RunningTask) -> None:
        pass

    def done(self, task: _RunningTask, output: Any) -> None:
        call_id = task.ctx.function_call.call_id
        if call_id not in self._released:
            return
        # a call stopped or failed with no end of its own: its task still hears that it ended
        error = "the call was cancelled" if task.exe_task.cancelled() else str(output)
        self._end(task.ctx, ToolError(error))

    async def stop(self, task: _RunningTask) -> None:
        await utils.aio.cancel_and_wait(task.exe_task)

    async def arguments(
        self, tool: FunctionTool | RawFunctionTool, raw: dict[str, Any], ctx: RunContext[Any, Any]
    ) -> dict[str, Any]:
        return raw

    def returned(self, ctx: RunContext[Any, Any], output: Any) -> Any:
        return output

    async def update(
        self,
        ctx: RunContext[Any, Any],
        pair: tuple[FunctionCall, FunctionCallOutput],
        text: str,
        message: Any,
        *,
        silent: bool,
    ) -> None:
        call_id = ctx.function_call.call_id
        if (task_id := self._released.get(call_id)) is None:
            # in its turn: progress on the turn's response, and under live the call's output
            hooks = current_run_hooks()
            if not silent and hooks is not None and hooks.on_update is not None:
                out = hooks.on_update(text)
                if inspect.isawaitable(out):
                    await out
            if self._live and ctx._first_update_fut is not None:
                self._released[call_id] = hooks.task_id or "" if hooks is not None else ""
                # marked on the run now: the work may end before the run reads its output
                assert ctx._run is not None
                ctx._run._background_calls.add(call_id)
                self.executor._report(ctx, message, silent=silent)
            return
        if not silent and self._on_update is not None:
            await self._on_update(task_id, text, False)
        self._queue.append(
            (task_id, make_tool_output(fnc_call=pair[0], output=message, exception=None), True)
        )

    async def say(self, ctx: RunContext[Any, Any], text: str) -> None:
        call_id = ctx.function_call.call_id
        if (task_id := self._released.get(call_id)) is None:
            assert ctx._run is not None
            await ctx._run._say(text)
            return
        if self._on_update is not None:
            await self._on_update(task_id, text, True)
        step = len(ctx._updates)
        pair = ctx._make_update_pair(
            ToolResult(DELIVERED.format(text=text), reply_required=False),
            call_id_suffix=f"_update_{step}",
        )
        ctx._updates.append(pair)
        out = make_tool_output(fnc_call=pair[0], output=pair[1].output, exception=None)
        self._queue.append((task_id, out, True))

    async def ended(self, ctx: RunContext[Any, Any], output: Any) -> None:
        # as in the turn: the model reads why, and may retry
        self._end(ctx, ToolError(output.message) if isinstance(output, ModelRetry) else output)

    def _end(self, ctx: RunContext[Any, Any], output: Any) -> None:
        task_id = self._released.pop(ctx.function_call.call_id)
        final_call, _ = ctx._make_update_pair(None, call_id_suffix="_final")
        out = make_tool_output(fnc_call=final_call, output=output, exception=None)
        self._queue.append((task_id, out, False))
        self._ended.set()

    def owed(self) -> list[str]:
        """The tasks whose work runs on, or whose queue waits."""
        tasks = [*self._released.values(), *(task_id for task_id, _, _ in self._queue)]
        return list(dict.fromkeys(tasks))

    def progress(self) -> list[ToolExecutionOutput]:
        """The progress queued before any end, which the next turn records before its message."""
        taken = 0
        while taken < len(self._queue) and self._queue[taken][2]:
            taken += 1
        progress, self._queue = self._queue[:taken], self._queue[taken:]
        return [out for _, out, _ in progress]

    async def take(
        self, wait: float, waiting: Container[str]
    ) -> tuple[list[str], list[ToolExecutionOutput]]:
        """What is delivered next, after waiting up to ``wait`` seconds for an end, and the tasks it
        ends or asks: the queue in call order, a question on its own, nothing of a task that waits
        on the person's answer, and nothing while only progress waits."""

        def ready() -> bool:
            return any(not progress and t not in waiting for t, _, progress in self._queue)

        if wait and not ready():
            self._ended.clear()
            try:
                await asyncio.wait_for(self._ended.wait(), wait)
            except asyncio.TimeoutError:
                pass
        if not ready():
            return [], []
        taken: list[tuple[str, ToolExecutionOutput, bool]] = []
        kept: list[tuple[str, ToolExecutionOutput, bool]] = []
        asked = closed = False
        for task_id, out, progress in self._queue:
            asks = isinstance(out.transition, InputRequired)
            if task_id not in waiting:
                closed = closed or bool(taken and (asks or asked))
            if task_id in waiting or closed:
                kept.append((task_id, out, progress))
                continue
            taken.append((task_id, out, progress))
            asked = asks
        self._queue = kept
        ends = dict.fromkeys(task_id for task_id, _, progress in taken if not progress)
        return list(ends), [out for _, out, _ in taken]


_run_hooks = contextvars.ContextVar[RunHooks | None]("agents_run_hooks", default=None)


def current_run_hooks() -> RunHooks | None:
    return _run_hooks.get()


_HooksToken = tuple[contextvars.Token[RunHooks | None], contextvars.Token[Any]]


def set_run_hooks(hooks: RunHooks | None) -> _HooksToken:
    """Installs the door's hooks for a turn. The turn is a root: a door reached in process
    from another run's tool, as a test does, is not nested in it."""
    from .agent_activity import _AgentActivityContextVar

    return _run_hooks.set(hooks), _AgentActivityContextVar.set(None)


def reset_run_hooks(token: _HooksToken) -> None:
    from .agent_activity import _AgentActivityContextVar

    _run_hooks.reset(token[0])
    _AgentActivityContextVar.reset(token[1])


class AgentRun(Generic[Run_T]):
    """One turn of an ``Agent`` on a history: what ``agent.run()`` awaits, and what
    ``agent.stream()`` hands out to iterate, the reply as the model writes it, one piece of text
    at a time, with ``result`` set once the loop ends."""

    def __init__(
        self,
        agent: Agent,
        message: str | ChatMessage,
        *,
        message_history: ChatContext | Sequence[ChatItem] | None,
        state: Any,
        deps: Any,
        usage: RunUsage | None,
        max_steps: int | None,
        output_type: type[Run_T] | None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        self._id = utils.shortuuid("run_")
        self._root = agent
        self._agent = agent
        self._stack: list[Agent] = []
        self._stack_dirty = False
        self._nested = False
        self._deps = deps
        self._state = state
        self._usage = usage if usage is not None else RunUsage()
        self._max_steps = max_steps if max_steps is not None else DEFAULT_MAX_STEPS
        self._output_type: type[Any] | None = output_type
        self._conn_options = conn_options

        if isinstance(message_history, ChatContext):
            self._chat_ctx = message_history.copy()
        else:
            self._chat_ctx = ChatContext(list(message_history) if message_history else [])

        if isinstance(message, str):
            message = ChatMessage(role="user", content=[message])
        self._message = message
        self._result: RunResult[Run_T] | None = None
        self._on_token: Callable[[str], Awaitable[None]] | None = None
        # exact text said since the last call output, recorded after it as the agent's message
        self._said: list[ChatMessage] = []
        # what the lane's other open tasks added that this run's model does not read
        self._hidden: frozenset[str] = frozenset()
        # the calls whose tool went on in the background at its first update
        self._background_calls: set[str] = set()
        self._tools = TextTools(on_update=None)

    async def __aiter__(self) -> AsyncIterator[str]:
        if self._result is not None:
            raise RuntimeError("the run already ended")

        deltas: asyncio.Queue[str | None] = asyncio.Queue()

        async def on_token(delta: str) -> None:
            deltas.put_nowait(delta)

        self._on_token = on_token
        task = asyncio.ensure_future(self._run())
        task.add_done_callback(lambda _: deltas.put_nowait(None))
        try:
            while (delta := await deltas.get()) is not None:
                yield delta
            await task
        finally:
            if not task.done():
                task.cancel()

    @property
    def result(self) -> RunResult[Run_T]:
        """What the run produced; set once it was awaited or iterated to its end."""
        if self._result is None:
            raise RuntimeError("the run has not ended")
        return self._result

    # -- the binding, what the agent and its nodes read --

    @property
    def id(self) -> str:
        return self._id

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def llm(self) -> LLM | RealtimeModel | None:
        return self._agent.llm if is_given(self._agent.llm) else None

    @property
    def llm_conn_options(self) -> APIConnectOptions:
        return self._conn_options

    @property
    def chat_ctx(self) -> ChatContext:
        """A read-only view of the history this run works on."""
        return _ReadOnlyChatContext(self._chat_ctx.items)

    @property
    def deps(self) -> Any:
        return self._deps

    @property
    def state(self) -> Any:
        return self._state

    @property
    def usage(self) -> RunUsage:
        return self._usage

    # -- the loop --

    async def _run(self) -> RunResult[Run_T]:
        from .agent_activity import _AgentActivityContextVar

        # a run started inside another run's tool is a delegation: a fresh conversation the
        # door never sees, on no stack and with no boundary (docs/multi-agent.md)
        self._nested = isinstance(_AgentActivityContextVar.get(None), AgentRun)
        hooks = None if self._nested else _run_hooks.get()
        if hooks is not None:
            self._stack = self._resolve_stack(hooks)
            if self._stack:
                self._agent = self._stack[-1]
            if hooks.tools is not None:
                self._tools = hooks.tools

        token = _AgentActivityContextVar.set(self)
        try:
            result = await self._run_bound()
        finally:
            _AgentActivityContextVar.reset(token)
        self._result = result
        if hooks is not None and hooks.on_result is not None:
            hooks.on_result(result)
        return result

    # -- the agent stack --

    def _resolve_stack(self, hooks: RunHooks) -> list[Agent]:
        """The loaded stack as agents, bottom first; an entry that cannot be rebuilt drops
        itself and everything above it, and the nearest ancestor speaks."""
        agents: list[Agent] = []
        for ref in hooks.agent_stack:
            agent: Agent | None
            if ref.args is None and ref.name == self._root.id:
                agent = self._root
            else:
                agent = hooks.rebuild(ref) if hooks.rebuild is not None else None
            if agent is None:
                logger.warning(
                    "dropping an agent stack entry the server cannot rebuild, and everything "
                    "above it",
                    extra={"refName": ref.name, "runId": self._id},
                )
                self._stack_dirty = True
                break
            if isinstance(agent, AgentTask):
                agent._restore(ref.state, call_id=ref.call_id)
            agents.append(agent)
        return agents

    def _stack_refs(self) -> list[Ref] | None:
        """The stack for the commit: when it moved, or a task on it may have changed state."""
        if not self._stack_dirty and not any(isinstance(a, AgentTask) for a in self._stack):
            return None
        return [
            Ref(
                name=type(a).__name__,
                args=a._arguments(),
                state=a._dump_state(),
                call_id=a._pending_call_id,
            )
            if isinstance(a, AgentTask)
            else Ref(name=a.id)
            for a in self._stack
        ]

    def _hand_off(self, agent: Agent, record: Callable[[ChatItem], None]) -> None:
        """A tool returned an agent: it speaks from the next step, on the same history."""
        record(AgentHandoff(old_agent_id=self._agent.id, new_agent_id=agent.id))
        if self._stack and not isinstance(self._stack[-1], AgentTask):
            self._stack[-1] = agent
        elif agent is not self._root or self._stack:
            self._stack.append(agent)
        self._agent = agent
        self._stack_dirty = True

    def _pop(
        self, result: Any, record: Callable[[ChatItem], None], *, verbatim: bool = False
    ) -> None:
        """The task on top ended: its value, or a ``ToolError``, is the pending call's output,
        and the agent below speaks again from the next step. A verbatim value was said as
        written; the agent below reads that it was."""
        task = self._stack.pop()
        assert isinstance(task, AgentTask)
        below = self._stack[-1] if self._stack else self._root
        call = next(
            (
                item
                for item in reversed(self._chat_ctx.items)
                if isinstance(item, FunctionCall) and item.call_id == task._pending_call_id
            ),
            None,
        )
        if call is not None:
            record(
                FunctionCallOutput(
                    name=call.name,
                    call_id=call.call_id,
                    output=str(result)
                    if isinstance(result, Exception)
                    else DELIVERED.format(text=_tool_text(result))
                    if verbatim
                    else _tool_text(result),
                    is_error=isinstance(result, Exception),
                )
            )
        record(AgentHandoff(old_agent_id=task.id, new_agent_id=below.id))
        if not task.done():
            task._complete(result)
        self._agent = below
        self._stack_dirty = True

    async def _run_bound(self) -> RunResult[Run_T]:
        chat_ctx = self._chat_ctx
        result = RunResult[Run_T](output_type=None)
        result._run_usage = self._usage

        # what this run adds, in order; the instructions are rendered per request and are
        # not part of it
        new_items: list[ChatItem] = []
        hooks = None if self._nested else _run_hooks.get()
        task_id = hooks.task_id if hooks is not None else None
        if hooks is not None and hooks.open_tasks:
            # each item with its task; an output belongs to its call's task
            call_tasks: dict[str, Any] = {}
            tagged: list[tuple[ChatItem, Any]] = []
            for item in chat_ctx.items:
                if isinstance(item, FunctionCallOutput):
                    task = call_tasks.get(item.call_id)
                elif isinstance(item, (ChatMessage, FunctionCall)):
                    task = item.extra.get(TASK_KEY)
                else:
                    task = None
                if isinstance(item, FunctionCall):
                    call_tasks[item.call_id] = task
                tagged.append((item, task))
            last = max((n for n, (_, task) in enumerate(tagged) if task == task_id), default=None)
            if last is not None:
                # what other open tasks added after this task's last item: the task never saw it
                self._hidden = frozenset(
                    item.id for item, task in tagged[last + 1 :] if task in hooks.open_tasks
                )

        def record(item: ChatItem) -> None:
            if task_id is not None and isinstance(item, (ChatMessage, FunctionCall)):
                item.extra.setdefault(TASK_KEY, task_id)
            chat_ctx.insert(item)
            new_items.append(item)
            if hooks is not None and hooks.on_item is not None:
                hooks.on_item(item)

        def record_said() -> None:
            said, self._said = self._said, []
            for message in said:
                record(message)

        # a run another agent asked does not phrase: the agent facing the person does
        facing_person = hooks is None or not hooks.delegated

        def finish(
            output: Any,
            *,
            input_required: str | None = None,
            interrupted: bool = False,
            verbatim: bool = False,
        ) -> RunResult[Run_T]:
            record_said()
            answer = input_required if input_required is not None else output
            result._verbatim = answer if verbatim and isinstance(answer, str) else None
            result._run_output = output
            result._input_required = input_required
            result._interrupted = interrupted
            result._run_messages = new_items
            result._run_history = chat_ctx.copy(exclude_instructions=True).items
            result._agent_stack = self._stack_refs()
            result._done_fut.set_result(None)
            return result

        async def ask(question: InputRequired, call: FunctionCall | None) -> RunResult[Run_T]:
            """The task pauses on ``question`` from ``call``: as written when it is verbatim or
            another agent asked, else in the agent's own words."""
            text = question.question
            if not question.verbatim and facing_person:
                text = await self._phrase(chat_ctx, text)
            record_said()
            record(ChatMessage(role="assistant", content=[text]))
            result._pending_call = call
            if question._answer_ref and call is not None:
                result._answer_tool, result._answer_ref = call.name, question._answer_ref
            return finish(None, input_required=text, verbatim=question.verbatim)

        def finish_task_turn(text: str) -> RunResult[Run_T]:
            # a task holds the conversation: every reply of its model waits on the person
            return finish(None, input_required=text)

        async def act(
            transition: Transition, fnc_call: FunctionCall | None
        ) -> RunResult[Run_T] | None:
            """A task's transition, from a tool or ``on_enter``: the result when it ends the turn."""
            if not self._stack and isinstance(self._agent, AgentTask):
                # the run's root is the task itself, as in a test: nothing below takes over, so
                # its end is the run's end, with its value or its error as the output
                if isinstance(transition, Completed):
                    self._agent._complete(transition.value)
                    return finish(transition.value, verbatim=transition.verbatim)
                if isinstance(transition, (Failed, Rejected)):
                    text = transition.error if isinstance(transition, Failed) else transition.reason
                    self._agent._complete(ToolError(text))
                    return finish(text)
            if isinstance(transition, Completed):
                if transition.verbatim:
                    await self._say(_tool_text(transition.value), record=False)
                self._pop(transition.value, record, verbatim=transition.verbatim)
            elif isinstance(transition, (Failed, Rejected)):
                text = transition.error if isinstance(transition, Failed) else transition.reason
                self._pop(ToolError(text), record)
            elif isinstance(transition, InputRequired):
                return await ask(transition, fnc_call)
            elif isinstance(transition, Working):
                logger.warning(
                    "Working is not applied in text yet; the task keeps the conversation",
                    extra={"agent": self._agent.id, "runId": self._id},
                )
            return None

        # the verbatim results of a step that end the turn, in call order: the answer, unless a
        # result of the step asks for a reply or a transition takes over
        held: list[str] = []

        async def release_held() -> None:
            while held:
                await self._say(held.pop(0), record=False)

        async def run_call(
            fnc_call: FunctionCall, tool_ctx: ToolContext, typed: bool
        ) -> ToolExecutionOutput | AgentTask[Any, Any]:
            tool = tool_ctx.get_function_tool(fnc_call.name)
            ref = answer_ref if fnc_call is forwarded else None
            out = await self._execute(tool, fnc_call, answer_ref=ref)
            said = out.raw_output if not isinstance(out, AgentTask) else None
            if isinstance(said, ToolResult) and said.verbatim and (said.reply_required or typed):
                # said when its tool returns; the model reads that it was. On an agent whose
                # output is not text, nothing but the model's typed output ends the turn
                await self._say(_tool_text(said.output), record=False)
            return out

        # what work in the background queued, in place of a message: one reply, or its question
        delivery = hooks.delivery if hooks is not None else ()
        for queued in delivery:
            for item in (queued.fnc_call, queued.fnc_call_out):
                item.created_at = time.time()  # told now, after all the history holds
                record(item)
            if isinstance(queued.transition, InputRequired):
                # as on a call, no reply is generated: the question is what the person hears
                return await ask(queued.transition, queued.fnc_call)
        hint = None
        if delivery:
            call_ids = [queued.fnc_call.call_id for queued in delivery]
            hint = _reply_instructions(self._tool_options(), at_tail=True, call_ids=call_ids)
        else:
            record(self._message)
        # a task that answers the person itself, a waiting graph, does before any model
        taken = (
            await self._agent._take(self._message)
            if isinstance(self._agent, AgentTask) and not delivery
            else None
        )
        if taken is not None and (ended := await act(taken, None)) is not None:
            return ended
        # the message answers a question whose tool takes it: that call comes first, no model
        answer_tool = hooks.answer_tool if hooks is not None else None
        answer_ref = hooks.answer_ref if hooks is not None else None
        forwarded: FunctionCall | None = None

        # a toolset whose setup failed is not tried again in this run; the next run tries again
        unready: set[Toolset] = set()
        # as on a call: the reply to a report calls no tools, the reply to a result may
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN
        for _ in range(self._max_steps):
            if hooks is not None and hooks.boundary is not None and await hooks.boundary():
                # a newer message waits for the lane: stop here, what ran is kept, no reply
                return finish(None, interrupted=True)
            agent = self._agent
            run_ctx = RunContext[Any, Any](function_call=None, run=self)
            update_instructions(
                chat_ctx,
                instructions=await agent._render_instructions(run_ctx, modality="text"),
                add_if_missing=True,
                modality="text",
            )

            tools: list[Tool | Toolset] = list(agent.tools)
            if hooks is not None and hooks.tools is not None and has_cancellable_tool(tools):
                # as on a call, a cancellable tool's work in the background can be cancelled
                tools += [cancel_task, get_running_tasks]
            for toolset in tools:
                if not isinstance(toolset, Toolset) or toolset in unready:
                    continue
                try:
                    await toolset.setup()  # once per toolset object; a no-op after
                except Exception as e:
                    # as on a call: the run goes on with the toolset's tools as they are
                    unready.add(toolset)
                    logger.warning(
                        "a toolset could not be set up",
                        extra={"toolset": toolset.id, "error": str(e)},
                    )
            wants_tool_output = self._output_type is not None and self._output_type is not str
            if wants_tool_output:
                tools.append(self._final_result_tool())
            tool_ctx = ToolContext(tools)

            forwarded = self._forward_answer(tool_ctx, answer_tool)
            answer_tool = None
            if forwarded is not None:
                text, calls = "", [forwarded]
            else:
                on_token = self._on_token or (hooks.on_token if hooks is not None else None)
                request = _without_pending_calls(chat_ctx, self._hidden)
                if hint is not None:
                    # as on a call: told last, next to what it is about, and not recorded
                    request.add_message(role="system", content=[hint])
                    hint = None
                text, calls, usage = await self._infer(
                    request,
                    tool_ctx,
                    on_token=on_token,
                    tool_choice=tool_choice,
                )
                self._usage._add_completion(usage)
                if tool_choice == "none" and calls:
                    logger.debug(
                        "dropping tool calls made in the reply to a report",
                        extra={"functions": [c.name for c in calls], "runId": self._id},
                    )
                    calls = []
            tool_choice = NOT_GIVEN

            if text:
                record(ChatMessage(role="assistant", content=[text]))

            if not calls:
                if isinstance(agent, AgentTask):
                    return finish_task_turn(text)
                if wants_tool_output:
                    record(ChatMessage(role="user", content=[_FINAL_RESULT_PROMPT]))
                    continue
                retry = await self._validate(run_ctx, text)
                if retry is not None:
                    record(ChatMessage(role="user", content=[retry]))
                    continue
                return finish(text)

            for fnc_call in calls:
                self._usage.tool_calls += 1
                record(fnc_call)
            # a verbatim result the model goes on from is said when its tool returns; one that
            # ends the turn waits for the other calls, as whether it is the answer depends on them
            running = {
                fnc_call.call_id: asyncio.ensure_future(
                    run_call(fnc_call, tool_ctx, wants_tool_output)
                )
                for fnc_call in calls
                if not (fnc_call.name == FINAL_RESULT_TOOL and wants_tool_output)
            }
            try:
                await asyncio.gather(*running.values())
            except BaseException:
                await utils.aio.cancel_and_wait(*running.values())
                raise
            outs = {call_id: task.result() for call_id, task in running.items()}

            replies = stopped = False
            final: tuple[Any] | None = None
            questions: list[tuple[FunctionCall, InputRequired]] = []
            for fnc_call in calls:
                out = outs.get(fnc_call.call_id)
                if out is None:
                    output, error = self._parse_final_result(fnc_call)
                    if error is None:
                        error = await self._validate(run_ctx, output)
                    record(
                        FunctionCallOutput(
                            name=fnc_call.name,
                            call_id=fnc_call.call_id,
                            output=error if error is not None else "final result recorded",
                            is_error=error is not None,
                        )
                    )
                    if error is None and final is None:
                        final = (output,)
                    continue
                if isinstance(out, AgentTask):
                    continue  # the call stays pending; the task takes the conversation
                said = out.raw_output
                if isinstance(said, ToolResult) and said.verbatim:
                    if not said.reply_required and not wants_tool_output:
                        held.append(_tool_text(said.output))
                    out.fnc_call_out.output = delivered(said).output
                record(out.fnc_call_out)
                replies = replies or out.fnc_call_out.reply_required
            record_said()

            if final is not None:
                await release_held()
                return finish(final[0])

            for fnc_call in calls:
                out = outs.get(fnc_call.call_id)
                if out is None:
                    continue
                if isinstance(out, AgentTask):
                    await release_held()
                    entered = await self._push(out, fnc_call, record)
                    if is_transition(entered) and (ended := await act(entered, None)) is not None:
                        return ended
                    break
                if out.transition is not None:
                    await release_held()
                if out.transition is not None and isinstance(agent, AgentTask):
                    if (ended := await act(out.transition, fnc_call)) is not None:
                        return ended
                    if not isinstance(out.transition, Working):
                        break  # the task ended, or it asked; the agent below speaks next
                elif isinstance(out.transition, InputRequired):
                    questions.append((fnc_call, out.transition))
                elif out.transition is not None:
                    logger.warning(
                        "a conversation tool returned a transition that only an AgentTask "
                        "acts on; recorded as its output",
                        extra={"function": fnc_call.name, "runId": self._id},
                    )
                if out.agent_task is not None:
                    self._hand_off(out.agent_task, record)
                if isinstance(out.raw_exception, StopResponse):
                    stopped = True
                    break

            if questions:
                # the step's questions are one, asked once its every output is known
                takers = [(c, q) for c, q in questions if q._answer_ref]
                if len(takers) > 1:
                    logger.warning(
                        "several questions of one step go back to their tools; the first takes "
                        "the answer",
                        extra={"functions": [c.name for c, _ in takers], "runId": self._id},
                    )
                call, first = takers[0] if takers else questions[0]
                question = InputRequired(
                    "\n\n".join(dict.fromkeys(q.question for _, q in questions)),
                    verbatim=all(q.verbatim for _, q in questions),
                    _answer_ref=first._answer_ref,
                )
                return await ask(question, call)
            if held and (stopped or not replies):
                # the one place the step's exact results become one text
                return finish("\n\n".join(held), verbatim=True)
            if stopped:
                return finish(None)
            reports = _reply_tool_choice(
                out.reply_tool_choice for out in outs.values() if not isinstance(out, AgentTask)
            )
            tool_choice = reports if reports is not None else NOT_GIVEN
            if tool_choice == "none" and not wants_tool_output and not isinstance(agent, AgentTask):
                # every call went to the background: its progress went out as it is and the
                # receiver words it, so the turn ends on the reply beside the calls, if any
                return finish(text or None)
            await release_held()

        raise UnexpectedModelBehavior(
            f"run `{self._id}` did not finish within {self._max_steps} steps"
        )

    async def _say(self, text: str, *, record: bool = True) -> None:
        """Exact text to the person now: the door sends it as it is said. With ``record`` the
        history keeps it as the agent's message, after the call output that said it; a verbatim
        result is kept as its output instead."""
        hooks = None if self._nested else _run_hooks.get()
        if hooks is not None and hooks.on_say is not None:
            out = hooks.on_say(text)
            if inspect.isawaitable(out):
                await out
        if record:
            self._said.append(ChatMessage(role="assistant", content=[text]))

    async def _phrase(self, chat_ctx: ChatContext, question: str) -> str:
        """A question in the agent's own words: one model request on the history, no tools.
        An agent with no model, a graph or a code task, asks it as written."""
        if self.llm is None:
            return question
        ctx = chat_ctx.copy()
        run_ctx = RunContext[Any, Any](function_call=None, run=self)
        update_instructions(
            ctx,
            instructions=await self._agent._render_instructions(run_ctx, modality="text"),
            add_if_missing=True,
            modality="text",
        )
        ctx.add_message(role="system", content=ASK_IN_OWN_WORDS.format(question=question))
        text, _, usage = await self._infer(
            _without_pending_calls(ctx, self._hidden), ToolContext([])
        )
        self._usage._add_completion(usage)
        return text.strip() or question

    async def _infer(
        self,
        chat_ctx: ChatContext,
        tool_ctx: ToolContext,
        *,
        on_token: Callable[[str], Awaitable[None]] | None = None,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
    ) -> tuple[str, list[FunctionCall], CompletionUsage | None]:
        """One model request through the agent's ``llm_node``: text, calls and usage; each text
        delta reaches ``on_token`` as it arrives."""
        node = self._agent.llm_node(
            chat_ctx, tool_ctx.flatten(), ModelSettings(tool_choice=tool_choice)
        )
        if inspect.iscoroutine(node):
            node = await node

        if isinstance(node, str):
            if on_token is not None and node:
                await on_token(node)
            return node, [], None
        if not isinstance(node, AsyncIterable):
            raise UnexpectedModelBehavior("llm_node returned neither text nor a stream")

        text_parts: list[str] = []
        calls: list[FunctionCall] = []
        usage: CompletionUsage | None = None
        generation_id = utils.shortuuid("item_")
        async for chunk in node:
            if isinstance(chunk, str):
                text_parts.append(chunk)
                if on_token is not None and chunk:
                    await on_token(chunk)
                continue
            if not isinstance(chunk, ChatChunk):
                continue
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.delta is None:
                continue
            if chunk.delta.content:
                text_parts.append(chunk.delta.content)
                if on_token is not None:
                    await on_token(chunk.delta.content)
            for tool_call in chunk.delta.tool_calls:
                if tool_call.type != "function":
                    continue
                calls.append(
                    FunctionCall(
                        id=f"{generation_id}/fnc_{len(calls)}",
                        call_id=tool_call.call_id,
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                        extra=tool_call.extra or {},
                    )
                )
        return "".join(text_parts), calls, usage

    async def _push(
        self,
        task: AgentTask[Any, Any],
        fnc_call: FunctionCall,
        record: Callable[[ChatItem], None],
    ) -> Any:
        """A tool awaited a task: it goes on the stack, the call it answers stays pending, and
        its ``on_enter`` runs; what that returns is the caller's to act on."""
        record(AgentHandoff(old_agent_id=self._agent.id, new_agent_id=task.id))
        task._pending_call_id = fnc_call.call_id
        self._stack.append(task)
        self._agent = task
        self._stack_dirty = True
        return await task._enter(RunContext[Any, Any](function_call=fnc_call, run=self))

    def _forward_answer(self, tool_ctx: ToolContext, name: str | None) -> FunctionCall | None:
        """The call that forwards this run's message, the person's answer, to the tool whose
        question it answers; ``None`` when nothing waits on it."""
        if name is None:
            return None
        tool = tool_ctx.get_function_tool(name)
        if tool is None:
            logger.warning(
                "the tool that takes the answer is not on the agent; the model reads it",
                extra={"tool": name, "runId": self._id},
            )
            return None
        arguments = answer_arguments(tool, self._message.text_content or "")
        if arguments is None:
            return None  # no string parameter: the model reads the answer
        return FunctionCall(
            name=name, arguments=json.dumps(arguments), call_id=utils.shortuuid("call_")
        )

    async def _execute(
        self,
        tool: FunctionTool | RawFunctionTool | None,
        fnc_call: FunctionCall,
        *,
        answer_ref: str | None = None,
    ) -> ToolExecutionOutput | AgentTask[Any, Any]:
        """Run one tool with a ``RunContext`` of this run and ``answer_ref``; the output is what
        the model reads, or the ``AgentTask`` the tool awaited."""
        if tool is None:
            return make_tool_output(
                fnc_call=fnc_call,
                output=None,
                exception=ToolError(f"unknown tool `{fnc_call.name}`"),
            )

        try:
            raw = parse_function_arguments(fnc_call.arguments or "{}")
        except ValueError as e:
            error = ToolError(f"Error parsing arguments for `{fnc_call.name}`: {e}")
            return make_tool_output(fnc_call=fnc_call, output=None, exception=error)
        run_ctx = RunContext[Any, Any](function_call=fnc_call, run=self)
        run_ctx._answer_ref = answer_ref
        executor = self._tools.executor
        executor.set_tool_options(self._tool_options())
        try:
            output = await executor.execute(tool=tool, run_ctx=run_ctx, raw_arguments=raw)
            work = run_ctx._work
            if work is not None and fnc_call.call_id not in self._background_calls:
                await asyncio.wait({work})  # its task ends right after its output
                if work.cancelled():
                    error = ToolError("the call was cancelled")
                    return make_tool_output(fnc_call=fnc_call, output=None, exception=error)
        except TaskAwaited as e:
            return e.task
        except ModelRetry as e:
            return make_tool_output(fnc_call=fnc_call, output=None, exception=ToolError(e.message))
        except Exception as e:
            if not isinstance(e, (ToolError, StopResponse)):
                logger.exception(
                    "exception occurred while executing tool",
                    extra={"function": fnc_call.name, "runId": self._id},
                )
            return make_tool_output(fnc_call=fnc_call, output=None, exception=e)
        except BaseException:
            if run_ctx._work is not None and fnc_call.call_id not in self._background_calls:
                run_ctx._work.cancel()
            raise

        if (
            fnc_call.call_id not in self._background_calls
            and isinstance(output, Completed)
            and not isinstance(self._agent, AgentTask)
        ):
            # a conversation tool never ends a task; the value is its output
            output = ToolResult(output.value, verbatim=True) if output.verbatim else output.value
        out = make_tool_output(fnc_call=fnc_call, output=output, exception=None)
        out.reply_tool_choice = run_ctx._reply_tool_choice
        return out

    def _tool_options(self) -> AsyncToolOptions:
        """The agent's templates for its tools' progress and duplicates, as on a call."""
        options = self._agent._async_tool_options
        return _resolve_async_tool_options(options if is_given(options) else None)

    async def _validate(self, run_ctx: RunContext[Any, Any], output: Any) -> str | None:
        """Run the agent's output validators; the retry reason when one asks for it."""
        try:
            await self._agent._validate_output(run_ctx, output)
        except ModelRetry as e:
            return _RETRY_PROMPT.format(reason=e.message)
        return None

    def _final_result_tool(self) -> RawFunctionTool:
        assert self._output_type is not None
        schema = TypeAdapter(self._output_type).json_schema()
        schema.pop("title", None)

        async def final_result(raw_arguments: dict[str, Any]) -> str:
            return "final result recorded"

        return function_tool(
            final_result,
            raw_schema={
                "name": FINAL_RESULT_TOOL,
                "description": "The final result of the run, once you have it.",
                "parameters": schema,
            },
        )

    def _parse_final_result(self, fnc_call: FunctionCall) -> tuple[Any, str | None]:
        assert self._output_type is not None
        try:
            arguments = json.loads(fnc_call.arguments or "{}")
            return TypeAdapter(self._output_type).validate_python(arguments), None
        except (ValueError, ValidationError) as e:
            return None, f"invalid {FINAL_RESULT_TOOL}: {e}"


def _without_pending_calls(
    chat_ctx: ChatContext, hidden: frozenset[str] = frozenset()
) -> ChatContext:
    """The history as the model reads it: a call still waiting on a task has no output yet,
    and a request cannot carry one, so it is left out until the task completes; an answered
    read-back says the call has not run; the ``hidden`` items of other open tasks are left out."""
    answered = {item.call_id for item in chat_ctx.items if isinstance(item, FunctionCallOutput)}
    pending = [
        item
        for item in chat_ctx.items
        if isinstance(item, FunctionCall) and item.call_id not in answered
    ]
    kept = [item for item in chat_ctx.items if item not in pending and item.id not in hidden]
    return ChatContext(as_model_reads(kept))


def _tool_text(value: Any) -> str:
    return value if isinstance(value, str) else _serialize_output(value)
