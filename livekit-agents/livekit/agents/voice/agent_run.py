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
import contextlib
import contextvars
import inspect
import json
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

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
    ANSWERS_KEY,
    ASK_IN_OWN_WORDS,
    ASKED_VERBATIM_KEY,
    CONFIRM_DUPLICATE_PARAM,
    DELIVERED,
    HEARD_KEY,
    TASK_KEY,
    WAITING_PREFIX,
    Completed,
    Failed,
    InputRequired,
    ModelRetry,
    Rejected,
    StopResponse,
    ToolError,
    ToolFlag,
    ToolResult,
    Transition,
    Working,
    _tasks_of,
    approval_granted,
    approval_scope,
    as_model_reads,
    delivered,
    function_tool,
    is_transition,
    joint_question,
    relays_answer,
)
from ..llm.utils import answer_arguments, prepare_function_arguments
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
from .tool_executor import NON_BLOCKING_KEY, REPLY_INSTRUCTIONS_AT_TAIL

if TYPE_CHECKING:
    from ..storage.hub import LaneHub
    from .agent import Agent

Run_T = TypeVar("Run_T")

DEFAULT_MAX_STEPS = 12
WATCH_INTERVAL = 0.5
"""How often a running tool's turn asks whether it was canceled or interrupted, in seconds,
when nothing in the process signals it."""
WATCH_BACKSTOP = 5.0
"""With a lane hub that signals a cancel or an interrupting message, how often it asks anyway."""
CUT_OFF_QUESTION = (
    "The person's message cut your question off: they had seen {heard} of it. Read the message "
    "with that in mind; it may not answer the question."
)
"""Told with the message that cut off a question no delegate relays: the model decides."""
JOINT_QUESTION = "After your reply the person is asked this, so do not ask it yourself: {question}"
"""Told with results that ended together when one of them asks the person a question."""
FINAL_RESULT_TOOL = "final_result"
_RETRY_PROMPT = "Validation failed, fix the output and try again: {reason}"
_FINAL_RESULT_PROMPT = (
    f"You have not provided the final output yet. Call `{FINAL_RESULT_TOOL}` with it; "
    "a plain text response alone is not enough."
)


class BoundaryCheck(Protocol):
    """Whether the turn yields now; ``read=True`` reads even right after the turn took its
    batch, because the caller was told that something landed."""

    def __call__(self, read: bool = False) -> Awaitable[bool]: ...


class _ReplyInterrupted(Exception):
    """A message that interrupts landed while the model's reply streamed: what was shown."""

    def __init__(self, shown: str) -> None:
        super().__init__("the reply was interrupted")
        self.shown = shown


@dataclass
class RunHooks:
    """What a door installs around the ``agent.run()`` calls made inside its turn.

    ``boundary`` is awaited before each model call; ``True`` makes the run yield there with what
    ran so far and no reply (the ``interrupt`` policy of plan 3.3.4). ``on_result`` receives every
    result, so the door can commit what the run added without the handler forwarding it.
    ``on_update`` receives a tool's ``ctx.update(text)``: the door forwards it as progress, a
    ``working`` frame on the text door, speech on a call. ``on_token`` receives the model's text
    as it arrives, so the text door shows the reply as it is written. ``thread_id`` is the session the run
    belongs to, what ``A2ADelegate`` sends as the expert's ``contextId`` (plan 7.5).
    ``agent_stack`` is the lane's stack as loaded and ``rebuild`` makes an agent of one entry,
    the server's registry; the run resolves them itself since only it knows the root.
    ``answer_call_id`` is the call whose question this message answers: the calls of that
    question that relay answers (a delegate's) take it as said before the model runs, and
    the task's model reads it; ``cut_off``, ``{question, heard, verbatim}``, says this
    message cut that question off. ``delegated`` says the turn's message
    came from another agent: the run does not phrase its questions, the agent facing the person
    does. The person has one question in front of them: ``front_question`` is another open one,
    and a question this run asks waits behind it unasked; ``relay`` is the oldest waiting
    question, ``{task_id, question, verbatim}``, which the reply asks when this run asks none.
    Under the ``live`` policy ``background`` takes a tool that called ``ctx.update()``: the
    update is the call's output, the run goes on, and the tool keeps running; ``resume`` is the
    final output of such tools, which a later turn of the task reads in place of a message.
    While a tool runs, or the model's reply streams, the run reads ``boundary`` when the lane's
    ``hub`` signals a cancel or a message, or every so often: when it trips, the reply stops, a
    client's cancel (``canceled``) stops the tool, an interrupt stops it only when it is
    ``CANCELLABLE``.
    ``lane`` is the door's lane; ``record_message`` writes the turn's message to it ahead of the
    commit, for a delegate about to send; ``running`` is the lane's calls in the background,
    which a tool's ``on_duplicate`` is checked against. ``task_id`` is the task the turn runs
    under, which marks what the run records (``TASK_KEY``); ``open_tasks`` are the lane's other
    tasks still open, whose items after this task's own last one the model does not read.
    """

    boundary: BoundaryCheck | None = None
    on_result: Callable[[RunResult[Any]], None] | None = None
    on_update: Callable[[str], Any] | None = None
    on_token: Callable[[str], Awaitable[None]] | None = None
    thread_id: str | None = None
    agent_stack: Sequence[Ref] = ()
    rebuild: Callable[[Ref], Agent | None] | None = None
    answer_call_id: str | None = None
    cut_off: Mapping[str, Any] | None = None
    delegated: bool = False
    front_question: str | None = None
    relay: Mapping[str, Any] | None = None
    background: Callable[[FunctionCall, asyncio.Task[Any], RunContext[Any, Any]], None] | None = (
        None
    )
    resume: Sequence[ToolExecutionOutput] | None = None
    canceled: Callable[[], bool] | None = None
    lane: str | None = None
    record_message: Callable[[], Awaitable[None]] | None = None
    running: Callable[[], Sequence[RunContext[Any, Any]]] | None = None
    hub: LaneHub | None = None
    task_id: str | None = None
    open_tasks: frozenset[str] = frozenset()


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
        self._hooks: RunHooks | None = None
        self._backgrounded = False
        # the items of other open tasks the model does not read
        self._hidden: frozenset[str] = frozenset()

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
        self._hooks = hooks
        if hooks is not None:
            self._stack = self._resolve_stack(hooks)
            if self._stack:
                self._agent = self._stack[-1]

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
        and the agent below speaks again from the next step. A verbatim value reached the
        person as written; the agent below reads that it did."""
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
            # the model reads none of what the lane's other open tasks added after this turn's
            # own last item: work the continued task did not see when it paused
            tasks = _tasks_of(chat_ctx.items)
            resumed = (f.fnc_call.extra.get(TASK_KEY) for f in hooks.resume or ())
            own = {task for task in (task_id, *resumed) if task}
            ends = [n for n, item in enumerate(chat_ctx.items) if tasks[item.id] in own]
            if ends:
                self._hidden = frozenset(
                    item.id
                    for item in chat_ctx.items[ends[-1] + 1 :]
                    if tasks[item.id] in hooks.open_tasks
                )

        def record(item: ChatItem) -> None:
            if task_id is not None and isinstance(item, (ChatMessage, FunctionCall)):
                item.extra.setdefault(TASK_KEY, task_id)
            chat_ctx.insert(item)
            new_items.append(item)

        # what this turn told the person beside its tool calls, in order, and whether as
        # written: it leads the question the turn ends on, and the verbatim parts lead the answer
        shown: list[tuple[str, bool]] = []
        # a run another agent asked does not phrase: the agent facing the person does
        facing_person = hooks is None or not hooks.delegated

        def finish(
            output: Any, *, input_required: str | None = None, interrupted: bool = False
        ) -> RunResult[Run_T]:
            verbatim = [t for t, exact in shown if exact]
            if verbatim and input_required is None and not interrupted:
                if output is None or isinstance(output, str):
                    output = "\n\n".join([*verbatim, *([output] if output else [])])
            result._verbatim = "\n\n".join(verbatim) if verbatim else None
            result._run_output = output
            result._input_required = input_required
            result._interrupted = interrupted
            result._run_messages = new_items
            result._run_history = chat_ctx.copy(exclude_instructions=True).items
            result._agent_stack = self._stack_refs()
            result._background = self._backgrounded
            result._done_fut.set_result(None)
            return result

        # another question is in front of the person: one this run asks waits behind it
        queues = facing_person and hooks is not None and hooks.front_question is not None

        async def yielded() -> bool:
            """A newer message interrupted the turn: it ends where it is, with what ran."""
            return hooks is not None and hooks.boundary is not None and await hooks.boundary()

        def relaying(call: FunctionCall | None) -> bool:
            if call is None:
                return False
            return relays_answer(ToolContext(self._agent.tools).get_function_tool(call.name))

        async def ask(
            questions: Sequence[tuple[InputRequired, FunctionCall | None]],
            *,
            answered: str | None = None,
        ) -> RunResult[Run_T]:
            """The task pauses once, on one question that holds every part its calls asked: as
            written when every part is verbatim or another agent asked, else in the agent's own
            words; unasked when it waits behind the question in front of the person. A newer
            message that waits cuts the question off: the turn waits on it unshown, and that
            message is its answer. ``answered`` is the reply that answers the tasks resumed
            with this one, which do not wait."""
            q = joint_question([part for part, _ in questions])
            calls = [call for _, call in questions if call is not None]
            for part, call in questions:
                if call is not None and part.verbatim:
                    call.extra[ASKED_VERBATIM_KEY] = True
            # the call the question's row names: a delegate's, when one relays the answer
            relay = next((call for call in calls if relaying(call)), None)
            result._pending_call = relay or (calls[0] if calls else None)
            result._answer_tool = relay.name if relay is not None else None
            result._answered = answered
            if not queues and await yielded():
                result._heard = ""
                ended = finish(None, input_required=q.question)
                if q.verbatim:
                    result._verbatim = q.question
                return ended
            text = q.question
            if not q.verbatim and facing_person and not queues:
                text = await self._phrase(chat_ctx, text)
            if not queues:
                record(ChatMessage(role="assistant", content=[text]))
            result._queued = queues
            # what the turn told the person first goes with the question: the one who relays it
            # sees only this message. Unasked, the question waits alone
            ended = finish(None, input_required=text if queues else asked(text))
            if q.verbatim:
                parts = [t for t, exact in shown if exact] + [text]
                # exact as a whole only when every part is
                result._verbatim = (
                    ended.input_required if all(e for _, e in shown) else ("\n\n".join(parts))
                )
            return ended

        def asked(question: str) -> str:
            return "\n\n".join([*(t for t, _ in shown), question])

        def open_question(task: str | None) -> list[tuple[InputRequired, FunctionCall]]:
            """The parts of the question ``task`` waits on: its calls that asked and that no
            message of the task, nor a relayed answer, came after."""
            items = chat_ctx.items
            tasks = _tasks_of(items)
            calls = {i.call_id: i for i in items if isinstance(i, FunctionCall)}
            parts: dict[str, tuple[InputRequired, FunctionCall]] = {}
            for item in items:
                if tasks.get(item.id) != task:
                    continue
                if isinstance(item, ChatMessage) and item.role == "user":
                    parts.clear()
                elif isinstance(item, FunctionCall) and item.extra.get(ANSWERS_KEY):
                    parts.pop(str(item.extra[ANSWERS_KEY]), None)
                elif (
                    isinstance(item, FunctionCallOutput)
                    and str(item.output).startswith(WAITING_PREFIX)
                    and item.call_id in calls
                ):
                    call = calls[item.call_id]
                    question = str(item.output).removeprefix(WAITING_PREFIX)
                    verbatim = bool(call.extra.get(ASKED_VERBATIM_KEY))
                    parts[item.call_id] = (InputRequired(question, verbatim=verbatim), call)
            return list(parts.values())

        async def finish_or_ask(text: str) -> RunResult[Run_T]:
            """The turn's reply ends it, unless its task still waits on a question, as a resumed
            turn's task can: the reply goes first, then that question again."""
            waiting = open_question(task_id) if hooks is not None and hooks.resume else []
            if not waiting:
                return finish(text)
            if text:
                shown.append((text, False))
            return await ask(waiting)

        def finish_task_turn(text: str) -> RunResult[Run_T]:
            # a task holds the conversation: every reply of its model waits on the person
            return finish(None, input_required=asked(text))

        async def act(
            transition: Transition, fnc_call: FunctionCall | None
        ) -> RunResult[Run_T] | None:
            """A task's transition, from a tool or ``on_enter``: the result when it ends the turn."""
            if not self._stack and isinstance(self._agent, AgentTask):
                # the run's root is the task itself, as in a test: nothing below takes over, so
                # its end is the run's end, with its value or its error as the output
                if isinstance(transition, Completed):
                    if transition.verbatim:
                        shown.append((_tool_text(transition.value), True))
                    self._agent._complete(transition.value)
                    return finish(transition.value)
                if isinstance(transition, (Failed, Rejected)):
                    text = transition.error if isinstance(transition, Failed) else transition.reason
                    self._agent._complete(ToolError(text))
                    return finish(text)
            if isinstance(transition, Completed):
                if transition.verbatim:
                    shown.append((_tool_text(transition.value), True))
                self._pop(transition.value, record, verbatim=transition.verbatim)
            elif isinstance(transition, (Failed, Rejected)):
                text = transition.error if isinstance(transition, Failed) else transition.reason
                self._pop(ToolError(text), record)
            elif isinstance(transition, InputRequired):
                return await ask([(transition, fnc_call)])
            elif isinstance(transition, Working):
                logger.warning(
                    "Working is not applied in text yet; the task keeps the conversation",
                    extra={"agent": self._agent.id, "runId": self._id},
                )
            return None

        # what the first model step is told on top of its instructions
        hint: str | None = None

        # of tasks resumed together, the question of the first one whose work asked, with what
        # that task already waited on: the others are answered first
        asking: list[tuple[InputRequired, FunctionCall | None]] = []
        # what the resumed work returned, but for the asking one: an expert's answer carries it
        told = ""

        async def steps() -> RunResult[Run_T]:
            nonlocal hint, asking, told
            resumed = hooks.resume if hooks is not None else None
            relays: list[FunctionCall] = []
            if resumed is not None:
                # the tools in the background ended, of one task or of several at once: their
                # outputs, not a message, are what this turn answers, in one reply
                if facing_person and resumed:
                    # no reply had these results in front of it, whatever it said while the
                    # work ran: this turn tells them
                    hint = REPLY_INSTRUCTIONS_AT_TAIL.format(
                        call_ids=", ".join(f.fnc_call.call_id for f in resumed)
                    )
                # a task that already waits on a question asks it again, joined with what its
                # work asks now
                waiting = {
                    owner: open_question(owner)
                    for owner in dict.fromkeys(f.fnc_call.extra.get(TASK_KEY) for f in resumed)
                }
                by_task: dict[Any, list[tuple[InputRequired, FunctionCall | None]]] = {}
                quiet = True  # every result reached the person as written, and wants no reply
                for finished in resumed:
                    record(finished.fnc_call)
                    said = finished.raw_output
                    if isinstance(said, ToolResult) and said.verbatim:
                        shown.append((_tool_text(said.output), True))
                        finished.fnc_call_out.output = delivered(said).output
                    record(finished.fnc_call_out)
                    if isinstance(finished.transition, InputRequired):
                        owner = finished.fnc_call.extra.get(TASK_KEY)
                        if owner not in by_task:
                            by_task[owner] = list(waiting[owner])
                        by_task[owner].append((finished.transition, finished.fnc_call))
                        quiet = False
                    elif not (
                        isinstance(said, ToolResult) and said.verbatim and not said.reply_required
                    ):
                        quiet = False
                for owner, parts in by_task.items():
                    if not asking:
                        asking = parts
                        continue
                    # another task's question: it waits behind the first
                    q = joint_question([part for part, _ in parts])
                    call = parts[0][1]
                    assert call is not None
                    relay = next((c for _, c in parts if relaying(c)), None)
                    result._waiting.append(
                        (
                            str(owner),
                            q.question,
                            relay or call,
                            relay.name if relay is not None else None,
                            q.verbatim,
                        )
                    )
                told = "\n\n".join(
                    _tool_text(f.fnc_call_out.output)
                    for f in resumed
                    if not isinstance(f.transition, InputRequired)
                )
                if asking and len(resumed) == 1:
                    return await ask(asking)
                if asking and hint is not None:
                    question = joint_question([part for part, _ in asking]).question
                    hint += "\n" + JOINT_QUESTION.format(question=question)
                if not asking and resumed and quiet:
                    return finish(None, interrupted=await yielded())
            else:
                # the open question this message answers, before the message is recorded: its
                # calls that relay take the answer as said
                answering = (
                    hooks.answer_call_id if hooks is not None and hooks.answer_call_id else None
                )
                owner = next(
                    (
                        i.extra.get(TASK_KEY)
                        for i in chat_ctx.items
                        if isinstance(i, FunctionCall) and i.call_id == answering
                    ),
                    task_id,
                )
                relays = [c for _, c in open_question(owner) if relaying(c)] if answering else []
                record(self._message)
                # a task that answers the person itself, a waiting graph, does before any model
                taken = (
                    await self._agent._take(self._message)
                    if isinstance(self._agent, AgentTask)
                    else None
                )
                if taken is not None and (ended := await act(taken, None)) is not None:
                    return ended
            cut_off = hooks.cut_off if hooks is not None and resumed is None else None
            heard = str(cut_off.get("heard") or "") if cut_off is not None else None
            if cut_off is not None and not relays:
                # the model reads the answer, knowing what the person had of the question
                hint = CUT_OFF_QUESTION.format(heard=f'only "{heard}"' if heard else "none")

            progress_only = False
            for _ in range(self._max_steps):
                if await yielded():
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
                for toolset in tools:
                    if isinstance(toolset, Toolset):
                        await toolset.setup()  # once per toolset object; a no-op after
                wants_tool_output = self._output_type is not None and self._output_type is not str
                if wants_tool_output:
                    tools.append(self._final_result_tool())
                tool_ctx = ToolContext(tools)

                answer_calls = self._answer_calls(tool_ctx, relays, heard)
                relays = []
                if answer_calls:
                    text, calls = "", answer_calls
                else:
                    on_token = self._on_token or (hooks.on_token if hooks is not None else None)
                    request = _without_pending_calls(chat_ctx, self._hidden)
                    if hint is not None:
                        # as on a call: told last, next to the results, and not recorded
                        request.add_message(role="system", content=[hint])
                        hint = None
                    text, calls, usage = await self._infer(
                        request,
                        tool_ctx,
                        on_token=on_token,
                        # as on a call, the reply to progress calls nothing: the work runs
                        tool_choice="none" if progress_only else NOT_GIVEN,
                    )
                    self._usage._add_completion(usage)
                    if progress_only and calls:
                        # a model that calls anyway (gemma does): the calls are not run
                        logger.debug(
                            "dropping tool calls made in the reply to progress",
                            extra={"calls": [c.name for c in calls], "runId": self._id},
                        )
                        calls = []

                if text:
                    record(ChatMessage(role="assistant", content=[text]))
                    if calls:
                        shown.append((text, False))

                if not calls:
                    if isinstance(agent, AgentTask):
                        return finish_task_turn(text)
                    if wants_tool_output:
                        record(ChatMessage(role="user", content=[_FINAL_RESULT_PROMPT]))
                        continue
                    if not text and resumed and not facing_person:
                        # an expert's answer must carry the result, also when its model was silent
                        text = told
                    retry = await self._validate(run_ctx, text)
                    if retry is not None:
                        record(ChatMessage(role="user", content=[retry]))
                        continue
                    if asking:
                        # the others are answered with the reply; the asking task waits
                        return await ask(asking, answered=text or told)
                    return await finish_or_ask(text)

                # the reply beside the calls told the person already: a call that went to the
                # background with its first update is progress, and asks for no second reply
                needs_reply = not text or wants_tool_output or isinstance(agent, AgentTask)
                # the step's questions are asked as one, once all its calls ran
                questions: list[tuple[InputRequired, FunctionCall | None]] = []
                for i, fnc_call in enumerate(calls):
                    self._usage.tool_calls += 1
                    record(fnc_call)

                    if fnc_call.name == FINAL_RESULT_TOOL and wants_tool_output:
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
                        if error is None:
                            return finish(output)
                        continue

                    tool = tool_ctx.get_function_tool(fnc_call.name)
                    out = await self._execute(tool, fnc_call)
                    if isinstance(out, AgentTask) or not fnc_call.extra.get(NON_BLOCKING_KEY):
                        needs_reply = True
                    if isinstance(out, AgentTask):
                        # the call stays pending; the task takes the conversation from here
                        for skipped in calls[i + 1 :]:
                            record(skipped)
                            record(
                                FunctionCallOutput(
                                    name=skipped.name,
                                    call_id=skipped.call_id,
                                    output=f"not run: {type(out).__name__} took the conversation",
                                    is_error=True,
                                )
                            )
                        entered = await self._push(out, fnc_call, record)
                        if (
                            is_transition(entered)
                            and (ended := await act(entered, None)) is not None
                        ):
                            return ended
                        break
                    said = out.raw_output
                    if isinstance(said, ToolResult) and said.verbatim:
                        # the person gets it as written; the model reads that they did
                        shown.append((_tool_text(said.output), True))
                        out.fnc_call_out.output = delivered(said).output
                    record(out.fnc_call_out)
                    if isinstance(said, ToolResult) and said.verbatim and not said.reply_required:
                        if questions:
                            return await ask(questions)
                        return finish(None, interrupted=await yielded())

                    if out.transition is not None and isinstance(agent, AgentTask):
                        if (ended := await act(out.transition, fnc_call)) is not None:
                            return ended
                        if not isinstance(out.transition, Working):
                            break  # the task ended, or it asked; the agent below speaks next
                    elif isinstance(out.transition, InputRequired):
                        questions.append((out.transition, fnc_call))
                    elif out.transition is not None:
                        logger.warning(
                            "a conversation tool returned a transition that only an AgentTask "
                            "acts on; recorded as its output",
                            extra={"function": fnc_call.name, "runId": self._id},
                        )
                    if out.agent_task is not None:
                        self._hand_off(out.agent_task, record)
                    if isinstance(out.raw_exception, StopResponse):
                        return await ask(questions) if questions else finish(None)
                if questions:
                    return await ask(questions)
                if not needs_reply and not asking:
                    return await finish_or_ask(text)
                # every call went to the background: the next step only reads their progress
                progress_only = all(c.extra.get(NON_BLOCKING_KEY) for c in calls)

            raise UnexpectedModelBehavior(
                f"run `{self._id}` did not finish within {self._max_steps} steps"
            )

        try:
            ended = await steps()
        except _ReplyInterrupted as e:
            # what was shown stays in the history, marked as cut off; the task ends with it
            if e.shown:
                record(ChatMessage(role="assistant", content=[e.shown], interrupted=True))
            ended = finish(e.shown or None, interrupted=True)
        relay = hooks.relay if hooks is not None and facing_person else None
        if (
            relay is not None
            and ended.input_required is None
            and not ended.interrupted
            and (ended._run_output is None or isinstance(ended._run_output, str))
        ):
            # this run asks nothing: the oldest waiting question is put to the person now
            question = str(relay.get("question") or "")
            if not relay.get("verbatim"):
                try:
                    question = await self._phrase(chat_ctx, question)
                except _ReplyInterrupted:
                    return ended  # a newer message is next: the question stays queued
            record(ChatMessage(role="assistant", content=[question]))
            said = ended._run_output
            ended._run_output = f"{said}\n\n{question}" if said else question
            ended._relayed = str(relay.get("task_id") or "") or None
            ended._run_history = chat_ctx.copy(exclude_instructions=True).items
        return ended

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
        settings = ModelSettings(tool_choice=tool_choice)
        node = self._agent.llm_node(chat_ctx, tool_ctx.flatten(), settings)
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

        async def read() -> None:
            nonlocal usage
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

        hooks = self._hooks
        if hooks is None or hooks.boundary is None or hooks.hub is None:
            await read()
            return "".join(text_parts), calls, usage

        boundary, hub = hooks.boundary, hooks.hub

        async def interrupted() -> None:
            # the lane's hub says when a message landed; the boundary says if it interrupts
            while True:
                seen = hub.signals
                if await boundary(read=True):
                    return
                await hub.since(seen, WATCH_BACKSTOP)

        reading, watching = asyncio.ensure_future(read()), asyncio.ensure_future(interrupted())
        try:
            done, _ = await asyncio.wait({reading, watching}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            watching.cancel()
            if not reading.done():
                reading.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await reading
                if isinstance(node, AsyncGenerator):
                    await node.aclose()
        if reading in done:
            reading.result()
            return "".join(text_parts), calls, usage
        watching.result()  # the boundary's own error, if it raised
        raise _ReplyInterrupted("".join(text_parts))

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

    def _answer_calls(
        self, tool_ctx: ToolContext, asking: Sequence[FunctionCall], heard: str | None = None
    ) -> list[FunctionCall]:
        """The calls that hand this run's message, as the person said it, to each asking call's
        tool that relays answers: each names the call it answers (``ANSWERS_KEY``), so a
        delegate continues the task it reached, and what the person had of a question they cut
        off (``HEARD_KEY``)."""
        calls: list[FunctionCall] = []
        for asked in asking:
            tool = tool_ctx.get_function_tool(asked.name)
            arguments = answer_arguments(tool, self._message.text_content or "") if tool else None
            if arguments is None:
                continue  # nothing takes it as said: the model reads the answer
            calls.append(
                FunctionCall(
                    name=asked.name,
                    arguments=json.dumps(arguments),
                    call_id=utils.shortuuid("call_"),
                    extra={
                        ANSWERS_KEY: asked.call_id,
                        **({HEARD_KEY: heard} if heard is not None else {}),
                    },
                )
            )
        return calls

    async def _execute(
        self, tool: FunctionTool | RawFunctionTool | None, fnc_call: FunctionCall
    ) -> ToolExecutionOutput | AgentTask[Any, Any]:
        """Run one tool with a ``RunContext`` of this run; the output is what the model reads,
        or the ``AgentTask`` the tool awaited."""
        if tool is None:
            return make_tool_output(
                fnc_call=fnc_call,
                output=None,
                exception=ToolError(f"unknown tool `{fnc_call.name}`"),
            )

        run_ctx = RunContext[Any, Any](function_call=fnc_call, run=self)
        granted = isinstance(tool, FunctionTool) and tool.info.requires_approval
        if granted:
            granted = approval_granted(self._chat_ctx.items, fnc_call)
        background = self._hooks.background if self._hooks is not None else None
        try:
            args, kwargs = prepare_function_arguments(
                fnc=tool, json_arguments=fnc_call.arguments, call_ctx=run_ctx, fnc_call=fnc_call
            )

            async def invoke() -> Any:
                with approval_scope(fnc_call, granted):
                    output = tool(*args, **kwargs)
                    if inspect.isawaitable(output):
                        output = await output
                return output

            duplicate = self._duplicate(tool, fnc_call)
            if duplicate is not None:
                return make_tool_output(fnc_call=fnc_call, output=None, exception=duplicate)
            # the tool runs in a task of its own, so a cancel can stop it alone
            work = asyncio.ensure_future(invoke())
            run_ctx._work = work
            waits: set[asyncio.Future[Any]] = {work}
            release: asyncio.Future[str] | None = None
            if background is not None:
                # the first update lets the run go on; the tool keeps running in the background
                release = asyncio.get_running_loop().create_future()
                run_ctx._release = release
                waits.add(release)
            watching = (
                asyncio.ensure_future(self._watch(tool, run_ctx))
                if self._hooks is not None and self._hooks.boundary is not None
                else None
            )
            try:
                await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
            finally:
                if watching is not None:
                    watching.cancel()
            if not work.done() and release is not None and background is not None:
                background(fnc_call, work, run_ctx)
                self._backgrounded = True
                fnc_call.extra[NON_BLOCKING_KEY] = True
                return make_tool_output(fnc_call=fnc_call, output=release.result(), exception=None)
            if work.cancelled():
                return make_tool_output(
                    fnc_call=fnc_call, output=None, exception=ToolError("the task was canceled")
                )
            output = work.result()
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

        if isinstance(output, Completed) and not isinstance(self._agent, AgentTask):
            # a conversation tool never ends a task; the value is its output
            output = ToolResult(output.value, verbatim=True) if output.verbatim else output.value
        return make_tool_output(fnc_call=fnc_call, output=output, exception=None)

    def _duplicate(
        self, tool: FunctionTool | RawFunctionTool, fnc_call: FunctionCall
    ) -> ToolError | None:
        """The tool's ``on_duplicate`` against the lane's calls in the background: refused, or
        to confirm, with the running calls named; the running ones canceled under
        ``replace``."""
        from .tool_executor import (
            DUPLICATE_CONFIRM,
            DUPLICATE_REJECT,
            _duplicate_key,
            _render,
        )

        hooks = self._hooks
        info = tool.info
        if hooks is None or hooks.running is None or info.on_duplicate == "allow":
            return None

        def key(call: FunctionCall) -> tuple[str, str | None]:
            try:
                raw = json.loads(call.arguments or "{}")
            except ValueError:
                raw = {}
            return _duplicate_key(
                fnc=tool, fnc_name=call.name, scope=info.duplicate_scope, raw_arguments=raw
            )

        mine = key(fnc_call)
        same = [c for c in hooks.running() if c.function_call.name == fnc_call.name]
        same = [c for c in same if key(c.function_call) == mine]
        if not same:
            return None
        if info.on_duplicate == "replace":
            if ToolFlag.CANCELLABLE not in info.flags:
                return ToolError(
                    f"cannot replace duplicate call of `{fnc_call.name}`: "
                    "the running call is not cancellable"
                )
            for running in same:
                running._cancel()
            return None
        calls = [c.function_call.model_dump_json() for c in same]
        args = {
            "function_name": fnc_call.name,
            "fnc_calls_json": calls,
            "fnc_calls_text": "\n".join(calls),
        }
        if info.on_duplicate == "reject":
            return ToolError(_render(DUPLICATE_REJECT, args))
        try:
            confirmed = bool(json.loads(fnc_call.arguments or "{}").get(CONFIRM_DUPLICATE_PARAM))
        except ValueError:
            confirmed = False
        return None if confirmed else ToolError(_render(DUPLICATE_CONFIRM, args))

    async def _watch(
        self, tool: FunctionTool | RawFunctionTool, run_ctx: RunContext[Any, Any]
    ) -> None:
        """While a tool runs: a client's cancel stops it, an interrupt only a ``CANCELLABLE``
        one; either stops it at its next wait point outside ``ctx.uninterruptible()``."""
        hooks = self._hooks
        assert hooks is not None and hooks.boundary is not None
        seen = hooks.hub.signals if hooks.hub is not None else 0
        while True:
            if hooks.hub is not None:
                await hooks.hub.since(seen, WATCH_BACKSTOP)
                seen = hooks.hub.signals
            else:
                await asyncio.sleep(WATCH_INTERVAL)
            if not await hooks.boundary(read=True):
                continue
            canceled = hooks.canceled is not None and hooks.canceled()
            if canceled or ToolFlag.CANCELLABLE in tool.info.flags:
                run_ctx._cancel()
            return

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
