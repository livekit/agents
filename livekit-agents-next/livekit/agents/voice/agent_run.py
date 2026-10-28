"""``agent.run()`` and ``agent.run_stream()``: one invocation of an agent on the core's
``AgentRunner``, over the history and context the caller supplies.

The core's ``AgentRun`` is the one decision authority (model steps, tools, approval, retries,
output acceptance). This module is glue: it captures the invocation's context (``_TextRun``,
what tools and hooks read), hands the runner the agent's definition and the request, and turns
the updates back into ``StreamEvent`` and ``RunResult``. ``RunHooks`` is what a door (the
managed A2A turn) installs around the runs made inside its turn.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

from .. import _core, llm, utils
from .._exceptions import UnexpectedModelBehavior
from .._generated.agent_run import (
    FunctionToolInfo,
    OutputNothing,
    OutputText,
    OutputValidated,
    ReadBack,
    RunError,
    RunErrorCancelled,
    RunErrorHost,
    RunErrorModel,
    RunErrorNoModel,
    RunErrorOutputRetriesExceeded,
    RunErrorToolRetriesExceeded,
    RunOutcomeCompleted,
    RunOutcomeInputRequired,
    RunRequest,
    RunSettled,
)
from ..llm import ChatContext, ChatItem, ChatMessage, FunctionTool, RawFunctionTool, ToolContext
from ..llm.chat_context import _ReadOnlyChatContext
from ..llm.tool_context import Toolset
from ..types import APIConnectOptions
from .agent import _BindingContextVar
from .events import Context_T
from .run_result import Output_T, RunCompleted, RunInputRequired, RunResult, RunUsage

if TYPE_CHECKING:
    from ..storage.context_store import ContextStore
    from .agent import Agent


@dataclass
class RunHooks:
    """What a door installs around the ``agent.run()`` calls made inside its turn.

    ``checkpoint`` is awaited before a run starts; ``True`` makes it yield with no reply (a newer
    message waits for the lane, or a client canceled the task), and ``interrupted`` is set.
    ``on_result`` receives every result, so the door can commit what the run added.
    ``on_update`` receives a tool's ``ctx.update(text)``; ``on_token`` the model's text as it
    arrives, from ``run()`` (``run_stream()`` hands delivery to its consumer). ``thread_id`` is
    the session the run belongs to. ``answer_tool`` names the tool the message is handed to
    before the model runs. ``delegated`` says the turn's message came from another agent: the
    run does not phrase its questions, the agent facing the person does.
    """

    checkpoint: Callable[[], Awaitable[bool]] | None = None
    on_result: Callable[[RunResult[Any]], None] | None = None
    on_update: Callable[[str], Any] | None = None
    on_token: Callable[[str], Awaitable[None]] | None = None
    thread_id: str | None = None
    store: ContextStore | None = None
    answer_tool: str | None = None
    delegated: bool = False
    interrupted: bool = False


_run_hooks = contextvars.ContextVar[RunHooks | None]("agents_run_hooks", default=None)


def current_run_hooks() -> RunHooks | None:
    return _run_hooks.get()


def door_hooks() -> RunHooks | None:
    """The door's hooks for a run started here; none inside another run's tool, a delegation
    the caller controls."""
    return None if isinstance(_BindingContextVar.get(None), _TextRun) else _run_hooks.get()


_HooksToken = tuple[contextvars.Token[RunHooks | None], contextvars.Token[Any]]


def set_run_hooks(hooks: RunHooks | None) -> _HooksToken:
    """Installs the door's hooks for a turn. The turn is a root: a door reached in process
    from another run's tool, as a test does, is not nested in it."""
    return _run_hooks.set(hooks), _BindingContextVar.set(None)


def reset_run_hooks(token: _HooksToken) -> None:
    _run_hooks.reset(token[0])
    _BindingContextVar.reset(token[1])


class _TextRun(Generic[Context_T]):
    """One invocation as its tools, instruction functions and validators see it: what
    ``RunContext`` and ``agent.chat_ctx`` read. Installed in the context the invocation's host
    callbacks run in."""

    def __init__(
        self,
        agent: Agent[Context_T, Any],
        *,
        history: list[ChatItem],
        context: Context_T,
        usage: RunUsage,
        hooks: RunHooks | None,
        conn_options: APIConnectOptions,
    ) -> None:
        self.id = utils.shortuuid("run_")
        self.agent = agent
        self.context = context
        self.usage = usage
        """Usage so far: the caller's accumulator, which this run's usage is added to."""
        self.hooks = hooks
        self.llm_conn_options = conn_options
        self.tools = ToolContext(agent.tools)
        # the runner of this invocation, once started: where its tools report progress
        self.runner: _core.AgentRunner | _core.TextSessionRunner | None = None
        self._history = history
        self.execution_id: str | None = None
        self.frame_id: Any = None
        self.context_scope: str | None = None
        self._toolsets: list[Toolset] = []

    @property
    def llm(self) -> llm.LLM:
        model = self.agent.llm
        if not isinstance(model, llm.LLM):
            raise RuntimeError(f"agent `{self.agent.id}` has no LLM to run on")
        return model

    @property
    def chat_ctx(self) -> ChatContext:
        """A read-only view of the history this run started from."""
        return _ReadOnlyChatContext(self._history)

    @property
    def store(self) -> ContextStore:
        if self.hooks is None or self.hooks.store is None:
            raise RuntimeError("this run has no managed context store")
        return self.hooks.store

    async def acquire_toolsets(self) -> None:
        """Sets up the agent's toolsets for this run (tools.md "Toolsets"); a failure releases
        the ones already acquired."""
        for toolset in self.agent.tools:
            if isinstance(toolset, Toolset):
                try:
                    await toolset._acquire()
                except BaseException:
                    await self.release_toolsets()
                    raise
                self._toolsets.append(toolset)

    async def release_toolsets(self) -> None:
        """Releases this run's toolsets once it settled, completed, failed or cancelled."""
        toolsets, self._toolsets = self._toolsets, []
        await asyncio.gather(*(t._release() for t in toolsets))


async def start_run(
    agent: Agent[Context_T, Output_T],
    message: str | ChatMessage,
    *,
    message_history: ChatContext | Sequence[ChatItem] | None,
    context: Context_T | None,
    usage: RunUsage | None,
    conn_options: APIConnectOptions,
) -> tuple[_TextRun[Context_T], _core.AgentRunUpdates | None]:
    """Starts one invocation; ``None`` updates when the door's checkpoint yields first."""
    from .._core_view import to_core_item, to_core_items

    hooks = door_hooks()
    if isinstance(message, str):
        message = ChatMessage(role="user", content=[message])
    history = list(
        message_history.items if isinstance(message_history, ChatContext) else message_history or []
    )
    run = _TextRun[Context_T](
        agent,
        history=[*history, message],
        context=context,  # type: ignore[arg-type]
        usage=usage if usage is not None else RunUsage(),
        hooks=hooks,
        conn_options=conn_options,
    )
    if hooks is not None and hooks.checkpoint is not None and await hooks.checkpoint():
        hooks.interrupted = True
        return run, None

    await run.acquire_toolsets()
    try:
        run.tools = ToolContext(agent.tools)
        captured = contextvars.copy_context()
        captured.run(_BindingContextVar.set, run)
        runner = _core.AgentRunner(agent._definition(run.tools), run.llm, agent._host)
        run.runner = runner
        updates = runner.run(
            RunRequest(
                invocation_id=run.id,
                message=to_core_item(message),
                history=to_core_items(history),
                answer_tool=hooks.answer_tool if hooks is not None else None,
                delegated=hooks.delegated if hooks is not None else False,
                conn_options=conn_options,
                host_context=(asyncio.get_running_loop(), captured),
            )
        )
    except BaseException:
        await run.release_toolsets()
        raise
    return run, updates


def settled_result(run: _TextRun[Any], settled: RunSettled) -> RunResult[Any]:
    """The settled invocation as the caller's ``RunResult``; this run's usage is added to the
    caller's accumulator."""
    from .._core_view import from_core_item

    result = settled.result
    usage = RunUsage(
        requests=result.usage.requests,
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        tool_calls=result.usage.tool_calls,
    )
    run.usage._add(usage)
    common: dict[str, Any] = {
        "new_messages": [from_core_item(i) for i in result.new_messages],
        "all_messages": [from_core_item(i) for i in result.all_messages],
        "usage": usage,
        "verbatim": result.verbatim,
    }
    match result.outcome:
        case RunOutcomeInputRequired(question=question, answer_tool=answer_tool):
            return RunInputRequired(question=question, answer_tool=answer_tool, **common)
        case RunOutcomeCompleted(output=OutputText(text=text)):
            return RunCompleted(output=text, **common)
        case RunOutcomeCompleted(output=OutputValidated()):
            return RunCompleted(output=settled.output, **common)
        case RunOutcomeCompleted(output=OutputNothing()):
            return RunCompleted(output=None, **common)


def interrupted_result(run: _TextRun[Any]) -> RunResult[Any]:
    """A run the door's checkpoint yielded before it started: it ran nothing."""
    message = run.chat_ctx.items[-1]
    return RunCompleted(
        output=None,
        new_messages=[message],
        all_messages=list(run.chat_ctx.items),
        usage=RunUsage(),
        verbatim=None,
    )


def run_error(error: RunError) -> BaseException:
    """The exception a run that settled without a result raises."""
    match error:
        case RunErrorCancelled():
            return asyncio.CancelledError("the run was cancelled")
        case RunErrorToolRetriesExceeded(tool=tool, retries=retries):
            return UnexpectedModelBehavior(f"tool `{tool}` exceeded max retries count of {retries}")
        case RunErrorOutputRetriesExceeded(retries=retries):
            return UnexpectedModelBehavior(
                f"exceeded maximum retries ({retries}) for output validation"
            )
        case RunErrorModel(message=message):
            from .._exceptions import APIError

            return APIError(f"the model request failed: {message}", retryable=False)
        case RunErrorHost(message=message):
            return RuntimeError(message)
        case RunErrorNoModel():
            return RuntimeError("the agent has no model")


def tool_infos(tools: ToolContext) -> list[FunctionToolInfo]:
    """The core's ``FunctionToolInfo`` of each function tool."""
    from .._core_view import tool_specs

    specs = {spec["name"]: spec for spec in tool_specs(tools.flatten())}
    infos = []
    for name, tool in tools.function_tools.items():
        approval: ReadBack | None = None
        if isinstance(tool, FunctionTool) and tool.info.needs_approval:
            approval = "function" if tool.info.read_back is not None else "default"
        infos.append(
            FunctionToolInfo(
                spec=specs[name],
                raw_schema=isinstance(tool, RawFunctionTool),
                needs_approval=approval,
                on_duplicate=tool.info.on_duplicate,
                duplicate_scope=tool.info.duplicate_scope,
            )
        )
    return infos
