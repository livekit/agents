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

from .. import _core, llm, telemetry, utils
from .._exceptions import UnexpectedModelBehavior
from .._generated.agent_run import (
    FunctionToolInfo,
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
    RunOutcomeNoOutput,
    RunRequest,
    RunSettled,
)
from .._generated.text_session import FrameId
from .._generated.usage import Usage
from ..llm import ChatContext, ChatItem, ChatMessage, FunctionTool, RawFunctionTool, ToolContext
from ..llm.chat_context import _ReadOnlyChatContext
from ..llm.tool_context import Toolset
from ..types import APIConnectOptions
from .agent import _BindingContextVar
from .events import Context_T
from .run_result import (
    Output_T,
    RunCompleted,
    RunInputRequired,
    RunNoOutput,
    RunResult,
    _SettledArgs,
)

if TYPE_CHECKING:
    from ..storage.context_store import ContextStore
    from ..storage.resolve import Thread
    from ..text_session import _FrameContexts
    from .agent import Agent, AgentRetries


@dataclass
class RunHooks:
    """What a door installs around the ``agent.run()`` calls made inside its turn.

    The native loop awaits ``checkpoint`` before model requests and result delivery. ``True``
    yields at that boundary; the settled result carries its history and marks ``interrupted``.
    ``on_result`` receives every result, so the door can commit what the run added.
    ``on_update`` receives a tool's ``ctx.update(text)``; ``on_token`` the model's text as it
    arrives, from ``run()`` (``run_stream()`` hands delivery to its consumer). ``thread`` and
    ``task_id`` are the thread and the A2A task of the turn, what ``ctx.thread`` and
    ``ctx.task_id`` read. ``answer_tool`` names the tool the message is handed to before the
    model runs. ``delegated`` says the turn's message came from another agent: the
    run does not phrase its questions, the agent facing the person does.
    """

    checkpoint: Callable[[], Awaitable[bool]] | None = None
    on_result: Callable[[RunResult[Any]], None] | None = None
    on_update: Callable[[str], Any] | None = None
    on_token: Callable[[str], Awaitable[None]] | None = None
    thread: Thread | None = None
    task_id: str | None = None
    store: ContextStore | None = None
    answer_tool: str | None = None
    delegated: bool = False
    interrupted: bool = False


_run_hooks = contextvars.ContextVar[RunHooks | None]("agents_run_hooks", default=None)


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
        hooks: RunHooks | None,
        conn_options: APIConnectOptions,
    ) -> None:
        self.id = utils.shortuuid("run_")
        self.agent = agent
        self.context = context
        self.hooks = hooks
        self.thread = hooks.thread if hooks is not None else None
        self._store: ContextStore | None = None
        self.llm_conn_options = conn_options
        self.tools = ToolContext(agent.tools)
        # the runner of this invocation, once started: where its tools report progress
        self.runner: _core.AgentRunner | _core.TextSessionRunner | None = None
        self._history = history
        self.execution_id: str | None = None
        self.frame_id: FrameId | None = None
        self.context_scope: str | None = None
        self.frame_contexts: _FrameContexts | None = None

    @property
    def usage(self) -> Usage:
        """The latest snapshot of the scope this run belongs to: the invocation, or the
        text-session turn it is a frame of; empty before it started."""
        if isinstance(self.runner, _core.AgentRunner):
            return self.runner.usage(self.id)
        if self.runner is not None:
            return self.runner.usage
        return Usage()

    @property
    def llm(self) -> llm.LLM:
        model = self.agent.llm
        if not isinstance(model, llm.LLM):
            raise RuntimeError(f"agent `{self.agent.name}` has no LLM to run on")
        return model

    @property
    def chat_ctx(self) -> ChatContext:
        """A read-only view of the history this run started from."""
        return _ReadOnlyChatContext(self._history)

    @property
    def store(self) -> ContextStore:
        store = self.hooks.store if self.hooks is not None else None
        if store is None:
            store = self._store
        if store is None:
            raise RuntimeError("this run has no managed context store")
        return store._for_scope(self.context_scope) if self.context_scope is not None else store

    def bind_resources(
        self,
        resources: _core._core_RunResources,
        *,
        thread: Thread | None = None,
        store: ContextStore | None = None,
    ) -> None:
        """Hydrate Python handles after the native request owner has acquired its resources."""
        from ..storage.backends import _StorageHandle
        from ..storage.context_store import ContextStore
        from ..storage.resolve import Thread

        self.thread = self.thread or thread
        if self.thread is None and (namespace := resources.namespace) is not None:
            assert resources.thread_id is not None
            self.thread = Thread(
                _StorageHandle(namespace),
                resources.thread_id,
                agent_name=self.agent.name,
            )
        if self.hooks is None or self.hooks.store is None:
            self._store = store
            if self._store is None:
                self._store = ContextStore(activation=resources.activation)
                resources.retain_hydration(self._store)


def run_resources(
    agent: Agent[Any, Any], hooks: RunHooks | None, agents: Sequence[Agent[Any, Any]]
) -> _core._core_RunResources:
    """Translate Python thread/store handles and authored toolsets to native capabilities."""
    thread = hooks.thread if hooks is not None else None
    store = hooks.store if hooks is not None else None
    return _core._core_RunResources(
        thread._storage.threads if thread is not None else None,
        thread.id if thread is not None else None,
        store._activation if store is not None else None,
        agent.name,
        [tool for configured in agents for tool in configured.tools if isinstance(tool, Toolset)],
    )


async def start_run(
    agent: Agent[Context_T, Output_T],
    stream: _core._core_RunStream,
    message: str | ChatMessage,
    *,
    message_history: ChatContext | Sequence[ChatItem] | None,
    context: Context_T | None,
    retries: int | AgentRetries | None,
    conn_options: APIConnectOptions,
) -> _TextRun[Context_T]:
    """Start one native invocation; its core owns managed checkpoints and settlement."""

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
        hooks=hooks,
        conn_options=conn_options,
    )
    resources = run_resources(agent, hooks, [agent])
    try:
        await resources.open()
        run.bind_resources(resources)
        run.tools = ToolContext(agent.tools)
        captured = contextvars.copy_context()
        captured.run(_BindingContextVar.set, run)
        runner = _core.AgentRunner(agent._definition(run.tools, retries), run.llm, agent._host)
        run.runner = runner
        runner.run_stream(
            RunRequest(
                invocation_id=run.id,
                execution_id=hooks.task_id if hooks is not None else None,
                message=message,
                history=history,
                answer_tool=hooks.answer_tool if hooks is not None else None,
                delegated=hooks.delegated if hooks is not None else False,
                conn_options=conn_options,
                host_context=(asyncio.get_running_loop(), captured),
                tools_host=run.tools.flatten(),
                parent=telemetry._current(),
            ),
            stream,
            resources,
        )
    except BaseException:
        await resources.rollback()
        raise
    return run


def settled_result(run: _TextRun[Any], settled: RunSettled) -> RunResult[Any]:
    """The settled invocation as the caller's ``RunResult``."""

    result = settled.result
    if result.interrupted and run.hooks is not None:
        run.hooks.interrupted = True
    common: _SettledArgs = {
        "new_messages": list(result.new_messages),
        "all_messages": list(result.all_messages),
        "usage": result.usage,
        "verbatim": result.verbatim,
        "_pending_call": result.pending_call,
        "input_requirements": list(result.input_requirements),
    }
    match result.outcome:
        case RunOutcomeInputRequired(question=question):
            return RunInputRequired(question=question, **common)
        case RunOutcomeCompleted(output=OutputText(text=text)):
            return RunCompleted(output=text, **common)
        case RunOutcomeCompleted(output=OutputValidated()):
            return RunCompleted(output=settled.output, **common)
        case RunOutcomeNoOutput():
            return RunNoOutput(**common)
    raise AssertionError(f"unsupported run outcome: {result.outcome!r}")


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
    from ..a2a.remote import _ExpertTool

    specs = {spec["name"]: spec for spec in tool_specs(list(tools.function_tools.values()))}
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
                retries=tool.info.retries,
                remote=tool.endpoint if isinstance(tool, _ExpertTool) else None,
            )
        )
    return infos
