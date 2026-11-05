"""``agent.run()`` and ``agent.run_stream()``: one invocation of an agent on the core's AgentRun,
its application callbacks and the tool templates rendered around background calls."""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

from .. import _core, llm, telemetry, utils
from .._bindings import UnexpectedModelBehavior
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
from ..llm.tool_context import Toolset, tools_for_run
from ..types import APIConnectOptions
from . import _BindingContextVar
from .events import Context_T, RunContext
from .run_result import (
    Output_T,
    RunCompleted,
    RunInputRequired,
    RunNoOutput,
    RunResult,
    _SettledArgs,
)

if TYPE_CHECKING:
    from ..storage import Thread
    from ..storage.context import ContextStore
    from ..text_session import _FrameContexts
    from . import Agent, AgentRetries
import copy
import dataclasses
import inspect
import uuid
from typing import TYPE_CHECKING, get_type_hints

from pydantic import TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError, to_jsonable_python

from .._generated.agent_run import (
    ArgumentIdentity,
    ArgumentIdentityFingerprint,
    ArgumentIdentityJson,
    Invocation,
    OutputCandidate,
    OutputCandidateText,
    OutputSpec,
    OutputSpecStructured,
    OutputSpecText,
    PreparedArguments,
    ReadBackAnswer,
    ReadBackAnswerInputRequired,
    ReadBackAnswerQuestion,
    SerializedOutput,
    ToolTemplate,
    ToolTemplateDuplicateConfirm,
    ToolTemplateDuplicateReject,
    ToolTemplateReplyAtTail,
    ToolTemplateReplyMaybeCovered,
    ValidatedOutput,
    ValidatedOutputAccepted,
    ValidatedOutputFailed,
    ValidatedOutputRetry,
)
from .._generated.tool_report import (
    ToolOutcomeCancelled,
    ToolOutcomeInternal,
    ToolOutcomeInvalid,
    ToolOutcomeModelRetry,
    ToolOutcomeToolError,
    ToolReport,
)
from .._generated.voice_session import VoiceCommandOriginTool
from ..llm import FunctionCall
from ..llm.tool_context import (
    CONFIRM_DUPLICATE_PARAM,
    Completed,
    Failed,
    InputRequired,
    ModelRetry,
    Tool,
    ToolError,
)
from ..llm.utils import (
    bind_function_arguments,
    function_arguments_to_pydantic_model,
    is_context_type,
    tool_report,
    validated_arguments,
)
from ..log import logger
from . import _VoiceBinding
from .io import SpeechHandle
from .run_result import _tool_body

if TYPE_CHECKING:
    from . import Agent
    from .agent_session import VoiceSession
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.tool_context import Tool, ToolHandlingOptions, Toolset


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
        # the agent's tools as this run uses them, resolved once (Toolset.for_run)
        self.agent_tools = tools_for_run(agent.tools, RunContext(function_call=None, run=self))
        self.tools = ToolContext(self.agent_tools)
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
        from ..storage import Thread, _StorageHandle
        from ..storage.context import ContextStore

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
    agent: Agent[Any, Any], hooks: RunHooks | None, runs: Sequence[_TextRun[Any]]
) -> _core._core_RunResources:
    """Translate Python thread/store handles and authored toolsets to native capabilities."""
    thread = hooks.thread if hooks is not None else None
    store = hooks.store if hooks is not None else None
    return _core._core_RunResources(
        thread._storage.threads if thread is not None else None,
        thread.id if thread is not None else None,
        store._activation if store is not None else None,
        agent.name,
        [tool for run in runs for tool in run.agent_tools if isinstance(tool, Toolset)],
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
    resources = run_resources(agent, hooks, [run])
    try:
        await resources.open()
        run.bind_resources(resources)
        run.tools = ToolContext(run.agent_tools)
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
            from .._bindings import APIError

            return APIError(f"the model request failed: {message}", retryable=False)
        case RunErrorHost(message=message):
            return RuntimeError(message)
        case RunErrorNoModel():
            return RuntimeError("the agent has no model")


def tool_infos(tools: ToolContext) -> list[FunctionToolInfo]:
    """The core's ``FunctionToolInfo`` of each function tool."""
    from .._bindings import tool_specs
    from ..a2a import _ExpertTool

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


class ToolArgumentError(Exception):
    """A call's arguments did not bind or validate: the message is the model's feedback."""


@dataclass
class _Prepared:
    """A call's validated arguments, held by the runtime between preparation and the body."""

    tool: FunctionTool[..., Any] | RawFunctionTool[..., Any]
    ctx: RunContext[Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


def prepare_arguments(
    tool: FunctionTool[..., Any] | RawFunctionTool[..., Any],
    arguments: dict[str, Any],
    ctx: RunContext[Any],
) -> PreparedArguments:
    """Binds a call's arguments (the framework's parse, with the schema's defaults) to the tool's
    signature and validates them, for a run and on a call alike: the prepared call and the
    identity approval binds to (the validated arguments as JSON, decision 15). Raises
    ``ToolArgumentError`` with the model's feedback."""
    from . import _LocalAgentTool

    name = ctx.function_call.name
    try:
        fields = (
            tool.validate(arguments)
            if isinstance(tool, _LocalAgentTool)
            else validated_arguments(tool, arguments)
        )
    except (ValidationError, ValueError, TypeError) as e:
        raise ToolArgumentError(f"Error parsing arguments for `{name}`: {e}") from e
    args, kwargs = (
        ((), fields)
        if isinstance(tool, _LocalAgentTool)
        else bind_function_arguments(tool, fields, ctx)
    )
    identified = {k: v for k, v in fields.items() if k != CONFIRM_DUPLICATE_PARAM}
    identity: ArgumentIdentity
    try:
        identity = ArgumentIdentityJson(value=to_jsonable_python(identified))
    except PydanticSerializationError:
        # no JSON for these arguments: a fresh fingerprint matches no earlier read-back, so the
        # call can never be approved by mistake
        identity = ArgumentIdentityFingerprint(value=uuid.uuid4().hex)
    return PreparedArguments(prepared=_Prepared(tool, ctx, args, kwargs), identity=identity)


async def render_read_back(prepared: _Prepared) -> ReadBackAnswer:
    """The tool's read-back function on the prepared arguments."""
    assert isinstance(prepared.tool, FunctionTool) and prepared.tool.info.read_back
    question = prepared.tool.info.read_back(*prepared.args, **prepared.kwargs)
    if inspect.isawaitable(question):
        question = await question
    if isinstance(question, InputRequired):
        return ReadBackAnswerInputRequired(question=question.question, verbatim=question.verbatim)
    return ReadBackAnswerQuestion(question=str(question))


class AgentCallbacks:
    """The host side of one ``Agent``'s invocations; shared by all of them."""

    def __init__(self, agent: Agent[Any, Any]) -> None:
        self._agent = agent
        self._output = OutputContract.of(agent)
        # the callback task of each operation in flight, for `cancel`; on a voice session, the
        # body of a tool call, which outlives its callback once its first `ctx.update()` returned
        self._running: dict[tuple[str, int], asyncio.Task[Any]] = {}
        # the context of each tool body running on a voice session, which a reply may answer
        self._bodies: dict[tuple[str, int], RunContext[Any]] = {}

    async def render_instructions(self, invocation: Invocation) -> str:
        async with self._operation(invocation):
            ctx = RunContext[Any](function_call=None, run=_current_run())
            return await self._agent._render_dynamic_instructions(ctx)

    async def checkpoint(self, invocation: Invocation) -> bool:
        """Forward the framework's verdict under this callback's captured invocation."""
        async with self._operation(invocation):
            hooks = _current_run().hooks
            return (
                await hooks.checkpoint()
                if hooks is not None and hooks.checkpoint is not None
                else False
            )

    async def list_tools(self, invocation: Invocation) -> list[FunctionToolInfo]:
        """Refresh this run's callable tools after toolset discovery."""

        async with self._operation(invocation) as binding:
            assert not isinstance(binding, _VoiceBinding), "a voice frame carries its tools"
            binding.tools.update_tools(binding.agent_tools)
            return tool_infos(binding.tools)

    async def prepare_tool(
        self, invocation: Invocation, arguments: dict[str, Any]
    ) -> PreparedArguments:
        """Binds and validates the call's arguments in the call's scope (``prepare_arguments``)."""
        async with self._operation(invocation) as binding:
            call = _function_call(invocation)
            ctx: RunContext[Any]
            if isinstance(binding, _VoiceBinding):
                tool = _voice_tools(binding).get_function_tool(call.name)
                assert isinstance(invocation.speech, _core.SpeechHandle)
                speech = SpeechHandle._of(invocation.speech)
                ctx = RunContext(function_call=call, session=binding.session, speech_handle=speech)
            else:
                tool = binding.tools.get_function_tool(call.name)
                ctx = RunContext[Any](function_call=call, run=binding)
            ctx._invocation = invocation
            if tool is None:
                raise ToolArgumentError(f"unknown tool `{call.name}`")
            return prepare_arguments(tool, arguments, ctx)

    async def read_back(self, invocation: Invocation, prepared: _Prepared) -> ReadBackAnswer:
        """The tool's read-back function on the call's arguments."""
        async with self._operation(invocation):
            return await render_read_back(prepared)

    async def run_tool(
        self, invocation: Invocation, prepared: _Prepared
    ) -> tuple[ToolReport, dict[str, Any]]:
        """Runs the body, sync or async, and classifies what it returned or raised."""
        async with self._operation(invocation) as binding:
            prepared.ctx._function_call = _function_call(invocation)
            prepared.ctx._invocation = invocation
            key = _operation_key(invocation)
            self._bodies[key] = prepared.ctx
            try:
                body = _tool_body(
                    self._agent,
                    prepared.ctx.function_call.name,
                    prepared.tool,
                    binding.session if isinstance(binding, _VoiceBinding) else None,
                )
                output = body(*prepared.args, **prepared.kwargs)
                if inspect.isawaitable(output):
                    output = await output
            except asyncio.CancelledError:
                return ToolReport(outcome=ToolOutcomeCancelled()), prepared.ctx.function_call.extra
            except Exception as e:
                return _classified(prepared.ctx.function_call, None, e), (
                    prepared.ctx.function_call.extra
                )
            finally:
                self._bodies.pop(key, None)
            report = _classified(prepared.ctx.function_call, output, None)
            if isinstance(output, Completed):
                report = dataclasses.replace(report, transition=output.value)
            elif isinstance(output, Failed):
                report = dataclasses.replace(report, transition=output)
            return report, prepared.ctx.function_call.extra

    async def validate_tool_completion(
        self, invocation: Invocation, prepared: _Prepared, report: ToolReport
    ) -> ToolReport:
        """Application type validation, requested by the voice core for a task completion."""
        async with self._operation(invocation):
            return await self._validated_completion(prepared.ctx, report.transition, report)

    async def _validated_completion(
        self, ctx: RunContext[Any], value: Any, report: ToolReport
    ) -> ToolReport:
        """A task's ``Completed(value)`` through its agent's ``output_type`` and output
        validators: the value the task returns, or the error its model reads and corrects."""
        try:
            value = TypeAdapter(self._agent.output_type).validate_python(value)
            value = await self._agent._validate_output(ctx, value)
        except ValidationError as e:
            return ToolReport(outcome=ToolOutcomeToolError(message=str(e)))
        except ModelRetry as e:
            return ToolReport(outcome=ToolOutcomeModelRetry(message=e.message))
        return dataclasses.replace(report, transition=value)

    async def validate_output(
        self, invocation: Invocation, candidate: OutputCandidate
    ) -> ValidatedOutput:
        """The output contract and the output validators on a candidate; the accepted value is
        kept for the run (a text output's too)."""
        async with self._operation(invocation) as binding:
            ctx: RunContext[Any]
            if isinstance(binding, _VoiceBinding):
                assert isinstance(invocation.speech, _core.SpeechHandle)
                speech = SpeechHandle._of(invocation.speech)
                ctx = RunContext(function_call=None, session=binding.session, speech_handle=speech)
            else:
                ctx = RunContext[Any](function_call=None, run=binding)
            try:
                if isinstance(candidate, OutputCandidateText):
                    text = str(await self._agent._validate_output(ctx, candidate.text))
                    return ValidatedOutputAccepted(
                        text=text, value=text, serialized=SerializedOutput(value=text)
                    )
                value = await self._output.accept(ctx, candidate.value)
                value = await self._agent._validate_output(ctx, value)
            except ValidationError as e:
                return ValidatedOutputRetry(reason=str(e))
            except (ModelRetry, ToolError) as e:
                return ValidatedOutputRetry(reason=e.message)
            except Exception as e:
                logger.exception("the output validation failed")
                return ValidatedOutputFailed(error=f"{type(e).__name__}: {e}")
            try:
                serialized: SerializedOutput | None = SerializedOutput(
                    value=to_jsonable_python(value)
                )
            except PydanticSerializationError:
                serialized = None
            return ValidatedOutputAccepted(text=None, value=value, serialized=serialized)

    async def render_template(self, invocation: Invocation, template: ToolTemplate) -> str | None:
        """The agent's ``tool_handling`` template, or the call's toolset's, rendered with the
        template's arguments; ``None`` keeps the framework's text."""
        call = invocation.function_call
        return render_template(
            self._agent._tool_handling,
            _template_tools(self._agent, _BindingContextVar.get(None)),
            _function_call(invocation).name if call is not None else None,
            _TEMPLATE_KEYS[type(template)],
            dataclasses.asdict(template),
        )

    def cancel(self, invocation: Invocation) -> None:
        """Cancels the callback in flight for ``invocation``; it still settles."""
        task = self._running.get(_operation_key(invocation))
        if task is not None:
            task.cancel()

    def answer_expert(self, invocation: Invocation, answer: str) -> None:
        """The person's reply to the question the expert of ``invocation``'s delegated call
        asked, for the call's body waiting on it."""
        ctx = self._bodies.get(_operation_key(invocation))
        waiting = None if ctx is None else ctx._expert_answer
        if waiting is not None and not waiting.done():
            waiting.set_result(answer)

    def _operation(self, invocation: Invocation) -> _Operation:
        return _Operation(self._running, invocation)


class _Operation:
    """Enters one callback's binding, and registers the current task as the callback of its
    operation while it runs."""

    def __init__(self, running: dict[tuple[str, int], asyncio.Task[Any]], invocation: Invocation):
        self._running = running
        self._key = _operation_key(invocation)
        self._invocation = invocation
        self._token: contextvars.Token[Any] | None = None

    async def __aenter__(self) -> _TextRun | _VoiceBinding:
        binding = _BindingContextVar.get(None)
        if isinstance(binding, _VoiceBinding):
            # a tool's commands are sent as its own
            call = self._invocation.function_call
            if call is not None:
                binding = binding._replace(
                    origin=VoiceCommandOriginTool(
                        invocation_id=self._invocation.invocation_id,
                        operation=self._invocation.operation,
                        call_id=_function_call(self._invocation).call_id,
                    )
                )
            self._token = _BindingContextVar.set(binding)
            self._register()
            return binding
        run = copy.copy(_current_run())
        run.id = self._invocation.invocation_id

        run._history = list(self._invocation.history)
        run.execution_id = self._invocation.execution_id
        run.frame_id = self._invocation.frame_id
        run.context_scope = self._invocation.context_scope
        if run.context_scope is not None:
            run.context = self._invocation.application_context
        self._token = _BindingContextVar.set(run)
        self._register()

        try:
            if run.frame_contexts is not None:
                await run.frame_contexts.resolve(run)
        except BaseException:
            await self.__aexit__()
            raise
        return run

    def _register(self) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._running[self._key] = task

    async def __aexit__(self, *exc: object) -> None:
        if self._running.get(self._key) is asyncio.current_task():
            self._running.pop(self._key, None)
        if self._token is not None:
            _BindingContextVar.reset(self._token)


class OutputContract:
    """``output_type`` as the core sees it (``OutputSpec``) and as the host accepts a candidate:
    ``str``, a type validated as its schema, or an output function whose parameters are the
    final output's arguments and whose return value is the output (agents.md "Structured
    output")."""

    def __init__(
        self,
        *,
        schema: dict[str, Any] | None,
        accept: Callable[[RunContext[Any], Any], Awaitable[Any]],
    ) -> None:
        self.schema = schema
        self.accept = accept

    @classmethod
    def of(cls, agent: Agent[Any, Any]) -> OutputContract:
        output = agent._output_spec
        if output is str:
            return cls(schema=None, accept=_unexpected_value)
        if isinstance(output, type) or not callable(output):
            adapter: TypeAdapter[Any] = TypeAdapter(output)
            schema = adapter.json_schema()
            schema.pop("title", None)

            async def validate(ctx: RunContext[Any], value: Any) -> Any:
                return adapter.validate_python(value)

            return cls(schema=schema, accept=validate)
        return cls._of_function(output)

    @classmethod
    def _of_function(cls, fn: Callable[..., Any]) -> OutputContract:
        parameters = function_arguments_to_pydantic_model(fn)
        schema = parameters.model_json_schema()
        schema.pop("title", None)
        hints = get_type_hints(fn, include_extras=True)
        returns: TypeAdapter[Any] | None = (
            TypeAdapter(hints["return"]) if hints.get("return") not in (None, Any) else None
        )
        context = [
            n
            for n, h in hints.items()
            if n != "return" and is_context_type(h, allow_subclasses=True)
        ]

        async def call(ctx: RunContext[Any], value: Any) -> Any:
            fields = dict(parameters.model_validate(value))
            result = fn(**fields, **dict.fromkeys(context, ctx))
            if inspect.isawaitable(result):
                result = await result
            return returns.validate_python(result) if returns is not None else result

        return cls(schema=schema, accept=call)

    def spec(self, *, validated: bool) -> OutputSpec:
        """The core's ``OutputSpec``."""
        if self.schema is None:
            return OutputSpecText(validated=validated)
        return OutputSpecStructured(schema=self.schema)


async def _unexpected_value(ctx: RunContext[Any], value: Any) -> Any:
    raise RuntimeError("a str output takes no structured candidate")


def _voice_tools(binding: _VoiceBinding) -> ToolContext:
    """A voice frame's tools: the agent's, then the session's it lacks (a session tool added
    since the frame was installed, as IVR's DTMF tool)."""
    agent_tools, session_tools = _call_tools(binding.agent, binding.session)
    own = ToolContext(agent_tools)
    extra = [
        tool
        for name, tool in ToolContext(session_tools).function_tools.items()
        if own.get_function_tool(name) is None
    ]
    return ToolContext([*agent_tools, *extra])


def _call_tools(
    agent: Agent[Any, Any], session: VoiceSession[Any]
) -> tuple[list[Tool | Toolset], list[Tool | Toolset]]:
    """The agent's and the session's tools as the call uses them (``Toolset.for_run``)."""
    ctx = RunContext[Any](function_call=None, session=session)
    return tools_for_run(agent.tools, ctx), tools_for_run(session._tools, ctx)


def _template_tools(agent: Agent[Any, Any], binding: object) -> list[Tool | Toolset]:
    """Where a tool's ``tool_handling`` override is looked up: the agent's tools, and on a call the
    session's (an ``AsyncToolset`` passed to ``VoiceSession(tools=)``)."""
    if isinstance(binding, _VoiceBinding):
        agent_tools, session_tools = _call_tools(agent, binding.session)
        return [*agent_tools, *session_tools]

    if isinstance(binding, _TextRun):
        return list(binding.agent_tools)
    return list(agent.tools)


_TEMPLATE_KEYS: dict[type[ToolTemplate], str] = {
    ToolTemplateDuplicateReject: "duplicate_reject_template",
    ToolTemplateDuplicateConfirm: "duplicate_confirm_template",
    ToolTemplateReplyAtTail: "reply_at_tail_template",
    ToolTemplateReplyMaybeCovered: "reply_maybe_covered_template",
}


def _classified(call: FunctionCall, output: Any, exc: BaseException | None) -> ToolReport:
    """What the body returned or raised, classified; an internal error or an invalid output is
    logged."""
    report = tool_report(output, exc)
    if isinstance(report.outcome, ToolOutcomeInternal):
        logger.error(
            "exception occurred while executing a tool",
            exc_info=exc,
            extra={"function": call.name, "call_id": call.call_id},
        )
    elif isinstance(report.outcome, ToolOutcomeInvalid):
        logger.error(
            f"tool `{call.name}` returned an invalid output",
            extra={"call_id": call.call_id, "lk.pii.output": output},
        )
    return report


def _current_run() -> _TextRun:

    run = _BindingContextVar.get(None)
    if not isinstance(run, _TextRun):
        raise RuntimeError("an agent callback ran outside its invocation's context")
    return run


def _function_call(invocation: Invocation) -> FunctionCall:
    call = invocation.function_call
    if call is None:
        raise RuntimeError("a tool callback has no function call")
    return call


def _operation_key(invocation: Invocation) -> tuple[str, int]:
    return invocation.invocation_id, invocation.operation


UPDATE_TEMPLATE = """The tool `{function_name}` has updated, message: {message}
The task is still running, so DON'T make up or give information not included in the message above."""


def render_template(
    agent_options: ToolHandlingOptions | None,
    tools: Sequence[Tool | Toolset],
    tool: str | None,
    key: str,
    args: dict[str, Any],
) -> str | None:
    """The ``key`` template rendered with ``args``: the innermost ``AsyncToolset`` among
    ``tools`` holding ``tool`` that sets it, else the agent's; ``None`` when neither does."""
    template = _toolset_template(tools, tool, key) if tool is not None else None
    if template is None:
        template = _template(agent_options, key)
    if template is None:
        return None
    return str(template(args) if callable(template) else template.format(**args))


def _toolset_template(tools: Sequence[Tool | Toolset], tool: str, key: str) -> Any:
    from ..llm.tool_context import AsyncToolset, ToolContext, Toolset

    for entry in tools:
        if not isinstance(entry, Toolset):
            continue
        template = _toolset_template(entry.tools, tool, key)
        if (
            template is None
            and isinstance(entry, AsyncToolset)
            and ToolContext([entry]).get_function_tool(tool) is not None
        ):
            template = _template(entry._tool_handling, key)
        if template is not None:
            return template
    return None


def _template(options: ToolHandlingOptions | None, key: str) -> Any:
    templates: Mapping[str, Any] = (options or {}).get("async_options") or {}
    return templates.get(key)
