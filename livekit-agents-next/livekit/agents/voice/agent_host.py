"""The application callbacks of an agent invocation, run for the core: a text run's
(``_core.AgentRunner``) and a voice session's (``_core.VoiceAgent``).

``lk-agents-core``'s ``AgentRun`` decides what runs, when, and what an outcome means
(continuation, approval, retries, phrasing); ``AgentHost`` only does host work: it binds and
validates tool arguments, runs tool bodies and classifies what they returned or raised, renders
the dynamic instructions, renders read-backs and validates output candidates. Each callback runs
in a copy of the context captured when the invocation was entered (``run()``, or the voice frame's
install), so what it serves is the binding ``_BindingContextVar`` holds there: a ``_TextRun`` or
a ``_VoiceBinding``. The records crossing are the generated models of ``lk_agents_abi::agent``
(``_generated.agent_run``).
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import dataclasses
import inspect
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_type_hints

from pydantic import TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError, to_jsonable_python

from .. import _core
from .._generated.agent_run import (
    ArgumentIdentity,
    ArgumentIdentityFingerprint,
    ArgumentIdentityJson,
    FunctionToolInfo,
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
    ValidatedOutput,
    ValidatedOutputAccepted,
    ValidatedOutputFailed,
    ValidatedOutputRetry,
)
from .._generated.tool_report import (
    ToolOutcomeCancelled,
    ToolOutcomeCompleted,
    ToolOutcomeInternal,
    ToolOutcomeInvalid,
    ToolOutcomeModelRetry,
    ToolOutcomeToolError,
    ToolOutcomeUpdate,
    ToolReport,
)
from ..llm import FunctionCall, ToolContext
from ..llm.tool_context import (
    CONFIRM_DUPLICATE_PARAM,
    FunctionTool,
    InputRequired,
    ModelRetry,
    RawFunctionTool,
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
from .agent import _BindingContextVar, _VoiceBinding
from .events import RunContext
from .run_result import _tool_body
from .speech_handle import SpeechHandle

if TYPE_CHECKING:
    from .agent import Agent
    from .agent_run import _TextRun

__all__ = ["AgentHost", "ToolArgumentError", "prepare_arguments", "render_read_back"]


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
    from .agent_tool import _LocalAgentTool

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


class AgentHost:
    """The host side of one ``Agent``'s invocations; shared by all of them."""

    def __init__(self, agent: Agent[Any, Any]) -> None:
        self._agent = agent
        self._output = OutputContract.of(agent)
        # the callback task of each operation in flight, for `cancel`; on a voice session, the
        # body of a tool call, which outlives its callback once its first `ctx.update()` returned
        self._running: dict[tuple[str, int], asyncio.Task[Any]] = {}

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
        from .agent_run import tool_infos

        async with self._operation(invocation) as binding:
            assert not isinstance(binding, _VoiceBinding), "a voice frame carries its tools"
            binding.tools.update_tools(self._agent.tools)
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
            prepared.ctx._operation = invocation.operation
            if isinstance(binding, _VoiceBinding):
                report = await self._run_voice_tool(invocation, binding, prepared)
                return report, prepared.ctx.function_call.extra
            try:
                body = _tool_body(self._agent, prepared.ctx.function_call.name, prepared.tool)
                output = body(*prepared.args, **prepared.kwargs)
                if inspect.isawaitable(output):
                    output = await output
            except asyncio.CancelledError:
                return ToolReport(outcome=ToolOutcomeCancelled()), prepared.ctx.function_call.extra
            except Exception as e:
                return _classified(prepared.ctx.function_call, None, e), (
                    prepared.ctx.function_call.extra
                )
            return _classified(prepared.ctx.function_call, output, None), (
                prepared.ctx.function_call.extra
            )

    async def _run_voice_tool(
        self, invocation: Invocation, binding: _VoiceBinding, prepared: _Prepared
    ) -> ToolReport:
        """A call's body on a voice session: its own task, so the call's step is released by the
        body's first ``ctx.update()`` while the body runs on; its end then reaches the session as
        the call's final update. In a ``VoiceTask``'s frame, a ``Completed`` ends the task once
        validated."""
        call = prepared.ctx.function_call
        body = asyncio.create_task(
            _call(
                _tool_body(self._agent, call.name, prepared.tool, binding.session),
                prepared.args,
                prepared.kwargs,
            ),
            name=f"tool_{call.name}",
        )
        key = _operation_key(invocation)
        self._running[key] = body
        body.add_done_callback(lambda _: self._running.pop(key, None))
        first_update = prepared.ctx._first_update
        assert first_update is not None, "a voice tool's context takes updates"
        await asyncio.wait({body, first_update}, return_when=asyncio.FIRST_COMPLETED)
        if body.done():
            report = _ended(call, body)
            if binding.task and isinstance(report.outcome, ToolOutcomeCompleted):
                return await self._validated_completion(prepared.ctx, body.result().value, report)
            return report

        def _final(task: asyncio.Task[Any]) -> None:
            binding.session._core.tool_update(call.call_id, _ended(call, task))

        body.add_done_callback(_final)
        message, raw = first_update.result()
        return ToolReport(outcome=ToolOutcomeUpdate(message=message, raw=raw))

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

    def cancel(self, invocation: Invocation) -> None:
        """Cancels the callback in flight for ``invocation``; it still settles."""
        task = self._running.get(_operation_key(invocation))
        if task is not None:
            task.cancel()

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
                    origin=f"tool:{_function_call(self._invocation).call_id}"
                )
            self._token = _BindingContextVar.set(binding)
            self._register()
            return binding
        run = copy.copy(_current_run())
        run.id = self._invocation.invocation_id
        from .._core_view import from_core_item

        run._history = [from_core_item(item) for item in self._invocation.history]
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
    own = ToolContext(binding.agent.tools)
    extra = [
        tool
        for name, tool in ToolContext(binding.session._tools).function_tools.items()
        if own.get_function_tool(name) is None
    ]
    return ToolContext([*binding.agent.tools, *extra])


async def _call(body: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """The body, sync or async."""
    out = body(*args, **kwargs)
    return await out if inspect.isawaitable(out) else out


def _ended(call: FunctionCall, task: asyncio.Task[Any]) -> ToolReport:
    if task.cancelled():
        return ToolReport(outcome=ToolOutcomeCancelled())
    if (exc := task.exception()) is not None:
        return _classified(call, None, exc)
    return _classified(call, task.result(), None)


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
    from .agent_run import _TextRun

    run = _BindingContextVar.get(None)
    if not isinstance(run, _TextRun):
        raise RuntimeError("an AgentHost callback ran outside its invocation's context")
    return run


def _function_call(invocation: Invocation) -> FunctionCall:
    from .._core_view import from_core_item

    call = invocation.function_call
    if call is None:
        raise RuntimeError("a tool callback has no function call")
    if isinstance(call, FunctionCall):
        return call
    if isinstance(call, dict) and "type" not in call:
        call = {**call, "type": "function_call"}
    item = from_core_item(call)
    assert isinstance(item, FunctionCall)
    return item


def _operation_key(invocation: Invocation) -> tuple[str, int]:
    return invocation.invocation_id, invocation.operation
