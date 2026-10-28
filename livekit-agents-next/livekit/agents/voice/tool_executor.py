"""The session's tool runner: the core decides which call runs and what its outcome means; this
runs the Python function and classifies what it returned or raised (``llm.utils.tool_report``)."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, get_type_hints

from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict

from .. import _core
from .._generated.tool_executor import AsyncToolOptions, ToolHandlingOptions
from .._generated.tool_report import (
    Approval,
    ToolOutcomeCancelled,
    ToolOutcomeCompleted,
    ToolOutcomeInternal,
    ToolOutcomeInvalid,
    ToolOutcomeModelRetry,
    ToolOutcomeToolError,
    ToolOutcomeUpdate,
    ToolReport,
)
from ..llm import ToolContext
from ..llm.tool_context import FunctionTool, InputRequired, ModelRetry, Tool, Toolset
from ..llm.utils import is_context_type, prepare_function_arguments, tool_report
from ..log import logger
from .agent import _BindingContextVar, _VoiceBinding
from .events import RunContext
from .run_result import _tool_body
from .speech_handle import SpeechHandle

if TYPE_CHECKING:
    from ..llm.chat_context import FunctionCall
    from .agent import Agent
    from .agent_session import VoiceSession

__all__ = ["AsyncToolOptions", "ToolHandlingOptions", "UpdatePromptArgs"]

UPDATE_TEMPLATE = """The tool `{function_name}` has updated, message: {message}
The task is still running, so DON'T make up or give information not included in the message above."""


class UpdatePromptArgs(TypedDict):
    """Args for the ``update`` template."""

    function_name: str
    call_id: str
    message: str


class DuplicatePromptArgs(TypedDict):
    """Args for the ``duplicate_reject`` / ``duplicate_confirm`` templates."""

    function_name: str
    fnc_calls_json: list[str]
    fnc_calls_text: str


class ReplyPromptArgs(TypedDict):
    """Args for the ``reply_at_tail`` / ``reply_maybe_covered`` templates."""

    call_ids: list[str]


def _render(template: str | Callable[[Any], str], args: dict[str, Any]) -> str:
    return template(args) if callable(template) else template.format(**args)


class ToolRunner:
    """One agent frame's tools, run for the core (``_core_run``/``_core_cancel``)."""

    def __init__(
        self, session: VoiceSession[Any], agent: Agent, tools: Sequence[Tool | Toolset]
    ) -> None:
        self._session = session
        self._agent = agent
        # read per call: a toolset fills its tools in setup() (an MCP server's), after this
        self._tools = tools
        self._running: dict[str, asyncio.Task[Any]] = {}

    async def _core_run(
        self, call: dict[str, Any], speech: _core.SpeechHandle, granted: bool, task: bool
    ) -> ToolReport:
        from .._core_view import from_core_item

        fnc_call = from_core_item(call)
        assert fnc_call.type == "function_call"
        # a session tool added since this frame was installed (IVR's DTMF tool) is the session's
        tools = ToolContext(self._tools)
        tool = tools.get_function_tool(fnc_call.name) or ToolContext(
            self._session._tools
        ).get_function_tool(fnc_call.name)
        if tool is None:
            names = list(tools.function_tools)
            message = _core.unknown_tool_message(fnc_call.name, names)
            return ToolReport(outcome=ToolOutcomeToolError(message=message))

        handle = SpeechHandle._of(speech)
        ctx = RunContext(function_call=fnc_call, session=self._session, speech_handle=handle)
        try:
            args, kwargs = prepare_function_arguments(
                fnc=tool, json_arguments=fnc_call.arguments, call_ctx=ctx, fnc_call=fnc_call
            )
        except Exception as e:
            return _outcome(fnc_call, None, e)

        binding = _VoiceBinding(self._session, self._agent, origin=f"tool:{fnc_call.call_id}")
        # a tool that requires approval runs only on the core's grant, else asks its read-back
        approval: list[Approval] = []  # the call's mark once decided
        gated = isinstance(tool, FunctionTool) and tool.info.needs_approval
        if gated and granted:
            approval.append("granted")
        token = _BindingContextVar.set(binding)
        try:
            work = (
                _read_back(tool, args, kwargs, approval)  # type: ignore[arg-type]
                if gated and not granted
                else _call(
                    _tool_body(self._agent, fnc_call.name, tool, self._session), args, kwargs
                )
            )
            body = asyncio.create_task(work, name=f"tool_{fnc_call.name}")
        finally:
            _BindingContextVar.reset(token)
        self._running[fnc_call.call_id] = body
        body.add_done_callback(lambda _: self._running.pop(fnc_call.call_id, None))
        first_update = ctx._first_update
        await asyncio.wait({body, first_update}, return_when=asyncio.FIRST_COMPLETED)
        if body.done():
            report = _ended(fnc_call, body, approval)
            # in a VoiceTask's speech a `Completed` ends the task, once validated; from an
            # ordinary tool it is the call's output, with a reply
            if task and isinstance(report.outcome, ToolOutcomeCompleted):
                return await self._validated(ctx, body.result().value, report)
            return report

        # the first ctx.update() released the step; the core reads the end as a final update
        def _final(task: asyncio.Task[Any]) -> None:
            self._session._core.tool_update(fnc_call.call_id, _ended(fnc_call, task, approval))

        body.add_done_callback(_final)
        message, raw = first_update.result()
        return ToolReport(outcome=ToolOutcomeUpdate(message=message, raw=raw))

    async def _validated(self, ctx: RunContext, value: Any, report: ToolReport) -> ToolReport:
        """A task's `Completed(value)` through its agent's `output_type` and output validators:
        the value the task returns, or the error its model reads and corrects."""
        try:
            value = TypeAdapter(self._agent.output_type).validate_python(value)
            value = await self._agent._validate_output(ctx, value)
        except ValidationError as e:
            return ToolReport(outcome=ToolOutcomeToolError(message=str(e)))
        except ModelRetry as e:
            return ToolReport(outcome=ToolOutcomeModelRetry(message=e.message))
        return dataclasses.replace(report, transition=value)

    def _core_cancel(self, call_id: str) -> None:
        if task := self._running.get(call_id):
            task.cancel()


async def _call(body: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """The body, sync or async."""
    out = body(*args, **kwargs)
    return await out if inspect.isawaitable(out) else out


async def _read_back(
    tool: FunctionTool, args: tuple[Any, ...], kwargs: dict[str, Any], approval: list[str]
) -> InputRequired:
    """The read-back of a call the person has not approved, asked as written. Transitional: the
    voice path moves onto the core's approval gate with E-3 (the text path's is the core's)."""
    if tool.info.read_back is not None:
        question = tool.info.read_back(*args, **kwargs)
        if inspect.isawaitable(question):
            question = await question
        if isinstance(question, InputRequired):
            return question  # something is missing before there is anything to approve
        approval.append("asked")
        return InputRequired(str(question), verbatim=True)
    approval.append("asked")
    sig = inspect.signature(tool)
    hints = get_type_hints(tool, include_extras=True)
    bound = sig.bind_partial(*args, **kwargs)
    shown = ", ".join(
        f"{k}={v!r}"
        for k, v in bound.arguments.items()
        if k not in ("self", "cls")
        and not is_context_type(hints.get(k, object), allow_subclasses=True)
    )
    description = tool.info.description or tool.info.name
    plain = description.strip().split(". ", 1)[0].splitlines()[0].rstrip(".") or "this"
    return InputRequired(
        f"Please confirm: {plain} ({shown}). Yes or no?"
        if shown
        else f"Please confirm: {plain}. Yes or no?",
        verbatim=True,
    )


def _ended(fnc_call: FunctionCall, task: asyncio.Task[Any], approval: list[Approval]) -> ToolReport:
    if task.cancelled():
        report = ToolReport(outcome=ToolOutcomeCancelled())
    elif (exc := task.exception()) is not None:
        report = _outcome(fnc_call, None, exc)
    else:
        report = _outcome(fnc_call, task.result(), None)
    if approval:
        report = dataclasses.replace(report, approval=approval[0])
    return report


def _outcome(fnc_call: FunctionCall, output: Any, exc: BaseException | None) -> ToolReport:
    report = tool_report(output, exc)
    if isinstance(report.outcome, ToolOutcomeInternal):
        logger.error("exception occurred while executing a tool", exc_info=exc)
    elif isinstance(report.outcome, ToolOutcomeInvalid):
        logger.error(
            f"AI function `{fnc_call.name}` returned an invalid output",
            extra={"call_id": fnc_call.call_id, "lk.pii.output": output},
        )
    return report
