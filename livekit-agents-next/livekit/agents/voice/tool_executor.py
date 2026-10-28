"""The session's tool runner: the core decides which call runs and what its outcome means; this
runs the Python function and classifies what it returned or raised (``llm.utils.tool_outcome``)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict

from .. import _core
from .._generated.tool_executor import AsyncToolOptions, ToolHandlingOptions
from ..llm import ToolContext
from ..llm.tool_context import (
    FunctionTool,
    ModelRetry,
    Tool,
    Toolset,
    _approval_scope,
    _ApprovalScope,
)
from ..llm.utils import prepare_function_arguments, tool_outcome
from ..log import logger
from .agent import _BindingContextVar, _VoiceBinding
from .events import RunContext
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
    ) -> dict[str, Any]:
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
            return {"kind": "tool_error", "text": _core.unknown_tool_message(fnc_call.name, names)}

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
        approval = _ApprovalScope(granted=granted) if _needs_approval(tool) else None
        token = _BindingContextVar.set(binding)
        scope_token = _approval_scope.set(approval)
        try:
            task = asyncio.create_task(tool(*args, **kwargs), name=f"tool_{fnc_call.name}")
        finally:
            _approval_scope.reset(scope_token)
            _BindingContextVar.reset(token)
        self._running[fnc_call.call_id] = task
        task.add_done_callback(lambda _: self._running.pop(fnc_call.call_id, None))
        first_update = ctx._first_update
        await asyncio.wait({task, first_update}, return_when=asyncio.FIRST_COMPLETED)
        if task.done():
            outcome = _ended(fnc_call, task, approval)
            # in a VoiceTask's speech a `Completed` ends the task, once validated
            if task and outcome["kind"] == "completed":
                return await self._validated(ctx, task.result().value, outcome)
            return outcome

        # the first ctx.update() released the step; the core reads the end as a final update
        def _final(task: asyncio.Task[Any]) -> None:
            self._session._core.tool_update(fnc_call.call_id, _ended(fnc_call, task, approval))

        task.add_done_callback(_final)
        message, raw = first_update.result()
        return {"kind": "update", "text": message, "raw": raw}

    async def _validated(
        self, ctx: RunContext, value: Any, outcome: dict[str, Any]
    ) -> dict[str, Any]:
        """A task's `Completed(value)` through its agent's `output_type` and output validators:
        the value the task returns, or the error its model reads and corrects."""
        try:
            value = TypeAdapter(self._agent.output_type).validate_python(value)
            value = await self._agent._validate_output(ctx, value)
        except ValidationError as e:
            return {"kind": "tool_error", "text": str(e)}
        except ModelRetry as e:
            return {"kind": "tool_error", "text": e.message}
        return {**outcome, "transition": value}

    def _core_cancel(self, call_id: str) -> None:
        if task := self._running.get(call_id):
            task.cancel()


def _needs_approval(tool: Tool) -> bool:
    return isinstance(tool, FunctionTool) and tool.info.needs_approval


def _ended(
    fnc_call: FunctionCall, task: asyncio.Task[Any], approval: _ApprovalScope | None
) -> dict[str, Any]:
    if task.cancelled():
        outcome: dict[str, Any] = {"kind": "cancelled"}
    elif (exc := task.exception()) is not None:
        outcome = _outcome(fnc_call, None, exc)
    else:
        outcome = _outcome(fnc_call, task.result(), None)
    if approval is not None and (approval.granted or approval.asked):
        outcome["approval"] = "granted" if approval.granted else "asked"
    return outcome


def _outcome(fnc_call: FunctionCall, output: Any, exc: BaseException | None) -> dict[str, Any]:
    outcome = tool_outcome(output, exc)
    if outcome["kind"] == "internal":
        logger.error("exception occurred while executing a tool", exc_info=exc)
    elif outcome["kind"] == "invalid":
        logger.error(
            f"AI function `{fnc_call.name}` returned an invalid output",
            extra={"call_id": fnc_call.call_id, "lk.pii.output": output},
        )
    return outcome
