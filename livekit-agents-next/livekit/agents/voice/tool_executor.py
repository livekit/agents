"""The session's tool runner: the core decides which call runs and what its output means; this
runs the Python function and reports what it did, as the marshal's dict the core reads."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from typing_extensions import TypedDict

from .. import _core
from .._generated.tool_executor import AsyncToolOptions, ToolHandlingOptions
from ..llm import ToolContext
from ..llm.tool_context import (
    ASK_IN_OWN_WORDS,
    InputRequired,
    StopResponse,
    Tool,
    ToolError,
    Toolset,
    Transition,
)
from ..llm.utils import prepare_function_arguments
from ..log import logger
from .agent import _BindingContextVar, _VoiceBinding
from .events import RunContext
from .speech_handle import SpeechHandle
from .tool_output import make_tool_output

if TYPE_CHECKING:
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
        self._tools = ToolContext(tools)
        self._running: dict[str, asyncio.Task[Any]] = {}

    async def _core_run(
        self, call: dict[str, Any], speech: _core.SpeechHandle, step: int
    ) -> dict[str, Any]:
        from .._core_view import from_core_item

        fnc_call = from_core_item(call)
        assert fnc_call.type == "function_call"
        # a session tool added since this frame was installed (IVR's DTMF tool) is the session's
        tool = self._tools.get_function_tool(fnc_call.name) or ToolContext(
            self._session._tools
        ).get_function_tool(fnc_call.name)
        if tool is None:
            names = list(self._tools.function_tools)
            return _raised(ToolError(_core.unknown_tool_message(fnc_call.name, names)))

        handle = SpeechHandle._of(speech)
        ctx = RunContext(function_call=fnc_call, session=self._session, speech_handle=handle)
        try:
            args, kwargs = prepare_function_arguments(
                fnc=tool, json_arguments=fnc_call.arguments, call_ctx=ctx, fnc_call=fnc_call
            )
        except Exception as e:
            return _raised(e)

        binding = _VoiceBinding(self._session, self._agent, origin=f"tool:{fnc_call.call_id}")
        token = _BindingContextVar.set(binding)
        try:
            task = asyncio.create_task(tool(*args, **kwargs), name=f"tool_{fnc_call.name}")
        finally:
            _BindingContextVar.reset(token)
        self._running[fnc_call.call_id] = task
        task.add_done_callback(lambda _: self._running.pop(fnc_call.call_id, None))
        first_update = ctx._first_update
        await asyncio.wait({task, first_update}, return_when=asyncio.FIRST_COMPLETED)
        if task.done():
            return self._answer(fnc_call, task, handle)

        # the first ctx.update() released the step; the core reads the end as a final update
        def _final(task: asyncio.Task[Any]) -> None:
            self._session._core.tool_update(
                fnc_call.call_id, "final", answer=self._answer(fnc_call, task, handle)
            )

        task.add_done_callback(_final)
        message, raw = first_update.result()
        return {"kind": "detached", "message": message, "raw": raw}

    def _core_cancel(self, call_id: str) -> None:
        if task := self._running.get(call_id):
            task.cancel()

    def _answer(
        self, fnc_call: Any, task: asyncio.Task[Any], handle: SpeechHandle
    ) -> dict[str, Any]:
        if task.cancelled():
            return {"kind": "cancelled"}
        if (exc := task.exception()) is not None:
            return _raised(exc)
        out = make_tool_output(fnc_call=fnc_call, output=task.result(), exception=None)
        if out.transition is not None:
            self._apply(out.transition, handle)
        if out.fnc_call_out.is_error:
            return {"kind": "raised", "error": "tool_error", "message": out.fnc_call_out.output}
        value = out.fnc_call_out.output if out.fnc_call_out.reply_required else None
        return {"kind": "returned", "value": value}

    def _apply(self, transition: Transition, handle: SpeechHandle) -> None:
        """Deliver human-input questions requested by tools."""
        if isinstance(transition, InputRequired):
            self._session._core.set_run_output(handle._core, value=transition, error=None)
            if transition.verbatim:
                self._session.say(transition.question)
            else:
                self._session.generate_reply(
                    instructions=ASK_IN_OWN_WORDS.format(question=transition.question),
                    tool_choice="none",
                )


def _raised(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, ToolError):
        return {"kind": "raised", "error": "tool_error", "message": exc.message}
    if isinstance(exc, StopResponse):
        return {"kind": "raised", "error": "stop_response"}
    logger.error("exception occurred while executing a tool", exc_info=exc)
    return {"kind": "raised", "error": "internal"}
