"""Prebuilt tools: end the call after the agent's goodbye, and send keypad (DTMF) tones. The
core decides when the call closes and sends the tones."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from .. import _core
from ..job import get_job_context
from ..llm import ToolFlag, Toolset, function_tool
from ..log import logger
from ..voice.events import CloseEvent, RunContext
from .workflows import DtmfEvent

__all__ = ["EndCallTool", "send_dtmf_events"]


class EndCallTool(Toolset):
    def __init__(
        self,
        *,
        extra_description: str = "",
        delete_room: bool = True,
        end_instructions: str | None = "say goodbye to the user",
        ignore_on_enter: bool = False,
        on_tool_called: Callable[[Toolset.ToolCalledEvent], Awaitable[None]] | None = None,
        on_tool_completed: Callable[[Toolset.ToolCompletedEvent], Awaitable[None]] | None = None,
    ):
        """
        This tool allows the agent to end the call and disconnect from the room.

        Args:
            extra_description: Additional description to add to the end call tool.
            delete_room: Whether to delete the room when the user ends the call. deleting the room disconnects all remote users, including SIP callers.
            end_instructions: Tool output to the LLM for generating the tool response.
            ignore_on_enter: Hide the tool during ``on_enter`` so the model can't end the call while greeting.
            on_tool_called: Callback to call when the tool is called.
            on_tool_completed: Callback to call when the tool is completed.
        """
        end_call_tool = function_tool(
            self._end_call,
            name="end_call",
            description=f"{_core.END_CALL_DESCRIPTION}\n{extra_description}",
            flags=ToolFlag.IGNORE_ON_ENTER if ignore_on_enter else ToolFlag.NONE,
        )
        super().__init__(id="end_call", tools=[end_call_tool])
        self._delete_room = delete_room
        self._end_instructions = end_instructions
        self._on_tool_called = on_tool_called
        self._on_tool_completed = on_tool_completed

    async def _end_call(self, ctx: RunContext) -> Any | None:
        logger.debug("end_call tool called")
        # the core closes the session once this speech (or a realtime model's own reply to the
        # tool result) is done
        ctx.session._core.end_call(ctx.speech_handle._core)
        ctx.session.once("close", self._on_session_close)

        if self._on_tool_called:
            await self._on_tool_called(Toolset.ToolCalledEvent(ctx=ctx, arguments={}))

        completed_ev = Toolset.ToolCompletedEvent(ctx=ctx, output=self._end_instructions)
        if self._on_tool_completed:
            await self._on_tool_completed(completed_ev)

        return completed_ev.output

    def _on_session_close(self, ev: CloseEvent) -> None:
        """The call is over: the job ends, deleting the room on its way out when asked."""
        job_ctx = get_job_context(required=False)
        if job_ctx is None:
            return
        if self._delete_room:

            async def _delete_room(reason: str) -> None:
                logger.info("deleting the room because the user ended the call")
                await job_ctx.delete_room()

            job_ctx.add_shutdown_callback(_delete_room)
        job_ctx.shutdown(reason=ev.reason.value)


@function_tool
async def send_dtmf_events(ctx: RunContext, events: list[DtmfEvent]) -> str:
    """
    Send a list of DTMF events to the telephony provider.

    Call when:
    - User wants to send DTMF events
    """
    room = ctx.session._room or get_job_context().room
    return await _core.send_dtmf_events(room, [event.value for event in events])
