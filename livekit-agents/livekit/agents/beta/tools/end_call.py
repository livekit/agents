import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from ...job import get_job_context
from ...llm import RealtimeModel, Tool, Toolset, function_tool
from ...log import logger
from ...voice.events import CloseEvent, RunContext, SpeechCreatedEvent
from ...voice.speech_handle import SpeechHandle

END_CALL_DESCRIPTION = """
Ends the current call and disconnects immediately.

Call when:
- The user clearly indicates they are done (e.g., “that’s all, bye”).
- The agent determines the conversation is complete and should end.

Do not call when:
- The user asks to pause, hold, or transfer.
- Intent is unclear.

This is the final action the agent can take.
Once called, no further interaction is possible with the user.
Don't generate any other text or response when the tool is called.
"""


class EndCallTool(Toolset):
    def __init__(
        self,
        *,
        extra_description: str = "",
        delete_room: bool = True,
        end_instructions: str | None = "say goodbye to the user",
        on_tool_called: Callable[[Toolset.ToolCalledEvent], Awaitable[None]] | None = None,
        on_tool_completed: Callable[[Toolset.ToolCompletedEvent], Awaitable[None]] | None = None,
    ):
        """
        This tool allows the agent to end the call and disconnect from the room.

        Args:
            extra_description: Additional description to add to the end call tool.
            delete_room: Whether to delete the room when the user ends the call. deleting the room disconnects all remote users, including SIP callers.
            end_instructions: Tool output to the LLM for generating the tool response.
            on_tool_called: Callback to call when the tool is called.
            on_tool_completed: Callback to call when the tool is completed.
        """
        super().__init__(id="end_call")
        self._delete_room = delete_room
        self._extra_description = extra_description

        self._end_instructions = end_instructions
        self._on_tool_called = on_tool_called
        self._on_tool_completed = on_tool_completed

        self._end_call_tool = function_tool(
            self._end_call,
            name="end_call",
            description=f"{END_CALL_DESCRIPTION}\n{extra_description}",
        )
        self._shutdown_session_task: asyncio.Task[None] | None = None

    async def _end_call(self, ctx: RunContext) -> Any | None:
        logger.debug("end_call tool called")
        llm_v = ctx.session.current_agent._get_activity_or_raise().llm

        def _on_speech_done(_: SpeechHandle) -> None:
            if (
                not isinstance(llm_v, RealtimeModel)
                or not llm_v.capabilities.auto_tool_reply_generation
            ):
                # tool reply will reuse the same speech handle, so we can shutdown the session
                # directly after this speech handle is done
                ctx.session.shutdown()
            else:
                self._shutdown_session_task = asyncio.create_task(
                    self._delayed_session_shutdown(ctx)
                )

        ctx.speech_handle.add_done_callback(_on_speech_done)
        ctx.session.once("close", self._on_session_close)

        if self._on_tool_called:
            await self._on_tool_called(Toolset.ToolCalledEvent(ctx=ctx, arguments={}))

        completed_ev = Toolset.ToolCompletedEvent(ctx=ctx, output=self._end_instructions)
        if self._on_tool_completed:
            await self._on_tool_completed(completed_ev)

        return completed_ev.output

    async def _delayed_session_shutdown(self, ctx: RunContext) -> None:
        """Shutdown the session after the tool reply is played out"""
        speech_created_fut = asyncio.Future[SpeechHandle]()

        @ctx.session.once("speech_created")
        def _on_speech_created(ev: SpeechCreatedEvent) -> None:
            if not speech_created_fut.done():
                speech_created_fut.set_result(ev.speech_handle)

        try:
            speech_handle = await asyncio.wait_for(speech_created_fut, timeout=5.0)
            await speech_handle
        except asyncio.TimeoutError:
            logger.warning("tool reply timed out, shutting down session")
        finally:
            ctx.session.off("speech_created", _on_speech_created)
            ctx.session.shutdown()

    def _on_session_close(self, ev: CloseEvent) -> None:
        """Close the job process when AgentSession is closed"""
        if self._shutdown_session_task:
            # cleanup
            self._shutdown_session_task.cancel()
            self._shutdown_session_task = None

        job_ctx = get_job_context()

        if self._delete_room:

            async def _on_shutdown() -> None:
                logger.info("deleting the room because the user ended the call")
                await job_ctx.delete_room()

            job_ctx.add_shutdown_callback(_on_shutdown)

        # shutdown the job process
        job_ctx.shutdown(reason=ev.reason.value)

    @property
    def tools(self) -> list[Tool]:
        return [self._end_call_tool]
