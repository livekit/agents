from collections.abc import Awaitable
from typing import Callable

from ...job import get_job_context
from ...llm import Tool, Toolset, function_tool
from ...log import logger
from ...voice.events import CloseEvent, RunContext

END_CALL_INSTRUCTIONS = """
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


async def _default_on_end_call(ctx: RunContext) -> None:
    await ctx.session.generate_reply(instructions="say goodbye to the user", tool_choice="none")


class EndCallTool(Toolset):
    def __init__(
        self,
        *,
        extra_instructions: str = "",
        delete_room: bool = True,
        on_end_call: Callable[[RunContext], Awaitable[None]] | None = _default_on_end_call,
    ):
        """
        This tool allows the agent to end the call and disconnect from the room.

        Args:
            extra_instructions: Additional instructions to add to the end call tool.
            delete_room: Whether to delete the room when the user ends the call.
            on_end_call: Callback to be called when the user ends the call.
        """
        super().__init__()
        self._delete_room = delete_room
        self._extra_instructions = extra_instructions
        self._on_end_call = on_end_call

        self._end_call_tool = function_tool(
            self._end_call,
            name="end_call",
            description=f"{END_CALL_INSTRUCTIONS}\n{extra_instructions}",
        )

    async def _end_call(self, ctx: RunContext) -> None:
        try:
            logger.debug("end_call tool called")
            ctx.session.once("close", self._on_session_close)
            if self._on_end_call:
                await self._on_end_call(ctx)
        finally:
            # close the AgentSession
            ctx.session.shutdown()

    def _on_session_close(self, ev: CloseEvent) -> None:
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
