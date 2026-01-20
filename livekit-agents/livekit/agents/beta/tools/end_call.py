from collections.abc import Awaitable
from typing import Callable

from ...job import get_job_context
from ...llm import Tool, Toolset, function_tool
from ...log import logger
from ...voice.events import CloseEvent, RunContext

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
        on_end: str | Callable[[RunContext], Awaitable[None]] | None = "say goodbye to the user",
    ):
        """
        This tool allows the agent to end the call and disconnect from the room.

        Args:
            extra_description: Additional description to add to the end call tool.
            delete_room: Whether to delete the room when the user ends the call. deleting the room disconnects all remote users, including SIP callers.
            on_end: If a string is provided, it will be used as the instructions of
                `session.generate_reply` when the user ends the call. If a callback, it will be called
                when the user ends the call.
        """
        super().__init__()
        self._delete_room = delete_room
        self._extra_description = extra_description
        self._on_end = on_end

        self._end_call_tool = function_tool(
            self._end_call,
            name="end_call",
            description=f"{END_CALL_DESCRIPTION}\n{extra_description}",
        )

    async def _end_call(self, ctx: RunContext) -> None:
        try:
            logger.debug("end_call tool called")
            ctx.session.once("close", self._on_session_close)
            if isinstance(self._on_end, str):
                await ctx.session.generate_reply(instructions=self._on_end, tool_choice="none")
            elif callable(self._on_end):
                await self._on_end(ctx)
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
