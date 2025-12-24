from ...llm import Tool, Toolset, function_tool
from ...log import logger
from ...voice.events import RunContext


class EndCallTool(Toolset):
    @function_tool(name="end_call")
    async def _end_call(self, ctx: RunContext) -> None:
        """
        Ends the current call and disconnects immediately.

        Call when:
        - The user clearly indicates they are done (e.g., “that’s all, bye”).
        - The agent determines the conversation is complete and should end.

        Do not call when:
        - The user asks to pause, hold, or transfer.
        - Intent is unclear.

        This is the final action the agent can take.
        Once called, no further interaction is possible with the user.
        """
        logger.debug("end_call tool called")
        ctx.session.shutdown()

    @property
    def tools(self) -> list[Tool]:
        return [self._end_call]
