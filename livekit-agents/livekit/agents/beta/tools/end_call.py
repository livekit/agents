from ...llm import FunctionTool, function_tool, ToolSet
from ...log import logger


class EndCallTool(ToolSet):
    @function_tool(name="end_call")
    async def _end_call(self) -> None:
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
        from ...voice.agent_activity import _AgentActivityContextVar

        activity = _AgentActivityContextVar.get(None)
        if not activity:
            logger.error("couldn't end the call, EndCallTool was called without any AgentSession")
            return

        activity.session.close_soon()

    def __livekit_tools__(self) -> list[FunctionTool]:
        return [self._end_call]
