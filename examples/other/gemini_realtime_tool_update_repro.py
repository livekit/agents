"""Repro for issue #6479: Gemini Live tool result lost when update_tools() restarts the session.

Scenario
--------
1. A function tool (`get_weather`) is registered on a Gemini Live realtime agent.
2. The user asks something that makes the model call the tool.
3. While the tool is running, `session.update_tools(...)` is called. On Gemini Live this
   forces the underlying websocket to restart (`_session_should_close` is set).
4. The tool returns. Before the fix, its result was sent on the dying socket and never
   replayed to the reconnected session, so the model hung waiting for a response it would
   never receive and the turn stalled.

With the fix, the tool result is buffered while the socket is restarting and replayed once the
new session is established, so the model receives it and continues the turn.

Run (needs GOOGLE_API_KEY):
    python examples/other/gemini_realtime_tool_update_repro.py console
"""

from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, function_tool
from livekit.plugins import google

load_dotenv()

logger = logging.getLogger("gemini-tool-update-repro")


class WeatherAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a voice assistant. When asked about the weather, call the "
                "get_weather tool and then tell the user the result in one sentence."
            ),
        )

    @function_tool
    async def get_weather(self, location: str) -> str:
        """Get the current weather for a location.

        Args:
            location: the city to get the weather for
        """
        logger.info("get_weather called for %s; triggering update_tools() mid-turn", location)

        # Force a session restart in the middle of the tool call, exactly like a real
        # update_tools() would while the model is awaiting this tool's result. This is the
        # window where the result used to be lost.
        session = self.session
        assert session._activity is not None
        realtime_session = session._activity.realtime_llm_session
        if realtime_session is not None:
            # change the tool set mid-turn; update_tools() only restarts the socket when the
            # tools actually differ, so clear them to force the restart this repro needs
            await realtime_session.update_tools([])
            logger.info("update_tools() done — session is now restarting")

        # simulate the tool doing a bit of work while the socket reconnects
        await asyncio.sleep(0.5)
        return f"It's 22°C and sunny in {location}."


async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        llm=google.beta.realtime.RealtimeModel(
            # any Gemini Live model; the fix is model-agnostic (capability-driven)
            voice="Puck",
        ),
    )

    await session.start(agent=WeatherAgent(), room=ctx.room)
    await session.generate_reply(instructions="Greet the user and ask how you can help.")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
