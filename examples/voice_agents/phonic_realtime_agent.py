import asyncio
import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    function_tool,
)
from livekit.plugins.phonic.realtime import RealtimeModel

logger = logging.getLogger("phonic-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice AI assistant named Sabrina.",
            llm=RealtimeModel(
                voice="sabrina",
                audio_speed=1.2,
            ),
        )

    @function_tool(
        description="Toggle a light on or off. Available lights are A05, A06, A07, and A08."
    )
    async def toggle_light(self, light_id: str, state: str) -> str:
        """Called when the user asks to toggle a light on or off.

        Args:
            light_id: The ID of the light to toggle
            state: Whether to turn the light on or off, e.g., 'on', 'off'
        """
        logger.info(f"Turning {state} light {light_id}")
        await asyncio.sleep(1.0)
        return f"Light {light_id} turned {state}"


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession()
    await session.start(agent=MyAgent(), room=ctx.room)
    await session.generate_reply(
        instructions="Greet the user, asking about their day.",
    )


if __name__ == "__main__":
    cli.run_app(server)
