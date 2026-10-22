import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
    function_tool,
)
from livekit.plugins import elevenlabs

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice.
            Keep your answers short and conversational, without any formatting, emojis or symbols.
            When the user asks about the weather, use the lookup_weather tool.""",
        )

    async def on_enter(self) -> None:
        self.session.generate_reply()

    @function_tool
    async def lookup_weather(self, context: RunContext, location: str) -> str:
        """Look up the current weather for a location.

        Args:
            location: The city or place to look up.
        """
        logger.info(f"looking up weather for {location}")
        return f"It is sunny and 21 degrees Celsius in {location}."


server = AgentServer()


@server.rtc_session()
async def my_agent(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        llm=elevenlabs.realtime.RealtimeModel(
            model="eleven_realtime_v1_mini",
            voice="brian",
            language="en",
        ),
    )

    await session.start(agent=Assistant(), room=ctx.room)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
