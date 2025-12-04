import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    function_tool,
    room_io,
)
from livekit.plugins import silero
from livekit.plugins.ultravox.realtime import RealtimeModel

logger = logging.getLogger("ultravox-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Jessica, a helpful assistant",
            llm=RealtimeModel(
                model="fixie-ai/ultravox",
                voice="Jessica",
                language_hint="en",
            ),
            vad=silero.VAD.load(),
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions="introduce yourself very briefly and ask about the user's day"
        )

    @function_tool
    async def get_weather(self, city: str):
        """Get the weather for a given city"""
        return f"The weather in {city} is sunny and 70 degrees"


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession()

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            text_output=room_io.TextOutputOptions(transcription_speed_factor=1.5),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
