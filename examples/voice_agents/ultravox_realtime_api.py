import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomOutputOptions,
    WorkerOptions,
    cli,
    function_tool,
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


async def entrypoint(ctx: JobContext):
    session = AgentSession()

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_output_options=RoomOutputOptions(
            transcription_speed_factor=1.5,
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
