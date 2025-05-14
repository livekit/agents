import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import google, silero

logger = logging.getLogger("gemini-video-agent")

load_dotenv()


class GeminiAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are gemini, a helpful assistant",
            llm=google.beta.realtime.RealtimeModel(),
            #  By default, additional video frames are transmitted while the user is speaking
            vad=silero.VAD.load(),
        )

    async def on_enter(self):
        if await self.session.ensure_silence_for(timeout=3):
            logger.info("generating proactive reply")
            self.session.generate_reply(
                instructions="introduce yourself very briefly and ask about the user's day"
            )


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession()

    await session.start(
        agent=GeminiAgent(),
        room=ctx.room,
        # by default, video is disabled
        room_input_options=RoomInputOptions(video_enabled=True),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
