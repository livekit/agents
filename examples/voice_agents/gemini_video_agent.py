import logging

from dotenv import load_dotenv
from google.genai import types

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import google

logger = logging.getLogger("gemini-video-agent")

load_dotenv()


class GeminiAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="you are gemini, a helpful assistant",
            llm=google.beta.realtime.RealtimeModel(
                input_audio_transcription=types.AudioTranscriptionConfig(),
                vertexai=True,
                project="project-id",
                location="us-central1",
            ),
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions="introduce yourself very briefly and ask about the user's day"
        )


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession()

    await ctx.wait_for_participant()

    await session.start(
        agent=GeminiAgent(),
        room=ctx.room,
        # by default, video is disabled
        room_input_options=RoomInputOptions(video_enabled=True),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
