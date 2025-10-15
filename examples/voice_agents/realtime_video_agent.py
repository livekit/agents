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
    voice,  # noqa: F401
)
from livekit.plugins import google, silero

logger = logging.getLogger("realtime-video-agent")

load_dotenv()


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=silero.VAD.load(),
        # both Gemini and OpenAI Realtime API support streaming video input
        llm=google.realtime.RealtimeModel(),
        # customize how video frames are sampled
        # by default it's 1fps while the user is speaking and 0.3fps when silent
        # video_sampler=voice.VoiceActivityVideoSampler(speaking_fps=1.0, silent_fps=0.1),
    )

    agent = Agent(
        instructions="You are an AI assistant that interacts with user via voice. You are able to see the user's video and hear the user's voice.",
    )

    await session.start(
        agent=agent,
        room=ctx.room,
        # by default, video is disabled
        room_input_options=RoomInputOptions(video_enabled=True),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

    await session.generate_reply(
        instructions="introduce yourself very briefly and ask about the user's day"
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
