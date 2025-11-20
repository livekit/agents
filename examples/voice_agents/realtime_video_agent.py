import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    room_io,
    voice,  # noqa: F401
)
from livekit.plugins import google, silero

logger = logging.getLogger("realtime-video-agent")

load_dotenv()

server = AgentServer()


@server.rtc_session()
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
        room_options=room_io.RoomOptions(
            video_input=True,
        ),
    )

    await session.generate_reply(
        instructions="introduce yourself very briefly and ask about the user's day"
    )


if __name__ == "__main__":
    cli.run_app(server)
