import os

from dotenv import load_dotenv
from openai.types.beta.realtime.session import TurnDetection

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, WorkerType, cli
from livekit.plugins import avatartalk, openai

load_dotenv()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            voice="ash",
            turn_detection=TurnDetection(
                type="server_vad",
                threshold=0.5,
                prefix_padding_ms=300,
                silence_duration_ms=500,
                create_response=True,
                interrupt_response=True,
            ),
        )
    )

    avatar = avatartalk.AvatarSession(
        api_url=os.getenv("AVATARTALK_API_URL"),
        avatar=os.getenv("AVATARTALK_AVATAR"),
        emotion=os.getenv("AVATARTALK_EMOTION"),
        api_secret=os.getenv("AVATARTALK_API_KEY"),
    )

    await avatar.start(session, room=ctx.room)
    await session.start(
        agent=Agent(
            instructions=(
                "You are a helpful AI assistant designed to "
                "communicate with users in multiple languages. "
                "Your primary directive is to always respond in "
                "the same language that the user writes in."
            )
        ),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))
