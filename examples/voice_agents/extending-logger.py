from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import openai

load_dotenv()


async def entrypoint(ctx: JobContext):
    user_id = "fake_user_id"
    ctx.log_context_fields = {"user_id": user_id}

    await ctx.connect()

    agent = Agent(instructions="You are a helpful assistant.")
    session = AgentSession(llm=openai.realtime.RealtimeModel())

    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
