"""Run with: python experiments/livekit/examples/agent.py dev."""

import os

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession
from livekit.plugins import openai, openspline

load_dotenv()
server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    session = AgentSession(llm=openai.realtime.RealtimeModel(model="gpt-realtime"))
    source = (
        {"avatar_id": os.environ["OPENSPLINE_AVATAR_ID"]}
        if os.getenv("OPENSPLINE_AVATAR_ID")
        else {"portrait": os.environ["PORTRAIT"]}
    )
    avatar = openspline.AvatarSession(**source)
    await avatar.start(session, room=ctx.room)
    await session.start(room=ctx.room, agent=Agent(instructions="Be helpful and concise."))
    await session.generate_reply(instructions="Greet the user.")


if __name__ == "__main__":
    agents.cli.run_app(server)
