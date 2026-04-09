import logging
import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import did, openai

logger = logging.getLogger("did-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="alloy"),
        resume_false_interruption=False,
    )

    agent_id = os.getenv("DID_AGENT_ID")
    did_avatar = did.AvatarSession(agent_id=agent_id)
    await did_avatar.start(session, room=ctx.room)

    # start the agent, it will join the room and wait for the avatar to join
    await session.start(
        agent=Agent(instructions="Talk to me!"),
        room=ctx.room,
    )

    session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(server)
