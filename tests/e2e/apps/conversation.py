"""A deterministic conversation app; the framework supplies only process configuration."""

from dataclasses import dataclass

from livekit.agents import AgentServer, SessionState, TextSession, persisted
from livekit.agents.a2a import A2ASessionContext

from .conversation_agent import agent

server = AgentServer()


@dataclass
class Conversation:
    session: SessionState = persisted(default_factory=SessionState)


@server.a2a_session(endpoint="desk", description="Remember a conversation")
async def text(ctx: A2ASessionContext) -> object:
    app = await ctx.store.load_context(Conversation())
    turn = await TextSession(agent=agent()).run(ctx.message, context=ctx.store.bind(app))
    return turn
