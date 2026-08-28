"""Delegating to an expert that runs somewhere else, over the A2A protocol.

`delegation.py` runs its expert in this process. Here the expert is an HTTP endpoint that
speaks A2A, and nothing else changes: the conversation model still gets one `delegate` tool,
and still receives progress while the work runs and facts when it finishes.

The endpoint does not have to be built with this framework. Any agent that speaks A2A 1.0
works — LangGraph, ADK, or anything else — because what travels is the protocol's own
`SendMessageRequest` and task stream, not a shape of ours. The agent card at
`<url>/.well-known/agent-card.json` is read once, so the endpoint describes its own skills and
auth rather than being a URL with an implicit contract.

    FARE_DESK_URL=https://experts.internal/fare-desk python delegation_remote.py console

What the remote placement gives up is the shared view of in-flight work. The expert is top
level in its own process, so a second delegation cannot cancel a booking the first one started
— no store fixes that, because cancelling needs the live task rather than a record of it. What
it buys is that the expert scales on its own, can be shared between agents, and need not be
written in Python.

Serving *our* expert as an A2A endpoint is not here yet: it needs the HTTP server from
livekit/agents#6856, and lands as `@server.a2a(path=...)`. See DELEGATION_DESIGN.md.
"""

import logging
import os

from delegation import VOICE_INSTRUCTIONS
from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, a2a, cli

logger = logging.getLogger("delegation-remote")

load_dotenv()

FARE_DESK_URL = os.getenv("FARE_DESK_URL", "http://127.0.0.1:8080")

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    delegate = a2a.A2ADelegate(FARE_DESK_URL)
    ctx.add_shutdown_callback(delegate.aclose)

    session = AgentSession(
        stt="deepgram/nova-3",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2",
        delegate=delegate,
        delegation_options={"metadata": {"room": ctx.room.name}},
    )

    await session.start(agent=Agent(instructions=VOICE_INSTRUCTIONS), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
