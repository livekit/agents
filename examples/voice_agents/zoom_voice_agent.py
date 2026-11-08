"""A voice agent running on Zoom Voice Agent (speech-to-speech), with one function tool.

Zoom handles recognition, the model, speech and turn-taking, so the session needs only the
realtime model. Needs ZOOM_AI_API_KEY (a Zoom AI Services key with Voice Agent access, from
Platform Studio):

    python examples/voice_agents/zoom_voice_agent.py console

Wear headphones in console mode so the agent doesn't hear itself.
"""

import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.agents.llm import function_tool
from livekit.plugins import zoom_ai

logger = logging.getLogger("zoom-voice-agent")

load_dotenv()


class OrderAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            # Zoom takes no per-reply instructions, so the greeting is described here
            instructions=(
                "You are a friendly order support agent. Greet the caller in one sentence, "
                "then help with their order. Keep answers short. Use lookup_order when the "
                "caller asks about an order."
            ),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply()

    @function_tool
    async def lookup_order(self, order_id: str) -> str:
        """Look up the status of an order.

        Args:
            order_id: The order number the caller gives.
        """
        logger.info(f"looking up order {order_id}")
        return f"Order {order_id} shipped and arrives Friday."


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session: AgentSession = AgentSession(llm=zoom_ai.realtime.RealtimeModel(voice="jenna"))
    await session.start(agent=OrderAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
