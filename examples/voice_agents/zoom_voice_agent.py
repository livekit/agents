"""Three agents in one conversation, the third running on Zoom Voice Agent.

Kelly (STT -> LLM -> TTS) answers first and can hand the caller to Alloy (OpenAI Realtime) or
Nova (Zoom Voice Agent, speech-to-speech). Each agent can hand back to the others.

Needs ZOOM_AI_API_KEY (a Zoom AI Services key with Voice Agent access, from Platform Studio),
OPENAI_API_KEY and LiveKit credentials:

    python examples/voice_agents/zoom_voice_agent.py console
"""

import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference
from livekit.agents.llm import function_tool
from livekit.plugins import openai, zoom_ai

logger = logging.getLogger("zoom-voice-agent")

load_dotenv()


@function_tool
async def lookup_order(order_id: str) -> str:
    """Look up the status of an order.

    Args:
        order_id: The order number the caller gives.
    """
    logger.info(f"looking up order {order_id}")
    return f"Order {order_id} shipped and arrives Friday."


class Kelly(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Kelly, the front desk. Greet the caller and ask how you can help. "
                "Send order questions to Nova and anything else to Alloy. Keep it short."
            ),
            stt=inference.STT("deepgram/nova-3"),
            llm=inference.LLM("openai/gpt-4.1-mini"),
            tts=inference.TTS("cartesia/sonic-3"),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply()

    @function_tool
    async def transfer_to_nova(self) -> Agent:
        """Transfer the caller to Nova, who handles orders."""
        return Nova()

    @function_tool
    async def transfer_to_alloy(self) -> Agent:
        """Transfer the caller to Alloy, for general questions."""
        return Alloy()


class Alloy(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your name is Alloy. Answer general questions briefly.",
            llm=openai.realtime.RealtimeModel(voice="alloy"),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply()

    @function_tool
    async def transfer_to_kelly(self) -> Agent:
        """Transfer the caller back to Kelly at the front desk."""
        return Kelly()

    @function_tool
    async def transfer_to_nova(self) -> Agent:
        """Transfer the caller to Nova, who handles orders."""
        return Nova()


class Nova(Agent):
    def __init__(self) -> None:
        super().__init__(
            # Zoom takes no per-reply instructions, so the greeting lives here. The
            # conversation so far is added to these instructions when Nova takes over.
            instructions=(
                "Your name is Nova and you handle orders. Introduce yourself in one sentence "
                "as soon as you join, then help with the caller's order. Use lookup_order "
                "for order status."
            ),
            llm=zoom_ai.realtime.RealtimeModel(voice="ava", vocabulary=["LiveKit"]),
            tools=[lookup_order],
        )

    async def on_enter(self) -> None:
        self.session.generate_reply()

    @function_tool
    async def transfer_to_kelly(self) -> Agent:
        """Transfer the caller back to Kelly at the front desk."""
        return Kelly()

    @function_tool
    async def transfer_to_alloy(self) -> Agent:
        """Transfer the caller to Alloy, for general questions."""
        return Alloy()


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session: AgentSession = AgentSession()
    await session.start(agent=Kelly(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
