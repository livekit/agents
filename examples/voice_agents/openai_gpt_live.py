"""OpenAI GPT-Live (alpha) full-duplex voice agent.

GPT-Live is a server-driven, full-duplex voice model: it listens and speaks at the
same time and decides when to reply, so unlike the Realtime API there is no client
`generate_reply`. Reasoning and tools are delegated to a backend Responses model
(``gpt-5.5``), so ordinary ``@function_tool`` methods and the hosted ``web_search``
tool work as usual.

Notes for this alpha:
- The model is reactive: it responds after the user speaks. There is no proactive
  greeting, so ``on_enter`` does not call ``generate_reply`` (it would raise).
- Barge-in is the model's own: it keeps listening while it speaks and decides when to yield,
  so the framework does not cut playback when you start talking.
- The Agent's ``instructions`` are the voice persona and are immutable once the
  session starts; the backend reasoning model is configured via ``backend_instructions``.
- Audio the model never transcribes — a backchannel, a laugh — still plays; it simply
  produces no chat item.

Run it in the terminal (needs OPENAI_API_KEY and alpha access):

    python examples/voice_agents/openai_gpt_live.py console
"""

import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
)
from livekit.agents.llm import function_tool
from livekit.plugins.openai.realtime import GPTLiveModel

logger = logging.getLogger("gpt-live-agent")

load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            # voice persona — the immutable top-level GPT-Live instructions
            instructions=(
                "You are a helpful voice assistant. Keep replies short and conversational. "
                "Do not use emojis, asterisks, or other special characters. "
                "Ask before taking any external action."
            ),
        )

    @function_tool
    async def lookup_weather(self, context: RunContext, location: str) -> str:
        """Look up the current weather for a location.

        Args:
            location: The city or region to look up.
        """
        logger.info("looking up weather for %s", location)
        # the backend Responses model calls this tool; the framework runs it here and
        # returns the result via delegation.function_call_output.create
        return f"The weather in {location} is 62 degrees and partly cloudy."


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        llm=GPTLiveModel(
            voice="marin",
            # backend Responses model that handles reasoning and tools
            backend_model="gpt-5.5",
            backend_instructions="Use tools when current information is required.",
            # hosted server-side tool, in addition to the @function_tool above
            web_search=True,
        ),
    )

    await session.start(agent=Assistant(), room=ctx.room)

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
