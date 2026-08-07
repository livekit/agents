"""OpenAI GPT-Live (alpha) full-duplex voice agent.

GPT-Live is a server-driven, full-duplex voice model: it listens and speaks at the
same time and decides for itself when to reply. Reasoning and tools are delegated to a
backend Responses model (``gpt-5.6-sol``), so ordinary ``@function_tool`` methods and
the hosted ``web_search`` tool work as usual.

Notes for this alpha:
- ``generate_reply`` asks rather than tells: it places a spoken instruction in the
  model's context, which is how ``on_enter`` greets first. The model may decline, and
  the call then fails rather than waiting forever.
- Barge-in is the model's own: it keeps listening while it speaks and decides when to yield,
  so the framework does not cut playback when you start talking.
- A ``chat_ctx`` passed before the session starts seeds the conversation as history;
  afterwards the API is append-only, so nothing can be edited or removed. A reconnect
  reseeds the whole conversation, so the model picks up where the dropped one stopped.
- ``delegation="client"`` hands work to the application instead of a backend model: it
  arrives as a ``GPTLiveDelegation`` on the session's ``delegation_created`` event and is
  answered with ``send_delegation_context``. There is no tool channel in that mode, so
  ``@function_tool`` is ignored. ``update_delegation`` switches between the two mid-session.
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
    ChatContext,
    JobContext,
    RunContext,
    cli,
)
from livekit.agents.llm import function_tool
from livekit.plugins.openai.realtime import GPTLiveModel

logger = logging.getLogger("gpt-live-agent")

load_dotenv()

ORDERS = {
    "A1042": "shipped, arriving Thursday",
    "B2231": "still being packed",
}


def prior_conversation() -> ChatContext:
    """A conversation this caller had earlier, seeded as GPT-Live startup history.

    Only what is here before the session starts becomes ``initial_items``; afterwards the API is
    append-only, so nothing can be edited or removed. The service accepts at most 128 messages and
    8192 rendered tokens, oldest dropped first.
    """
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="Hi, I ordered a standing desk last week.")
    chat_ctx.add_message(
        role="assistant", content="Thanks for calling. I have your order A1042 on file."
    )
    chat_ctx.add_message(role="user", content="I had to run, I will call back about delivery.")
    return chat_ctx


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            # voice persona — the immutable top-level GPT-Live instructions
            instructions=(
                "You are a helpful voice assistant for an online furniture store. "
                "Keep replies short and conversational. "
                "Do not use emojis, asterisks, or other special characters. "
                "Ask before taking any external action."
            ),
            chat_ctx=prior_conversation(),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "greet the caller by picking up where the earlier conversation left off, "
                "and ask how you can help."
            )
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

    @function_tool
    async def check_order_status(self, context: RunContext, order_id: str) -> str:
        """Check the delivery status of an order.

        Args:
            order_id: The order reference, such as A1042.
        """
        logger.info("checking order %s", order_id)
        status = ORDERS.get(order_id.upper())
        return f"Order {order_id} is {status}." if status else f"I cannot find order {order_id}."

    @function_tool
    async def schedule_delivery(self, context: RunContext, order_id: str, day: str) -> str:
        """Book a delivery day for an order.

        Args:
            order_id: The order reference, such as A1042.
            day: The requested day, such as Tuesday.
        """
        logger.info("scheduling %s for %s", order_id, day)
        if order_id.upper() not in ORDERS:
            return f"I cannot find order {order_id}, so I did not schedule anything."
        return f"Delivery for order {order_id} is booked for {day}."


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        llm=GPTLiveModel(
            voice="marin",
            # backend Responses model that handles reasoning and tools
            backend_model="gpt-5.6-sol",
            backend_instructions="Use tools when current information is required.",
            # hosted server-side tool, in addition to the @function_tool above
            web_search=True,
        ),
        turn_handling={"interruption": {"enabled": False}},
    )

    await session.start(agent=Assistant(), room=ctx.room)

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
