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
from livekit.plugins.openai.tools import WebSearch

logger = logging.getLogger("gpt-live-agent")

load_dotenv()

ORDERS = {
    "A1042": "shipped, arriving Thursday",
    "B2231": "still being packed",
}


def prior_conversation() -> ChatContext:
    """A conversation this caller had earlier, seeded as GPT-Live startup history.

    Only what is here before the session starts becomes startup ``input``; afterwards the API is
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
            # hosted tool the backend Responses model runs on its own, no client round trip
            tools=[WebSearch()],
        )

    async def on_enter(self) -> None:
        # an ask is a request the model may decline, and it declines one that contradicts the
        # seeded history, so the greeting picks up where that conversation left off
        self.session.generate_reply(
            instructions=(
                "Welcome the caller back to Acme and ask whether they are calling about the "
                "delivery of order A1042."
            )
        )

    @function_tool
    async def lookup_weather(self, context: RunContext, location: str) -> str:
        """Look up the current weather for a location.

        Args:
            location: The city or region to look up.
        """
        logger.info("looking up weather", extra={"lk.pii.location": location})
        # the backend Responses model calls this tool; the framework runs it here and the
        # plugin returns the result to the backend
        return f"The weather in {location} is 62 degrees and partly cloudy."

    @function_tool
    async def check_order_status(self, context: RunContext, order_id: str) -> str:
        """Check the delivery status of an order.

        Args:
            order_id: The order reference, such as A1042.
        """
        logger.info("checking order", extra={"lk.pii.order_id": order_id})
        status = ORDERS.get(order_id.upper())
        return f"Order {order_id} is {status}." if status else f"I cannot find order {order_id}."

    @function_tool
    async def schedule_delivery(self, context: RunContext, order_id: str, day: str) -> str:
        """Book a delivery day for an order.

        Args:
            order_id: The order reference, such as A1042.
            day: The requested day, such as Tuesday.
        """
        logger.info("scheduling delivery", extra={"lk.pii.order_id": order_id, "lk.pii.day": day})
        if order_id.upper() not in ORDERS:
            return f"I cannot find order {order_id}, so I did not schedule anything."
        return f"Delivery for order {order_id} is booked for {day}."


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"lk.pii.room": ctx.room.name}

    session = AgentSession(
        llm=GPTLiveModel(
            voice="marin",
            # backend Responses model that handles reasoning and tools
            responses_options={
                "model": "gpt-5.6-luna",
                "instructions": "Use tools when current information is required.",
            },
        ),
    )

    async def log_usage() -> None:
        # gpt-live reports usage about once a minute, and the last of it only on close
        logger.info("usage: %s", session.usage)

    # shutdown callbacks are triggered when the session is over
    ctx.add_shutdown_callback(log_usage)

    await session.start(agent=Assistant(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
