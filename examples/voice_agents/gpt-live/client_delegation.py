import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ClientDelegation,
    JobContext,
    cli,
    inference,
)
from livekit.agents.llm import function_tool
from livekit.plugins.openai.realtime import GPTLiveModel

logger = logging.getLogger("gpt-live-client-delegation")
load_dotenv()

ORDERS = {
    "A1042": "shipped, arriving Thursday",
    "B2231": "still being packed",
}


# These tools run through the SDK executor and belong to the backend.
@function_tool
async def check_order_status(order_id: str) -> str:
    """Check the delivery status of an order.

    Args:
        order_id: The order reference, such as A1042.
    """
    logger.info("desk tool: checking order", extra={"lk.pii.order_id": order_id})
    status = ORDERS.get(order_id.upper())
    return f"Order {order_id} is {status}." if status else f"There is no order {order_id}."


@function_tool
async def lookup_weather(location: str) -> str:
    """Look up the current weather for a location.

    Args:
        location: The city or region to look up.
    """
    logger.info("desk tool: looking up weather", extra={"lk.pii.location": location})
    return f"The weather in {location} is 62 degrees and partly cloudy."


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant for an online furniture store. "
                "Delegate order and weather requests, including corrections immediately. "
                "Keep replies short and conversational."
            )
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="Greet the caller and ask how you can help.")


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"lk.pii.room": ctx.room.name}
    model = inference.LLM("openai/gpt-5.5")
    ctx.add_shutdown_callback(model.aclose)
    backend = ClientDelegation(
        model=model,
        tools=[check_order_status, lookup_weather],
        # One conversation task: new input updates its retained history. Applications
        # with independent operations can select distinct task IDs instead.
        select_task=lambda request: "desk",
        instructions=(
            "You resolve order and weather requests for the voice assistant. "
            "Use verified tool results and the latest corrections. Address outstanding "
            "independent requests too; do not repeat completed operations. "
            "Ask for clarification when needed. Return facts in one short sentence, "
            "under 500 UTF-8 bytes."
        ),
    )
    session: AgentSession = AgentSession(
        llm=GPTLiveModel(voice="marin", delegation="client"),
        tools=[backend],
        max_tool_steps=4,
    )
    await session.start(agent=Assistant(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
