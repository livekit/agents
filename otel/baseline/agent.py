"""A 1.8.10 agent that exercises the telemetry surface: tools, a handoff, interruptions,
STT/LLM/TTS through LiveKit inference, VAD, the turn detector and on_session_end."""

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
    function_tool,
)
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

ORDERS = {"1234": "shipped yesterday, arriving Friday", "5678": "still being packed"}


class Billing(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are the billing specialist of an online shop, on a phone call. Keep answers "
                "short. You can refund an order with the refund tool once the caller gives the "
                "order number and confirms."
            )
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Introduce yourself as billing and ask how to help."
        )

    @function_tool
    async def refund(self, ctx: RunContext, order_id: str) -> str:
        """Refund an order.

        Args:
            order_id: the order number
        """
        return f"order {order_id} refunded, the money arrives in 3 to 5 days"


class FrontDesk(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are the front desk of an online shop, on a phone call. Keep answers short. "
                "Look up order status with lookup_order. For refunds or charges, transfer to "
                "billing with transfer_to_billing."
            )
        )

    @function_tool
    async def lookup_order(self, ctx: RunContext, order_id: str) -> str:
        """Look up the status of an order.

        Args:
            order_id: the order number
        """
        return ORDERS.get(order_id, "no order with that number")

    @function_tool
    async def transfer_to_billing(self, ctx: RunContext) -> tuple[Agent, str]:
        """Transfer the caller to billing for refunds and charges."""
        return Billing(), "Transferring you to billing."


server = AgentServer()


async def on_session_end(ctx: JobContext) -> None:
    report = ctx.make_session_report()
    print("session report items:", len(report.chat_history.items))


@server.rtc_session(on_session_end=on_session_end)
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        stt="deepgram/nova-3",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )
    await session.start(agent=FrontDesk(), room=ctx.room)
    await session.generate_reply(instructions="Greet the caller and ask how you can help.")


if __name__ == "__main__":
    cli.run_app(server)
