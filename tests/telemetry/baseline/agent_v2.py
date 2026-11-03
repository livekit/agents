"""The Agents 2.0 twin of agent.py: the same agents, instructions, tools, handoff and models,
so a capture of it diffs against schema_1_8_10.json."""

from livekit.agents import Agent, AgentServer, JobContext, RunContext, VoiceSession, cli, inference
from livekit.plugins import silero

ORDERS = {"1234": "shipped yesterday, arriving Friday", "5678": "still being packed"}

billing = Agent(
    name="billing",
    instructions=(
        "You are the billing specialist of an online shop, on a phone call. Keep answers "
        "short. You can refund an order with the refund tool once the caller gives the "
        "order number and confirms."
    ),
)


@billing.on_enter
async def introduce(session: VoiceSession) -> None:
    session.generate_reply(instructions="Introduce yourself as billing and ask how to help.")


@billing.tool
async def refund(ctx: RunContext, order_id: str) -> str:
    """Refund an order.

    Args:
        order_id: the order number
    """
    return f"order {order_id} refunded, the money arrives in 3 to 5 days"


front_desk = Agent(
    name="front_desk",
    instructions=(
        "You are the front desk of an online shop, on a phone call. Keep answers short. "
        "Look up order status with lookup_order. For refunds or charges, transfer to "
        "billing with transfer_to_billing."
    ),
)


@front_desk.tool
async def lookup_order(ctx: RunContext, order_id: str) -> str:
    """Look up the status of an order.

    Args:
        order_id: the order number
    """
    return ORDERS.get(order_id, "no order with that number")


@front_desk.tool
async def transfer_to_billing(ctx: RunContext) -> str:
    """Transfer the caller to billing for refunds and charges."""
    assert ctx.live_session is not None
    ctx.live_session.update_agent(billing)
    return "Transferring you to billing."


server = AgentServer()


async def on_session_end(ctx: JobContext) -> None:
    report = ctx.make_session_report()
    print("session report items:", len(report.chat_history))


@server.rtc_session(on_session_end=on_session_end)
async def entrypoint(ctx: JobContext) -> None:
    session = VoiceSession(
        stt="deepgram/nova-3",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2",
        vad=silero.VAD.load(),
        turn_detection=inference.TurnDetector(),
    )
    await session.start(agent=front_desk, room=ctx.room)
    await session.generate_reply(instructions="Greet the caller and ask how you can help.")


if __name__ == "__main__":
    cli.run_app(server)
