"""Prism room agent. Install livekit-plugins-rime[realtime] and set RIME_API_KEY."""

import os

from dotenv import load_dotenv
from rimelabs_sdk.realtime import PrismFiller, TurnDetection

from livekit.agents import Agent, AgentServer, JobContext, cli, function_tool
from livekit.plugins import rime

load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Give short, clear answers. Use lookup_order for order status."
        )

    @function_tool
    async def lookup_order(self, order_id: str) -> str:
        """Look up an order's delivery status."""
        return f"Order {order_id} has shipped."

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="Greet the caller.")


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    model = rime.realtime.RealtimeModel(
        endpoint=os.getenv("PRISM_URL"),
        voice=os.getenv("PRISM_VOICE"),
        turn_detection=TurnDetection(eagerness="low", end_of_turn_timeout_ms=5000),
        response_metadata={"example": "livekit-prism"},
        # Optional on deployments that support history truncation.
        truncation="auto",
        tool_fillers={"lookup_order": PrismFiller(type="text", text="Checking that order.")},
    )
    session = rime.realtime.AgentSession(llm=model)
    ctx.add_shutdown_callback(model.aclose)
    await session.start(agent=Assistant(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
