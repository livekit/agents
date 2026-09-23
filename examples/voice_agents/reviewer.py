"""Review an agent's own replies with TypeSafe (Jev).

Every reply this agent produces is judged against its own system prompt and
tool catalog. When it drifts, a correction is appended to the chat context and
the next reply pulls itself back on course. The caller hears nothing.

The review runs off the reply path, so it costs nothing in time-to-first-token.
Set TYPESAFE_API_KEY to enable it.
"""

import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    CloseEvent,
    CloseReason,
    JobContext,
    RunContext,
    cli,
    inference,
)
from livekit.agents.llm import function_tool
from livekit.plugins import typesafe

logger = logging.getLogger("reviewer-agent")

load_dotenv()


class SupportAgent(Agent):
    def __init__(self) -> None:
        # These rules are what the checks judge against, so there is no second
        # copy of the policy to keep in sync.
        super().__init__(
            instructions=(
                "You are a support agent for an online store, speaking over voice. "
                "Keep replies to one or two sentences. "
                "Always call lookup_order before saying anything about an order's status. "
                "Never quote a price, promise a refund, or give a delivery date. "
                "If the caller asks for any of those, say a human will follow up."
            ),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="greet the caller and ask how you can help")

    @function_tool
    async def lookup_order(self, context: RunContext, order_id: str) -> str:
        """Look up the current status of a customer's order.

        Args:
            order_id: The order number the caller gave you.
        """
        logger.info(f"looking up order {order_id}")
        return "shipped, in transit"


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
    )

    def on_verdict(verdict: typesafe.Verdict) -> None:
        if verdict.triggered_checks:
            logger.warning(
                f"{verdict.triggered_checks} triggered in {verdict.duration * 1000:.0f}ms"
            )
        else:
            logger.debug(f"clear in {verdict.duration * 1000:.0f}ms")

    reviewer = typesafe.Reviewer(
        # Thresholds are starting points. Tune them against recordings of your
        # own calls before trusting them.
        checks=typesafe.default_checks(unsupported_claim=0.7, severity=2.0),
        on_verdict=on_verdict,
    )
    reviewer.attach(session)

    @session.on("close")
    def _on_close(ev: CloseEvent) -> None:
        # A caller who hangs up is the cheapest outcome label available, and the
        # only one you get for free on every call. Pair it with what the checks
        # saw on the way and the thresholds above stop being guesses: compare
        # the probability distributions on calls that ended this way against
        # calls that ran to completion, and move each threshold to where it
        # would actually have separated them.
        if ev.reason is not CloseReason.PARTICIPANT_DISCONNECTED:
            return

        turns = reviewer.results
        logger.info(
            f"caller disconnected after {len(turns)} judged turns; "
            f"{sum(1 for v in turns if v.triggered_checks)} triggered"
        )
        for verdict in turns[-3:]:
            logger.info(f"  {verdict.triggered_checks or 'clear'}: {verdict.reviewed_reply!r}")

    await session.start(agent=SupportAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
