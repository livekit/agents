"""Post-call GTM/CRM telemetry example.

Demonstrates how to wire a PostCallTelemetryCollector into an AgentSession so that
a structured PostCallReport is sent to a webhook (and printed as Salesforce/HubSpot
payloads) at the end of every call.

The flush hook uses @server.rtc_session(on_session_end=...) which runs after
session.aclose() with a 300s default budget (WorkerOptions.session_end_timeout),
comfortably covering the dispatcher's ~31.5s retry worst case. JobContext.add_shutdown_callback
is NOT used: those callbacks run under the ~10s shutdown_process_timeout, which cannot cover
the full retry budget.
"""

import logging
import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, RunContext, cli, inference
from livekit.agents.beta.gtm_telemetry import (
    PostCallTelemetryCollector,
    WebhookDispatcher,
)
from livekit.agents.llm import function_tool

logger = logging.getLogger("gtm-telemetry-agent")

load_dotenv()

# Module-level registry keyed by job ID so the on_session_end hook can find
# the collector for the closing session. Using job ID (not room name) avoids
# collisions when two overlapping jobs share the same room under threaded execution.
_collectors: dict[str, PostCallTelemetryCollector] = {}


class SalesAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a sales assistant for Acme Corp. "
                "Help the caller with product information and qualification."
            ),
        )

    @function_tool
    async def lookup_salesforce_contact(self, context: RunContext, email: str) -> str:
        """Look up a Salesforce contact record by email address."""
        # Mock implementation — in production, call the Salesforce REST API
        logger.info("Looking up contact: %s", email)
        return (
            f'{{"Name": "Jane Smith", "Email": "{email}", '
            f'"Account": "Acme Corp", "Status": "Active"}}'
        )


async def on_session_end(ctx: JobContext) -> None:
    """Flush the collector and print CRM adapter outputs."""
    collector = _collectors.pop(ctx.job.id, None)
    if collector is None:
        return

    try:
        await collector.aflush()
        report = collector.generate_report()
        logger.info("Generated PostCallReport with %d turns", len(report.turns))

        # NOTE: To prevent PII/transcript leakage into application log aggregation
        # systems, avoid logging the full report or adapter payloads in production.
        # If you need to inspect payloads during local development, uncomment below:
        # print(report.model_dump_json(indent=2))
    finally:
        await collector.aclose()


server = AgentServer()


@server.rtc_session(on_session_end=on_session_end)
async def gtm_agent(ctx: JobContext) -> None:
    await ctx.connect()

    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
    )

    # Build the optional webhook dispatcher from environment
    webhook_url = os.environ.get("POST_CALL_WEBHOOK_URL")
    dispatcher: WebhookDispatcher | None = None
    if webhook_url:
        dispatcher = WebhookDispatcher(
            webhook_url,
            webhook_secret=os.environ.get("POST_CALL_WEBHOOK_SECRET"),
        )

    collector = PostCallTelemetryCollector(
        session,
        room_name=ctx.room.name,
        metadata={"campaign": "demo"},
        dispatcher=dispatcher,
    )
    collector.attach()

    # Store so on_session_end can find it (keyed by job ID for thread safety)
    _collectors[ctx.job.id] = collector

    await session.start(agent=SalesAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
