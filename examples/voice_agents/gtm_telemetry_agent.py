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
    to_hubspot_engagement,
    to_salesforce_task,
)
from livekit.agents.llm import function_tool

logger = logging.getLogger("gtm-telemetry-agent")

load_dotenv()

# Module-level registry keyed by room name so the on_session_end hook can find
# the collector for the closing session.
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
    collector = _collectors.pop(ctx.room.name, None)
    if collector is None:
        return

    try:
        await collector.aflush()
        report = collector.generate_report()
        logger.info("PostCallReport:\n%s", report.model_dump_json(indent=2))
        logger.info("Salesforce Task payload: %s", to_salesforce_task(report))
        logger.info("HubSpot Engagement payload: %s", to_hubspot_engagement(report))
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

    # Store so on_session_end can find it
    _collectors[ctx.room.name] = collector

    await session.start(agent=SalesAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
