import logging
import os

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
    inference,
    room_io,
)
from livekit.agents.beta.gtm_telemetry import (
    PostCallTelemetryCollector,
    PostCallWebhookDispatcher,
    WebhookConfig,
    WebhookDeliveryError,
)
from livekit.agents.llm import function_tool

logger = logging.getLogger("gtm-telemetry-agent")

load_dotenv()

# Beta feature: livekit.agents.beta.gtm_telemetry is opt-in and not covered by semver
# stability guarantees. Nothing here runs unless you construct and attach a collector
# yourself. collector.finalize() also works without a JobContext (e.g. `console` mode
# or a bare script) — job_id/room_id/room_name/participant_identity are simply None.
#
# Transcripts and tool arguments/results may contain personal or confidential
# information — review CollectorConfig before enabling this in production, and only
# point the webhook at a destination you trust.


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your name is Kelly, a sales assistant built by LiveKit. "
            "Keep responses concise and to the point. You will speak english to the "
            "user over voice."
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="greet the user and introduce yourself")

    @function_tool
    async def lookup_pricing(self, context: RunContext, plan: str) -> str:
        """Called when the user asks about pricing for a plan.

        Args:
            plan: The plan name the user is asking about.
        """
        return f"the {plan} plan is $49/month"


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session: AgentSession = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
    )

    collector = PostCallTelemetryCollector(
        metadata={"campaign": "inbound-demo"},
    )
    collector.attach(session, job_ctx=ctx)

    async def deliver_post_call_report() -> None:
        report = collector.finalize()

        webhook_url = os.environ.get("POST_CALL_WEBHOOK_URL")
        webhook_secret = os.environ.get("POST_CALL_WEBHOOK_SECRET")
        if not webhook_url or not webhook_secret:
            logger.info("post-call webhook not configured, skipping delivery")
            return

        dispatcher = PostCallWebhookDispatcher(
            WebhookConfig(url=webhook_url, secret=webhook_secret)
        )
        try:
            await dispatcher.send(report)
        except WebhookDeliveryError:
            logger.exception("failed to deliver post-call report")
        finally:
            await dispatcher.aclose()

    # shutdown callbacks are triggered when the session is over
    ctx.add_shutdown_callback(deliver_post_call_report)

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(),
    )


if __name__ == "__main__":
    cli.run_app(server)
