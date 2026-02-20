import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    cli,
    inference,
    metrics,
)
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("basic-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are reaching out to a customer with a phone call. "
                "You are calling to see if they are home. "
                "You might encounter an answering machine with a DTMF menu or IVR system. "
                "If you do, you will try to leave a message to ask them to call back."
            ),
        )

    async def on_enter(self):
        result = await self.session.amd_detection_result()
        if result.is_human:
            logger.info("human answered the call, proceeding with normal conversation")
            return

        async with self.session.disable_preemptive_generation():
            if result.category == "machine-dtmf":
                logger.info("dtmf menu detected, starting IVR detection")
                await self.session.start_ivr_detection(transcript=result.transcript)
                return

            if result.category == "machine-vm":
                logger.info("voicemail detected, leaving a message")
                speech_handle = self.session.generate_reply(
                    instructions=(
                        "You've reached voicemail. Leave a brief message asking "
                        "the customer to call back."
                    ),
                )
                await speech_handle.wait_for_playout()
            else:
                logger.info("mailbox unavailable, ending call")
            self.session.shutdown()


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
        amd="openai/gpt-5-mini",
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
