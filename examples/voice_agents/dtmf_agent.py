import logging
import os

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    AgentTask,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomOutputOptions,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.beta.tools.dtmf import DtmfEvent, collect_dtmf_inputs, send_dtmf_events
from livekit.agents.llm.chat_context import ChatContext
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("dtmf-agent")

load_dotenv()


DTMF_AGENT_DISPATCH_NAME = os.getenv("DTMF_AGENT_DISPATCH_NAME", "my-telephony-agent")


class CollectConsent(AgentTask[list[str]]):
    def __init__(self, chat_ctx: ChatContext) -> None:
        super().__init__(
            instructions="""You are a voice assistant that collect information from the user through phone call.""",
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="""Ask user to collect the phone number, make sure to specify the format of the phone number""",
        )
        numbers = await collect_dtmf_inputs(max_digits=10, terminator_key=DtmfEvent.POUND)
        self.complete(numbers)


class DtmfAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a voice assistant that can press number on the phone to interact with the user or a IVR system."
            ),
            tools=[send_dtmf_events],
        )

    async def on_enter(self) -> None:
        numbers = await CollectConsent(chat_ctx=self.chat_ctx)
        await self.session.generate_reply(
            user_input="User has provided the phone number: " + ", ".join(numbers)
        )


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session: AgentSession = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=openai.LLM(model="gpt-5"),
        stt=deepgram.STT(model="nova-3", language="multi"),
        tts=openai.TTS(voice="ash"),
        turn_detection=MultilingualModel(),
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage() -> None:
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=DtmfAgent(),
        room=ctx.room,
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            agent_name=DTMF_AGENT_DISPATCH_NAME,
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
