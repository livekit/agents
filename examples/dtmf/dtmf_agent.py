import logging
import os

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomOutputOptions,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.beta.workflows.dtmf_inputs import (
    GetDtmfTask,
)
from livekit.agents.llm.tool_context import function_tool
from livekit.agents.voice.events import RunContext
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("dtmf-agent")

load_dotenv()


DTMF_AGENT_DISPATCH_NAME = os.getenv("DTMF_AGENT_DISPATCH_NAME", "my-telephony-agent")


class DtmfAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=("You are a voice assistant. First ask user to provide a phone number."),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="Ask user to provide a phone number")

    @function_tool
    async def ask_for_phone_number(self, context: RunContext) -> str:
        """Ask user to provide a phone number."""
        self.session.generate_reply(instructions="Ask user to provide a 10-digit phone number")
        await context.wait_for_playout()

        while True:
            result = await GetDtmfTask(
                name="phone number",
                num_digits=10,
                chat_ctx=self.chat_ctx.copy(exclude_instructions=True, exclude_function_call=True),
                ask_for_confirmation=True,
            )

            if result is not None:
                break

            self.session.generate_reply(
                instructions="Tell user the provided phone number is invalid. Please try again."
            )
            await context.wait_for_playout()

        return f"User's phone number is {result}"


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session: AgentSession = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm="azure/gpt-4.1-mini",
        stt="deepgram/nova-3",
        tts="cartesia/sonic-2",
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
