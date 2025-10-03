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
from livekit.agents.llm.tool_context import ToolError, function_tool
from livekit.agents.voice.events import RunContext
from livekit.plugins import deepgram, elevenlabs, openai, silero
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
        while True:
            try:
                result = await GetDtmfTask(
                    num_digits=10,
                    chat_ctx=self.chat_ctx.copy(
                        exclude_instructions=True, exclude_function_call=True
                    ),
                    ask_for_confirmation=True,
                    extra_instructions="Ask user to provide a phone number in the format of 123 456 7890",
                )
            except ToolError as e:
                self.session.generate_reply(instructions=e.message, allow_interruptions=False)
                await context.wait_for_playout()
                continue

            break

        return f"User's phone number is {result.user_input}"


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session: AgentSession = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=openai.LLM(model="gpt-4.1-mini"),
        stt=deepgram.STT(model="nova-3"),
        tts=elevenlabs.TTS(model="eleven_multilingual_v2"),
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
