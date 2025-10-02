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
from livekit.agents.beta.tools.dtmf import send_dtmf_events
from livekit.plugins import silero

logger = logging.getLogger("phone-tree-agent")

load_dotenv()


PHONE_TREE_AGENT_DISPATCH_NAME = os.getenv("PHONE_TREE_AGENT_DISPATCH_NAME", "my-phone-tree-agent")


class PhoneTreeAgent(Agent):
    def __init__(self, goal: str) -> None:
        super().__init__(
            instructions=(
                "You are a voice assistant that can help users navigate a phone tree IVR system by pressing numbers on the phone. "
                "You have access to a tool to send a sequence of dtmf number inputs. Prefer using the tool to send number input over using your own voice. "
                f"You are connected to a automatic IVR system and your goal is {goal}. "
                "Listen to the IVR instructions and follow them carefully to navigate to the correct place to enter the account number. "
            ),
            tools=[send_dtmf_events],
        )


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session: AgentSession = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm="google/gemini-2.5-pro",
        stt="deepgram/nova-3",
        tts="elevenlabs/eleven_multilingual_v2",
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
        agent=PhoneTreeAgent(
            goal=(
                # "record your 6-digits account number to LiveKit"
                "ask about 1) current order status in sales section and 2) warranty covers in billing section"
            )
        ),
        room=ctx.room,
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            agent_name=PHONE_TREE_AGENT_DISPATCH_NAME,
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
