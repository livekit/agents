import logging
import os

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    RoomOutputOptions,
    cli,
    metrics,
)
from livekit.agents.beta.tools.send_dtmf import send_dtmf_events
from livekit.plugins import deepgram, elevenlabs, openai, silero

logger = logging.getLogger("phone-tree-agent")

load_dotenv()


server = AgentServer()


PHONE_TREE_AGENT_DISPATCH_NAME = os.getenv("PHONE_TREE_AGENT_DISPATCH_NAME", "dtmf-agent")


class DtmfAgent(Agent):
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


@server.realtime_session(agent_name=PHONE_TREE_AGENT_DISPATCH_NAME)
async def dtmf_session(ctx: JobContext) -> None:
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session: AgentSession = AgentSession(
        vad=silero.VAD.load(),
        llm=openai.LLM(model="gpt-4.1"),
        stt=deepgram.STT(model="nova-3"),
        tts=elevenlabs.TTS(model="eleven_multilingual_v2"),
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
        agent=DtmfAgent(goal=("ask about current account balance")),
        room=ctx.room,
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(server)
