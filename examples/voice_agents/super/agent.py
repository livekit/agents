# agent.py

import asyncio
import logging
from typing import AsyncIterable

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    UserInputTranscribedEvent,
    AgentStateChangedEvent,
    cli,
    metrics,
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import noise_cancellation

from interrupt_gate import InterruptionGate

logger = logging.getLogger("basic-agent")
load_dotenv()

# -------------------------------------------------
# AGENT
# -------------------------------------------------

class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Sahbaz. You interact with users via voice. "
                "Be concise, friendly, and helpful. "
                "Do not use emojis or special characters. "
                "Start by greeting the user."
            ),
            allow_interruptions=True,
        )

        self._gate = InterruptionGate()

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ):
        logger.info(f"Looking up weather for {location}")
        return "sunny with a temperature of 70 degrees."


# -------------------------------------------------
# SERVER
# -------------------------------------------------

server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt="deepgram/nova-2-phonecall",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],

        # ⚠️ IMPORTANT: these eliminate lag
        preemptive_generation=True,
        allow_interruptions=True,
        discard_audio_if_uninterruptible=True,
        min_interruption_duration=0.6,
        min_interruption_words=2,
    )

    # -------------------
    # METRICS (unchanged)
    # -------------------

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    # -------------------------------------------------
    # 🔥 FAST INTERRUPTION LAYER (NEW)
    # -------------------------------------------------

    agent_is_speaking = {"value": False}
    gate = InterruptionGate()

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        agent_is_speaking["value"] = ev.new_state == "speaking"

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev: UserInputTranscribedEvent):
        text = (ev.transcript or "").strip()
        if not text or not ev.is_final:
            return

        if not agent_is_speaking["value"]:
            return

        decision = gate.classify(text,True)

        if decision == "IGNORE":
            # instant ignore, zero lag
            session.clear_user_turn()
            return

        if decision == "INTERRUPT":
            session.interrupt(force=True)
            return

    # -------------------------------------------------

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
