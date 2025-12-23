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
    cli,
    metrics,
    room_io,
    stt,
)
from livekit.agents.llm import function_tool
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import noise_cancellation

from interrupt_gate import InterruptionGate
from config import TTS_DRAIN_SECONDS

logger = logging.getLogger("basic-agent")
load_dotenv()

# -------------------------------------------------
# INTELLIGENT AGENT
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

        self._is_speaking = False
        self._gate = InterruptionGate()

    async def on_enter(self):

    # ---------------------------
    # TTS lifecycle tracking
    # ---------------------------
    def tts_node(self, text: AsyncIterable[str], model_settings):
        async def _wrapped():
            stream = super().tts_node(text, model_settings)
            try:
                async for frame in stream:
                    self._is_speaking = True
                    yield frame
            finally:
                await asyncio.sleep(TTS_DRAIN_SECONDS)
                self._is_speaking = False

        return _wrapped()

    # ---------------------------
    # STT interception
    # ---------------------------
    def stt_node(self, audio: AsyncIterable[rtc.AudioFrame], model_settings):
        async def _logic():
            upstream = super().stt_node(audio, model_settings)

            async for event in upstream:

                # VAD start → advisory only
                if event.type == stt.SpeechEventType.START_OF_SPEECH:
                    if self._is_speaking:
                        continue
                    yield event
                    continue

                # Transcript events
                if event.type in (
                    stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    stt.SpeechEventType.FINAL_TRANSCRIPT,
                ):
                    text = event.alternatives[0].text
                    decision = self._gate.classify(text, self._is_speaking)

                    if decision == "IGNORE":
                        continue

                    if decision == "INTERRUPT":
                        if self._activity:
                            await self._activity.interrupt()
                        yield event
                        continue

                    yield event
                    continue

                yield event

        return _logic()

    # ---------------------------
    # Example tool (unchanged)
    # ---------------------------
    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ):
        logger.info(f"Looking up weather for {location}")
        return "sunny with a temperature of 70 degrees."

# -------------------------------------------------
# SERVER BOOTSTRAP 
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
        turn_detection=MultilingualModel(unlikely_threshold=0.3),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,
        resume_false_interruption=True,
        false_interruption_timeout=0.4,
    )

    # ---- metrics (kept exactly) ----
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

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
