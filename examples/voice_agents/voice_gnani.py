"""Gnani Vachana STT + TTS smoke voice agent for LiveKit.

Pipeline (streaming end-to-end):
  room audio → Gnani STT WebSocket (wss://.../stt/v3/stream)
  → Groq LLM → Gnani TTS WebSocket (wss://.../api/v1/tts) → room audio

Defaults to timbre-v2.5 (voice: Nalini, language: en-IN, sample_rate: 16000).
Override via GNANI_STT_SAMPLE_RATE, GNANI_STT_STREAMING (true | false; default true for
WebSocket, false for REST), GNANI_TTS_MODEL, GNANI_TTS_VOICE, GNANI_TTS_LANGUAGE, GNANI_TTS_SAMPLE_RATE
(8000 | 16000 | 22050 | 44100), and GNANI_TTS_SYNTHESIZE_METHOD (rest | sse | websocket).

Requires GNANI_API_KEY, GROQ_API_KEY, and LIVEKIT_* vars in examples/.env
for dev/start modes. Console mode only needs GNANI_API_KEY and GROQ_API_KEY.

Dev mode uses explicit agent dispatch (GNANI_AGENT_NAME, default ``gnani-voice``).
Tokens and the playground must request that agent name.

Run:
  uv run examples/voice_agents/voice_gnani.py console
  uv run examples/voice_agents/voice_gnani.py dev
  ./scripts/smoke-gnani/run-bot.sh dev
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    TurnHandlingOptions,
    cli,
    metrics,
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.plugins import gnani, groq, silero

logger = logging.getLogger("voice-gnani")

_env_file = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_file)

GNANI_STT_LANGUAGE = os.environ.get("GNANI_STT_LANGUAGE", "en-IN")
# Gnani STT WebSocket stream supports 8000, 16000, 44100, 48000.
GNANI_STT_SAMPLE_RATE = int(os.environ.get("GNANI_STT_SAMPLE_RATE", "16000"))
GNANI_STT_STREAMING = os.environ.get("GNANI_STT_STREAMING", "true").lower() in ("true", "1", "yes")
# timbre-v2.5: 42 voices; timbre-v2.0: Pranav, Kaveri, Shubhra, Deepak
GNANI_TTS_VOICE = os.environ.get("GNANI_TTS_VOICE", "Nalini")
GNANI_TTS_MODEL = os.environ.get("GNANI_TTS_MODEL", "timbre-v2.5")
GNANI_TTS_LANGUAGE = os.environ.get("GNANI_TTS_LANGUAGE", "en-IN")
# Supported: 8000, 16000, 22050, 44100 — room output is matched automatically.
GNANI_TTS_SAMPLE_RATE = int(os.environ.get("GNANI_TTS_SAMPLE_RATE", "16000"))
# AgentSession uses TTS.stream() (WebSocket). synthesize_method applies to synthesize().
GNANI_TTS_SYNTHESIZE_METHOD = os.environ.get("GNANI_TTS_SYNTHESIZE_METHOD", "websocket")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
# Explicit dispatch name — must match create-token.sh --agent and playground agent picker.
GNANI_AGENT_NAME = os.environ.get("GNANI_AGENT_NAME", "gnani-voice")


class GnaniVoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant powered by Gnani Vachana speech AI. "
                "You interact with users via voice, so keep your responses concise and natural. "
                "Do not use emojis, asterisks, markdown, or other special characters. "
                "You can help with general questions, weather lookups, and casual conversation."
            ),
        )

    async def on_enter(self) -> None:
        # on_enter runs in parallel with job_ctx.connect(); wait_for_ready() blocks
        # until the room is connected, a participant is linked, and audio I/O is up.
        try:
            await asyncio.wait_for(self.session.room_io.wait_for_ready(), timeout=30.0)
        except TimeoutError:
            logger.warning("Room not ready within 30s; skipping greeting")
            return

        # Brief settle after subscription before first TTS (AEC warmup is also active).
        await asyncio.sleep(0.3)

        self.session.generate_reply(
            instructions="Greet the user and introduce yourself as a Gnani-powered voice assistant.",
            allow_interruptions=False,
        )

    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ) -> str:
        """Called when the user asks for weather related information.

        Args:
            location: The location they are asking for
            latitude: The latitude of the location
            longitude: The longitude of the location
        """
        logger.info("Looking up weather for %s", location)
        return "sunny with a temperature of 30 degrees celsius."


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name=GNANI_AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    stt_kwargs: dict[str, str | int | bool] = {
        "language": GNANI_STT_LANGUAGE,
        "sample_rate": GNANI_STT_SAMPLE_RATE,
        "streaming": GNANI_STT_STREAMING,
    }

    tts_kwargs: dict[str, str | int] = {
        "voice": GNANI_TTS_VOICE,
        "model": GNANI_TTS_MODEL,
        "language": GNANI_TTS_LANGUAGE,
        "sample_rate": GNANI_TTS_SAMPLE_RATE,
        "synthesize_method": GNANI_TTS_SYNTHESIZE_METHOD,
    }

    logger.info(
        "Gnani TTS: voice=%s model=%s language=%s sample_rate=%s method=%s",
        GNANI_TTS_VOICE,
        GNANI_TTS_MODEL,
        GNANI_TTS_LANGUAGE,
        GNANI_TTS_SAMPLE_RATE,
        GNANI_TTS_SYNTHESIZE_METHOD,
    )

    session: AgentSession = AgentSession(
        stt=gnani.STT(**stt_kwargs),
        llm=groq.LLM(model=GROQ_MODEL),
        tts=gnani.TTS(**tts_kwargs),
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            interruption={
                "resume_false_interruption": True,
                # Longer timeout reduces false cuts when mic picks up agent audio.
                "false_interruption_timeout": 2.0,
            },
        ),
        aec_warmup_duration=5.0,
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)

    async def log_usage() -> None:
        logger.info("Usage: %s", session.usage)

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=GnaniVoiceAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            # Match room capture rate to Gnani STT WebSocket stream.
            audio_input=room_io.AudioInputOptions(
                sample_rate=GNANI_STT_SAMPLE_RATE,
                # Disable AGC — browser AGC + agent playback echo can pump volume.
                auto_gain_control=False,
            ),
            # Match room playback rate to Gnani TTS output to avoid resampling artifacts.
            audio_output=room_io.AudioOutputOptions(sample_rate=GNANI_TTS_SAMPLE_RATE),
            # Bypass TranscriptSynchronizer on the audio path — audio goes straight to
            # RoomIO playback. lk_pipeline WAVs are good; sync barrier was stalling live output.
            text_output=room_io.TextOutputOptions(sync_transcription=False),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
