"""Fully local voice agent — zero cloud calls.

Every stage of the voice loop runs on your own hardware:

    STT  — speaches (faster-whisper) via its OpenAI-compatible endpoint
    LLM  — Ollama via ``openai.LLM.with_ollama()``
    TTS  — kokoro-fastapi via its OpenAI-compatible endpoint
    VAD  — silero (local ONNX model)
    Turn detection — LiveKit turn-detector (local model)

Prerequisites (see README.md for setup):

    Ollama serving a chat model      http://localhost:11434
    speaches serving faster-whisper  http://localhost:8000
    kokoro-fastapi serving kokoro    http://localhost:8880

No API keys required. The OPENAI_API_KEY values passed below are
placeholders — the openai plugin requires a non-empty key, but the local
servers never check it.
"""

import logging
import os

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    cli,
    metrics,
)
from livekit.plugins import openai, silero

# NOTE: livekit.plugins.turn_detector is marked deprecated in favor of
# livekit.agents.inference.TurnDetector — but the replacement routes through
# LiveKit's inference gateway (it takes api_key/api_secret), while this plugin
# runs the turn-detection model locally. For a fully offline pipeline the
# plugin is currently the only option.
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("local-only-agent")

# Local endpoints — override via environment if your servers live elsewhere.
STT_BASE_URL = os.environ.get("LOCAL_STT_URL", "http://localhost:8000/v1")
LLM_BASE_URL = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/v1")
TTS_BASE_URL = os.environ.get("LOCAL_TTS_URL", "http://localhost:8880/v1")

STT_MODEL = os.environ.get("LOCAL_STT_MODEL", "Systran/faster-whisper-small")
# Use a non-thinking model for the voice path. Thinking models (e.g. qwen3)
# stream reasoning tokens first and content tokens seconds later — the agent
# stays silent while it deliberates. See README "Failure modes encountered".
LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "llama3.2:3b")
TTS_MODEL = os.environ.get("LOCAL_TTS_MODEL", "kokoro")
TTS_VOICE = os.environ.get("LOCAL_TTS_VOICE", "af_heart")


class LocalAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a voice assistant running entirely on local hardware. "
                "Keep responses concise and conversational — they will be spoken aloud. "
                "Do not use emojis, asterisks, markdown, or other special characters."
            ),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Greet the user briefly and mention you are running fully offline."
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session: AgentSession = AgentSession(
        stt=openai.STT(
            model=STT_MODEL,
            base_url=STT_BASE_URL,
            api_key="not-needed",
        ),
        llm=openai.LLM.with_ollama(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
        ),
        tts=openai.TTS(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            base_url=TTS_BASE_URL,
            api_key="not-needed",
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    # Log per-stage metrics — with a fully local stack these numbers are
    # your hardware's honest answer, not a provider's SLA.
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)

    await session.start(agent=LocalAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
