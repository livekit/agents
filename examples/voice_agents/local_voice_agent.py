"""Fully local voice pipeline: STT and TTS via Speaches (https://speaches.ai),
LLM via Ollama (https://ollama.com).

Start Speaches, Ollama, pull the models, then run the example:

    docker run -d --rm -p 8000:8000 ghcr.io/speaches-ai/speaches:latest-cpu

    uvx speaches-cli model download Systran/faster-whisper-small
    uvx speaches-cli model download speaches-ai/Kokoro-82M-v1.0-ONNX
    ollama pull gemma3:4b

    uv run examples/voice_agents/local_voice_agent.py console
"""

import os

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
)
from livekit.plugins import openai, silero

load_dotenv()

SPEACHES_BASE_URL = os.getenv("SPEACHES_BASE_URL", "http://localhost:8000/v1")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "Systran/faster-whisper-small")
KOKORO_MODEL = os.getenv("KOKORO_MODEL", "speaches-ai/Kokoro-82M-v1.0-ONNX")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")


class LocalVoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a local voice assistant. Keep replies short, conversational, and plain."
            ),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="Greet the user briefly")


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session: AgentSession = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=openai.STT(model=WHISPER_MODEL, base_url=SPEACHES_BASE_URL, api_key="not-needed"),
        llm=openai.LLM.with_ollama(model=OLLAMA_MODEL),
        tts=openai.TTS(
            model=KOKORO_MODEL,
            voice=KOKORO_VOICE,
            base_url=SPEACHES_BASE_URL,
            api_key="not-needed",
            response_format="wav",
            stream_format="audio",
        ),
    )

    await session.start(agent=LocalVoiceAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
