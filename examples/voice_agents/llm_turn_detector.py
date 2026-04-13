"""Voice agent using LLMTurnDetector for end-of-turn classification.

Demonstrates plugging ``LLMTurnDetector`` into an ``AgentSession`` so a small
LLM decides when the user has finished speaking, instead of relying solely on
VAD silence timeouts.

Run with:
    OPENAI_API_KEY=... DEEPGRAM_API_KEY=... \
        python examples/voice_agents/llm_turn_detector.py dev
"""

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
)
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector import LLMTurnDetector

load_dotenv()

server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(),
        # A cheap model is usually plenty for turn-completion classification.
        turn_detection=LLMTurnDetector(llm=openai.LLM(model="gpt-4o-mini")),
    )

    await session.start(
        agent=Agent(instructions="You are a friendly assistant. Keep answers short."),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
