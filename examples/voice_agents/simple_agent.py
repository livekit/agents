import logging
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
)
from livekit.plugins import silero

logger = logging.getLogger("simple-agent")
load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice AI assistant. "
            "Keep your responses concise and to the point."
        )

    async def on_enter(self):
        self.session.generate_reply()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4o-mini",
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=ctx.proc.userdata["vad"],
        # Filler detection configuration (built into AgentSession)
        filler_languages=['en', 'hi'],  # Multi-language support
        filler_min_confidence=0.3,
        filler_enable_logging=True,
    )

    await session.start(agent=MyAgent(), room=ctx.room)
    
    # Log statistics when session closes
    @session.on("close")
    def on_close(ev):
        if session._filler_detector:
            stats = session._filler_detector.get_stats()
            logger.info(f"Filler Detection Statistics: {stats}")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

__all__ = ["MyAgent", "prewarm", "entrypoint"]
