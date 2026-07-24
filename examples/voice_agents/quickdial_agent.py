import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import openai, quickdial, silero

# Quickdial provides cheap, fast, real-time STT + TTS on CPU (priced per character).
# Set QUICKDIAL_API_KEY in your environment (get a key at https://web.quickdial.ai).

logger = logging.getLogger("quickdial-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly voice assistant powered by Quickdial. "
                "Keep replies short, natural, and conversational. "
                "Do not use emojis, asterisks, or markdown."
            )
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="Warmly greet the user and ask how you can help.")


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=quickdial.STT(language="en"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=quickdial.TTS(voice="alba"),
        vad=silero.VAD.load(),
    )

    await session.start(agent=MyAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
