import logging

from dotenv import load_dotenv

from livekit.agents import JobContext, JobProcess, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, silero

logger = logging.getLogger("vad-realtime-example")
logger.setLevel(logging.INFO)

load_dotenv()


class AlloyAgent(Agent):
    def __init__(self, vad: silero.VAD | None = None) -> None:
        super().__init__(
            instructions="You are Alloy.",
            llm=openai.realtime.RealtimeModel(voice="alloy", turn_detection=None),
            vad=vad,
        )

    async def on_enter(self):
        self.session.generate_reply()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        turn_detection="vad",  # or not set
        allow_interruptions=True,
    )
    await session.start(
        agent=AlloyAgent(vad=ctx.proc.userdata["vad"]),
        room=ctx.room,
    )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
