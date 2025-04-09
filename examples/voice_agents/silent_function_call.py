import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, openai, silero

logger = logging.getLogger("silent-function-call")
logger.setLevel(logging.INFO)

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a voice agent. Call the turn_on_light function when user asks to turn on the light."  # noqa: E501
            ),
        )
        self.light_on = False

    @function_tool()
    async def turn_on_light(self) -> str:
        """Called when user asks to turn on the light."""
        self.light_on = True
        logger.info("Light is now on")

    @function_tool()
    async def turn_off_light(self) -> str:
        """Called when user asks to turn off the light."""
        self.light_on = False
        logger.info("Light is now off")


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    agent = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        # llm=openai.realtime.RealtimeModel(voice="alloy"),
    )

    await ctx.wait_for_participant()
    await agent.start(agent=MyAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
