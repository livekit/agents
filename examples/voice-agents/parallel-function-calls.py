import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import AgentTask, RunContext, VoiceAgent
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import cartesia, deepgram, openai

logger = logging.getLogger("parallel-functions")
logger.setLevel(logging.INFO)

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a voice agent. Call the get_weather_today and get_weather_tomorrow functions when user asks for the weather."  # noqa: E501
                "Tell the user when you are calling the functions."
            ),
        )

    @function_tool()
    async def get_weather_today(self) -> str:
        """Called when user asks for the weather."""
        return "The weather is sunny today."

    @function_tool()
    async def get_weather_tomorrow(self) -> str:
        """Called when user asks for the weather."""
        return "The weather is rainy tomorrow."



async def entrypoint(ctx: JobContext):
    await ctx.connect()

    agent = AgentSession(
        agent=MyAgent(),
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
    )

    await ctx.wait_for_participant()
    await agent.start(
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
