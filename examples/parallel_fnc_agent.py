import logging

from dotenv import load_dotenv

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import ai_function
from livekit.agents.voice import AgentTask, RunContext, VoiceAgent
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import cartesia, deepgram, openai

# from livekit.plugins import noise_cancellation

logger = logging.getLogger("roomio-example")
logger.setLevel(logging.INFO)

load_dotenv()


class EchoTask(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Echo. Call the get_weather_today and get_weather_tomorrow functions when user asks for the weather."  # noqa: E501
                "Tell the user when you are calling the functions."
            ),
            # llm=openai.realtime.RealtimeModel(voice="echo"),
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=cartesia.TTS(),
        )

    @ai_function()
    async def talk_to_alloy(self, context: RunContext):
        return AlloyTask(), "Transferring you to Alloy."

    @ai_function()
    async def get_weather_today(self) -> str:
        """Called when user asks for the weather."""
        return "The weather is sunny today."

    @ai_function()
    async def get_weather_tomorrow(self) -> str:
        """Called when user asks for the weather."""
        return "The weather is rainy tomorrow."


class AlloyTask(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Alloy. Call the get_weather_today and get_weather_tomorrow functions when user asks for the weather."  # noqa: E501
                "Tell the user when you are calling the functions."
            ),
            llm=openai.realtime.RealtimeModel(voice="alloy"),
        )

    @ai_function()
    async def talk_to_echo(self, context: RunContext):
        return EchoTask(), "Transferring you to Echo."

    @ai_function()
    async def get_weather_today(self) -> str:
        """Called when user asks for the weather."""
        return "The weather is sunny today."

    @ai_function()
    async def get_weather_tomorrow(self) -> str:
        """Called when user asks for the weather."""
        return "The weather is rainy tomorrow."


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    agent = VoiceAgent(
        task=EchoTask(),
    )

    await agent.start(
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # noise_cancellation=noise_cancellation.BVC(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
