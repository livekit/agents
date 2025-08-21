import logging

from dotenv import load_dotenv
from google.genai.types import Modality  # noqa: F401

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.room_io import RoomOutputOptions
from livekit.plugins import google, openai  # noqa: F401

logger = logging.getLogger("realtime-with-tts")
logger.setLevel(logging.INFO)

load_dotenv()

# This example is showing a half-duplex realtime LLM usage where we:
# - use a multimodal/realtime LLM that takes audio input, generating text output
# - then use a separate TTS to synthesize audio output
#
# This approach fully utilizes the realtime LLM's ability to understand directly from audio
# and yet maintains control of the pipeline, including using custom voices with TTS


class WeatherAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant.",
            llm=openai.realtime.RealtimeModel(modalities=["text"]),
            # llm=google.beta.realtime.RealtimeModel(modalities=[Modality.TEXT]),
            tts=openai.TTS(voice="ash"),
        )

    @function_tool
    async def get_weather(self, location: str):
        """Called when the user asks about the weather.

        Args:
            location: The location to get the weather for
        """

        logger.info(f"getting weather for {location}")
        return f"The weather in {location} is sunny, and the temperature is 20 degrees Celsius."


async def entrypoint(ctx: JobContext):
    session = AgentSession()

    await session.start(
        agent=WeatherAgent(),
        room=ctx.room,
        room_output_options=RoomOutputOptions(
            transcription_enabled=True,
            audio_enabled=True,  # you can also disable audio output to use text modality only
        ),
    )
    session.generate_reply(instructions="say hello to the user in English")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
