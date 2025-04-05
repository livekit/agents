import asyncio
import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    JobContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import openai

logger = logging.getLogger("background-audio")

load_dotenv()


class FakeWebSearchAgent(Agent):
    def __init__(self):
        super().__init__(instructions="You are a helpful assistant")

    @function_tool
    async def search_web(self, query: str) -> str:
        """
        Search the web for information based on the given query.

        Always use this function whenever the user requests a web search

        Args:
            query: The search query to look up on the web.
        """

        # simulate a long web search to demonstrate the background "thinking" audio
        logger.info("FakeWebSearchAgent thinking...")
        await asyncio.sleep(5)
        return "The request failed, give the users some information based on your knowledge"


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(llm=openai.realtime.RealtimeModel())
    await session.start(FakeWebSearchAgent(), room=ctx.room)

    # by default BackgroundAudio will play a office ambience sound.
    background_audio = BackgroundAudioPlayer(
        # ambient_sound="background-sound.ogg",
        thinking_sound=[
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.8),
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.7),
        ],
    )

    await background_audio.start(room=ctx.room, agent_session=session)

    # Play another audio file at any time using the play method:
    # background_audio.play("filepath.ogg")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
