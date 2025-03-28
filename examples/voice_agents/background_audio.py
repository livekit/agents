import asyncio
import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    BackgroundAudio,
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

        Use this function whenever the user requests a web search or seeks information
        that may require real-time or up-to-date data from the internet.

        Args:
            query (str): The search query to look up on the web.
        """

        # simulate a long web search to demonstrate the background "thinking" audio
        logger.info("FakeWebSearchAgent thinking...")
        await asyncio.sleep(5)
        return "The request failed, give the users some information based on your knowledge"


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(llm=openai.realtime.RealtimeModel())
    await session.start(FakeWebSearchAgent(), room=ctx.room)

    # by default BackgroundAudio will play a office ambience sound and a keyboard typing sound
    # when the agent is thinking.
    background_audio = BackgroundAudio(
        # ambient_sound="background-sound.ogg",
        # thinking_sound="beep.ogg",
    )

    await background_audio.start(room=ctx.room, agent_session=session)

    # Play another audio file at any time using the play method:
    # background_audio.play("filepath.ogg")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
