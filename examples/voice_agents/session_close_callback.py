import asyncio
import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, CloseEvent, JobContext, WorkerOptions, cli, llm
from livekit.plugins import cartesia, deepgram, openai, silero

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

load_dotenv()


# This example shows how to close the agent session when the linked participant disconnects
# or when the worker is shutting down. When closing the session, agent will be interrupted
# and the last agent message will be added to the chat context.


class MyAgent(Agent):
    def __init__(self):
        super().__init__(instructions="You are a helpful assistant.")

        self._closing_task: asyncio.Task[None] | None = None

    @llm.function_tool
    async def close_session(self):
        """Called when user want to leave the conversation"""

        logger.info("Closing session from function tool")
        await self.session.generate_reply(instructions="say goodbye to the user")

        # don't await it, the function call will be awaited before closing
        self._closing_task = asyncio.create_task(self.session.aclose())


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
    )

    # session will be closed automatically when the linked participant disconnects
    # with reason CLIENT_INITIATED, ROOM_DELETED, or USER_REJECTED
    # or you can disable it by setting the RoomInputOptions.close_on_disconnect to False
    await session.start(agent=MyAgent(), room=ctx.room)

    @session.on("close")
    def on_close(_: CloseEvent):
        print("Agent Session closed, Chat History:")
        print("=" * 20)
        for item in session.history.items:
            if item.type == "message":
                text = f"{item.role}: {item.text_content.replace('\n', '\\n')}"
                if item.interrupted:
                    text += " (interrupted)"
                print(text)
        print("=" * 20)

        # Optionally, you can delete the room when the session is closed
        # this will stop the worker immediately
        ctx.delete_room()

    # close the session when the worker is shutting down
    ctx.add_shutdown_callback(session.aclose)

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
