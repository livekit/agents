import logging

from dotenv import load_dotenv

from livekit import rtc  # noqa: F401
from livekit.agents import (
    Agent,
    AgentSession,
    CloseEvent,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import cartesia, deepgram, openai, silero

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

load_dotenv()


# This example shows how to close the agent session when the linked participant disconnects
# or when the worker is shutting down. When closing the session, agent will be interrupted
# and the last agent message will be added to the chat context.


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
    )

    # session will be closed automatically when the linked participant disconnects
    # with reason CLIENT_INITIATED, ROOM_DELETED, or USER_REJECTED
    # or you can specify the disconnect reasons to close the session in RoomInputOptions
    await session.start(
        agent=Agent(instructions="You are a helpful assistant."),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # close_on_disconnect_reasons=[rtc.DisconnectReason.CLIENT_INITIATED]
        ),
    )

    @session.on("close")
    def on_close(_: CloseEvent):
        print("Agent Session closed, Chat History:")
        print("=" * 20)
        for item in session.history.items:
            if item.type == "message":
                print(f"{item.role}: {item.text_content.replace('\n', '\\n')}")
        print("=" * 20)

        # Optionally, you can delete the room when the session is closed
        # this will stop the worker immediately
        ctx.delete_room()

    # close the session when the worker is shutting down
    ctx.add_shutdown_callback(session.aclose)

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
