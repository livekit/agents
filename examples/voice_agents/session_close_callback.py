import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    CloseEvent,
    JobContext,
    cli,
    room_io,
    utils,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.plugins import silero

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

load_dotenv()


# This example shows how to close the agent session when the linked participant disconnects
# or when the worker is shutting down. When closing the session, agent will be interrupted
# and the last agent message will be added to the chat context.

server = AgentServer()


class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful assistant.",
            tools=[EndCallTool()],
        )

    @utils.log_exceptions(logger=logger)
    async def on_exit(self) -> None:
        logger.info("exiting the agent")
        if self.session.current_speech:
            await self.session.current_speech

        logger.info("generating goodbye message")
        await self.session.generate_reply(
            instructions="say goodbye to the user", tool_choice="none"
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt="assemblyai/universal-streaming",
        llm="openai/gpt-4.1-mini",
        tts="rime/arcana",
        vad=silero.VAD.load(),
    )

    # session will be closed automatically when the linked participant disconnects
    # with reason CLIENT_INITIATED, ROOM_DELETED, or USER_REJECTED
    # or you can disable it by setting the RoomInputOptions.close_on_disconnect to False
    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            delete_room_on_close=True,
        ),
    )

    @session.on("close")
    def on_close(ev: CloseEvent):
        print(f"Agent Session closed, reason: {ev.reason}")
        print("=" * 20)
        print("Chat History:")
        for item in session.history.items:
            if item.type == "message":
                # New code
                content = item.text_content.replace("\n", "\\n")
                text = f"{item.role}: {content}"

                if item.interrupted:
                    text += " (interrupted)"

            elif item.type == "function_call":
                text = f"function_call: {item.name}, arguments: {item.arguments}"

            elif item.type == "function_call_output":
                text = f"{item.name}: '{item.output}'"
                if item.is_error:
                    text += " (error)"

            elif item.type == "agent_handoff":
                text = f"agent_handoff: {item.old_agent_id} -> {item.new_agent_id}"

            else:
                raise ValueError(f"unknown item type: {item.type}")

            print(text)

        print("=" * 20)


if __name__ == "__main__":
    cli.run_app(server)
