import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    inference,
    room_io,
)

logger = logging.getLogger("text-only")
logger.setLevel(logging.INFO)

load_dotenv()

## This example demonstrates a text-only agent.
## When using with LiveKit SDKs, this agent is automatically wired up to text input and output:
## - Send text input using TextStream to topic `lk.chat` (https://docs.livekit.io/home/client/data/text-streams)
## - The agent output is sent through TextStream to the `lk.transcription` topic
## You can also transport text via other means and directly send them to the agent
## - Send text input via: `generate_reply(user_input="user's input text")`
## - Receive agent's response via `session.on("conversation_item_added", ev)`. docs: https://docs.livekit.io/agents/build/events/#conversation_item_added


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant.",
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        llm=inference.LLM("openai/gpt-4.1-mini"),
        # note that no TTS or STT are needed here
    )
    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            text_input=True,
            text_output=True,
            audio_input=False,
            audio_output=False,
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
