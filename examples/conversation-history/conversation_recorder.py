import asyncio
import logging
import wave

from dotenv import load_dotenv
from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai


class ConversationRecorder:
    def __init__(
        self,
        *,
        session: AgentSession | None,
        file_name: str | None,
    ):
        """
        Initializes a ConversationRecorder instance which records the audio of a conversation.

        Args:
            model (AgentSession): an instance of an AgentSession
            file_name (str): the name of the audio file
        """
        super().__init__()

        self._session = session
        self._file_name = file_name

        with wave.open(file_name, "wb") as file:
            file.setnchannels(2)
            file.setframerate(24000)
            file.setsampwidth(2)
        self._file = file

    @property
    def session(self) -> AgentSession | None:
        return self._session

    async def record_input(self) -> None:
        async for audioframe in self._session._input.audio:
            self._file.writeframes(audioframe)

    async def record_output(self) -> None:
        ...
        # capture agent output audio frames

    async def _main_atask(self) -> None:
        recordinput = asyncio.create_task(self.record_input())
        recordoutput = asyncio.create_task(self.record_output())
        await asyncio.gather(recordinput, recordoutput)

    def start(self) -> None:
        self._main_task = asyncio.create_task(self._main_atask())


class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful assistant that can answer questions and help with tasks.",
        )

    @function_tool()
    async def open_door(self) -> str:
        await self.session.generate_reply(instructions="Opening the door..")

        return "The door is open!"


load_dotenv()

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    session = AgentSession(llm=openai.realtime.RealtimeModel())

    cr = ConversationRecorder(session=session, file_name="recording.wav")
    cr.start()

    await session.start(agent=MyAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
