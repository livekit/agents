import asyncio
import logging
import wave

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.agents.voice import AgentTask, VoiceAgent


class ConversationRecorder:
    def __init__(
        self,
        *,
        model: VoiceAgent | None,
        file_name: str | None,
    ):
        """
        Initializes a ConversationRecorder instance which records the audio of a conversation.

        Args:
            model (VoiceAgent): an instance of a VoiceAgent
            file_name (str): the name of the audio file
        """
        super().__init__()

        self._model = model
        self._file_name = file_name

        with wave.open(file_name, "wb") as file:
            file.setnchannels(2)
            file.setframerate(24000)
        self._file = file

    @property
    def model(self) -> VoiceAgent | None:
        return self._model

    async def record_input(self) -> None:
        async for audioframe in self._model._input.audio:
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


load_dotenv()

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    agent = VoiceAgent(task=AgentTask())
    recorder = ConversationRecorder(model=agent, file_name="output.wav")

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()
    agent.start(ctx.room, participant)
    recorder.start()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
