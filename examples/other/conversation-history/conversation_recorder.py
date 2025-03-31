import logging
import wave
from collections.abc import AsyncIterable

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli, stt
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.agent import ModelSettings
from livekit.plugins import deepgram, openai, silero


class MyAgent(Agent):
    def __init__(self, file_name: str | None):
        super().__init__(
            instructions="""You are a helpful assistant that can answer questions and help with
                            tasks. The conversation will be recorded.""",
        )
        if file_name:
            self._file = wave.open(file_name, "wb")
            self._file.setnchannels(2)
            self._file.setframerate(24000)
            self._file.setsampwidth(2)

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> AsyncIterable[stt.SpeechEvent]:
        async def record_audio():
            async for frame in audio:
                wavframe = frame.data
                self._file.writeframes(wavframe)
                yield frame

        async for event in super().stt_node(record_audio(), model_settings):
            yield event

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        async for frame in super().tts_node(text, model_settings):
            wavframe = frame.data
            self._file.writeframes(wavframe)
            yield frame

    @function_tool()
    async def open_door(self) -> str:
        await self.session.generate_reply(instructions="Opening the door..")

        return "The door is open!"


load_dotenv()

logger = logging.getLogger("my-recorder")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    session = AgentSession(
        stt=deepgram.STT(), llm=openai.LLM(), tts=openai.TTS(), vad=silero.VAD.load()
    )

    await session.start(agent=MyAgent(file_name="recording.wav"), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
