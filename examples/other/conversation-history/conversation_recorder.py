import asyncio
import logging
import wave
from collections.abc import AsyncIterable
from functools import update_wrapper

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.agents.voice.agent import ModelSettings
from livekit.plugins import deepgram, openai, silero
from livekit.rtc import AudioFrame, AudioResampler

# Default settings for the file
FRAMERATE = 48000
CHANNELS = 1
SAMPLEWIDTH = 2


def stt_recorder(*, stt, recorder):
    """
    Wraps Agent's default STT node with stt_node_wrapper.
    The AudioFrames are caught before transcribing.
    """

    async def stt_node_wrapper(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ):
        async def record_audio():
            async for frame in audio:
                await recorder.queue_audio(frame)
                yield frame

        return stt(self, record_audio(), model_settings)

    update_wrapper(stt_node_wrapper, stt)
    return stt_node_wrapper


def tts_recorder(*, tts, recorder):
    """
    Wraps Agent's default TTS node with tts_node_wrapper.
    The AudioFrames are caught before playout.
    """

    async def tts_node_wrapper(self, text: AsyncIterable[str], model_settings: ModelSettings):
        async for frame in tts(self, text, model_settings=model_settings):
            await recorder.queue_audio(frame)
            yield frame

    update_wrapper(tts_node_wrapper, tts)
    return tts_node_wrapper


class SessionRecorder:
    def __init__(self, session: AgentSession, file_name: str):
        """
        Initializes an instance of a SessionRecorder which records an AgentSession to a wav file.
        The STT and TTS nodes of the default Agent class are wrapped to catch and queue AudioFrames.
        AudioFrames are also converted to a uniform format before being recorded.

        Args:
            session (AgentSession): Session to be recorded
            file_name (str): The name of the wav file
        """
        self._session = session
        self._audio_q = asyncio.Queue[AudioFrame]()

        self._file_name = file_name
        self._file = wave.open(self._file_name, "wb")
        self._file.setnchannels(CHANNELS)
        self._file.setframerate(FRAMERATE)
        self._file.setsampwidth(SAMPLEWIDTH)

        Agent.default.stt_node = stt_recorder(stt=Agent.default.stt_node, recorder=self)
        Agent.default.tts_node = tts_recorder(tts=Agent.default.tts_node, recorder=self)

    async def queue_audio(self, audioframe: AudioFrame):
        self._audio_q.put_nowait(audioframe)

    async def _main_atask(self) -> None:
        self._current_input_rate = 0
        self._audio_resampler = None

        while True:
            frame = await self._audio_q.get()
            if frame is None:
                break

            elif frame.sample_rate != FRAMERATE:
                if frame.sample_rate != self._current_input_rate:
                    if self._audio_resampler:
                        frames = self._audio_resampler.flush()
                        if frames:
                            for frame in frames:
                                self._file.writeframes(frame.data.tobytes())
                    else:
                        self._audio_resampler = AudioResampler(
                            input_rate=frame.sample_rate, output_rate=FRAMERATE
                        )
                        self._current_input_rate = frame.sample_rate
                        frame = self._audio_resampler.push(frame)
                        if frame:
                            self._file.writeframes(frame[0].data.tobytes())
                else:
                    frame = self._audio_resampler.push(frame)
                    if frame:
                        self._file.writeframes(frame[0].data.tobytes())
            else:
                self._file.writeframes(frame.data.tobytes())

    async def aclose(self) -> None:
        self._audio_q.put_nowait(None)
        await self._main_atask

    def start(self) -> None:
        self._main_atask = asyncio.create_task(self._main_atask())


class AgentA(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are Agent A, you can answer questions and help with
                            tasks. You can only open doors.""",
        )

    @function_tool()
    async def open_door(self) -> str:
        await self.session.generate_reply(instructions="Opening the door..", tool_choice="none")

        return "The door is open!"

    @function_tool()
    async def transfer_to_agentB(self, context: RunContext) -> tuple[Agent, str]:
        return context.userdata["AgentB"], "Taking you to Agent B."


class AgentB(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are Agent B, you can answer questions and help with
                            tasks. You can only close doors.""",
        )

    @function_tool()
    async def close_door(self) -> str:
        await self.session.generate_reply(instructions="Closing the door..", tool_choice="none")
        return "The door is closed!"

    @function_tool()
    async def transfer_to_agentA(self, context: RunContext) -> tuple[Agent, str]:
        return context.userdata["AgentA"], "Taking you to Agent A."


load_dotenv()

logger = logging.getLogger("my-recorder")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    agent_bank = {"AgentA": AgentA(), "AgentB": AgentB()}
    session = AgentSession(
        userdata=agent_bank,
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=deepgram.TTS(),
        vad=silero.VAD.load(),
    )
    recorder = SessionRecorder(session=session, file_name="recording.wav")

    await session.start(agent=session.userdata["AgentA"], room=ctx.room)
    recorder.start()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
