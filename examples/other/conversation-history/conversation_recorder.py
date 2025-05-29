import asyncio
import logging
import wave
from collections.abc import AsyncIterable, AsyncIterator
from functools import update_wrapper

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.agents.voice.agent import ModelSettings
from livekit.plugins import deepgram, openai, silero
from livekit.rtc import AudioFrame, AudioMixer, AudioResampler

# Default settings for the file
FRAMERATE = 48000
CHANNELS = 1
SAMPLEWIDTH = 2


def stt_recorder(*, stt, recorder):
    """
    Wraps Agent's default STT node with stt_node_wrapper. The AudioFrames are caught
    before transcribing and SpeechEvents are yielded.
    """

    async def stt_node_wrapper(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ):
        async def record_audio():
            async for frame in audio:
                await recorder.queue_stt_audio(frame)
                yield frame

        async for event in stt(self, record_audio(), model_settings):
            yield event

    update_wrapper(stt_node_wrapper, stt)
    return stt_node_wrapper


def tts_recorder(*, tts, recorder):
    """
    Wraps Agent's default TTS node with tts_node_wrapper.
    The AudioFrames are caught before playout.
    """

    async def tts_node_wrapper(self, text: AsyncIterable[str], model_settings: ModelSettings):
        async for frame in tts(self, text, model_settings):
            # FIXME will handle interruptions so full playout isn't recorded
            if recorder.session.current_speech.interrupted:
                ...

            await recorder.queue_tts_audio(frame)
            yield frame

    update_wrapper(tts_node_wrapper, tts)
    return tts_node_wrapper


class SessionRecorder:
    def __init__(self, session: AgentSession, file_name: str):
        """
        Initializes an instance of a SessionRecorder which records an AgentSession to a wav file.
        The STT and TTS nodes of the default Agent class are wrapped to catch and queue AudioFrames.
        AudioFrames are also converted to a uniform format and mixed before being recorded.
        Args:
            session (AgentSession): Session to be recorded
            file_name (str): The name of the wav file
        """
        self._session = session
        self._user_audio = asyncio.Queue[AudioFrame]()
        self._agent_audio = asyncio.Queue[AudioFrame]()

        self._stt_resampler: AudioResampler | None = None
        self._tts_resampler: AudioResampler | None = None
        self._audio_mixer: AudioMixer | None = None

        self._file_name = file_name
        self._file = wave.open(self._file_name, "wb")
        self._file.setnchannels(CHANNELS)
        self._file.setframerate(FRAMERATE)
        self._file.setsampwidth(SAMPLEWIDTH)

        Agent.default.stt_node = stt_recorder(stt=Agent.default.stt_node, recorder=self)
        Agent.default.tts_node = tts_recorder(tts=Agent.default.tts_node, recorder=self)

    @property
    def session(self) -> AgentSession:
        return self._session

    async def queue_stt_audio(self, audioframe: AudioFrame):
        """
        Resamples given AudioFrame and adds to the STT queue. An empty AudioFrame is added to the
        TTS queue to align the two streams for the AudioMixer.
        """
        if audioframe.sample_rate != FRAMERATE:
            if not self._stt_resampler:
                self._stt_resampler = AudioResampler(
                    input_rate=audioframe.sample_rate, output_rate=FRAMERATE, quality="very_high"
                )
            frames = self._stt_resampler.push(audioframe)
            if frames:
                for resampled_frame in frames:
                    self._user_audio.put_nowait(resampled_frame)
        else:
            self._user_audio.put_nowait(audioframe)

        if self._agent_audio.empty():
            await self.queue_tts_audio(
                audioframe=AudioFrame.create(
                    sample_rate=FRAMERATE, num_channels=CHANNELS, samples_per_channel=1660
                )
            )

    async def queue_tts_audio(self, audioframe: AudioFrame):
        """
        Resamples given AudioFrame and adds to the TTS queue.
        Interruptions will be handled accordingly.
        """
        if audioframe.sample_rate != FRAMERATE:
            if not self._tts_resampler:
                self._tts_resampler = AudioResampler(
                    input_rate=audioframe.sample_rate, output_rate=FRAMERATE, quality="very_high"
                )
            frames = self._tts_resampler.push(audioframe)
            if frames:
                for resampled_frame in frames:
                    self._agent_audio.put_nowait(resampled_frame)
        else:
            self._agent_audio.put_nowait(audioframe)

    async def stream_audio_queue(self, queue) -> AsyncIterator[AudioFrame]:
        while (frame := await queue.get()) is not None:
            yield frame

    def user_audio_stream(self) -> AsyncIterator[AudioFrame]:
        return self.stream_audio_queue(self._user_audio)

    def agent_audio_stream(self) -> AsyncIterator[AudioFrame]:
        return self.stream_audio_queue(self._agent_audio)

    async def _main_atask(self) -> None:
        """
        Creates an AudioMixer to combine the STT and TTS streams. The mixed bytes are then written
        to the specified wav file.
        """
        self._audio_mixer = AudioMixer(sample_rate=FRAMERATE, num_channels=1, stream_timeout_ms=300)

        self._audio_mixer.add_stream(self.user_audio_stream())
        self._audio_mixer.add_stream(self.agent_audio_stream())

        async for mixed_frame in self._audio_mixer:
            self._file.writeframes(mixed_frame.data.tobytes())

    async def aclose(self) -> None:
        await self._main_atask
        await self._audio_mixer.aclose()

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
