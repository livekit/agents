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
from livekit.agents.voice.events import AgentStateChangedEvent
from livekit.plugins import cartesia, deepgram, openai, silero
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
            await recorder.queue_tts_audio(frame)
            yield frame

    update_wrapper(tts_node_wrapper, tts)
    return tts_node_wrapper


class SessionRecorder:
    def __init__(
        self,
        session: AgentSession,
        file_name: str,
    ):
        """
        Initializes an instance of a SessionRecorder which records an AgentSession to a wav file.
        The STT and TTS nodes of the default Agent class are wrapped to catch and queue AudioFrames.
        AudioFrames are also converted to a uniform format before being recorded.
        Args:
            session (AgentSession): Session to be recorded
            file_name (str): The name of the wav file
        """
        self._session = session
        self._user_audio = asyncio.Queue[AudioFrame]()
        self._agent_audio = asyncio.Queue[AudioFrame]()

        self._audio_mixer: AudioMixer | None = None
        self._stt_resampler: AudioResampler | None = None
        self._tts_resampler: AudioResampler | None = None
        self._current_tts_input_rate: int = 0

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

    @property
    def user_queue(self) -> asyncio.Queue[AudioFrame]:
        return self._user_audio

    @property
    def agent_queue(self) -> asyncio.Queue[AudioFrame]:
        return self._agent_audio

    async def clear_agent_queue(self) -> None:
        while not self._agent_audio.empty():
            self._agent_audio.get_nowait()

    async def queue_stt_audio(self, audioframe: AudioFrame):
        """
        Resamples via an AudioResampler and queues the AudioFrame to the user audio queue.
        If there is no agent AudioFrame to pair with, an empty AudioFrame is queued to the
        TTS queue to ensure level queues.

        Args:
            audioframe (AudioFrame): The AudioFrame to be queued and recorded
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
        Resamples via an AudioResampler and queues the AudioFrame to the agent audio queue.
        In the case that the input frame rate changes due to different TTS providers, new
        AudioResamplers are created accordingly.

        Args:
            audioframe (AudioFrame): The AudioFrame to be queued and recorded
        """
        if audioframe.sample_rate != FRAMERATE:
            if self._tts_resampler and audioframe.sample_rate != self._current_tts_input_rate:
                flushed_frames = self._tts_resampler.flush()
                for frame in flushed_frames:
                    self._agent_audio.put_nowait(frame)
                self._tts_resampler = None
            if not self._tts_resampler:
                self._tts_resampler = AudioResampler(
                    input_rate=audioframe.sample_rate, output_rate=FRAMERATE, quality="very_high"
                )
                self._current_tts_input_rate = audioframe.sample_rate
            frames = self._tts_resampler.push(audioframe)
            if frames:
                for resampled_frame in frames:
                    self._agent_audio.put_nowait(resampled_frame)
        else:
            self._agent_audio.put_nowait(audioframe)

    async def stream_audio_queue(
        self, queue: asyncio.Queue[AudioFrame]
    ) -> AsyncIterator[AudioFrame]:
        """
        Creates an AsyncIterator for the contents of a queue.

        Args:
            queue (asyncio.Queue[AudioFrame]): A queue to be returned as an AsyncIterator
        """
        while (frame := await queue.get()) is not None:
            yield frame

    def user_audio_stream(self) -> AsyncIterator[AudioFrame]:
        return self.stream_audio_queue(self._user_audio)

    def agent_audio_stream(self) -> AsyncIterator[AudioFrame]:
        return self.stream_audio_queue(self._agent_audio)

    async def _main_atask(self) -> None:
        """
        Creates an AudioMixer to combine the STT and TTS audio streams
        and writes the mixed bytes to the wav file.
        """
        self._audio_mixer = AudioMixer(
            sample_rate=FRAMERATE, num_channels=1, stream_timeout_ms=3000
        )

        self._audio_mixer.add_stream(self.user_audio_stream())
        self._audio_mixer.add_stream(self.agent_audio_stream())

        async for mixed_frame in self._audio_mixer:
            self._file.writeframes(mixed_frame.data.tobytes())

    async def aclose(self) -> None:
        await self._audio_mixer.aclose()
        if self._clear_task:
            await self._clear_task
        await self._main_atask
        self._file.close()

    def start(self) -> None:
        """
        Begins the recording and keeps track of agent interruptions.
        """
        self._main_atask = asyncio.create_task(self._main_atask())

        @self._session.on("agent_state_changed")
        def check_interruption(ev: AgentStateChangedEvent):
            if (
                self._session.current_speech is not None
                and self._session.current_speech.interrupted
                and ev.old_state == "speaking"
            ):
                self._clear_task = asyncio.create_task(self.clear_agent_queue())

        @self._session.on("close")
        def on_close(ev):
            self._clear_task = asyncio.create_task(self.clear_agent_queue())
            self._close_task = asyncio.create_task(self.aclose())


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
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
    )
    recorder = SessionRecorder(session=session, file_name="recording.wav")

    await session.start(agent=session.userdata["AgentA"], room=ctx.room)
    recorder.start()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
