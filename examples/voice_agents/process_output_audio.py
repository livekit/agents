import logging
from collections.abc import AsyncIterable

import numpy as np
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli, utils
from livekit.plugins import deepgram, openai, silero

try:
    import librosa
except ImportError:
    raise ImportError(
        "librosa is required to run this example, install it with `pip install librosa`"
    ) from None


logger = logging.getLogger("basic-agent")
logging.getLogger("numba").setLevel(logging.WARNING)

load_dotenv()

## This example demonstrates how to add post-processing to the output audio of the agent.


class MyAgent(Agent):
    def __init__(self, *, speed_factor: float = 1.2) -> None:
        super().__init__(
            instructions="Your name is Jenna. You would interact with users via voice."
            "with that in mind keep your responses concise and to the point."
            "You are curious and friendly, and have a sense of humor.",
        )
        self.speed_factor = speed_factor

    async def audio_output_node(
        self, audio: AsyncIterable[rtc.AudioFrame]
    ) -> AsyncIterable[rtc.AudioFrame]:
        stream: utils.audio.AudioByteStream | None = None
        async for frame in audio:
            if stream is None:
                stream = utils.audio.AudioByteStream(
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                    samples_per_channel=frame.sample_rate // 10,  # 100ms
                )
            # TODO: find a streamed way to process the audio
            for f in stream.push(frame.data):
                yield self._process_audio(f)

        for f in stream.flush():
            yield self._process_audio(f)

    def _process_audio(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        # time-stretch without pitch change
        audio_data = np.frombuffer(frame.data, dtype=np.int16)

        stretched = librosa.effects.time_stretch(
            audio_data.astype(np.float32) / np.iinfo(np.int16).max,
            rate=self.speed_factor,
        )
        stretched = (stretched * np.iinfo(np.int16).max).astype(np.int16)
        return rtc.AudioFrame(
            data=stretched.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            samples_per_channel=stretched.shape[-1],
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

    # warmup the librosa
    librosa.effects.time_stretch(
        np.random.randn(16000).astype(np.float32),
        rate=1.2,
    )


async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "user_id": "your user_id",
    }
    await ctx.connect()

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # any combination of STT, LLM, TTS, or realtime API can be used
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(model="nova-3"),
        tts=openai.TTS(voice="ash"),
    )
    await session.start(agent=MyAgent(), room=ctx.room)
    session.say("Hello, how can I help you today?")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
