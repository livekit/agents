import logging
import os
from typing import AsyncGenerator, AsyncIterator

import cv2
import numpy as np
from bithuman_runtime import AsyncBithumanRuntime
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli, utils
from livekit.agents.llm import ai_function
from livekit.agents.voice import AgentTask, CallContext, VoiceAgent
from livekit.agents.voice.avatar import (
    AudioSegmentEnd,
    AvatarOptions,
    AvatarRunner,
    QueueAudioSink,
    VideoGenerator,
)
from livekit.agents.voice.room_io import RoomOutputOptions
from livekit.plugins import cartesia, deepgram, openai

logger = logging.getLogger("avatar-example")
logger.setLevel(logging.INFO)

logging.getLogger("numba").setLevel(logging.WARNING)

load_dotenv()


class BithumanGenerator(VideoGenerator):
    def __init__(self, avatar_model: str, token: str):
        self._runtime = AsyncBithumanRuntime(token=token)
        self._avatar_model = avatar_model

    async def start(self) -> None:
        await self._runtime.set_avatar_model(self._avatar_model)
        await self._runtime.start()

    @property
    def video_resolution(self) -> tuple[int, int]:
        frame = self._runtime.get_first_frame()
        if frame is None:
            raise ValueError("Failed to read frame")
        return frame.shape[1], frame.shape[0]

    @property
    def video_fps(self) -> int:
        return self._runtime.settings.FPS

    @property
    def audio_sample_rate(self) -> int:
        return self._runtime.settings.INPUT_SAMPLE_RATE

    @utils.log_exceptions(logger=logger)
    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        if isinstance(frame, AudioSegmentEnd):
            await self._runtime.flush()
            return
        await self._runtime.push_audio(bytes(frame.data), frame.sample_rate, last_chunk=False)

    def clear_buffer(self) -> None:
        self._runtime.interrupt()

    def __aiter__(self) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        return self._stream_impl()

    async def _stream_impl(
        self,
    ) -> AsyncGenerator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd, None]:
        def create_video_frame(image: np.ndarray) -> rtc.VideoFrame:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
            return rtc.VideoFrame(
                width=image.shape[1],
                height=image.shape[0],
                type=rtc.VideoBufferType.RGBA,
                data=image.tobytes(),
            )

        async for frame in self._runtime.run():
            if frame.bgr_image is not None:
                video_frame = create_video_frame(frame.bgr_image)
                yield video_frame

            audio_chunk = frame.audio_chunk
            if audio_chunk is not None:
                audio_frame = rtc.AudioFrame(
                    data=audio_chunk.bytes,
                    sample_rate=audio_chunk.sample_rate,
                    num_channels=1,
                    samples_per_channel=len(audio_chunk.array),
                )
                yield audio_frame

            if frame.end_of_speech:
                yield AudioSegmentEnd()

    async def stop(self) -> None:
        await self._runtime.stop()


class EchoTask(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Echo.",
            # llm=openai.realtime.RealtimeModel(voice="echo"),
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=cartesia.TTS(),
        )

    @ai_function
    async def talk_to_alloy(self, context: CallContext):
        return AlloyTask(), "Transferring you to Alloy."


class AlloyTask(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Alloy.",
            llm=openai.realtime.RealtimeModel(voice="alloy"),
        )

    @ai_function
    async def talk_to_echo(self, context: CallContext):
        return EchoTask(), "Transferring you to Echo."


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # create agent
    agent = VoiceAgent(
        task=AlloyTask(),
    )
    agent.output.audio = QueueAudioSink()

    # create video generator
    avatar_model = os.getenv("BITHUMAN_AVATAR_MODEL")
    token = os.getenv("BITHUMAN_RUNTIME_TOKEN")
    if not avatar_model or not token:
        raise ValueError("BITHUMAN_AVATAR_MODEL and BITHUMAN_RUNTIME_TOKEN are required")
    video_gen = BithumanGenerator(avatar_model=avatar_model, token=token)
    await video_gen.start()

    output_width, output_height = video_gen.video_resolution
    avatar_options = AvatarOptions(
        video_width=output_width,
        video_height=output_height,
        video_fps=video_gen.video_fps,
        audio_sample_rate=video_gen.audio_sample_rate,
        audio_channels=1,
    )

    # create avatar runner
    avatar_runner = AvatarRunner(
        room=ctx.room,
        video_gen=video_gen,
        audio_recv=agent.output.audio,
        options=avatar_options,
    )
    await avatar_runner.start()

    # start agent
    await agent.start(
        room=ctx.room,
        room_output_options=RoomOutputOptions(
            audio_enabled=False, audio_sample_rate=video_gen.audio_sample_rate
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, job_memory_warn_mb=1500))
