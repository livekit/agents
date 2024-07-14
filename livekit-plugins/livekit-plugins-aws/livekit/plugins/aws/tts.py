from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import dataclass
from typing import Optional

import aiohttp
from livekit import rtc
from livekit.agents import tts, utils
from livekit.agents.utils import codecs
from aiobotocore.session import get_session, AioSession

from .log import logger
from .models import TTSVoices

TTS_SAMPLE_RATE = 16000
TTS_CHANNELS = 1
AWS_REGION = "us-east-1"


@dataclass
class _TTSOptions:
    voice: TTSVoices
    sample_rate: int    # sample rate in Hz
    output_format: str  # pcm or mp3


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: TTSVoices = "Ruth",
        aws_session: AioSession | None = None,
        sample_rate: int = 16000,
        output_format: str = "pcm",
    ) -> None:
        super().__init__(
            streaming_supported=False,
            sample_rate=TTS_SAMPLE_RATE,
            num_channels=TTS_CHANNELS,
        )

        self._opts = _TTSOptions(
            voice=voice,
            sample_rate=sample_rate,
            output_format=output_format,
        )
        self._session = aws_session

    def _ensure_session(self) -> AioSession:
        if not self._session:
            self._session = get_session()

        return self._session

    def synthesize(
        self,
        text: str,
    ) -> "ChunkedStream":
        return ChunkedStream(text, self._opts, self._ensure_session())


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self, text: str, opts: _TTSOptions, session: AioSession
    ) -> None:
        self._opts = opts
        self._text = text
        self._session = session
        self._decoder = codecs.Mp3StreamDecoder()
        self._main_task: asyncio.Task | None = None
        self._queue = asyncio.Queue[Optional[tts.SynthesizedAudio]]()

    async def _run(self):
        try:
            async with self._session.create_client('polly', region_name=AWS_REGION) as client:
                response = await client.synthesize_speech(
                    Text=self._text, OutputFormat=self._opts.output_format, Engine="generative",
                    VoiceId=self._opts.voice, TextType="text", SampleRate=str(self._opts.sample_rate)
                )
                if "AudioStream" in response:
                    async with response['AudioStream'] as resp:
                        if self._opts.output_format == "mp3":
                            async for data, _ in resp.content.iter_chunks():
                                frames = self._decoder.decode_chunk(data)
                                for frame in frames:
                                    self._queue.put_nowait(
                                        tts.SynthesizedAudio(text=self._text, data=frame)
                                    )
                        else:
                            bytes_per_frame = (self._opts.sample_rate // 100) * 2
                            buf = bytearray()

                            async for data, _ in resp.content.iter_chunks():
                                buf.extend(data)

                                while len(buf) >= bytes_per_frame:
                                    frame_data = buf[:bytes_per_frame]
                                    buf = buf[bytes_per_frame:]

                                    self._queue.put_nowait(
                                        tts.SynthesizedAudio(
                                            text=self._text,
                                            data=rtc.AudioFrame(
                                                data=frame_data,
                                                sample_rate=self._opts.sample_rate,
                                                num_channels=1,
                                                samples_per_channel=len(frame_data) // 2,
                                            ),
                                        )
                                    )

                            # send any remaining data
                            if len(buf) > 0:
                                self._queue.put_nowait(
                                    tts.SynthesizedAudio(
                                        text=self._text,
                                        data=rtc.AudioFrame(
                                            data=buf,
                                            sample_rate=self._opts.sample_rate,
                                            num_channels=1,
                                            samples_per_channel=len(buf) // 2,
                                        ),
                                    )
                                )
                else:
                    logger.error("polly tts failed to synthesizes speech")

        except Exception:
            logger.exception("polly tts main task failed in chunked stream")
        finally:
            self._queue.put_nowait(None)

    async def __anext__(self) -> tts.SynthesizedAudio:
        if not self._main_task:
            self._main_task = asyncio.create_task(self._run())

        frame = await self._queue.get()
        if frame is None:
            raise StopAsyncIteration

        return frame

    async def aclose(self) -> None:
        if not self._main_task:
            return

        self._main_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task