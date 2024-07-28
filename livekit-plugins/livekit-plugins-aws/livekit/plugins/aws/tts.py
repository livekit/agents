# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import dataclass
from typing import Optional

from livekit import rtc
from livekit.agents import tts, utils
from livekit.agents.utils import codecs
import boto3
from aiobotocore.session import get_session, AioSession

from .log import logger

TTS_SAMPLE_RATE: int = 16000
TTS_NUM_CHANNELS: int = 1


@dataclass
class _TTSOptions:
    # https://docs.aws.amazon.com/polly/latest/dg/generative-voices.html
    voice: str | None = None
    output_format: str | None = None  # pcm or mp3
    speech_engine: str | None = None  # generative, neural, standard
    speech_region: str | None = None


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str | None = "Ruth",
        aws_session: AioSession | None = None,
        output_format: str = "pcm",
        speech_engine: str = "generative",
        speech_region: str = "us-east-1",
        speech_key: str | None = None,
        speech_secret: str | None = None,
    ) -> None:
        super().__init__(
            streaming_supported=False,
            sample_rate=TTS_SAMPLE_RATE,
            num_channels=TTS_NUM_CHANNELS,
        )
        credentials = boto3.Session().get_credentials()

        speech_key = speech_key or os.environ.get("AWS_ACCESS_KEY_ID") or credentials.access_key
        if not speech_key:
            raise ValueError("AWS_ACCESS_KEY_ID must be set")

        speech_secret = speech_secret or os.environ.get("AWS_SECRET_ACCESS_KEY") or credentials.secret_key
        if not speech_secret:
            raise ValueError("AWS_SECRET_ACCESS_KEY must be set")

        speech_region = speech_region or os.environ.get("AWS_DEFAULT_REGION")
        if not speech_region:
            raise ValueError("AWS_DEFAULT_REGION must be set")

        self._opts = _TTSOptions(
            voice=voice,
            output_format=output_format,
            speech_engine=speech_engine,
            speech_region=speech_region,
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
            async with self._session.create_client('polly', region_name=self._opts.speech_region) as client:
                response = await client.synthesize_speech(
                    Text=self._text, OutputFormat=self._opts.output_format, Engine=self._opts.speech_engine,
                    VoiceId=self._opts.voice, TextType="text", SampleRate=str(TTS_SAMPLE_RATE)
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