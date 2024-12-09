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

import os
from dataclasses import dataclass

import boto3
from aiobotocore.session import AioSession, get_session
from livekit import rtc
from livekit.agents import tts, utils

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
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=TTS_SAMPLE_RATE,
            num_channels=TTS_NUM_CHANNELS,
        )
        credentials = boto3.Session().get_credentials()

        speech_key = (
            speech_key or os.environ.get("AWS_ACCESS_KEY_ID") or credentials.access_key
        )
        if not speech_key:
            raise ValueError("AWS_ACCESS_KEY_ID must be set")

        speech_secret = (
            speech_secret
            or os.environ.get("AWS_SECRET_ACCESS_KEY")
            or credentials.secret_key
        )
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
    def __init__(self, text: str, opts: _TTSOptions, session: AioSession) -> None:
        super().__init__()
        self._text, self._opts, self._session = text, opts, session

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        request_id = utils.shortuuid()
        segment_id = utils.shortuuid()

        async with self._session.create_client(
            "polly", region_name=self._opts.speech_region
        ) as client:
            response = await client.synthesize_speech(
                Text=self._text,
                OutputFormat=self._opts.output_format,
                Engine=self._opts.speech_engine,
                VoiceId=self._opts.voice,
                TextType="text",
                SampleRate=str(TTS_SAMPLE_RATE),
            )
            if "AudioStream" in response:
                decoder = utils.codecs.Mp3StreamDecoder()
                async with response["AudioStream"] as resp:
                    async for data, _ in resp.content.iter_chunks():
                        if self._opts.output_format == "mp3":
                            frames = decoder.decode_chunk(data)
                            for frame in frames:
                                self._event_ch.send_nowait(
                                    tts.SynthesizedAudio(
                                        request_id=request_id,
                                        segment_id=segment_id,
                                        frame=frame,
                                    )
                                )
                        else:
                            self._event_ch.send_nowait(
                                tts.SynthesizedAudio(
                                    request_id=request_id,
                                    segment_id=segment_id,
                                    frame=rtc.AudioFrame(
                                        data=data,
                                        sample_rate=TTS_SAMPLE_RATE,
                                        num_channels=1,
                                        samples_per_channel=len(data) // 2,  # 16-bit
                                    ),
                                )
                            )
            else:
                logger.error("polly tts failed to synthesizes speech")
