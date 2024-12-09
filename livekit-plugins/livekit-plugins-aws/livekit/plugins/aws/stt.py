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
import os
from dataclasses import dataclass
from typing import Optional

import boto3
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.model import TranscriptEvent, TranscriptResultStream
from livekit import rtc
from livekit.agents import stt, utils

from .log import logger


@dataclass
class STTOptions:
    speech_region: str
    sample_rate: int
    num_channels: int
    languages: list[
        str
    ]  # see https://docs.aws.amazon.com/transcribe/latest/dg/supported-languages.html


class STT(stt.STT):
    def __init__(
        self,
        *,
        speech_region: str = "us-east-1",
        speech_key: str | None = None,
        speech_secret: str | None = None,
        sample_rate: int = 48000,
        num_channels: int = 1,
        languages: list[str] = ["en-US"],
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=True, interim_results=True)
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

        self._config = STTOptions(
            speech_region=speech_region,
            languages=languages,
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

    async def recognize(
        self,
        *,
        buffer: utils.AudioBuffer,
        language: str | None = None,
    ) -> stt.SpeechEvent:
        raise NotImplementedError(
            "Amazon Transcribe does not support single frame recognition"
        )

    def stream(
        self,
        *,
        language: str | None = None,
    ) -> "SpeechStream":
        return SpeechStream(self._config)


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        opts: STTOptions,
        sample_rate: int = 48000,
        num_channels: int = 1,
        max_retry: int = 32,
    ) -> None:
        super().__init__()
        self._opts = opts
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._max_retry = max_retry

        self._client = TranscribeStreamingClient(region=self._opts.speech_region)

    async def _run(self, max_retry: int) -> None:
        while not self._input_ch.closed:
            try:
                # aws requires a async generator when calling start_stream_transcription
                stream = await self._client.start_stream_transcription(
                    language_code="en-US",
                    media_sample_rate_hz=self._sample_rate,
                    media_encoding="pcm",
                )

                # this function basically convert the queue into a async generator
                async def input_generator():
                    try:
                        async for frame in self._input_ch:
                            if isinstance(frame, rtc.AudioFrame):
                                await stream.input_stream.send_audio_event(
                                    audio_chunk=frame.data.tobytes()
                                )
                        await stream.input_stream.end_stream()
                    except Exception as e:
                        logger.exception(
                            f"an error occurred while streaming inputs: {e}"
                        )

                # try to connect
                handler = TranscriptEventHandler(stream.output_stream, self._event_ch)
                await asyncio.gather(input_generator(), handler.handle_events())
            except Exception as e:
                logger.exception(f"an error occurred while streaming inputs: {e}")


def _streaming_recognize_response_to_speech_data(
    resp: None,
) -> stt.SpeechData:
    lg = "en-US"
    data = stt.SpeechData(
        language=lg,
        start_time=0,
        end_time=0,
        confidence=0.0,
        text=resp.alternatives[0].transcript,
    )

    return data


class TranscriptEventHandler:
    def __init__(
        self,
        transcript_result_stream: TranscriptResultStream,
        event_ch: asyncio.Queue[Optional[stt.SpeechEvent]],
    ):
        self._transcript_result_stream = transcript_result_stream
        self._event_ch = event_ch

    async def handle_events(self):
        """Process generic incoming events from Amazon Transcribe
        and delegate to appropriate sub-handlers.
        """
        async for event in self._transcript_result_stream:
            if isinstance(event, TranscriptEvent):
                await self.handle_transcript_event(event)

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        # This handler can be implemented to handle transcriptions as needed.
        stream = transcript_event.transcript.results
        for resp in stream:
            if resp.start_time == 0.0:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                )

            if resp.end_time > 0.0:
                if resp.is_partial:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                            alternatives=[
                                _streaming_recognize_response_to_speech_data(resp)
                            ],
                        )
                    )

                else:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[
                                _streaming_recognize_response_to_speech_data(resp)
                            ],
                        )
                    )

            if not resp.is_partial:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                )
