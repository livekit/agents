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
from contextlib import suppress
from dataclasses import dataclass
from typing import List, Optional, Union

from livekit import rtc
from livekit.agents import stt
from livekit.agents.utils import AudioBuffer

import boto3
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.model import TranscriptEvent, TranscriptResultStream

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
        languages: list[str] = ["en-US"],  # when empty, auto-detect the language
    ):
        super().__init__(streaming_supported=True)
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

        self._config = STTOptions(
            speech_region=speech_region,
            languages=languages,
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

    async def recognize(
        self,
        *,
        buffer: AudioBuffer,
        language: str | None = None,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Amazon Transcribe does not support single frame recognition")

    def stream(
        self,
        *,
        language: str | None = None,
    ) -> "SpeechStream":
        return SpeechStream(self._config)


class SpeechStream(stt.SpeechStream):
    def __init__(self, opts: STTOptions) -> None:
        super().__init__()
        self._opts = opts
        self._queue = asyncio.Queue[Union[rtc.AudioFrame, str]]()
        self._event_queue = asyncio.Queue[Optional[stt.SpeechEvent]]()
        self._closed = False
        self._speaking = False

        self._client = TranscribeStreamingClient(region=self._opts.speech_region)
        self._main_task = asyncio.create_task(self._run(max_retry=32))

        self._final_events: List[stt.SpeechEvent] = []
        self._need_bos = True
        self._need_eos = False

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            raise ValueError("cannot push frame to closed stream")
        self._queue.put_nowait(frame)

    async def aclose(self, *, wait: bool = True) -> None:
        if self._closed:
            return

        self._closed = True
        if not wait:
            self._main_task.cancel()

        self._queue.put_nowait(None)
        with suppress(asyncio.CancelledError):
            await self._main_task

    async def _run(self, max_retry: int) -> None:
        retry_count = 0
        while not self._closed:
            try:
                # aws requires a async generator when calling start_stream_transcription
                stream = await self._client.start_stream_transcription(
                    language_code="en-US",
                    media_sample_rate_hz=48000,
                    media_encoding="pcm",
                )
                # this function basically convert the queue into a async generator
                async def input_generator():
                    try:
                        # Start transcription to generate our async stream
                        while True:
                            frame = await self._queue.get()
                            if frame is None:
                                break

                            # frame = frame.remix_and_resample(
                            #     self._sample_rate, self._num_channels
                            # )
                            await stream.input_stream.send_audio_event(audio_chunk=frame.data.tobytes())
                        await stream.input_stream.end_stream()
                    except Exception as e:
                        logger.error(f"an error occurred while streaming inputs: {e}")

                # try to connect
                asyncio.get_event_loop().create_task(input_generator())
                retry_count = 0  # connection successful, reset retry count

                await self._run_stream(stream)
            except Exception as e:
                if retry_count >= max_retry:
                    logger.error(
                        f"failed to connect to aws stt after {max_retry} tries",
                        exc_info=e,
                    )
                    break

                retry_delay = min(retry_count * 2, 5)  # max 5s
                retry_count += 1
                logger.warning(
                    f"aws stt connection failed, retrying in {retry_delay}s",
                    exc_info=e,
                )
                await asyncio.sleep(retry_delay)
            finally:
                self._event_queue.put_nowait(None)

    async def _run_stream(
        self, stream: TranscribeStreamingClient
    ):
        # Instantiate our handler and start processing events
        handler = TranscriptEventHandler(stream.output_stream, self._event_queue)
        await handler.handle_events()

    async def __anext__(self) -> stt.SpeechEvent:
        evt = await self._event_queue.get()
        if evt is None:
            raise StopAsyncIteration

        return evt

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
            event_queue: asyncio.Queue[stt.SpeechEvent | None]
        ):
        self._transcript_result_stream = transcript_result_stream
        self._event_queue = event_queue
        self._final_events: List[stt.SpeechEvent] = []
        self._need_bos = True
        self._need_eos = False

    async def handle_events(self):
        """Process generic incoming events from Amazon Transcribe
        and delegate to appropriate sub-handlers.
        """
        async for event in self._transcript_result_stream:
            if isinstance(event, TranscriptEvent):
                await self.handle_transcript_event(event)

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        # This handler can be implemented to handle transcriptions as needed.
        stream =  transcript_event.transcript.results
        for resp in stream:
            if (
                resp.start_time == 0.0
            ):
                if self._need_eos:
                    self._send_eos()

            if self._need_bos:
                self._send_bos()

            if (
                resp.end_time > 0.0
            ):
                if resp.is_partial:
                    iterim_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        alternatives=[
                            _streaming_recognize_response_to_speech_data(resp)
                        ],
                    )
                    self._event_queue.put_nowait(iterim_event)

                else:
                    final_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[
                            _streaming_recognize_response_to_speech_data(resp)
                        ],
                    )
                    self._final_events.append(final_event)
                    self._event_queue.put_nowait(final_event)

            if self._need_eos:
                self._send_eos()

            if (
                resp.is_partial == False
            ):
                self._need_eos = True

        if not self._need_bos:
            self._send_eos()

    def _send_bos(self) -> None:
        self._need_bos = False
        start_event = stt.SpeechEvent(
            type=stt.SpeechEventType.START_OF_SPEECH,
        )
        self._event_queue.put_nowait(start_event)

    def _send_eos(self) -> None:
        self._need_eos = False
        self._need_bos = True

        if self._final_events:
            lg = self._final_events[0].alternatives[0].language

            sentence = ""
            confidence = 0.0
            for alt in self._final_events:
                sentence += f"{alt.alternatives[0].text.strip()} "
                confidence += alt.alternatives[0].confidence

            sentence = sentence.rstrip()
            confidence /= len(self._final_events)  # avg. of confidence

            end_event = stt.SpeechEvent(
                type=stt.SpeechEventType.END_OF_SPEECH,
                alternatives=[
                    stt.SpeechData(
                        language=lg,
                        start_time=self._final_events[0].alternatives[0].start_time,
                        end_time=self._final_events[-1].alternatives[0].end_time,
                        confidence=confidence,
                        text=sentence,
                    )
                ],
            )

            self._final_events = []
            self._event_queue.put_nowait(end_event)
        else:
            end_event = stt.SpeechEvent(
                type=stt.SpeechEventType.END_OF_SPEECH,
                alternatives=[
                    stt.SpeechData(
                        language="",
                        start_time=0,
                        end_time=0,
                        confidence=0,
                        text="",
                    )
                ],
            )

            self._event_queue.put_nowait(end_event)
