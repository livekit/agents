# Copyright 2023 LiveKit, Inc.
#
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

from typing import Optional
from google.auth import credentials
from google.cloud.speech_v2 import SpeechAsyncClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.gapic_v1.client_info import ClientInfo
from livekit import rtc, agents
from livekit.agents import stt
from typing import Union
from dataclasses import dataclass
from .models import SpeechModels, SpeechLanguages
import asyncio
import logging


@dataclass
class StreamOptions(stt.StreamOptions):
    model: SpeechModels = "long"
    language: Union[SpeechLanguages, str] = "en-US"


@dataclass
class RecognizeOptions(stt.RecognizeOptions):
    model: SpeechModels = "long"
    language: Union[SpeechLanguages, str] = "en-US"


class STT(stt.STT):
    def __init__(
        self,
        *,
        credentials_info: Optional[dict] = None,
        credentials_file: Optional[str] = None,
    ):
        """
        if no credentials is provided, it will use the credentials on the environment
        GOOGLE_APPLICATION_CREDENTIALS (Default behavior of Google SpeechAsyncClient)
        """
        super().__init__(streaming_supported=True)

        if credentials_info:
            self._client = SpeechAsyncClient.from_service_account_info(credentials_info)
        elif credentials_file:
            self._client = SpeechAsyncClient.from_service_account_file(credentials_file)
        else:
            self._client = SpeechAsyncClient()

        self._creds = self._client.transport._credentials

    @property
    def _recognizer(self) -> str:
        # TODO(theomonnom): should we use recognizers?
        # Recognizers may improve latency https://cloud.google.com/speech-to-text/v2/docs/recognizers#understand_recognizers
        return f"projects/{self._creds.project_id}/locations/global/recognizers/_"  # type: ignore

    async def recognize(
        self, buffer: agents.AudioBuffer, opts: RecognizeOptions = RecognizeOptions()
    ) -> stt.SpeechEvent:
        buffer = agents.utils.merge_frames(buffer)
        config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=buffer.sample_rate,
                audio_channel_count=buffer.num_channels,
            ),
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=opts.punctuate,
            ),
            language_codes=[opts.language],
            model=opts.model,
        )

        resp = await self._client.recognize(
            cloud_speech.RecognizeRequest(
                recognizer=self._recognizer,
                config=config,
                content=buffer.data.tobytes(),
            )
        )
        return recognize_response_to_speech_event(opts, resp)

    def stream(self, opts: StreamOptions = StreamOptions()) -> "SpeechStream":
        return SpeechStream(opts, self._client, self._creds, self._recognizer)


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        opts: StreamOptions,
        client: SpeechAsyncClient,
        creds: credentials.Credentials,
        recognizer: str,
    ) -> None:
        super().__init__()
        self._opts = opts
        self._client = client
        self._creds = creds
        self._recognizer = recognizer
        self._queue = asyncio.Queue[rtc.AudioFrame]()
        self._transcript_queue = asyncio.Queue[stt.SpeechEvent]()
        self._closed = False

        self._main_task = asyncio.create_task(self._run(max_retry=32))

        def log_exception(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                logging.error(f"google speech task failed: {task.exception()}")

        self._main_task.add_done_callback(log_exception)

    async def _run(self, max_retry: int) -> None:
        """Try to connect to Google Speech API with exponential backoff and forward frames"""
        retry_count = 0
        while True:
            try:
                config = cloud_speech.StreamingRecognitionConfig(
                    config=cloud_speech.RecognitionConfig(
                        explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                            encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                            sample_rate_hertz=self._opts.sample_rate,
                            audio_channel_count=self._opts.num_channels,
                        ),
                        language_codes=[self._opts.language],
                        model=self._opts.model,
                        features=cloud_speech.RecognitionFeatures(
                            enable_automatic_punctuation=self._opts.punctuate,
                        ),
                    ),
                    streaming_features=cloud_speech.StreamingRecognitionFeatures(
                        interim_results=True,
                    ),
                )

                # convert our input queue to a generator (needed by the Google Speech client in Python)
                async def input_gen():
                    nonlocal config
                    try:
                        yield cloud_speech.StreamingRecognizeRequest(
                            recognizer=self._recognizer,
                            streaming_config=config,
                        )
                        while True:
                            frame = await self._queue.get()
                            frame = frame.remix_and_resample(
                                self._opts.sample_rate, self._opts.num_channels
                            )
                            yield cloud_speech.StreamingRecognizeRequest(
                                recognizer=self._recognizer,
                                streaming_config=config,
                                audio=frame.data.tobytes(),
                            )
                            self._queue.task_done()
                    except Exception as e:
                        logging.error(f"an error occurred while streaming inputs: {e}")

                stream = await self._client.streaming_recognize(requests=input_gen())
                retry_count = 0

                async for resp in stream:
                    self._transcript_queue.put_nowait(
                        streaming_recognize_response_to_speech_event(self._opts, resp)
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                if retry_count > max_retry and max_retry > 0:
                    logging.error(f"failed to connect to Google Speech: {e}")
                    break

                retry_delay = 2**retry_count - 1
                retry_count += 1
                logging.warning(
                    f"failed to connect to Google Speech: {e} - retrying in {retry_delay}s"
                )
                await asyncio.sleep(retry_delay)

        self._closed = True

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            raise ValueError("cannot push frame to closed stream")

        self._queue.put_nowait(frame)

    async def flush(self) -> None:
        await self._queue.join()

    async def close(self) -> None:
        self._main_task.cancel()
        try:
            await self._main_task
        except asyncio.CancelledError:
            pass

    def __aiter__(self) -> "SpeechStream":
        return self

    async def __anext__(self) -> stt.SpeechEvent:
        if self._closed and self._transcript_queue.empty():
            raise StopAsyncIteration

        return await self._transcript_queue.get()


def recognize_response_to_speech_event(
    opts: RecognizeOptions, resp: cloud_speech.RecognizeResponse
) -> stt.SpeechEvent:
    result = resp.results[0]
    gg_alts = result.alternatives
    return stt.SpeechEvent(
        is_final=True,
        alternatives=[
            stt.SpeechData(
                language=result.language_code,
                start_time=alt.words[0].start_offset.seconds if alt.words else 0,
                end_time=alt.words[-1].end_offset.seconds if alt.words else 0,
                confidence=alt.confidence,
                text=alt.transcript,
            )
            for alt in gg_alts
        ],
    )


def streaming_recognize_response_to_speech_event(
    opts: RecognizeOptions, resp: cloud_speech.StreamingRecognizeResponse
) -> stt.SpeechEvent:
    result = resp.results[0]
    gg_alts = result.alternatives
    return stt.SpeechEvent(
        is_final=result.is_final,
        alternatives=[
            stt.SpeechData(
                language=result.language_code,
                start_time=alt.words[0].start_offset.seconds if alt.words else 0,
                end_time=alt.words[-1].end_offset.seconds if alt.words else 0,
                confidence=alt.confidence,
                text=alt.transcript,
            )
            for alt in gg_alts
        ],
    )
