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

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import os
from dataclasses import dataclass
from typing import AsyncIterable, List, Optional, Union

from livekit import agents, rtc
from livekit.agents import stt
from livekit.agents.utils import AudioBuffer

from google.cloud.speech_v2 import SpeechAsyncClient
from google.cloud.speech_v2.types import cloud_speech

from .log import logger
from .models import SpeechLanguages, SpeechModels

LgType = Union[SpeechLanguages, str]
LanguageCode = Union[LgType, List[LgType]]


# This class is only be used internally to encapsulate the options
@dataclass
class STTOptions:
    languages: List[LgType]
    detect_language: bool
    interim_results: bool
    punctuate: bool
    spoken_punctuation: bool
    model: SpeechModels


class STT(stt.STT):
    def __init__(
        self,
        *,
        languages: LanguageCode = "en-US",  # Google STT can accept multiple languages
        detect_language: bool = True,
        interim_results: bool = True,
        punctuate: bool = True,
        spoken_punctuation: bool = True,
        model: SpeechModels = "long",
        credentials_info: dict | None = None,
        credentials_file: str | None = None,
    ):
        """
        if no credentials is provided, it will use the credentials on the environment
        GOOGLE_APPLICATION_CREDENTIALS (default behavior of Google SpeechAsyncClient)
        """
        super().__init__(streaming_supported=True)

        self._client: SpeechAsyncClient | None = None
        self._credentials_info = credentials_info
        self._credentials_file = credentials_file

        if credentials_file is None and credentials_info is None:
            creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if not creds:
                raise ValueError(
                    "GOOGLE_APPLICATION_CREDENTIALS must be set if no credentials is provided"
                )

        if isinstance(languages, str):
            languages = [languages]

        self._config = STTOptions(
            languages=languages,
            detect_language=detect_language,
            interim_results=interim_results,
            punctuate=punctuate,
            spoken_punctuation=spoken_punctuation,
            model=model,
        )

    def _ensure_client(self) -> SpeechAsyncClient:
        if self._credentials_info:
            self._client = SpeechAsyncClient.from_service_account_info(
                self._credentials_info
            )
        elif self._credentials_file:
            self._client = SpeechAsyncClient.from_service_account_file(
                self._credentials_file
            )
        else:
            self._client = SpeechAsyncClient()

        assert self._client is not None
        return self._client

    @property
    def _recognizer(self) -> str:
        # TODO(theomonnom): should we use recognizers?
        # recognizers may improve latency https://cloud.google.com/speech-to-text/v2/docs/recognizers#understand_recognizers

        # TODO(theomonnom): find a better way to access the project_id
        project_id = self._ensure_client().transport._credentials.project_id  # type: ignore
        return f"projects/{project_id}/locations/global/recognizers/_"

    def _sanitize_options(
        self,
        *,
        language: str | None = None,
    ) -> STTOptions:
        config = dataclasses.replace(self._config)

        if language:
            config.languages = [language]

        if not isinstance(config.languages, list):
            config.languages = [config.languages]
        elif not config.detect_language:
            if len(config.languages) > 1:
                logger.warning(
                    "multiple languages provided, but language detection is disabled"
                )
            config.languages = [config.languages[0]]

        return config

    async def recognize(
        self,
        *,
        buffer: AudioBuffer,
        language: SpeechLanguages | str | None = None,
    ) -> stt.SpeechEvent:
        config = self._sanitize_options(language=language)
        frame = agents.utils.merge_frames(buffer)

        config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=frame.sample_rate,
                audio_channel_count=frame.num_channels,
            ),
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=config.punctuate,
                enable_spoken_punctuation=config.spoken_punctuation,
                enable_word_time_offsets=True,
            ),
            model=config.model,
            language_codes=config.languages,
        )

        raw = await self._ensure_client().recognize(
            cloud_speech.RecognizeRequest(
                recognizer=self._recognizer,
                config=config,
                content=frame.data.tobytes(),
            )
        )
        return _recognize_response_to_speech_event(raw)

    def stream(
        self,
        *,
        language: SpeechLanguages | str | None = None,
    ) -> "SpeechStream":
        config = self._sanitize_options(language=language)
        return SpeechStream(
            self._ensure_client(),
            self._recognizer,
            config,
        )


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        client: SpeechAsyncClient,
        recognizer: str,
        config: STTOptions,
        sample_rate: int = 48000,
        num_channels: int = 1,
        max_retry: int = 32,
    ) -> None:
        super().__init__()

        self._client = client
        self._recognizer = recognizer
        self._config = config
        self._sample_rate = sample_rate
        self._num_channels = num_channels

        self._queue = asyncio.Queue[Optional[rtc.AudioFrame]]()
        self._event_queue = asyncio.Queue[Optional[stt.SpeechEvent]]()
        self._closed = False
        self._main_task = asyncio.create_task(self._run(max_retry=max_retry))

        self._final_events: List[stt.SpeechEvent] = []
        self._need_bos = True
        self._need_eos = False

        self._streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=cloud_speech.RecognitionConfig(
                explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                    encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self._sample_rate,
                    audio_channel_count=self._num_channels,
                ),
                language_codes=self._config.languages,
                model=self._config.model,
                features=cloud_speech.RecognitionFeatures(
                    enable_automatic_punctuation=self._config.punctuate,
                    enable_word_time_offsets=True,
                ),
            ),
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                enable_voice_activity_events=True,
                interim_results=self._config.interim_results,
            ),
        )

        def log_exception(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                logger.error(f"google stt task failed: {task.exception()}")

        self._main_task.add_done_callback(log_exception)

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            raise ValueError("cannot push frame to closed stream")

        self._queue.put_nowait(frame)

    async def aclose(self, *, wait: bool = True) -> None:
        self._closed = True
        if not wait:
            self._main_task.cancel()

        self._queue.put_nowait(None)
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task

    async def _run(self, max_retry: int) -> None:
        retry_count = 0
        while not self._closed:
            try:
                # google requires a async generator when calling streaming_recognize
                # this function basically convert the queue into a async generator
                async def input_generator():
                    try:
                        # first request should contain the config
                        yield cloud_speech.StreamingRecognizeRequest(
                            recognizer=self._recognizer,
                            streaming_config=self._streaming_config,
                        )
                        while True:
                            frame = await self._queue.get()
                            if frame is None:
                                break

                            frame = frame.remix_and_resample(
                                self._sample_rate, self._num_channels
                            )
                            yield cloud_speech.StreamingRecognizeRequest(
                                audio=frame.data.tobytes(),
                            )
                    except Exception as e:
                        logger.error(f"an error occurred while streaming inputs: {e}")

                # try to connect
                stream = await self._client.streaming_recognize(
                    requests=input_generator()
                )
                retry_count = 0  # connection successful, reset retry count

                await self._run_stream(stream)
            except Exception as e:
                if retry_count >= max_retry:
                    logger.error(
                        f"failed to connect to google stt after {max_retry} tries",
                        exc_info=e,
                    )
                    break

                retry_delay = min(retry_count * 2, 5)  # max 5s
                retry_count += 1
                logger.warning(
                    f"google stt connection failed, retrying in {retry_delay}s",
                    exc_info=e,
                )
                await asyncio.sleep(retry_delay)

        self._event_queue.put_nowait(None)

    async def _run_stream(
        self, stream: AsyncIterable[cloud_speech.StreamingRecognizeResponse]
    ):
        async for resp in stream:
            if (
                resp.speech_event_type
                == cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_BEGIN
            ):
                if self._need_eos:
                    self._send_eos()

            if self._need_bos:
                self._send_bos()

            if (
                resp.speech_event_type
                == cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_EVENT_TYPE_UNSPECIFIED
            ):
                result = resp.results[0]
                if not result.is_final:
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
                resp.speech_event_type
                == cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_END
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

    async def __anext__(self) -> stt.SpeechEvent:
        evt = await self._event_queue.get()
        if evt is None:
            raise StopAsyncIteration

        return evt


def _recognize_response_to_speech_event(
    resp: cloud_speech.RecognizeResponse,
) -> stt.SpeechEvent:
    text = ""
    confidence = 0.0
    for result in resp.results:
        text += result.alternatives[0].transcript
        confidence += result.alternatives[0].confidence

    # not sure why start_offset and end_offset returns a timedelta
    start_offset = resp.results[0].alternatives[0].words[0].start_offset
    end_offset = resp.results[-1].alternatives[0].words[-1].end_offset

    confidence /= len(resp.results)
    lg = resp.results[0].language_code
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            stt.SpeechData(
                language=lg,
                start_time=start_offset.total_seconds(),  # type: ignore
                end_time=end_offset.total_seconds(),  # type: ignore
                confidence=confidence,
                text=text,
            )
        ],
    )


def _streaming_recognize_response_to_speech_data(
    resp: cloud_speech.StreamingRecognizeResponse,
) -> stt.SpeechData:
    text = ""
    confidence = 0.0
    for result in resp.results:
        text += result.alternatives[0].transcript
        confidence += result.alternatives[0].confidence

    confidence /= len(resp.results)
    lg = resp.results[0].language_code

    data = stt.SpeechData(
        language=lg,
        start_time=0,
        end_time=0,
        confidence=confidence,
        text=text,
    )

    return data
