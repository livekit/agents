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
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterable, Dict, List

from livekit import agents, rtc
from livekit.agents import stt
from livekit.agents.utils import AudioBuffer

from google.auth import credentials  # type: ignore
from google.cloud.speech_v2 import SpeechAsyncClient
from google.cloud.speech_v2.types import cloud_speech

from .models import SpeechLanguages, SpeechModels

LgType = SpeechLanguages | str
LanguageCode = LgType | List[LgType]


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
        credentials_info: Dict[str, Any] | None = None,
        credentials_file: str | None = None,
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
        self._creds = self._client.transport._credentials

    @property
    def _recognizer(self) -> str:
        # TODO(theomonnom): should we use recognizers?
        # Recognizers may improve latency https://cloud.google.com/speech-to-text/v2/docs/recognizers#understand_recognizers
        return f"projects/{self._creds.project_id}/locations/global/recognizers/_"  # type: ignore

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
                logging.warning(
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
        buffer = agents.utils.merge_frames(buffer)

        config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=buffer.sample_rate,
                audio_channel_count=buffer.num_channels,
            ),
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=config.punctuate,
                enable_spoken_punctuation=config.spoken_punctuation,
            ),
            model=config.model,
            language_codes=config.languages,
        )

        return recognize_response_to_speech_event(
            await self._client.recognize(
                cloud_speech.RecognizeRequest(
                    recognizer=self._recognizer,
                    config=config,
                    content=buffer.data.tobytes(),
                )
            )
        )

    def stream(
        self,
        *,
        language: SpeechLanguages | str | None = None,
    ) -> "SpeechStream":
        config = self._sanitize_options(language=language)
        return SpeechStream(
            self._client,
            self._creds,
            self._recognizer,
            config,
        )


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        client: SpeechAsyncClient,
        creds: credentials.Credentials,
        recognizer: str,
        config: STTOptions,
        sample_rate: int = 24000,
        num_channels: int = 1,
        max_retry: int = 32,
    ) -> None:
        super().__init__()

        self._client = client
        self._creds = creds
        self._recognizer = recognizer
        self._config = config
        self._sample_rate = sample_rate
        self._num_channels = num_channels

        self._queue = asyncio.Queue[rtc.AudioFrame | None]()
        self._event_queue = asyncio.Queue[stt.SpeechEvent | None]()
        self._closed = False
        self._main_task = asyncio.create_task(self._run(max_retry=max_retry))

        self._final_events: List[stt.SpeechEvent] = []
        self._speaking = False

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
                ),
            ),
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                enable_voice_activity_events=True,
                interim_results=self._config.interim_results,
            ),
        )

        def log_exception(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                logging.error(f"google stt task failed: {task.exception()}")

        self._main_task.add_done_callback(log_exception)

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            raise ValueError("cannot push frame to closed stream")

        self._queue.put_nowait(frame)

    async def aclose(self, wait: bool = True) -> None:
        self._closed = True
        if not wait:
            self._main_task.cancel()

        self._queue.put_nowait(None)
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task

    async def _run(self, max_retry: int) -> None:
        retry_count = 0
        try:
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
                                frame = (
                                    await self._queue.get()
                                )  # wait for a new rtc.AudioFrame
                                if frame is None:
                                    break  # None is sent inside aclose

                                self._queue.task_done()
                                frame = frame.remix_and_resample(
                                    self._sample_rate, self._num_channels
                                )
                                yield cloud_speech.StreamingRecognizeRequest(
                                    audio=frame.data.tobytes(),
                                )
                        except Exception as e:
                            logging.error(
                                f"an error occurred while streaming inputs: {e}"
                            )

                    # try to connect
                    stream = await self._client.streaming_recognize(
                        requests=input_generator()
                    )
                    retry_count = 0  # connection successful, reset retry count

                    await self._run_stream(stream)
                except Exception as e:
                    if retry_count >= max_retry:
                        logging.error(
                            f"failed to connect to google stt after {max_retry} tries",
                            exc_info=e,
                        )
                        break

                    retry_delay = min(retry_count * 2, 10)  # max 10s
                    retry_count += 1
                    logging.warning(
                        f"google stt connection failed, retrying in {retry_delay}s",
                        exc_info=e,
                    )
                    await asyncio.sleep(retry_delay)
        finally:
            self._event_queue.put_nowait(None)

    async def _run_stream(
        self, stream: AsyncIterable[cloud_speech.StreamingRecognizeResponse]
    ):
        async for resp in stream:
            if (
                resp.speech_event_type
                == cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_BEGIN
            ):
                self._speaking = True
                start_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.START_OF_SPEECH,
                )
                self._event_queue.put_nowait(start_event)

            if (
                resp.speech_event_type
                == cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_EVENT_TYPE_UNSPECIFIED
            ):
                result = resp.results[0]
                if not result.is_final:
                    # interim results
                    iterim_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        alternatives=streaming_recognize_response_to_speech_data(resp),
                    )
                    self._event_queue.put_nowait(iterim_event)

                else:
                    final_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=streaming_recognize_response_to_speech_data(resp),
                    )
                    self._final_events.append(final_event)
                    self._event_queue.put_nowait(final_event)

                    if not self._speaking:
                        # With Google STT, we receive the final event after the END_OF_SPEECH event
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
                                    language=result.language_code,
                                    start_time=self._final_events[0]
                                    .alternatives[0]
                                    .start_time,
                                    end_time=self._final_events[-1]
                                    .alternatives[0]
                                    .end_time,
                                    confidence=confidence,
                                    text=sentence,
                                )
                            ],
                        )

                        self._final_events = []
                        self._event_queue.put_nowait(end_event)

            if (
                resp.speech_event_type
                == cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_END
            ):
                self._speaking = False

    async def __anext__(self) -> stt.SpeechEvent:
        evt = await self._event_queue.get()
        if evt is None:
            raise StopAsyncIteration

        return evt


def recognize_response_to_speech_event(
    resp: cloud_speech.RecognizeResponse,
) -> stt.SpeechEvent:
    result = resp.results[0]
    gg_alts = result.alternatives
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
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


def streaming_recognize_response_to_speech_data(
    resp: cloud_speech.StreamingRecognizeResponse,
) -> List[stt.SpeechData]:
    result = resp.results[0]
    gg_alts = result.alternatives
    return [
        stt.SpeechData(
            language=result.language_code,
            start_time=alt.words[0].start_offset.seconds if alt.words else 0,
            end_time=alt.words[-1].end_offset.seconds if alt.words else 0,
            confidence=alt.confidence,
            text=alt.transcript,
        )
        for alt in gg_alts
    ]
