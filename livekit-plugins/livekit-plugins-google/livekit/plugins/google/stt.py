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
import dataclasses
import time
import weakref
from dataclasses import dataclass
from typing import Callable, Union

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import DeadlineExceeded, GoogleAPICallError
from google.auth import default as gauth_default
from google.auth.exceptions import DefaultCredentialsError
from google.cloud.speech_v2 import SpeechAsyncClient
from google.cloud.speech_v2.types import cloud_speech
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)

from .log import logger
from .models import SpeechLanguages, SpeechModels

LgType = Union[SpeechLanguages, str]
LanguageCode = Union[LgType, list[LgType]]

# Google STT has a timeout of 5 mins, we'll attempt to restart the session
# before that timeout is reached
_max_session_duration = 240

# Google is very sensitive to background noise, so we'll ignore results with low confidence
_min_confidence = 0.65


# This class is only be used internally to encapsulate the options
@dataclass
class STTOptions:
    languages: list[LgType]
    detect_language: bool
    interim_results: bool
    punctuate: bool
    spoken_punctuation: bool
    model: SpeechModels | str
    sample_rate: int
    keywords: list[tuple[str, float]] | None

    def build_adaptation(self) -> cloud_speech.SpeechAdaptation | None:
        if self.keywords:
            return cloud_speech.SpeechAdaptation(
                phrase_sets=[
                    cloud_speech.SpeechAdaptation.AdaptationPhraseSet(
                        inline_phrase_set=cloud_speech.PhraseSet(
                            phrases=[
                                cloud_speech.PhraseSet.Phrase(value=keyword, boost=boost)
                                for keyword, boost in self.keywords
                            ]
                        )
                    )
                ]
            )
        return None


class STT(stt.STT):
    def __init__(
        self,
        *,
        languages: LanguageCode = "en-US",  # Google STT can accept multiple languages
        detect_language: bool = True,
        interim_results: bool = True,
        punctuate: bool = True,
        spoken_punctuation: bool = False,
        model: SpeechModels | str = "latest_long",
        location: str = "global",
        sample_rate: int = 16000,
        credentials_info: dict | None = None,
        credentials_file: str | None = None,
        keywords: list[tuple[str, float]] | None = None,
    ):
        """
        Create a new instance of Google STT.

        Credentials must be provided, either by using the ``credentials_info`` dict, or reading
        from the file specified in ``credentials_file`` or via Application Default Credentials as
        described in https://cloud.google.com/docs/authentication/application-default-credentials

        args:
            languages(LanguageCode): list of language codes to recognize (default: "en-US")
            detect_language(bool): whether to detect the language of the audio (default: True)
            interim_results(bool): whether to return interim results (default: True)
            punctuate(bool): whether to punctuate the audio (default: True)
            spoken_punctuation(bool): whether to use spoken punctuation (default: False)
            model(SpeechModels): the model to use for recognition default: "latest_long"
            location(str): the location to use for recognition default: "global"
            sample_rate(int): the sample rate of the audio default: 16000
            credentials_info(dict): the credentials info to use for recognition (default: None)
            credentials_file(str): the credentials file to use for recognition (default: None)
            keywords(List[tuple[str, float]]): list of keywords to recognize (default: None)
        """
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))

        self._location = location
        self._credentials_info = credentials_info
        self._credentials_file = credentials_file

        if credentials_file is None and credentials_info is None:
            try:
                gauth_default()
            except DefaultCredentialsError:
                raise ValueError(
                    "Application default credentials must be available "
                    "when using Google STT without explicitly passing "
                    "credentials through credentials_info or credentials_file."
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
            sample_rate=sample_rate,
            keywords=keywords,
        )
        self._streams = weakref.WeakSet[SpeechStream]()
        self._pool = utils.ConnectionPool[SpeechAsyncClient](
            max_session_duration=_max_session_duration,
            connect_cb=self._create_client,
        )

    async def _create_client(self) -> SpeechAsyncClient:
        # Add support for passing a specific location that matches recognizer
        # see: https://cloud.google.com/speech-to-text/v2/docs/speech-to-text-supported-languages
        client_options = None
        client: SpeechAsyncClient | None = None
        if self._location != "global":
            client_options = ClientOptions(api_endpoint=f"{self._location}-speech.googleapis.com")
        if self._credentials_info:
            client = SpeechAsyncClient.from_service_account_info(
                self._credentials_info,
                client_options=client_options,
            )
        elif self._credentials_file:
            client = SpeechAsyncClient.from_service_account_file(
                self._credentials_file,
                client_options=client_options,
            )
        else:
            client = SpeechAsyncClient(
                client_options=client_options,
            )
        assert client is not None
        return client

    def _get_recognizer(self, client: SpeechAsyncClient) -> str:
        # TODO(theomonnom): should we use recognizers?
        # recognizers may improve latency https://cloud.google.com/speech-to-text/v2/docs/recognizers#understand_recognizers

        # TODO(theomonnom): find a better way to access the project_id
        try:
            project_id = client.transport._credentials.project_id  # type: ignore
        except AttributeError:
            from google.auth import default as ga_default

            _, project_id = ga_default()
        return f"projects/{project_id}/locations/{self._location}/recognizers/_"

    def _sanitize_options(self, *, language: str | None = None) -> STTOptions:
        config = dataclasses.replace(self._config)

        if language:
            config.languages = [language]

        if not isinstance(config.languages, list):
            config.languages = [config.languages]
        elif not config.detect_language:
            if len(config.languages) > 1:
                logger.warning("multiple languages provided, but language detection is disabled")
            config.languages = [config.languages[0]]

        return config

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: SpeechLanguages | str | None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        config = self._sanitize_options(language=language)
        frame = rtc.combine_audio_frames(buffer)

        config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=frame.sample_rate,
                audio_channel_count=frame.num_channels,
            ),
            adaptation=config.build_adaptation(),
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=config.punctuate,
                enable_spoken_punctuation=config.spoken_punctuation,
                enable_word_time_offsets=True,
            ),
            model=config.model,
            language_codes=config.languages,
        )

        try:
            async with self._pool.connection() as client:
                raw = await client.recognize(
                    cloud_speech.RecognizeRequest(
                        recognizer=self._get_recognizer(client),
                        config=config,
                        content=frame.data.tobytes(),
                    ),
                    timeout=conn_options.timeout,
                )

                return _recognize_response_to_speech_event(raw)
        except DeadlineExceeded:
            raise APITimeoutError()
        except GoogleAPICallError as e:
            raise APIStatusError(
                e.message,
                status_code=e.code or -1,
            )
        except Exception as e:
            raise APIConnectionError() from e

    def stream(
        self,
        *,
        language: SpeechLanguages | str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        config = self._sanitize_options(language=language)
        stream = SpeechStream(
            stt=self,
            pool=self._pool,
            recognizer_cb=self._get_recognizer,
            config=config,
            conn_options=conn_options,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        languages: LanguageCode | None = None,
        detect_language: bool | None = None,
        interim_results: bool | None = None,
        punctuate: bool | None = None,
        spoken_punctuation: bool | None = None,
        model: SpeechModels | None = None,
        location: str | None = None,
        keywords: list[tuple[str, float]] | None = None,
    ):
        if languages is not None:
            if isinstance(languages, str):
                languages = [languages]
            self._config.languages = languages
        if detect_language is not None:
            self._config.detect_language = detect_language
        if interim_results is not None:
            self._config.interim_results = interim_results
        if punctuate is not None:
            self._config.punctuate = punctuate
        if spoken_punctuation is not None:
            self._config.spoken_punctuation = spoken_punctuation
        if model is not None:
            self._config.model = model
        if location is not None:
            self._location = location
            # if location is changed, fetch a new client and recognizer as per the new location
            self._pool.invalidate()
        if keywords is not None:
            self._config.keywords = keywords

        for stream in self._streams:
            stream.update_options(
                languages=languages,
                detect_language=detect_language,
                interim_results=interim_results,
                punctuate=punctuate,
                spoken_punctuation=spoken_punctuation,
                model=model,
                keywords=keywords,
            )

    async def aclose(self) -> None:
        await self._pool.aclose()
        await super().aclose()


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        pool: utils.ConnectionPool[SpeechAsyncClient],
        recognizer_cb: Callable[[SpeechAsyncClient], str],
        config: STTOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=config.sample_rate)

        self._pool = pool
        self._recognizer_cb = recognizer_cb
        self._config = config
        self._reconnect_event = asyncio.Event()
        self._session_connected_at: float = 0

    def update_options(
        self,
        *,
        languages: LanguageCode | None = None,
        detect_language: bool | None = None,
        interim_results: bool | None = None,
        punctuate: bool | None = None,
        spoken_punctuation: bool | None = None,
        model: SpeechModels | None = None,
        keywords: list[tuple[str, float]] | None = None,
    ):
        if languages is not None:
            if isinstance(languages, str):
                languages = [languages]
            self._config.languages = languages
        if detect_language is not None:
            self._config.detect_language = detect_language
        if interim_results is not None:
            self._config.interim_results = interim_results
        if punctuate is not None:
            self._config.punctuate = punctuate
        if spoken_punctuation is not None:
            self._config.spoken_punctuation = spoken_punctuation
        if model is not None:
            self._config.model = model
        if keywords is not None:
            self._config.keywords = keywords

        self._reconnect_event.set()

    async def _run(self) -> None:
        # google requires a async generator when calling streaming_recognize
        # this function basically convert the queue into a async generator
        async def input_generator(client: SpeechAsyncClient, should_stop: asyncio.Event):
            try:
                # first request should contain the config
                yield cloud_speech.StreamingRecognizeRequest(
                    recognizer=self._recognizer_cb(client),
                    streaming_config=self._streaming_config,
                )

                async for frame in self._input_ch:
                    # when the stream is aborted due to reconnect, this input_generator
                    # needs to stop consuming frames
                    # when the generator stops, the previous gRPC stream will close
                    if should_stop.is_set():
                        return

                    if isinstance(frame, rtc.AudioFrame):
                        yield cloud_speech.StreamingRecognizeRequest(audio=frame.data.tobytes())

            except Exception:
                logger.exception("an error occurred while streaming input to google STT")

        async def process_stream(client: SpeechAsyncClient, stream):
            has_started = False
            async for resp in stream:
                if (
                    resp.speech_event_type
                    == cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_BEGIN
                ):
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                    )
                    has_started = True

                if (
                    resp.speech_event_type
                    == cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_EVENT_TYPE_UNSPECIFIED
                ):
                    result = resp.results[0]
                    speech_data = _streaming_recognize_response_to_speech_data(resp)
                    if speech_data is None:
                        continue

                    if not result.is_final:
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                alternatives=[speech_data],
                            )
                        )
                    else:
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                alternatives=[speech_data],
                            )
                        )
                        if time.time() - self._session_connected_at > _max_session_duration:
                            logger.debug(
                                "Google STT maximum connection time reached. Reconnecting..."
                            )
                            self._pool.remove(client)
                            if has_started:
                                self._event_ch.send_nowait(
                                    stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                                )
                                has_started = False
                            self._reconnect_event.set()
                            return

                if (
                    resp.speech_event_type
                    == cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_END
                ):
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                    )
                    has_started = False

        while True:
            try:
                async with self._pool.connection() as client:
                    self._streaming_config = cloud_speech.StreamingRecognitionConfig(
                        config=cloud_speech.RecognitionConfig(
                            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                                sample_rate_hertz=self._config.sample_rate,
                                audio_channel_count=1,
                            ),
                            adaptation=self._config.build_adaptation(),
                            language_codes=self._config.languages,
                            model=self._config.model,
                            features=cloud_speech.RecognitionFeatures(
                                enable_automatic_punctuation=self._config.punctuate,
                                enable_word_time_offsets=True,
                            ),
                        ),
                        streaming_features=cloud_speech.StreamingRecognitionFeatures(
                            interim_results=self._config.interim_results,
                        ),
                    )

                    should_stop = asyncio.Event()
                    stream = await client.streaming_recognize(
                        requests=input_generator(client, should_stop),
                    )
                    self._session_connected_at = time.time()

                    process_stream_task = asyncio.create_task(process_stream(client, stream))
                    wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())

                    try:
                        done, _ = await asyncio.wait(
                            [process_stream_task, wait_reconnect_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for task in done:
                            if task != wait_reconnect_task:
                                task.result()
                        if wait_reconnect_task not in done:
                            break
                        self._reconnect_event.clear()
                    finally:
                        await utils.aio.gracefully_cancel(process_stream_task, wait_reconnect_task)
                        should_stop.set()
            except DeadlineExceeded:
                raise APITimeoutError()
            except GoogleAPICallError as e:
                raise APIStatusError(
                    e.message,
                    status_code=e.code or -1,
                )
            except Exception as e:
                raise APIConnectionError() from e


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
) -> stt.SpeechData | None:
    text = ""
    confidence = 0.0
    for result in resp.results:
        if len(result.alternatives) == 0:
            continue
        text += result.alternatives[0].transcript
        confidence += result.alternatives[0].confidence

    confidence /= len(resp.results)
    lg = resp.results[0].language_code

    if confidence < _min_confidence:
        return None
    if text == "":
        return None

    data = stt.SpeechData(language=lg, start_time=0, end_time=0, confidence=confidence, text=text)

    return data
