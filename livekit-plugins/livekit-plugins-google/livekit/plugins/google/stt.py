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
from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Union, cast, get_args

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import DeadlineExceeded, GoogleAPICallError
from google.auth import default as gauth_default
from google.auth.exceptions import DefaultCredentialsError
from google.cloud.speech_v1 import SpeechAsyncClient as SpeechAsyncClientV1
from google.cloud.speech_v1.types import cloud_speech as cloud_speech_v1, resource as resource_v1
from google.cloud.speech_v2 import SpeechAsyncClient as SpeechAsyncClientV2
from google.cloud.speech_v2.types import cloud_speech as cloud_speech_v2
from google.protobuf.duration_pb2 import Duration
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
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TimedString

from .log import logger
from .models import SpeechLanguages, SpeechModels, SpeechModelsV2

LgType = Union[SpeechLanguages, str]
LanguageCode = Union[LgType, list[LgType]]

# Google STT has a timeout of 5 mins, we'll attempt to restart the session
# before that timeout is reached
_max_session_duration = 240

# Google is very sensitive to background noise, so we'll ignore results with low confidence
_default_min_confidence = 0.65


# This class is only be used internally to encapsulate the options
@dataclass
class STTOptions:
    languages: list[LgType]
    detect_language: bool
    interim_results: bool
    punctuate: bool
    spoken_punctuation: bool
    enable_word_time_offsets: bool
    enable_word_confidence: bool
    enable_voice_activity_events: bool
    model: SpeechModels | str
    sample_rate: int
    min_confidence_threshold: float
    profanity_filter: bool
    denoise_audio: NotGivenOr[bool] = NOT_GIVEN
    snr_threshold: NotGivenOr[float] = NOT_GIVEN
    keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN

    @property
    def version(self) -> int:
        return 2 if self.model in get_args(SpeechModelsV2) else 1

    def build_adaptation(
        self,
    ) -> cloud_speech_v2.SpeechAdaptation | resource_v1.SpeechAdaptation | None:
        if is_given(self.keywords):
            if self.version == 2:
                return cloud_speech_v2.SpeechAdaptation(
                    phrase_sets=[
                        cloud_speech_v2.SpeechAdaptation.AdaptationPhraseSet(
                            inline_phrase_set=cloud_speech_v2.PhraseSet(
                                phrases=[
                                    cloud_speech_v2.PhraseSet.Phrase(value=keyword, boost=boost)
                                    for keyword, boost in self.keywords
                                ]
                            )
                        )
                    ]
                )
            return resource_v1.SpeechAdaptation(
                phrase_sets=[
                    resource_v1.PhraseSet(
                        name="keywords",
                        phrases=[
                            resource_v1.PhraseSet.Phrase(value=keyword, boost=boost)
                            for keyword, boost in self.keywords
                        ],
                    )
                ]
            )
        return None

    def build_denoiser_config(self) -> cloud_speech_v2.DenoiserConfig | None:
        """Build DenoiserConfig for V2 API only. Returns None for V1 API."""
        if self.version != 2:
            return None

        if not is_given(self.denoise_audio) and not is_given(self.snr_threshold):
            return None

        kwargs: dict[str, bool | float] = {}
        if is_given(self.denoise_audio):
            kwargs["denoise_audio"] = self.denoise_audio
        if is_given(self.snr_threshold):
            kwargs["snr_threshold"] = self.snr_threshold

        return cloud_speech_v2.DenoiserConfig(**kwargs)


class STT(stt.STT):
    def __init__(
        self,
        *,
        languages: LanguageCode = "en-US",  # Google STT can accept multiple languages
        detect_language: bool = True,
        interim_results: bool = True,
        punctuate: bool = True,
        spoken_punctuation: bool = False,
        enable_word_time_offsets: NotGivenOr[bool] = NOT_GIVEN,
        enable_word_confidence: bool = False,
        enable_voice_activity_events: bool = False,
        model: SpeechModels | str = "latest_long",
        location: str = "global",
        profanity_filter: bool = False,
        sample_rate: int = 16000,
        min_confidence_threshold: float = _default_min_confidence,
        credentials_info: NotGivenOr[dict] = NOT_GIVEN,
        credentials_file: NotGivenOr[str] = NOT_GIVEN,
        keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN,
        use_streaming: NotGivenOr[bool] = NOT_GIVEN,
        denoise_audio: NotGivenOr[bool] = NOT_GIVEN,
        snr_threshold: NotGivenOr[float] = NOT_GIVEN,
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
            enable_word_time_offsets(bool): whether to enable word time offsets (default: None)
            enable_word_confidence(bool): whether to enable word confidence (default: False)
            enable_voice_activity_events(bool): whether to enable voice activity events (default: False)
            model(SpeechModels): the model to use for recognition default: "latest_long"
            location(str): the location to use for recognition default: "global"
            profanity_filter(bool): whether to filter out profanities default: False
            sample_rate(int): the sample rate of the audio default: 16000
            min_confidence_threshold(float): minimum confidence threshold for recognition
            (default: 0.65)
            credentials_info(dict): the credentials info to use for recognition (default: None)
            credentials_file(str): the credentials file to use for recognition (default: None)
            keywords(List[tuple[str, float]]): list of keywords to recognize (default: None)
            use_streaming(bool): whether to use streaming for recognition (default: True)
            denoise_audio(bool): whether to enable audio denoising to reduce background noise
            (default: None). Setting to True can help reduce background music or noises like
            rain and street traffic. Note: cannot remove background human voices.
            snr_threshold(float): signal-to-noise ratio threshold to control the minimum
            loudness of speech required for transcription. This helps filter out non-speech
            audio or background noise. A higher value means the user needs to speak louder
            for transcription. Recommended values: 10.0-100.0 when denoise_audio=True,
            0.5-5.0 when denoise_audio=False. (default: None)
        """
        if not is_given(use_streaming):
            use_streaming = True

        if model == "chirp_3":
            if is_given(enable_word_time_offsets) and enable_word_time_offsets:
                logger.warning(
                    "Chirp 3 does not support word timestamps, setting 'enable_word_time_offsets' to False."
                )
            enable_word_time_offsets = False
        elif is_given(enable_word_time_offsets):
            enable_word_time_offsets = enable_word_time_offsets
        else:
            enable_word_time_offsets = True

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=use_streaming,
                interim_results=True,
                aligned_transcript="word" if enable_word_time_offsets and use_streaming else False,
            )
        )

        self._location = location
        self._credentials_info = credentials_info
        self._credentials_file = credentials_file

        if not is_given(credentials_file) and not is_given(credentials_info):
            try:
                gauth_default()  # type: ignore
            except DefaultCredentialsError:
                raise ValueError(
                    "Application default credentials must be available "
                    "when using Google STT without explicitly passing "
                    "credentials through credentials_info or credentials_file."
                ) from None

        if isinstance(languages, str):
            languages = [languages]

        self._config = STTOptions(
            languages=languages,
            detect_language=detect_language,
            interim_results=interim_results,
            punctuate=punctuate,
            spoken_punctuation=spoken_punctuation,
            enable_word_time_offsets=enable_word_time_offsets,
            enable_word_confidence=enable_word_confidence,
            enable_voice_activity_events=enable_voice_activity_events,
            model=model,
            profanity_filter=profanity_filter,
            sample_rate=sample_rate,
            min_confidence_threshold=min_confidence_threshold,
            denoise_audio=denoise_audio,
            snr_threshold=snr_threshold,
            keywords=keywords,
        )
        self._streams = weakref.WeakSet[SpeechStream]()
        self._pool = utils.ConnectionPool[SpeechAsyncClientV2 | SpeechAsyncClientV1](
            max_session_duration=_max_session_duration,
            connect_cb=self._create_client,
        )

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def provider(self) -> str:
        return "Google Cloud Platform"

    async def _create_client(self, timeout: float) -> SpeechAsyncClientV2 | SpeechAsyncClientV1:
        # Add support for passing a specific location that matches recognizer
        # see: https://cloud.google.com/speech-to-text/v2/docs/speech-to-text-supported-languages
        # TODO(long): how to set timeout?
        client_options = None
        client: SpeechAsyncClientV2 | SpeechAsyncClientV1 | None = None
        client_cls = SpeechAsyncClientV2 if self._config.version == 2 else SpeechAsyncClientV1
        if self._location != "global":
            client_options = ClientOptions(api_endpoint=f"{self._location}-speech.googleapis.com")
        if is_given(self._credentials_info):
            client = client_cls.from_service_account_info(
                self._credentials_info, client_options=client_options
            )
        elif is_given(self._credentials_file):
            client = client_cls.from_service_account_file(
                self._credentials_file, client_options=client_options
            )
        else:
            client = client_cls(client_options=client_options)
        assert client is not None
        return client

    def _get_recognizer(self, client: SpeechAsyncClientV2) -> str:
        # TODO(theomonnom): should we use recognizers?
        # recognizers may improve latency https://cloud.google.com/speech-to-text/v2/docs/recognizers#understand_recognizers

        # TODO(theomonnom): find a better way to access the project_id
        try:
            project_id = client.transport._credentials.project_id  # type: ignore
        except AttributeError:
            from google.auth import default as ga_default

            _, project_id = ga_default()  # type: ignore
        return f"projects/{project_id}/locations/{self._location}/recognizers/_"

    def _sanitize_options(self, *, language: NotGivenOr[str] = NOT_GIVEN) -> STTOptions:
        config = dataclasses.replace(self._config)

        if is_given(language):
            config.languages = [language]

        if not isinstance(config.languages, list):
            config.languages = [config.languages]
        elif not config.detect_language:
            if len(config.languages) > 1:
                logger.warning("multiple languages provided, but language detection is disabled")
            config.languages = [config.languages[0]]

        return config

    def _build_recognition_config(
        self,
        sample_rate: int,
        num_channels: int,
        language: NotGivenOr[SpeechLanguages | str] = NOT_GIVEN,
    ) -> cloud_speech_v2.RecognitionConfig | cloud_speech_v1.RecognitionConfig:
        config = self._sanitize_options(language=language)
        if self._config.version == 2:
            recognition_config_kwargs: dict[str, Any] = {
                "explicit_decoding_config": cloud_speech_v2.ExplicitDecodingConfig(
                    encoding=cloud_speech_v2.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=sample_rate,
                    audio_channel_count=num_channels,
                ),
                "adaptation": config.build_adaptation(),
                "features": cloud_speech_v2.RecognitionFeatures(
                    enable_automatic_punctuation=config.punctuate,
                    enable_spoken_punctuation=config.spoken_punctuation,
                    enable_word_time_offsets=config.enable_word_time_offsets,
                    enable_word_confidence=config.enable_word_confidence,
                    profanity_filter=config.profanity_filter,
                ),
                "model": config.model,
                "language_codes": config.languages,
            }
            denoiser_config = config.build_denoiser_config()
            if denoiser_config is not None:
                recognition_config_kwargs["denoiser_config"] = denoiser_config
            return cloud_speech_v2.RecognitionConfig(**recognition_config_kwargs)
        return cloud_speech_v1.RecognitionConfig(
            encoding=cloud_speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            audio_channel_count=num_channels,
            adaptation=config.build_adaptation(),
            language_code=config.languages[0],
            alternative_language_codes=config.languages[1:],
            enable_word_time_offsets=config.enable_word_time_offsets,
            enable_word_confidence=config.enable_word_confidence,
            enable_automatic_punctuation=config.punctuate,
            enable_spoken_punctuation=config.spoken_punctuation,
            profanity_filter=config.profanity_filter,
            model=config.model,
        )

    def _build_recognition_request(
        self,
        client: SpeechAsyncClientV2 | SpeechAsyncClientV1,
        config: cloud_speech_v2.RecognitionConfig | cloud_speech_v1.RecognitionConfig,
        content: bytes,
    ) -> cloud_speech_v2.RecognizeRequest | cloud_speech_v1.RecognizeRequest:
        if self._config.version == 2:
            return cloud_speech_v2.RecognizeRequest(
                recognizer=self._get_recognizer(cast(SpeechAsyncClientV2, client)),
                config=config,
                content=content,
            )

        return cloud_speech_v1.RecognizeRequest(
            config=config,
            audio=cloud_speech_v1.RecognitionAudio(content=content),
        )

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[SpeechLanguages | str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        frame = rtc.combine_audio_frames(buffer)

        config = self._build_recognition_config(
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            language=language,
        )

        try:
            async with self._pool.connection(timeout=conn_options.timeout) as client:
                raw = await client.recognize(
                    self._build_recognition_request(client, config, frame.data.tobytes()),
                    timeout=conn_options.timeout,
                )
                return _recognize_response_to_speech_event(raw)
        except DeadlineExceeded:
            raise APITimeoutError() from None
        except GoogleAPICallError as e:
            raise APIStatusError(f"{e.message} {e.details}", status_code=e.code or -1) from e
        except Exception as e:
            raise APIConnectionError() from e

    def stream(
        self,
        *,
        language: NotGivenOr[SpeechLanguages | str] = NOT_GIVEN,
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
        languages: NotGivenOr[LanguageCode] = NOT_GIVEN,
        detect_language: NotGivenOr[bool] = NOT_GIVEN,
        interim_results: NotGivenOr[bool] = NOT_GIVEN,
        punctuate: NotGivenOr[bool] = NOT_GIVEN,
        spoken_punctuation: NotGivenOr[bool] = NOT_GIVEN,
        profanity_filter: NotGivenOr[bool] = NOT_GIVEN,
        model: NotGivenOr[SpeechModels] = NOT_GIVEN,
        location: NotGivenOr[str] = NOT_GIVEN,
        keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN,
        denoise_audio: NotGivenOr[bool] = NOT_GIVEN,
        snr_threshold: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(languages):
            if isinstance(languages, str):
                languages = [languages]
            self._config.languages = cast(list[LgType], languages)
        if is_given(detect_language):
            self._config.detect_language = detect_language
        if is_given(interim_results):
            self._config.interim_results = interim_results
        if is_given(punctuate):
            self._config.punctuate = punctuate
        if is_given(spoken_punctuation):
            self._config.spoken_punctuation = spoken_punctuation
        if is_given(profanity_filter):
            self._config.profanity_filter = profanity_filter
        if is_given(model):
            old_version = self._config.version
            self._config.model = model
            if self._config.version != old_version:
                self._pool.invalidate()

        if is_given(location):
            self._location = location
            # if location is changed, fetch a new client and recognizer as per the new location
            self._pool.invalidate()
        if is_given(keywords):
            self._config.keywords = keywords
        if is_given(denoise_audio):
            self._config.denoise_audio = denoise_audio
        if is_given(snr_threshold):
            self._config.snr_threshold = snr_threshold

        for stream in self._streams:
            stream.update_options(
                languages=languages,
                detect_language=detect_language,
                interim_results=interim_results,
                punctuate=punctuate,
                spoken_punctuation=spoken_punctuation,
                profanity_filter=profanity_filter,
                model=model,
                keywords=keywords,
                denoise_audio=denoise_audio,
                snr_threshold=snr_threshold,
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
        pool: utils.ConnectionPool[SpeechAsyncClientV2 | SpeechAsyncClientV1],
        recognizer_cb: Callable[[SpeechAsyncClientV2], str],
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
        languages: NotGivenOr[LanguageCode] = NOT_GIVEN,
        detect_language: NotGivenOr[bool] = NOT_GIVEN,
        interim_results: NotGivenOr[bool] = NOT_GIVEN,
        punctuate: NotGivenOr[bool] = NOT_GIVEN,
        spoken_punctuation: NotGivenOr[bool] = NOT_GIVEN,
        profanity_filter: NotGivenOr[bool] = NOT_GIVEN,
        model: NotGivenOr[SpeechModels] = NOT_GIVEN,
        min_confidence_threshold: NotGivenOr[float] = NOT_GIVEN,
        keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN,
        denoise_audio: NotGivenOr[bool] = NOT_GIVEN,
        snr_threshold: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(languages):
            if isinstance(languages, str):
                languages = [languages]
            self._config.languages = cast(list[LgType], languages)
        if is_given(detect_language):
            self._config.detect_language = detect_language
        if is_given(interim_results):
            self._config.interim_results = interim_results
        if is_given(punctuate):
            self._config.punctuate = punctuate
        if is_given(spoken_punctuation):
            self._config.spoken_punctuation = spoken_punctuation
        if is_given(profanity_filter):
            self._config.profanity_filter = profanity_filter
        if is_given(model):
            old_version = self._config.version
            self._config.model = model
            if self._config.version != old_version:
                self._pool.invalidate()
        if is_given(min_confidence_threshold):
            self._config.min_confidence_threshold = min_confidence_threshold
        if is_given(keywords):
            self._config.keywords = keywords
        if is_given(denoise_audio):
            self._config.denoise_audio = denoise_audio
        if is_given(snr_threshold):
            self._config.snr_threshold = snr_threshold

        self._reconnect_event.set()

    def _build_streaming_config(
        self,
    ) -> cloud_speech_v2.StreamingRecognitionConfig | cloud_speech_v1.StreamingRecognitionConfig:
        if self._config.version == 2:
            recognition_config_kwargs: dict[str, Any] = {
                "explicit_decoding_config": cloud_speech_v2.ExplicitDecodingConfig(
                    encoding=cloud_speech_v2.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self._config.sample_rate,
                    audio_channel_count=1,
                ),
                "adaptation": self._config.build_adaptation(),
                "language_codes": self._config.languages,
                "model": self._config.model,
                "features": cloud_speech_v2.RecognitionFeatures(
                    enable_automatic_punctuation=self._config.punctuate,
                    enable_word_time_offsets=self._config.enable_word_time_offsets,
                    enable_spoken_punctuation=self._config.spoken_punctuation,
                    enable_word_confidence=self._config.enable_word_confidence,
                    profanity_filter=self._config.profanity_filter,
                ),
            }
            denoiser_config = self._config.build_denoiser_config()
            if denoiser_config is not None:
                recognition_config_kwargs["denoiser_config"] = denoiser_config
            return cloud_speech_v2.StreamingRecognitionConfig(
                config=cloud_speech_v2.RecognitionConfig(**recognition_config_kwargs),
                streaming_features=cloud_speech_v2.StreamingRecognitionFeatures(
                    interim_results=self._config.interim_results,
                    enable_voice_activity_events=self._config.enable_voice_activity_events,
                ),
            )

        return cloud_speech_v1.StreamingRecognitionConfig(
            config=cloud_speech_v1.RecognitionConfig(
                encoding=cloud_speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self._config.sample_rate,
                audio_channel_count=1,
                adaptation=self._config.build_adaptation(),
                language_code=self._config.languages[0],
                alternative_language_codes=self._config.languages[1:],
                enable_word_time_offsets=self._config.enable_word_time_offsets,
                enable_word_confidence=self._config.enable_word_confidence,
                enable_automatic_punctuation=self._config.punctuate,
                enable_spoken_punctuation=self._config.spoken_punctuation,
                profanity_filter=self._config.profanity_filter,
                model=self._config.model,
            ),
            interim_results=self._config.interim_results,
            enable_voice_activity_events=self._config.enable_voice_activity_events,
        )

    def _build_init_request(
        self,
        client: SpeechAsyncClientV2 | SpeechAsyncClientV1,
    ) -> cloud_speech_v2.StreamingRecognizeRequest | cloud_speech_v1.StreamingRecognizeRequest:
        if self._config.version == 2:
            return cloud_speech_v2.StreamingRecognizeRequest(
                recognizer=self._recognizer_cb(cast(SpeechAsyncClientV2, client)),
                streaming_config=self._streaming_config,
            )
        return cloud_speech_v1.StreamingRecognizeRequest(
            streaming_config=self._streaming_config,
        )

    def _build_audio_request(
        self,
        frame: rtc.AudioFrame,
    ) -> cloud_speech_v2.StreamingRecognizeRequest | cloud_speech_v1.StreamingRecognizeRequest:
        if self._config.version == 2:
            return cloud_speech_v2.StreamingRecognizeRequest(audio=frame.data.tobytes())
        return cloud_speech_v1.StreamingRecognizeRequest(audio_content=frame.data.tobytes())

    async def _run(self) -> None:
        audio_pushed = False

        # google requires a async generator when calling streaming_recognize
        # this function basically convert the queue into a async generator
        async def input_generator(
            client: SpeechAsyncClientV2 | SpeechAsyncClientV1, should_stop: asyncio.Event
        ) -> AsyncGenerator[
            cloud_speech_v2.StreamingRecognizeRequest | cloud_speech_v1.StreamingRecognizeRequest,
            None,
        ]:
            nonlocal audio_pushed
            try:
                yield self._build_init_request(client)

                async for frame in self._input_ch:
                    # when the stream is aborted due to reconnect, this input_generator
                    # needs to stop consuming frames
                    # when the generator stops, the previous gRPC stream will close
                    if should_stop.is_set():
                        return

                    if isinstance(frame, rtc.AudioFrame):
                        yield self._build_audio_request(frame)
                        if not audio_pushed:
                            audio_pushed = True

            except Exception:
                logger.exception("an error occurred while streaming input to google STT")

        async def process_stream(
            client: SpeechAsyncClientV2 | SpeechAsyncClientV1,
            stream: AsyncIterable[
                cloud_speech_v2.StreamingRecognizeResponse
                | cloud_speech_v1.StreamingRecognizeResponse
            ],
        ) -> None:
            has_started = False
            async for resp in stream:
                if resp.speech_event_type == (
                    cloud_speech_v2.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_BEGIN
                    if self._config.version == 2
                    else cloud_speech_v1.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_BEGIN
                ):
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                    )
                    has_started = True

                if (
                    resp.speech_event_type
                    == (
                        cloud_speech_v2.StreamingRecognizeResponse.SpeechEventType.SPEECH_EVENT_TYPE_UNSPECIFIED
                        if self._config.version == 2
                        else cloud_speech_v1.StreamingRecognizeResponse.SpeechEventType.SPEECH_EVENT_UNSPECIFIED
                    )
                    and resp.results
                ):
                    result = resp.results[0]
                    speech_data = _streaming_recognize_response_to_speech_data(
                        resp,
                        min_confidence_threshold=self._config.min_confidence_threshold,
                        start_time_offset=self.start_time_offset,
                    )
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

                if resp.speech_event_type == (
                    cloud_speech_v2.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_END
                    if self._config.version == 2
                    else cloud_speech_v1.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_END
                ):
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                    )
                    has_started = False

        while True:
            audio_pushed = False
            try:
                async with self._pool.connection(timeout=self._conn_options.timeout) as client:
                    self._streaming_config = self._build_streaming_config()

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
                        should_stop.set()
                        if not process_stream_task.done() and not wait_reconnect_task.done():
                            # try to gracefully stop the process_stream_task
                            try:
                                await asyncio.wait_for(process_stream_task, timeout=1.0)
                            except asyncio.TimeoutError:
                                pass

                        await utils.aio.gracefully_cancel(process_stream_task, wait_reconnect_task)
            except DeadlineExceeded:
                raise APITimeoutError() from None
            except GoogleAPICallError as e:
                if e.code == 409:
                    if audio_pushed:
                        logger.debug("stream timed out, restarting.")
                else:
                    raise APIStatusError(
                        f"{e.message} {e.details}", status_code=e.code or -1
                    ) from e
            except Exception as e:
                raise APIConnectionError() from e


def _duration_to_seconds(duration: Duration | timedelta) -> float:
    # Proto Plus may auto-convert Duration to timedelta; handle both.
    # https://proto-plus-python.readthedocs.io/en/latest/marshal.html
    if isinstance(duration, timedelta):
        return duration.total_seconds()
    return duration.seconds + duration.nanos / 1e9


def _get_start_time(word: cloud_speech_v2.WordInfo | cloud_speech_v1.WordInfo) -> float:
    if hasattr(word, "start_offset"):
        return _duration_to_seconds(word.start_offset)
    return _duration_to_seconds(word.start_time)


def _get_end_time(word: cloud_speech_v2.WordInfo | cloud_speech_v1.WordInfo) -> float:
    if hasattr(word, "end_offset"):
        return _duration_to_seconds(word.end_offset)
    return _duration_to_seconds(word.end_time)


def _recognize_response_to_speech_event(
    resp: cloud_speech_v2.RecognizeResponse | cloud_speech_v1.RecognizeResponse,
) -> stt.SpeechEvent:
    text = ""
    confidence = 0.0
    for result in resp.results:
        text += result.alternatives[0].transcript
        confidence += result.alternatives[0].confidence

    alternatives = []

    # Google STT may return empty results when spoken_lang != stt_lang
    if resp.results:
        try:
            start_time = _get_start_time(resp.results[0].alternatives[0].words[0])
            end_time = _get_end_time(resp.results[-1].alternatives[0].words[-1])
        except IndexError:
            # When enable_word_time_offsets=False, there are no "words" to access
            start_time = end_time = 0

        confidence /= len(resp.results)
        lg = resp.results[0].language_code

        alternatives = [
            stt.SpeechData(
                language=lg,
                start_time=start_time,
                end_time=end_time,
                confidence=confidence,
                text=text,
                words=[
                    TimedString(
                        text=word.word,
                        start_time=_get_start_time(word),
                        end_time=_get_end_time(word),
                    )
                    for word in resp.results[0].alternatives[0].words
                ]
                if resp.results[0].alternatives[0].words
                else None,
            )
        ]

    return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=alternatives)


@utils.log_exceptions(logger=logger)
def _streaming_recognize_response_to_speech_data(
    resp: cloud_speech_v2.StreamingRecognizeResponse | cloud_speech_v1.StreamingRecognizeResponse,
    *,
    min_confidence_threshold: float,
    start_time_offset: float,
) -> stt.SpeechData | None:
    text = ""
    confidence = 0.0
    final_result = None
    words: list[cloud_speech_v2.WordInfo | cloud_speech_v1.WordInfo] = []
    for result in resp.results:
        if len(result.alternatives) == 0:
            continue
        else:
            if result.is_final:
                final_result = result
                break
            else:
                text += result.alternatives[0].transcript
                confidence += result.alternatives[0].confidence
                words.extend(result.alternatives[0].words)

    if final_result is not None:
        text = final_result.alternatives[0].transcript
        confidence = final_result.alternatives[0].confidence
        words = list(final_result.alternatives[0].words)
        lg = final_result.language_code
    else:
        confidence /= len(resp.results)
        if confidence < min_confidence_threshold:
            return None
        lg = resp.results[0].language_code

    if text == "" or not words:
        if text and not words:
            data = stt.SpeechData(
                language=lg,
                start_time=start_time_offset,
                end_time=start_time_offset,
                confidence=confidence,
                text=text,
            )
            return data
        return None

    data = stt.SpeechData(
        language=lg,
        start_time=_get_start_time(words[0]) + start_time_offset,
        end_time=_get_end_time(words[-1]) + start_time_offset,
        confidence=confidence,
        text=text,
        words=[
            TimedString(
                text=word.word,
                start_time=_get_start_time(word) + start_time_offset,
                end_time=_get_end_time(word) + start_time_offset,
                start_time_offset=start_time_offset,
                confidence=word.confidence,
            )
            for word in words
        ],
    )

    return data
