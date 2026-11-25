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
import time
import weakref
from collections.abc import AsyncGenerator, AsyncIterable, Callable
from dataclasses import dataclass
from datetime import timedelta

from grpc.aio import StreamStreamCall

import google.auth
import google.auth.credentials
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import DeadlineExceeded, GoogleAPICallError
from google.auth.exceptions import DefaultCredentialsError
from google.cloud.speech_v2 import SpeechAsyncClient
from google.cloud.speech_v2.types import cloud_speech
from google.protobuf.duration_pb2 import Duration
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.agents.utils.aio import ChanClosed
from livekit.agents.voice.io import TranscriptChunk

from .log import logger
from .models import EndpointingSensitivity, SpeechLanguages, SpeechModels
from .utils import api_status_error, auth

gauth_default = auth.default

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

LgType = SpeechLanguages | str
LanguagesInput = LgType | list[LgType]

# Google STT has a timeout of 5 mins, we'll attempt to restart the session
# before that timeout is reached
_max_session_duration = 240

# Google is very sensitive to background noise, so we'll ignore results with low confidence
_default_min_confidence = 0.65

# Default boost applied to keyterms set via the provider-agnostic keyterm hook.
# Google accepts boosts in roughly 0-20; a moderate value biases toward the terms without
# over-triggering false positives.
_DEFAULT_KEYTERM_BOOST = 10.0


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
    denoiser_config: NotGivenOr[cloud_speech.DenoiserConfig] = NOT_GIVEN
    adaptation: NotGivenOr[cloud_speech.SpeechAdaptation] = NOT_GIVEN
    keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN
    speech_start_timeout: NotGivenOr[float] = NOT_GIVEN
    speech_end_timeout: NotGivenOr[float] = NOT_GIVEN
    endpointing_sensitivity: NotGivenOr[EndpointingSensitivity] = NOT_GIVEN
    custom_prompt_config: NotGivenOr[cloud_speech.CustomPromptConfig] = NOT_GIVEN

    def build_adaptation(self) -> cloud_speech.SpeechAdaptation | None:
        if is_given(self.adaptation):
            return self.adaptation
        if is_given(self.keywords):
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
        languages: LanguagesInput = "en-US",  # Google STT can accept multiple languages
        detect_language: bool = True,
        interim_results: bool = True,
        punctuate: bool = True,
        spoken_punctuation: bool = False,
        enable_word_time_offsets: NotGivenOr[bool] = NOT_GIVEN,
        enable_word_confidence: bool = False,
        enable_voice_activity_events: bool = False,
        model: SpeechModels | str = "long",
        location: str = "global",
        project: NotGivenOr[str] = NOT_GIVEN,
        profanity_filter: bool = False,
        sample_rate: int = 16000,
        min_confidence_threshold: float = _default_min_confidence,
        denoiser_config: NotGivenOr[cloud_speech.DenoiserConfig] = NOT_GIVEN,
        adaptation: NotGivenOr[cloud_speech.SpeechAdaptation] = NOT_GIVEN,
        credentials_info: NotGivenOr[dict[str, object]] = NOT_GIVEN,
        credentials_file: NotGivenOr[str] = NOT_GIVEN,
        credentials: NotGivenOr[google.auth.credentials.Credentials] = NOT_GIVEN,
        keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN,
        speech_start_timeout: NotGivenOr[float] = NOT_GIVEN,
        speech_end_timeout: NotGivenOr[float] = NOT_GIVEN,
        endpointing_sensitivity: NotGivenOr[EndpointingSensitivity] = NOT_GIVEN,
        custom_prompt_config: NotGivenOr[cloud_speech.CustomPromptConfig] = NOT_GIVEN,
        use_streaming: NotGivenOr[bool] = NOT_GIVEN,
    ):
        """
        Create a new instance of Google STT.

        Credentials must be provided, either as a ``google.auth.credentials.Credentials`` object
        via ``credentials``, by using the ``credentials_info`` dict, by reading from the file
        specified in ``credentials_file``, or via Application Default Credentials as
        described in https://cloud.google.com/docs/authentication/application-default-credentials

        args:
            languages(LanguagesInput): list of language codes to recognize (default: "en-US")
            detect_language(bool): whether to detect the language of the audio (default: True)
            interim_results(bool): whether to return interim results (default: True)
            punctuate(bool): whether to punctuate the audio (default: True)
            spoken_punctuation(bool): whether to use spoken punctuation (default: False)
            enable_word_time_offsets(bool): whether to enable word time offsets (default: None)
            enable_word_confidence(bool): whether to enable word confidence (default: False)
            enable_voice_activity_events(bool): whether to enable voice activity events (default: False)
            model(SpeechModels): the Speech-to-Text V2 model to use for recognition
                (default: "long")
            location(str): the location to use for recognition default: "global"
            project(str): the Google Cloud project to use for recognition
            profanity_filter(bool): whether to filter out profanities default: False
            sample_rate(int): the sample rate of the audio default: 16000
            min_confidence_threshold(float): minimum confidence threshold for recognition
            (default: 0.65)
            denoiser_config (DenoiserConfig): the denoiser configuration (default: None)
            adaptation (SpeechAdaptation): speech adaptation for biasing specific words and phrases (default: None)
            credentials_info(dict): the credentials info to use for recognition (default: None)
            credentials_file(str): the credentials file to use for recognition (default: None)
            credentials(google.auth.credentials.Credentials): a credentials object to use
                directly, e.g. from Workload Identity Federation, where credentials are
                obtained in memory and never exist on disk. Takes precedence over
                ``credentials_info`` and ``credentials_file`` (default: NOT_GIVEN)
            keywords(List[tuple[str, float]]): list of keywords to recognize (default: None)
            speech_start_timeout(float): maximum seconds to wait for speech to begin before timeout (default: None)
            speech_end_timeout(float): seconds of silence before marking utterance as complete (default: None)
            endpointing_sensitivity(EndpointingSensitivity): controls the trade-off between latency
                and accuracy when detecting end-of-speech. Only supported with chirp_3.
                Options: ENDPOINTING_SENSITIVITY_STANDARD (default),
                ENDPOINTING_SENSITIVITY_SHORT, ENDPOINTING_SENSITIVITY_SUPERSHORT (default: None)
            custom_prompt_config (CustomPromptConfig): custom prompt configuration for recognition (default: None)
            use_streaming(bool): whether to use streaming for recognition (default: True)
        """
        if is_given(endpointing_sensitivity) and model != "chirp_3":
            logger.warning(
                "endpointing_sensitivity is only supported with the chirp_3 model; ignoring."
            )
            endpointing_sensitivity = NOT_GIVEN

        if is_given(adaptation):
            if is_given(keywords):
                logger.warning(
                    "Both 'adaptation' and 'keywords' are set; 'keywords' will be ignored."
                )

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

        if is_given(custom_prompt_config) and model != "chirp_3":
            logger.warning(
                "custom_prompt_config is only supported with the chirp_3 model; ignoring."
            )
            custom_prompt_config = NOT_GIVEN

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=use_streaming,
                interim_results=True,
                aligned_transcript="word" if enable_word_time_offsets and use_streaming else False,
                # adaptation shadows keywords (see build_adaptation), so keyterms can't be applied
                keyterms=not is_given(adaptation),
                # usage events carry Google's billed duration
                reports_usage=True,
            )
        )

        self._location = location
        self._credentials_info = credentials_info
        self._credentials_file = credentials_file
        self._credentials = credentials
        self._project_id: str | None = project if is_given(project) else None

        if (
            not is_given(credentials)
            and not is_given(credentials_file)
            and not is_given(credentials_info)
        ):
            try:
                gauth_default()
            except DefaultCredentialsError:
                raise ValueError(
                    "Application default credentials must be available "
                    "when using Google STT without explicitly passing "
                    "credentials through credentials_info or credentials_file."
                ) from None

        if isinstance(languages, str):
            languages = [LanguageCode(languages)]
        else:
            languages = [LanguageCode(lg) for lg in languages]

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
            adaptation=adaptation,
            keywords=keywords,
            denoiser_config=denoiser_config,
            speech_start_timeout=speech_start_timeout,
            speech_end_timeout=speech_end_timeout,
            endpointing_sensitivity=endpointing_sensitivity,
            custom_prompt_config=custom_prompt_config,
        )
        # user-tuned (phrase, boost) pairs, kept separate so keyterm updates can't clobber them
        self._user_keywords: list[tuple[str, float]] = list(keywords) if is_given(keywords) else []
        self._session_keyterms: list[str] = []  # framework-managed; merged with user keywords
        self._streams = weakref.WeakSet[SpeechStream]()
        self._pool = utils.ConnectionPool[SpeechAsyncClient](
            max_session_duration=_max_session_duration,
            connect_cb=self._create_client,
        )

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def provider(self) -> str:
        return "google_cloud"

    async def _create_client(self, conn_options: APIConnectOptions) -> SpeechAsyncClient:
        # a regional recognizer is only reachable through its regional endpoint
        client_options = None
        if self._location != "global":
            client_options = ClientOptions(api_endpoint=f"{self._location}-speech.googleapis.com")
        credentials: google.auth.credentials.Credentials | None = None
        project_id: str | None = None
        if is_given(self._credentials):
            credentials = self._credentials
            # in-memory credentials (e.g. Workload Identity Federation) may carry the project
            # directly, where Application Default Credentials typically are not available
            for field in ("project_id", "quota_project_id"):
                if isinstance(value := getattr(credentials, field, None), str) and value:
                    project_id = value
                    break
        elif is_given(self._credentials_info):
            credentials, project_id = auth.load_credentials_from_dict(
                self._credentials_info, scopes=_SCOPES
            )
        elif is_given(self._credentials_file):
            credentials, project_id = auth.load_credentials_from_file(
                self._credentials_file, scopes=_SCOPES
            )
        if self._project_id is None:
            self._project_id = project_id
        return SpeechAsyncClient(credentials=credentials, client_options=client_options)

    def _get_recognizer(self) -> str:
        project_id = self._project_id
        if project_id is None:
            try:
                _, project_id = auth.default()
            except DefaultCredentialsError as e:
                raise APIConnectionError(
                    "google stt: could not determine the GCP project id: the supplied "
                    "credentials expose no project_id/quota_project_id and Application "
                    "Default Credentials are unavailable. Pass `project`, or credentials "
                    "that carry one."
                ) from e
        if not project_id:
            raise APIConnectionError(
                "google stt: could not determine the GCP project id; pass `project`"
            )
        return f"projects/{project_id}/locations/{self._location}/recognizers/_"

    def _sanitize_options(self, *, language: NotGivenOr[str] = NOT_GIVEN) -> STTOptions:
        config = dataclasses.replace(self._config)

        if is_given(language):
            config.languages = [LanguageCode(language)]

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
    ) -> cloud_speech.RecognitionConfig:
        config = self._sanitize_options(language=language)
        return cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sample_rate,
                audio_channel_count=num_channels,
            ),
            adaptation=config.build_adaptation(),
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=config.punctuate,
                enable_spoken_punctuation=config.spoken_punctuation,
                enable_word_time_offsets=config.enable_word_time_offsets,
                enable_word_confidence=config.enable_word_confidence,
                profanity_filter=config.profanity_filter,
                custom_prompt_config=config.custom_prompt_config
                if is_given(config.custom_prompt_config)
                else None,
            ),
            denoiser_config=config.denoiser_config if is_given(config.denoiser_config) else None,
            model=config.model,
            language_codes=config.languages,
        )

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
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
            async with self._pool.connection(conn_options=conn_options) as client:
                request = cloud_speech.RecognizeRequest(
                    recognizer=self._get_recognizer(),
                    config=config,
                    content=frame.data.tobytes(),
                )
                raw = await client.recognize(request, timeout=None, retry=None)
                return _recognize_response_to_speech_event(raw)
        except DeadlineExceeded:
            raise APITimeoutError() from None
        except GoogleAPICallError as e:
            raise api_status_error(e) from e
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
        languages: NotGivenOr[LanguagesInput] = NOT_GIVEN,
        detect_language: NotGivenOr[bool] = NOT_GIVEN,
        interim_results: NotGivenOr[bool] = NOT_GIVEN,
        punctuate: NotGivenOr[bool] = NOT_GIVEN,
        spoken_punctuation: NotGivenOr[bool] = NOT_GIVEN,
        profanity_filter: NotGivenOr[bool] = NOT_GIVEN,
        model: NotGivenOr[SpeechModels | str] = NOT_GIVEN,
        location: NotGivenOr[str] = NOT_GIVEN,
        denoiser_config: NotGivenOr[cloud_speech.DenoiserConfig] = NOT_GIVEN,
        adaptation: NotGivenOr[cloud_speech.SpeechAdaptation] = NOT_GIVEN,
        keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN,
        speech_start_timeout: NotGivenOr[float] = NOT_GIVEN,
        speech_end_timeout: NotGivenOr[float] = NOT_GIVEN,
        endpointing_sensitivity: NotGivenOr[EndpointingSensitivity] = NOT_GIVEN,
        custom_prompt_config: NotGivenOr[cloud_speech.CustomPromptConfig] = NOT_GIVEN,
    ) -> None:
        if is_given(languages):
            if isinstance(languages, str):
                self._config.languages = [LanguageCode(languages)]
            else:
                self._config.languages = [LanguageCode(lg) for lg in languages]
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
            self._config.model = model
        if is_given(location):
            self._location = location
            # if location is changed, fetch a new client and recognizer as per the new location
            self._pool.invalidate()
        if is_given(denoiser_config):
            self._config.denoiser_config = denoiser_config
        if is_given(adaptation):
            if is_given(keywords) or is_given(self._config.keywords):
                logger.warning(
                    "Both 'adaptation' and 'keywords' are set; 'keywords' will be ignored."
                )
            self._config.adaptation = adaptation
        if is_given(keywords):
            if is_given(self._config.adaptation) and not is_given(adaptation):
                logger.warning(
                    "Both 'adaptation' and 'keywords' are set; 'keywords' will be ignored."
                )
            self._user_keywords = list(keywords)
            # re-merge with the active session keyterms so a user update doesn't drop them,
            # and forward the merged value to the streams below (not the raw user keywords)
            keywords = self._get_merged_keywords()
            self._config.keywords = keywords
        if is_given(speech_start_timeout):
            self._config.speech_start_timeout = speech_start_timeout
        if is_given(speech_end_timeout):
            self._config.speech_end_timeout = speech_end_timeout
        if is_given(endpointing_sensitivity):
            if self._config.model != "chirp_3":
                logger.warning(
                    "endpointing_sensitivity is only supported with the chirp_3 model; ignoring."
                )
                endpointing_sensitivity = NOT_GIVEN
            else:
                self._config.endpointing_sensitivity = endpointing_sensitivity
        if is_given(custom_prompt_config):
            if self._config.model != "chirp_3":
                logger.warning(
                    "custom_prompt_config is only supported with the chirp_3 model; ignoring."
                )
                custom_prompt_config = NOT_GIVEN
            else:
                self._config.custom_prompt_config = custom_prompt_config
        elif (
            is_given(model)
            and self._config.model != "chirp_3"
            and is_given(self._config.custom_prompt_config)
        ):
            # model was switched away from chirp_3; a previously configured prompt is no
            # longer valid for the new model and must not be sent with it.
            logger.warning(
                "model changed away from chirp_3; clearing previously configured "
                "custom_prompt_config."
            )
            self._config.custom_prompt_config = NOT_GIVEN

        for stream in self._streams:
            stream.update_options(
                languages=languages,
                detect_language=detect_language,
                interim_results=interim_results,
                punctuate=punctuate,
                spoken_punctuation=spoken_punctuation,
                profanity_filter=profanity_filter,
                model=model,
                denoiser_config=denoiser_config,
                adaptation=adaptation,
                keywords=keywords,
                speech_start_timeout=speech_start_timeout,
                speech_end_timeout=speech_end_timeout,
                endpointing_sensitivity=endpointing_sensitivity,
                custom_prompt_config=custom_prompt_config,
            )

    def _get_merged_keywords(self) -> list[tuple[str, float]]:
        # Google biases via (phrase, boost) pairs; the session hook carries no per-term weight,
        # so keep the user keyword boosts and bias session terms no stronger than the weakest
        # user term (or a moderate default when the user gave none).
        user_phrases = {phrase for phrase, _ in self._user_keywords}
        session_boost = (
            min(boost for _, boost in self._user_keywords)
            if self._user_keywords
            else _DEFAULT_KEYTERM_BOOST
        )
        return self._user_keywords + [
            (term, session_boost) for term in self._session_keyterms if term not in user_phrases
        ]

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        if is_given(self._config.adaptation):
            logger.warning("'adaptation' is set; ignoring keyterms update")
            return
        if keyterms == self._session_keyterms:
            return
        self._session_keyterms = list(keyterms)
        merged = self._get_merged_keywords()
        self._config.keywords = merged
        for stream in self._streams:
            stream.update_session_keyterms(merged)

    async def aclose(self) -> None:
        await self._pool.aclose()
        await super().aclose()


class SpeechStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        pool: utils.ConnectionPool[SpeechAsyncClient],
        recognizer_cb: Callable[[], str],
        config: STTOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=config.sample_rate)

        self._pool = pool
        self._recognizer_cb = recognizer_cb
        self._config = config
        self._reconnect_event = asyncio.Event()
        self._session_connected_at: float = 0
        self._speaking = False
        # keywords set while the user is speaking; applied at END_OF_SPEECH (latest wins)
        self._pending_keywords: list[tuple[str, float]] | None = None

    def update_session_keyterms(self, keywords: list[tuple[str, float]]) -> None:
        if self._speaking:
            self._pending_keywords = keywords
        else:
            self.update_options(keywords=keywords)

    def update_options(
        self,
        *,
        languages: NotGivenOr[LanguagesInput] = NOT_GIVEN,
        detect_language: NotGivenOr[bool] = NOT_GIVEN,
        interim_results: NotGivenOr[bool] = NOT_GIVEN,
        punctuate: NotGivenOr[bool] = NOT_GIVEN,
        spoken_punctuation: NotGivenOr[bool] = NOT_GIVEN,
        profanity_filter: NotGivenOr[bool] = NOT_GIVEN,
        model: NotGivenOr[SpeechModels | str] = NOT_GIVEN,
        min_confidence_threshold: NotGivenOr[float] = NOT_GIVEN,
        denoiser_config: NotGivenOr[cloud_speech.DenoiserConfig] = NOT_GIVEN,
        adaptation: NotGivenOr[cloud_speech.SpeechAdaptation] = NOT_GIVEN,
        keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN,
        speech_start_timeout: NotGivenOr[float] = NOT_GIVEN,
        speech_end_timeout: NotGivenOr[float] = NOT_GIVEN,
        endpointing_sensitivity: NotGivenOr[EndpointingSensitivity] = NOT_GIVEN,
        custom_prompt_config: NotGivenOr[cloud_speech.CustomPromptConfig] = NOT_GIVEN,
    ) -> None:
        if is_given(languages):
            if isinstance(languages, str):
                self._config.languages = [LanguageCode(languages)]
            else:
                self._config.languages = [LanguageCode(lg) for lg in languages]
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
            self._config.model = model
        if is_given(min_confidence_threshold):
            self._config.min_confidence_threshold = min_confidence_threshold
        if is_given(denoiser_config):
            self._config.denoiser_config = denoiser_config
        if is_given(adaptation):
            self._config.adaptation = adaptation
        if is_given(keywords):
            self._config.keywords = keywords
            self._pending_keywords = None
        if is_given(speech_start_timeout):
            self._config.speech_start_timeout = speech_start_timeout
        if is_given(speech_end_timeout):
            self._config.speech_end_timeout = speech_end_timeout
        if is_given(endpointing_sensitivity):
            self._config.endpointing_sensitivity = endpointing_sensitivity
        if is_given(custom_prompt_config):
            self._config.custom_prompt_config = custom_prompt_config
        elif (
            is_given(model)
            and self._config.model != "chirp_3"
            and is_given(self._config.custom_prompt_config)
        ):
            # model was switched away from chirp_3; a previously configured prompt is no
            # longer valid for the new model and must not be sent with it.
            self._config.custom_prompt_config = NOT_GIVEN

        self._reconnect_event.set()

    def _on_end_of_speech(self) -> None:
        if self._pending_keywords is not None:
            self.update_options(keywords=self._pending_keywords)
            self._pending_keywords = None

    def _build_streaming_config(self) -> cloud_speech.StreamingRecognitionConfig:
        # Build voice activity timeout if either timeout is specified
        voice_activity_timeout = None
        if is_given(self._config.speech_start_timeout) or is_given(self._config.speech_end_timeout):
            voice_activity_timeout = (
                cloud_speech.StreamingRecognitionFeatures.VoiceActivityTimeout()
            )
            if is_given(self._config.speech_start_timeout):
                voice_activity_timeout.speech_start_timeout = Duration(
                    seconds=int(self._config.speech_start_timeout),
                    nanos=int((self._config.speech_start_timeout % 1) * 1e9),
                )
            if is_given(self._config.speech_end_timeout):
                voice_activity_timeout.speech_end_timeout = Duration(
                    seconds=int(self._config.speech_end_timeout),
                    nanos=int((self._config.speech_end_timeout % 1) * 1e9),
                )

        return cloud_speech.StreamingRecognitionConfig(
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
                    enable_word_time_offsets=self._config.enable_word_time_offsets,
                    enable_spoken_punctuation=self._config.spoken_punctuation,
                    enable_word_confidence=self._config.enable_word_confidence,
                    profanity_filter=self._config.profanity_filter,
                    custom_prompt_config=self._config.custom_prompt_config
                    if is_given(self._config.custom_prompt_config)
                    else None,
                ),
                denoiser_config=self._config.denoiser_config
                if is_given(self._config.denoiser_config)
                else None,
            ),
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                interim_results=self._config.interim_results,
                # Auto-enable voice activity events when voice_activity_timeout is specified,
                # as per Google API documentation requirements
                enable_voice_activity_events=self._config.enable_voice_activity_events
                or (voice_activity_timeout is not None),
                voice_activity_timeout=voice_activity_timeout,
                # proto-plus reads an enum field from its name
                endpointing_sensitivity=self._config.endpointing_sensitivity
                if is_given(self._config.endpointing_sensitivity)
                else None,
            ),
        )

    async def _run(self) -> None:
        audio_pushed = False

        # google requires a async generator when calling streaming_recognize
        # this function basically convert the queue into a async generator
        async def input_generator(
            should_stop: asyncio.Event,
        ) -> AsyncGenerator[cloud_speech.StreamingRecognizeRequest, None]:
            nonlocal audio_pushed
            async with utils.aio.TaskSet() as tasks:
                stop_task = tasks.spawn(should_stop.wait())
                try:
                    yield cloud_speech.StreamingRecognizeRequest(
                        recognizer=self._recognizer_cb(),
                        streaming_config=self._streaming_config,
                    )

                    while True:
                        # Race the next-frame await against should_stop so this generator
                        # can exit even when no audio is flowing. Without this, on reconnect
                        # the generator stays parked on input and pins the previous
                        # gRPC streaming call, leaking it across iterations.
                        frame_task = tasks.spawn(self.input.recv())
                        try:
                            first = await tasks.first_completed(
                                frame_task, asyncio.shield(stop_task)
                            )
                        except ChanClosed:
                            return
                        if first is not frame_task:
                            return

                        frame = frame_task.result()
                        if isinstance(frame, rtc.AudioFrame):
                            yield cloud_speech.StreamingRecognizeRequest(audio=frame.data.tobytes())
                            if not audio_pushed:
                                audio_pushed = True

                except Exception:
                    logger.exception("an error occurred while streaming input to google STT")

        async def process_stream(
            client: SpeechAsyncClient,
            stream: AsyncIterable[cloud_speech.StreamingRecognizeResponse],
        ) -> None:
            events = cloud_speech.StreamingRecognizeResponse.SpeechEventType
            self._speaking = False
            last_usage_event_time: float = 0.0
            async for resp in stream:
                if resp.speech_event_type == events.SPEECH_ACTIVITY_BEGIN:
                    self.send(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
                    self._speaking = True

                if resp.speech_event_type == events.SPEECH_EVENT_TYPE_UNSPECIFIED and resp.results:
                    result = resp.results[0]
                    speech_data = _streaming_recognize_response_to_speech_data(
                        resp,
                        min_confidence_threshold=self._config.min_confidence_threshold,
                        start_time_offset=self.start_time_offset,
                    )
                    if speech_data is None:
                        continue

                    if not result.is_final:
                        self.send(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                alternatives=[speech_data],
                            )
                        )
                    else:
                        self.send(
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
                            if self._speaking:
                                self.send(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
                                self._speaking = False
                                self._on_end_of_speech()
                            self._reconnect_event.set()
                            return

                if resp.speech_event_type == events.SPEECH_ACTIVITY_END:
                    self.send(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
                    self._speaking = False
                    self._on_end_of_speech()

                if (audio_duration := _get_audio_duration(resp, last_usage_event_time)) > 0:
                    self.send(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.RECOGNITION_USAGE,
                            request_id=resp.metadata.request_id,
                            recognition_usage=stt.RecognitionUsage(audio_duration=audio_duration),
                        )
                    )
                    last_usage_event_time += audio_duration

        while True:
            audio_pushed = False
            try:
                async with self._pool.connection(conn_options=self._conn_options) as client:
                    self._streaming_config = self._build_streaming_config()

                    should_stop = asyncio.Event()
                    stream = await client.streaming_recognize(
                        requests=input_generator(should_stop), timeout=None, retry=None
                    )
                    self._session_connected_at = time.time()

                    async with utils.aio.TaskSet() as tasks:
                        process_stream_task = tasks.spawn(process_stream(client, stream))
                        reconnect = tasks.spawn(self._reconnect_event.wait())
                        first: asyncio.Future[None] | asyncio.Future[bool] | None = None
                        try:
                            first = await tasks.first_completed(
                                asyncio.shield(process_stream_task), reconnect
                            )
                            if first is not reconnect:
                                break
                            self._reconnect_event.clear()
                        finally:
                            should_stop.set()
                            # Cancel the streaming RPC so its underlying call object releases
                            # its read/write tasks and request iterator. Without this the
                            # call (and the input_generator that yielded into it) stays
                            # pinned across reconnects and leaks ~0.4 MB per cycle.
                            if isinstance(stream, StreamStreamCall):
                                stream.cancel()
                            if first is None and not process_stream_task.done():
                                # try to gracefully stop the process_stream_task
                                with contextlib.suppress(TimeoutError):
                                    await asyncio.wait_for(process_stream_task, timeout=1.0)
            except DeadlineExceeded:
                raise APITimeoutError() from None
            except GoogleAPICallError as e:
                if e.code == 409:
                    if audio_pushed:
                        logger.debug("stream timed out, restarting.")
                else:
                    raise api_status_error(e) from e
            except Exception as e:
                raise APIConnectionError() from e


def _duration_to_seconds(duration: Duration | timedelta) -> float:
    # Proto Plus may auto-convert Duration to timedelta; handle both.
    # https://proto-plus-python.readthedocs.io/en/latest/marshal.html
    if isinstance(duration, timedelta):
        return duration.total_seconds()
    return duration.seconds + duration.nanos / 1e9


def _recognize_response_to_speech_event(
    resp: cloud_speech.RecognizeResponse,
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
            start_time = _duration_to_seconds(resp.results[0].alternatives[0].words[0].start_offset)
            end_time = _duration_to_seconds(resp.results[-1].alternatives[0].words[-1].end_offset)
        except IndexError:
            # When enable_word_time_offsets=False, there are no "words" to access
            start_time = end_time = 0

        confidence /= len(resp.results)
        lg = LanguageCode(resp.results[0].language_code)

        alternatives = [
            stt.SpeechData(
                language=lg,
                start_time=start_time,
                end_time=end_time,
                confidence=confidence,
                text=text,
                words=[
                    TranscriptChunk(
                        text=word.word,
                        start_time=_duration_to_seconds(word.start_offset),
                        end_time=_duration_to_seconds(word.end_offset),
                    )
                    for word in resp.results[0].alternatives[0].words
                ]
                if resp.results[0].alternatives[0].words
                else None,
            )
        ]

    return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=alternatives)


def _streaming_recognize_response_to_speech_data(
    resp: cloud_speech.StreamingRecognizeResponse,
    *,
    min_confidence_threshold: float,
    start_time_offset: float,
) -> stt.SpeechData | None:
    text = ""
    confidence = 0.0
    final_result = None
    words: list[cloud_speech.WordInfo] = []
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
        lg = LanguageCode(final_result.language_code)
    else:
        confidence /= len(resp.results)
        if confidence < min_confidence_threshold:
            return None
        lg = LanguageCode(resp.results[0].language_code)

    if text == "" or not words:
        if text and not words:
            # No words to time the text with (e.g. chirp_3 forces
            # enable_word_time_offsets=False): report the timing as unknown,
            # like the REST path, instead of the stream offset. The SDK reads
            # any end_time > 0 as the provider's end of speech, which would
            # otherwise freeze the end-of-turn anchor in turn_detection="stt".
            data = stt.SpeechData(
                language=lg,
                start_time=0.0,
                end_time=0.0,
                confidence=confidence,
                text=text,
            )
            return data
        return None

    data = stt.SpeechData(
        language=lg,
        start_time=_duration_to_seconds(words[0].start_offset) + start_time_offset,
        end_time=_duration_to_seconds(words[-1].end_offset) + start_time_offset,
        confidence=confidence,
        text=text,
        words=[
            TranscriptChunk(
                text=word.word,
                start_time=_duration_to_seconds(word.start_offset) + start_time_offset,
                end_time=_duration_to_seconds(word.end_offset) + start_time_offset,
                start_time_offset=start_time_offset,
                confidence=word.confidence,
            )
            for word in words
        ],
    )

    return data


def _get_audio_duration(
    resp: cloud_speech.StreamingRecognizeResponse,
    last_usage_event_time: float,
) -> float:
    """Calculate the audio duration from the response.

    References:
        - https://docs.cloud.google.com/speech-to-text/docs/reference/rest/v2/StreamingRecognitionResult
    """
    # total_billed_duration is only set on the last response in the stream;
    # use the speech event offset before it is received
    if resp.metadata.total_billed_duration:
        return _duration_to_seconds(resp.metadata.total_billed_duration) - last_usage_event_time
    return _duration_to_seconds(resp.speech_event_offset) - last_usage_event_time
