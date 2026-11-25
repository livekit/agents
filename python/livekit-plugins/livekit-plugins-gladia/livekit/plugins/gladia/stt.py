# Copyright 2025 LiveKit, Inc.
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

import base64
import dataclasses
import os
import weakref
from dataclasses import dataclass
from enum import Enum
from typing import Literal, NotRequired, TypedDict, assert_never
from urllib.parse import urlencode

import numpy as np

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    APIError,
    LanguageCode,
    NotGivenOr,
    http,
    stt,
)
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TranscriptChunk

from .models import GladiaModels, LiveSession
from .version import __version__

BASE_URL = "https://api.gladia.io/v2/live"

MAGIC_NUMBER_THRESHOLD = 0.004**2


class AudioEnergyFilter:
    class State(Enum):
        START = 0
        SPEAKING = 1
        SILENCE = 2
        END = 3

    def __init__(self, *, min_silence: float = 1.5, rms_threshold: float = MAGIC_NUMBER_THRESHOLD):
        self._cooldown_seconds = min_silence
        self._cooldown = min_silence
        self._state = self.State.SILENCE
        self._rms_threshold = rms_threshold

    def update(self, frame: rtc.AudioFrame) -> State:
        arr = np.frombuffer(frame.data, dtype=np.int16)
        float_arr = arr.astype(np.float32) / 32768.0
        rms = np.mean(np.square(float_arr))

        if rms > self._rms_threshold:
            self._cooldown = self._cooldown_seconds
            if self._state in (self.State.SILENCE, self.State.END):
                self._state = self.State.START
            else:
                self._state = self.State.SPEAKING
        else:
            if self._cooldown <= 0:
                if self._state in (self.State.SPEAKING, self.State.START):
                    self._state = self.State.END
                elif self._state == self.State.END:
                    self._state = self.State.SILENCE
            else:
                self._cooldown -= frame.duration
                self._state = self.State.SPEAKING

        return self._state


@dataclass
class LanguageConfiguration:
    languages: list[str] | None = None
    code_switching: bool = True


@dataclass
class TranslationConfiguration:
    enabled: bool = False
    target_languages: list[str] = dataclasses.field(default_factory=list[str])
    model: str = "base"
    match_original_utterances: bool = True
    lipsync: bool = True
    context_adaptation: bool = False
    context: str | None = None
    informal: bool = False


@dataclass
class PreProcessingConfiguration:
    audio_enhancer: bool = False
    speech_threshold: float = 0.6


@dataclass
class STTOptions:
    model: GladiaModels
    language_config: LanguageConfiguration
    interim_results: bool
    sample_rate: int
    bit_depth: Literal[8, 16, 24, 32]
    channels: int
    endpointing: float
    maximum_duration_without_endpointing: float
    region: Literal["us-west", "eu-west"]
    encoding: Literal["wav/pcm", "wav/alaw", "wav/ulaw"]
    translation_config: TranslationConfiguration = dataclasses.field(
        default_factory=TranslationConfiguration
    )
    energy_filter: AudioEnergyFilter | bool = False
    custom_vocabulary: list[str | dict[str, object]] | None = None
    custom_spelling: dict[str, list[str]] | None = None
    pre_processing: PreProcessingConfiguration = dataclasses.field(
        default_factory=PreProcessingConfiguration
    )


class _LanguageConfig(TypedDict):
    languages: list[str]
    code_switching: bool


class _VocabularyConfig(TypedDict):
    vocabulary: list[str | dict[str, object]]


class _SpellingConfig(TypedDict):
    spelling_dictionary: dict[str, list[str]]


class _TranslationConfig(TypedDict):
    target_languages: list[str]
    model: str
    match_original_utterances: bool
    lipsync: bool
    context_adaptation: bool
    informal: bool
    context: NotRequired[str]


class _RealtimeProcessing(TypedDict):
    words_accurate_timestamps: bool
    custom_vocabulary: NotRequired[bool]
    custom_vocabulary_config: NotRequired[_VocabularyConfig]
    custom_spelling: NotRequired[bool]
    custom_spelling_config: NotRequired[_SpellingConfig]
    translation: NotRequired[bool]
    translation_config: NotRequired[_TranslationConfig]


class _MessagesConfig(TypedDict):
    receive_partial_transcripts: bool
    receive_final_transcripts: bool


class _PreProcessing(TypedDict):
    audio_enhancer: bool
    speech_threshold: float


class _CustomMetadata(TypedDict):
    livekit: str


class _LiveConfig(TypedDict):
    """The live session's configuration, posted at its initialization."""

    encoding: str
    sample_rate: int
    model: str
    endpointing: float
    maximum_duration_without_endpointing: float
    bit_depth: int
    channels: int
    language_config: _LanguageConfig
    realtime_processing: _RealtimeProcessing
    messages_config: _MessagesConfig
    custom_metadata: _CustomMetadata
    pre_processing: _PreProcessing


def _build_streaming_config(opts: STTOptions) -> _LiveConfig:
    """Build the streaming configuration for Gladia API."""
    processing = _RealtimeProcessing(words_accurate_timestamps=False)
    if opts.custom_vocabulary:
        processing["custom_vocabulary"] = True
        processing["custom_vocabulary_config"] = _VocabularyConfig(
            vocabulary=opts.custom_vocabulary
        )
    if opts.custom_spelling:
        processing["custom_spelling"] = True
        processing["custom_spelling_config"] = _SpellingConfig(
            spelling_dictionary=opts.custom_spelling
        )
    if opts.translation_config.enabled:
        tr = opts.translation_config
        translation = _TranslationConfig(
            target_languages=tr.target_languages,
            model=tr.model,
            match_original_utterances=tr.match_original_utterances,
            lipsync=tr.lipsync,
            context_adaptation=tr.context_adaptation,
            informal=tr.informal,
        )
        if tr.context:
            translation["context"] = tr.context
        processing["translation"] = True
        processing["translation_config"] = translation
    return _LiveConfig(
        encoding=opts.encoding,
        sample_rate=opts.sample_rate,
        model=opts.model,
        endpointing=opts.endpointing,
        maximum_duration_without_endpointing=opts.maximum_duration_without_endpointing,
        bit_depth=opts.bit_depth,
        channels=opts.channels,
        language_config=_LanguageConfig(
            languages=opts.language_config.languages or [],
            code_switching=opts.language_config.code_switching,
        ),
        realtime_processing=processing,
        messages_config=_MessagesConfig(
            receive_partial_transcripts=opts.interim_results, receive_final_transcripts=True
        ),
        custom_metadata=_CustomMetadata(livekit=__version__),
        pre_processing=_PreProcessing(
            audio_enhancer=opts.pre_processing.audio_enhancer,
            speech_threshold=opts.pre_processing.speech_threshold,
        ),
    )


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: GladiaModels = "solaria-1",
        interim_results: bool = True,
        languages: list[str] | None = None,
        code_switching: bool = True,
        sample_rate: int = 16000,
        bit_depth: Literal[8, 16, 24, 32] = 16,
        endpointing: float = 0.05,
        maximum_duration_without_endpointing: float = 5,
        channels: int = 1,
        region: Literal["us-west", "eu-west"] = "eu-west",
        encoding: Literal["wav/pcm", "wav/alaw", "wav/ulaw"] = "wav/pcm",
        api_key: str | None = None,
        http_session: http.HttpSession | None = None,
        base_url: str = BASE_URL,
        energy_filter: AudioEnergyFilter | bool = False,
        translation_enabled: bool = False,
        translation_target_languages: list[str] | None = None,
        translation_model: str = "base",
        translation_match_original_utterances: bool = True,
        translation_lipsync: bool = True,
        translation_context_adaptation: bool = False,
        translation_context: str | None = None,
        translation_informal: bool = False,
        custom_vocabulary: list[str | dict[str, object]] | None = None,
        custom_spelling: dict[str, list[str]] | None = None,
        pre_processing_audio_enhancer: bool = False,
        pre_processing_speech_threshold: float = 0.6,
    ) -> None:
        """Create a new instance of Gladia STT.

        Args:
            model: The model to use for recognition. Defaults to "solaria-1".
            interim_results: Whether to return interim (non-final) transcription results.
                            Defaults to True.
            languages: List of language codes to use for recognition. Defaults to None
                    (auto-detect).
            code_switching: Whether to allow switching between languages during recognition.
                            Defaults to True.
            sample_rate: The sample rate of the audio in Hz. Defaults to 16000.
            bit_depth: The bit depth of the audio. Defaults to 16.
            endpointing: Endpointing is the duration of silence in seconds which will cause an utterance to be considered as finished. Defaults to 0.05.
            maximum_duration_without_endpointing: If endpointing is not detected after this duration in seconds, current utterance will be considered as finished. Defaults to 5.
            channels: The number of audio channels. Defaults to 1.
            region: The region to use for the Gladia API. Defaults to "eu-west".
            encoding: The encoding of the audio. Defaults to "wav/pcm".
            api_key: Your Gladia API key. If not provided, will look for GLADIA_API_KEY
                        environment variable.
            http_session: Optional HttpSession to use for requests.
            base_url: The base URL for Gladia API. Defaults to "https://api.gladia.io/v2/live".
            energy_filter: Replaces low-energy audio with silence, preserving frame duration and
                         transcript timestamps. Can be a boolean or AudioEnergyFilter instance.
                         Defaults to False.
            translation_enabled: Whether to enable translation. Defaults to False.
            translation_target_languages: List of target languages for translation.
                                        Required if translation_enabled is True.
            translation_model: Translation model to use. Defaults to "base".
            translation_match_original_utterances: Whether to match original utterances with
                                                    translations. Defaults to True.
            translation_lipsync: If True, enables lipsync generation for translations.
            translation_context_adaptation: If True, adapts translation to the context.
            translation_context: A string providing context for translation.
            translation_informal: If True, uses informal translation style.
            custom_vocabulary: A list of custom vocabulary to use for recognition.
            custom_spelling: A dictionary of custom spelling to use for transcription.
            pre_processing_audio_enhancer: Whether to enable audio enhancement pre-processing.
            pre_processing_speech_threshold: The speech threshold for pre-processing.

        Raises:
            ValueError: If no API key is provided or found in environment variables.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True, interim_results=interim_results, aligned_transcript="word"
            )
        )
        self._base_url = base_url

        api_key = api_key or os.environ.get("GLADIA_API_KEY")
        if not api_key:
            raise ValueError("Gladia API key is required. Set GLADIA_API_KEY or pass api_key")

        self._api_key = api_key

        language_config = LanguageConfiguration(languages=languages, code_switching=code_switching)

        translation_config = TranslationConfiguration(
            enabled=translation_enabled,
            target_languages=translation_target_languages or [],
            model=translation_model,
            match_original_utterances=translation_match_original_utterances,
            lipsync=translation_lipsync,
            context_adaptation=translation_context_adaptation,
            context=translation_context,
            informal=translation_informal,
        )

        pre_processing_config = PreProcessingConfiguration(
            audio_enhancer=pre_processing_audio_enhancer,
            speech_threshold=pre_processing_speech_threshold,
        )

        if translation_enabled and not translation_target_languages:
            raise ValueError(
                "translation_target_languages is required when translation_enabled is True"
            )

        self._opts = STTOptions(
            model=model,
            language_config=language_config,
            interim_results=interim_results,
            sample_rate=sample_rate,
            bit_depth=bit_depth,
            channels=channels,
            region=region,
            encoding=encoding,
            endpointing=endpointing,
            maximum_duration_without_endpointing=maximum_duration_without_endpointing,
            translation_config=translation_config,
            pre_processing=pre_processing_config,
            energy_filter=energy_filter,
            custom_vocabulary=custom_vocabulary,
            custom_spelling=custom_spelling,
        )
        self._session = http_session.with_request_id("X-Request-ID") if http_session else None
        self._streams: weakref.WeakSet[SpeechStream] = weakref.WeakSet()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "gladia"

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session().with_request_id("X-Request-ID")

        return self._session

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        """One recording, recognized on a live session: its audio, then ``stop_recording``; the
        final utterances until Gladia ends the session."""
        config = self._sanitize_options(languages=[language] if is_given(language) else None)
        # the whole recording is transcribed: no energy filter cuts it
        whole = dataclasses.replace(config, energy_filter=False)
        stream = SpeechStream(stt=self, conn_options=conn_options, opts=whole)
        alternatives: list[stt.SpeechData] = []
        request_id = ""
        try:
            for frame in buffer if isinstance(buffer, list) else [buffer]:
                stream.push_frame(frame)
            stream.end_input()
            async for event in stream:
                if event.type is stt.SpeechEventType.FINAL_TRANSCRIPT and event.alternatives:
                    alternatives.append(event.alternatives[0])
                    request_id = event.request_id
        finally:
            await stream.aclose()
        if not alternatives:
            languages = config.language_config.languages
            alternatives.append(
                stt.SpeechData(
                    language=LanguageCode(languages[0] if languages else "en"),
                    start_time=0,
                    end_time=0,
                    confidence=1.0,
                    text="",
                    words=[],
                )
            )
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=request_id,
            alternatives=alternatives,
        )

    async def _init_live_session(
        self, opts: STTOptions, conn_options: APIConnectOptions
    ) -> LiveSession:
        """Initialize a live transcription session with Gladia."""
        async with self._ensure_session().post(
            f"{self._base_url}?{urlencode({'region': opts.region})}",
            json=_build_streaming_config(opts),
            headers={"X-Gladia-Key": self._api_key},
            conn_options=conn_options,
        ) as res:
            # the status's kind decides the retry
            await res.raise_for_status()
            text = await res.text()
            try:
                return LiveSession.model_validate_json(text)
            except ValueError as e:
                raise APIError(
                    "Gladia returned an invalid session", kind="protocol", detail=text
                ) from e

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        config = self._sanitize_options(languages=[language] if is_given(language) else None)
        stream = SpeechStream(stt=self, conn_options=conn_options, opts=config)
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        model: GladiaModels | None = None,
        languages: list[str] | None = None,
        code_switching: bool | None = None,
        interim_results: bool | None = None,
        sample_rate: int | None = None,
        bit_depth: Literal[8, 16, 24, 32] | None = None,
        channels: int | None = None,
        region: Literal["us-west", "eu-west"] | None = None,
        endpointing: float | None = None,
        maximum_duration_without_endpointing: float | None = None,
        encoding: Literal["wav/pcm", "wav/alaw", "wav/ulaw"] | None = None,
        translation_enabled: bool | None = None,
        translation_target_languages: list[str] | None = None,
        translation_model: str | None = None,
        translation_match_original_utterances: bool | None = None,
        translation_lipsync: bool | None = None,
        translation_context_adaptation: bool | None = None,
        translation_context: str | None = None,
        translation_informal: bool | None = None,
        custom_vocabulary: list[str | dict[str, object]] | None = None,
        custom_spelling: dict[str, list[str]] | None = None,
        pre_processing_audio_enhancer: bool | None = None,
        pre_processing_speech_threshold: float | None = None,
    ) -> None:
        if languages is not None or code_switching is not None:
            language_config = dataclasses.replace(
                self._opts.language_config,
                languages=languages
                if languages is not None
                else self._opts.language_config.languages,
                code_switching=code_switching
                if code_switching is not None
                else self._opts.language_config.code_switching,
            )
            self._opts.language_config = language_config

        if (
            translation_enabled is not None
            or translation_target_languages is not None
            or translation_model is not None
            or translation_match_original_utterances is not None
            or translation_lipsync is not None
            or translation_context_adaptation is not None
            or translation_context is not None
            or translation_informal is not None
        ):
            translation_config = dataclasses.replace(
                self._opts.translation_config,
                enabled=translation_enabled
                if translation_enabled is not None
                else self._opts.translation_config.enabled,
                target_languages=translation_target_languages
                if translation_target_languages is not None
                else self._opts.translation_config.target_languages,
                model=translation_model
                if translation_model is not None
                else self._opts.translation_config.model,
                match_original_utterances=translation_match_original_utterances
                if translation_match_original_utterances is not None
                else self._opts.translation_config.match_original_utterances,
                lipsync=translation_lipsync
                if translation_lipsync is not None
                else self._opts.translation_config.lipsync,
                context_adaptation=translation_context_adaptation
                if translation_context_adaptation is not None
                else self._opts.translation_config.context_adaptation,
                context=translation_context
                if translation_context is not None
                else self._opts.translation_config.context,
                informal=translation_informal
                if translation_informal is not None
                else self._opts.translation_config.informal,
            )
            self._opts.translation_config = translation_config

        if pre_processing_audio_enhancer is not None or pre_processing_speech_threshold is not None:
            self._opts.pre_processing = dataclasses.replace(
                self._opts.pre_processing,
                audio_enhancer=pre_processing_audio_enhancer
                if pre_processing_audio_enhancer is not None
                else self._opts.pre_processing.audio_enhancer,
                speech_threshold=pre_processing_speech_threshold
                if pre_processing_speech_threshold is not None
                else self._opts.pre_processing.speech_threshold,
            )

        if model is not None:
            self._opts.model = model
        if endpointing is not None:
            self._opts.endpointing = endpointing
        if maximum_duration_without_endpointing is not None:
            self._opts.maximum_duration_without_endpointing = maximum_duration_without_endpointing
        if interim_results is not None:
            self._opts.interim_results = interim_results
        if sample_rate is not None:
            self._opts.sample_rate = sample_rate
        if bit_depth is not None:
            self._opts.bit_depth = bit_depth
        if channels is not None:
            self._opts.channels = channels
        if region is not None:
            self._opts.region = region
        if encoding is not None:
            self._opts.encoding = encoding
        if custom_vocabulary is not None:
            self._opts.custom_vocabulary = custom_vocabulary
        if custom_spelling is not None:
            self._opts.custom_spelling = custom_spelling

        for stream in self._streams:
            stream.update_options(
                model=model,
                languages=languages,
                code_switching=code_switching,
                interim_results=interim_results,
                sample_rate=sample_rate,
                bit_depth=bit_depth,
                channels=channels,
                region=region,
                endpointing=endpointing,
                maximum_duration_without_endpointing=maximum_duration_without_endpointing,
                encoding=encoding,
                translation_enabled=translation_enabled,
                translation_target_languages=translation_target_languages,
                translation_model=translation_model,
                translation_match_original_utterances=translation_match_original_utterances,
                translation_lipsync=translation_lipsync,
                translation_context_adaptation=translation_context_adaptation,
                translation_context=translation_context,
                translation_informal=translation_informal,
                custom_vocabulary=custom_vocabulary,
                custom_spelling=custom_spelling,
                pre_processing_audio_enhancer=pre_processing_audio_enhancer,
                pre_processing_speech_threshold=pre_processing_speech_threshold,
            )

    def _sanitize_options(self, *, languages: list[str] | None = None) -> STTOptions:
        config = dataclasses.replace(self._opts)
        if languages is not None:
            language_config = dataclasses.replace(
                config.language_config,
                languages=languages,
            )
            config.language_config = language_config
        return config


class _Word(TypedDict):
    word: str
    start: float
    end: float


class _Utterance(TypedDict):
    text: str
    start: NotRequired[float]
    end: NotRequired[float]
    language: NotRequired[str | None]
    confidence: NotRequired[float]
    words: NotRequired[list[_Word]]


class _TranscriptData(TypedDict):
    is_final: bool
    utterance: _Utterance


class _TranslationData(TypedDict):
    utterance: NotRequired[_Utterance]
    translated_utterance: NotRequired[_Utterance | None]
    target_language: NotRequired[str]


class _Transcript(TypedDict):
    type: Literal["transcript"]
    data: _TranscriptData


class _Translation(TypedDict):
    type: Literal["translation"]
    data: _TranslationData


class _PostFinalTranscript(TypedDict):
    type: Literal["post_final_transcript"]


class _ErrorData(TypedDict):
    message: NotRequired[str]


class _Error(TypedDict):
    type: Literal["error"]
    data: NotRequired[_ErrorData]


class _AudioChunkData(TypedDict):
    chunk: str


class _AudioChunk(TypedDict):
    type: Literal["audio_chunk"]
    data: _AudioChunkData


class _StopRecording(TypedDict):
    type: Literal["stop_recording"]


_STOP = _StopRecording(type="stop_recording")


class SpeechStream(
    stt.WebSocketStream[
        _Transcript | _Translation | _PostFinalTranscript | _Error, _AudioChunk | _StopRecording
    ]
):
    frame_ms = 100
    # stop_recording ends Gladia's session: the audio after a flush goes on a new one
    new_session_per_flush = True

    def __init__(self, *, stt: STT, opts: STTOptions, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._gladia_stt = stt
        self._opts = opts
        self._audio_energy_filter: AudioEnergyFilter | None = None
        if opts.energy_filter:
            if isinstance(opts.energy_filter, AudioEnergyFilter):
                self._audio_energy_filter = opts.energy_filter
            else:
                self._audio_energy_filter = AudioEnergyFilter()
        self._session_id = ""

    def update_options(
        self,
        *,
        model: GladiaModels | None = None,
        languages: list[str] | None = None,
        code_switching: bool | None = None,
        interim_results: bool | None = None,
        sample_rate: int | None = None,
        bit_depth: Literal[8, 16, 24, 32] | None = None,
        channels: int | None = None,
        region: Literal["us-west", "eu-west"] | None = None,
        endpointing: float | None = None,
        maximum_duration_without_endpointing: float | None = None,
        encoding: Literal["wav/pcm", "wav/alaw", "wav/ulaw"] | None = None,
        translation_enabled: bool | None = None,
        translation_target_languages: list[str] | None = None,
        translation_model: str | None = None,
        translation_match_original_utterances: bool | None = None,
        translation_lipsync: bool | None = None,
        translation_context_adaptation: bool | None = None,
        translation_context: str | None = None,
        translation_informal: bool | None = None,
        custom_vocabulary: list[str | dict[str, object]] | None = None,
        custom_spelling: dict[str, list[str]] | None = None,
        pre_processing_audio_enhancer: bool | None = None,
        pre_processing_speech_threshold: float | None = None,
    ) -> None:
        if languages is not None or code_switching is not None:
            language_config = dataclasses.replace(
                self._opts.language_config,
                languages=languages
                if languages is not None
                else self._opts.language_config.languages,
                code_switching=code_switching
                if code_switching is not None
                else self._opts.language_config.code_switching,
            )
            self._opts.language_config = language_config

        if (
            translation_enabled is not None
            or translation_target_languages is not None
            or translation_model is not None
            or translation_match_original_utterances is not None
            or translation_lipsync is not None
            or translation_context_adaptation is not None
            or translation_context is not None
            or translation_informal is not None
        ):
            translation_config = dataclasses.replace(
                self._opts.translation_config,
                enabled=translation_enabled
                if translation_enabled is not None
                else self._opts.translation_config.enabled,
                target_languages=translation_target_languages
                if translation_target_languages is not None
                else self._opts.translation_config.target_languages,
                model=translation_model
                if translation_model is not None
                else self._opts.translation_config.model,
                match_original_utterances=translation_match_original_utterances
                if translation_match_original_utterances is not None
                else self._opts.translation_config.match_original_utterances,
                lipsync=translation_lipsync
                if translation_lipsync is not None
                else self._opts.translation_config.lipsync,
                context_adaptation=translation_context_adaptation
                if translation_context_adaptation is not None
                else self._opts.translation_config.context_adaptation,
                context=translation_context
                if translation_context is not None
                else self._opts.translation_config.context,
                informal=translation_informal
                if translation_informal is not None
                else self._opts.translation_config.informal,
            )
            self._opts.translation_config = translation_config

        if pre_processing_audio_enhancer is not None or pre_processing_speech_threshold is not None:
            self._opts.pre_processing = dataclasses.replace(
                self._opts.pre_processing,
                audio_enhancer=pre_processing_audio_enhancer
                if pre_processing_audio_enhancer is not None
                else self._opts.pre_processing.audio_enhancer,
                speech_threshold=pre_processing_speech_threshold
                if pre_processing_speech_threshold is not None
                else self._opts.pre_processing.speech_threshold,
            )

        if model is not None:
            self._opts.model = model
        if endpointing is not None:
            self._opts.endpointing = endpointing
        if maximum_duration_without_endpointing is not None:
            self._opts.maximum_duration_without_endpointing = maximum_duration_without_endpointing
        if interim_results is not None:
            self._opts.interim_results = interim_results
        if sample_rate is not None:
            self._opts.sample_rate = sample_rate
        if bit_depth is not None:
            self._opts.bit_depth = bit_depth
        if channels is not None:
            self._opts.channels = channels
        if region is not None:
            self._opts.region = region
        if encoding is not None:
            self._opts.encoding = encoding
        if custom_vocabulary is not None:
            self._opts.custom_vocabulary = custom_vocabulary
        if custom_spelling is not None:
            self._opts.custom_spelling = custom_spelling

        # the options are read at each connection's session initialization
        self.reconnect("options changed")

    async def connect(self, conn_options: APIConnectOptions) -> stt.WebSocketTarget:
        session = await self._gladia_stt._init_live_session(self._opts, conn_options)
        self._session_id = session.id
        return stt.WebSocketTarget(session.url)

    def encode(
        self, item: rtc.AudioFrame | stt.Flush | stt.EndOfAudio
    ) -> _AudioChunk | _StopRecording:
        match item:
            case rtc.AudioFrame():
                if self._audio_energy_filter is not None:
                    state = self._audio_energy_filter.update(item)
                    if state in (AudioEnergyFilter.State.SILENCE, AudioEnergyFilter.State.END):
                        # Provider timestamps count uploaded samples, including quiet stretches.
                        return _audio_chunk(bytes(item.data.nbytes))
                return _audio_chunk(bytes(item.data))
            case stt.Flush() | stt.EndOfAudio():
                return _STOP
            case _:
                assert_never(item)

    def on_message(
        self, message: _Transcript | _Translation | _PostFinalTranscript | _Error
    ) -> None:
        if message["type"] == "transcript":
            data = message["data"]
            # with translation, the translation is the final
            final = data["is_final"]
            if (final and not self._opts.translation_config.enabled) or (
                not final and self._opts.interim_results
            ):
                self.report(self._transcript(data["utterance"], final=final))
        elif message["type"] == "translation":
            # https://docs.gladia.io/reference/realtime-messages/translation
            translation = message["data"]
            translated = translation.get("translated_utterance")
            if self._opts.translation_config.enabled and translated is not None:
                original = translation.get("utterance")
                original_language = original.get("language") if original else None
                self.report(
                    self._transcript(
                        translated,
                        final=True,
                        language=translated.get("language") or translation.get("target_language"),
                        source_languages=[LanguageCode(original_language)]
                        if original_language
                        else None,
                        source_texts=[original["text"] if original else ""]
                        if original_language
                        else None,
                    )
                )
        elif message["type"] == "post_final_transcript":
            # the last message of a session stop_recording ended
            self.report(stt.SessionFinished())
        elif message["type"] == "error":
            raise APIError(
                "Gladia WebSocket error",
                kind="invalid_request",
                detail=message.get("data", {}).get("message"),
            )
        else:
            self._skip_unknown_message(message)

    def _transcript(
        self,
        utterance: _Utterance,
        *,
        final: bool,
        language: str | None = None,
        source_languages: list[LanguageCode] | None = None,
        source_texts: list[str] | None = None,
    ) -> stt.Transcript:
        offset = self.start_time_offset
        languages = self._opts.language_config.languages
        return stt.Transcript(
            utterance["text"].strip(),
            LanguageCode(
                language or utterance.get("language") or (languages[0] if languages else "en")
            ),
            final=final,
            request_id=self._session_id,
            start_time=utterance.get("start", 0.0) + offset,
            end_time=utterance.get("end", 0.0) + offset,
            confidence=utterance.get("confidence", 1.0),
            source_languages=source_languages,
            source_texts=source_texts,
            words=[
                TranscriptChunk(
                    text=word["word"],
                    start_time=word["start"] + offset,
                    end_time=word["end"] + offset,
                    start_time_offset=offset,
                )
                for word in utterance.get("words", [])
            ],
        )


def _audio_chunk(pcm: bytes) -> _AudioChunk:
    return _AudioChunk(
        type="audio_chunk", data=_AudioChunkData(chunk=base64.b64encode(pcm).decode("utf-8"))
    )
