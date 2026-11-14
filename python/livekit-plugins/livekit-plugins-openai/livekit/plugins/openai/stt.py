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

import base64
import dataclasses
import os
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, NotRequired, TypedDict
from urllib.parse import urlencode, urlparse

import httpx2 as httpx
from pydantic import Field, TypeAdapter

import openai
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    VAD,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    TokenUsage,
    VADEventType,
    VADStream,
    http,
    inference,
    stt,
    utils,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from openai.types.audio import Transcription, TranscriptionVerbose
from openai.types.beta.realtime.transcription_session_update_param import (
    SessionTurnDetection,
)
from openai.types.realtime import NoiseReductionType
from openai.types.realtime.audio_transcription import AudioTranscription
from openai.types.realtime.realtime_audio_formats import AudioPCM
from openai.types.realtime.realtime_transcription_session_audio import (
    RealtimeTranscriptionSessionAudio,
)
from openai.types.realtime.realtime_transcription_session_audio_input import (
    NoiseReduction,
    RealtimeTranscriptionSessionAudioInput,
)
from openai.types.realtime.realtime_transcription_session_audio_input_turn_detection import (
    RealtimeTranscriptionSessionAudioInputTurnDetection,
)
from openai.types.realtime.realtime_transcription_session_create_request import (
    RealtimeTranscriptionSessionCreateRequest,
)
from openai.types.realtime.session_update_event import SessionUpdateEvent

from .log import logger
from .models import STTModels
from .utils import AsyncAzureADTokenProvider, error_kind, http_timeout

# OpenAI Realtime API has a timeout of 15 mins, we'll attempt to restart the session
# before that timeout is reached
_max_session_duration = 10.0 * 60
# emit interim transcriptions every 0.5 seconds
_delta_transcript_interval = 0.5
SAMPLE_RATE = 24000
NUM_CHANNELS = 1
# turn_detection is a discriminated union, so a plain dict needs an adapter to validate
_TURN_DETECTION: TypeAdapter[RealtimeTranscriptionSessionAudioInputTurnDetection] = TypeAdapter(
    RealtimeTranscriptionSessionAudioInputTurnDetection
)

# realtime-only, and with no server-side endpointing: they reject turn_detection and emit no
# speech_started/stopped, so the client commits the buffer to close a segment
_REALTIME_ONLY_MODELS = ("gpt-realtime-whisper", "gpt-live-transcribe")
# these take a plural `languages` list and `keywords`; earlier models take neither
_CONTEXT_HINT_MODELS = ("gpt-transcribe", "gpt-live-transcribe")


def _is_realtime_only(model: str) -> bool:
    return model.startswith(_REALTIME_ONLY_MODELS)


def _supports_context_hints(model: str) -> bool:
    return model.startswith(_CONTEXT_HINT_MODELS)


def _as_languages(language: str | list[str]) -> list[str]:
    """Wrap a single code into a list, dropping empty strings."""
    if isinstance(language, str):
        return [language] if language else []
    return [code for code in language if code]


def _as_turn_detection(
    turn_detection: SessionTurnDetection,
) -> RealtimeTranscriptionSessionAudioInputTurnDetection:
    """Validate a turn-detection dict, filling in the `type` the union is tagged on."""
    return _TURN_DETECTION.validate_python({"type": "server_vad", **turn_detection})


def _validate_context(model: str, languages: list[str], keywords: list[str]) -> None:
    """Raise when keywords or several languages go to a model that takes neither."""
    if _supports_context_hints(model):
        return
    supported = " and ".join(_CONTEXT_HINT_MODELS)
    if keywords:
        raise ValueError(f"keywords are only supported by {supported}, not {model}")
    if len(languages) > 1:
        raise ValueError(f"{model} accepts a single language; only {supported} accept a list")


def _transcript_language(languages: list[str]) -> LanguageCode:
    """The code to tag transcripts with, empty unless exactly one language is set."""
    return LanguageCode(languages[0]) if len(languages) == 1 else LanguageCode("")


@dataclass
class _STTOptions:
    model: STTModels | str
    languages: list[str]
    detect_language: bool
    turn_detection: RealtimeTranscriptionSessionAudioInputTurnDetection
    keywords: list[str] = field(default_factory=list[str])
    prompt: NotGivenOr[str] = NOT_GIVEN
    noise_reduction_type: NotGivenOr[NoiseReductionType] = NOT_GIVEN
    temperature: NotGivenOr[float] = NOT_GIVEN


def _transcription(opts: _STTOptions) -> AudioTranscription:
    """The transcription config: `languages` and `keywords`, or a single `language`."""
    transcription = AudioTranscription(model=opts.model)
    # a field left out of a session.update keeps its previous value, so anything that can
    # be cleared is always sent
    if is_given(opts.prompt):
        transcription.prompt = opts.prompt
    if _supports_context_hints(opts.model):
        transcription.keywords = opts.keywords
        # `languages` rejects both an empty array and null, so it can only be replaced
        if opts.languages:
            transcription.languages = list(
                dict.fromkeys(LanguageCode(lang).language for lang in opts.languages)
            )
    elif opts.languages:
        transcription.language = LanguageCode(opts.languages[0]).language
    return transcription


def _session_update(opts: _STTOptions) -> SessionUpdateEvent:
    """The `session.update` event for the current config."""
    audio_input = RealtimeTranscriptionSessionAudioInput(
        format=AudioPCM(rate=SAMPLE_RATE, type="audio/pcm"),
        transcription=_transcription(opts),
    )
    # leave the field unset for models that reject it; the rest get server-side VAD
    if not _is_realtime_only(opts.model):
        audio_input.turn_detection = opts.turn_detection

    if opts.noise_reduction_type:
        audio_input.noise_reduction = NoiseReduction(type=opts.noise_reduction_type)

    return SessionUpdateEvent(
        type="session.update",
        session=RealtimeTranscriptionSessionCreateRequest(
            type="transcription",
            audio=RealtimeTranscriptionSessionAudio(input=audio_input),
        ),
    )


class STT(stt.STT):
    def __init__(
        self,
        *,
        language: str | list[str] = "en",
        detect_language: bool = False,
        model: STTModels | str = "gpt-4o-mini-transcribe",
        prompt: NotGivenOr[str] = NOT_GIVEN,
        keywords: NotGivenOr[list[str]] = NOT_GIVEN,
        turn_detection: NotGivenOr[SessionTurnDetection] = NOT_GIVEN,
        noise_reduction_type: NotGivenOr[NoiseReductionType] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        client: openai.AsyncClient | None = None,
        use_realtime: NotGivenOr[bool] = NOT_GIVEN,
        vad: NotGivenOr[VAD | None] = NOT_GIVEN,
    ):
        """
        Create a new instance of OpenAI STT.

        Args:
            language: The language code to use for transcription (e.g., "en" for English).
                gpt-transcribe and gpt-live-transcribe accept a list for code-switched audio.
            detect_language: Whether to automatically detect the language.
            model: The OpenAI model to use for transcription.
            prompt: Optional free-form description of the audio, such as its topic or setting.
            keywords: Literal terms to expect, such as product names or acronyms. Only for
                gpt-transcribe and gpt-live-transcribe, and only a hint.
            turn_detection: When using realtime transcription, this controls how model detects the user is done speaking.
                Final transcripts are generated only after the turn is over. See: https://platform.openai.com/docs/guides/realtime-vad
                Ignored for `gpt-realtime-whisper` and `gpt-live-transcribe`, which do not
                support server-side turn detection.
            noise_reduction_type: Type of noise reduction to apply. "near_field" or "far_field"
                This isn't needed when using LiveKit's noise cancellation.
            temperature: Sampling temperature between 0 and 1. Lower values make the
                transcription more deterministic. Not supported for realtime transcription.
            base_url: Custom base URL for OpenAI API.
            api_key: Your OpenAI API key. If not provided, will use the OPENAI_API_KEY environment variable.
            client: Optional pre-configured OpenAI AsyncClient instance.
            use_realtime: Whether to use the realtime transcription API. Defaults to True for
                `gpt-realtime-whisper` and `gpt-live-transcribe`, which are served only there,
                and to False otherwise.
            vad: Optional Voice Activity Detector used to commit the audio buffer when the model
                does not support server-side turn detection (`gpt-realtime-whisper`,
                `gpt-live-transcribe`).
                When not provided and the model requires it, the bundled Silero VAD is used with
                default settings. Pass `vad=None` to opt out and drive
                `input_audio_buffer.commit` yourself.
        """  # noqa: E501

        if not is_given(use_realtime):
            use_realtime = _is_realtime_only(model)

        if use_realtime and is_given(temperature):
            logger.warning(
                "temperature is not supported for realtime transcription; "
                "ignoring the provided value"
            )
            temperature = NOT_GIVEN

        if use_realtime and _is_realtime_only(model):
            if is_given(turn_detection):
                logger.warning(
                    "turn_detection is not supported for %s; ignoring the provided value", model
                )
                turn_detection = NOT_GIVEN
            if not is_given(vad):
                vad = inference.VAD(model="silero")

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=use_realtime,
                interim_results=use_realtime,
                aligned_transcript=False,
                keyterms=_supports_context_hints(model),
                # streaming usage carries the server's speech timing or billed seconds
                reports_usage=True,
            )
        )
        # the last language asked for, kept while detection is on so it can be restored
        self._specified_languages = _as_languages(language)
        languages = [] if detect_language else self._specified_languages
        resolved_keywords = list(keywords) if is_given(keywords) else []
        _validate_context(model, languages, resolved_keywords)

        if not is_given(turn_detection):
            turn_detection = {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 600,
                "silence_duration_ms": 350,
            }

        self._opts = _STTOptions(
            languages=languages,
            detect_language=detect_language,
            model=model,
            prompt=prompt,
            keywords=resolved_keywords,
            turn_detection=_as_turn_detection(turn_detection),
            temperature=temperature,
        )
        if is_given(noise_reduction_type):
            self._opts.noise_reduction_type = noise_reduction_type

        # user keywords; _opts.keywords holds the effective set (user + session)
        self._user_keywords: list[str] = list(self._opts.keywords)
        self._session_keyterms: list[str] = []

        self._vad = vad if is_given(vad) else None
        # an explicit `vad=None` means the caller commits the audio buffer itself
        self._vad_opted_out = vad is None

        if is_given(api_key) and not api_key:
            raise ValueError(
                "OpenAI API key is required, either as argument or set"
                " OPENAI_API_KEY environment variable"
            )

        self._client = client or openai.AsyncClient(
            max_retries=0,
            api_key=api_key if is_given(api_key) else None,
            base_url=base_url if is_given(base_url) else None,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            ),
        )

        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return self._client.base_url.netloc.decode("utf-8")

    @staticmethod
    def with_azure(
        *,
        language: str | list[str] = "en",
        detect_language: bool = False,
        model: STTModels | str = "gpt-4o-mini-transcribe",
        prompt: NotGivenOr[str] = NOT_GIVEN,
        keywords: NotGivenOr[list[str]] = NOT_GIVEN,
        turn_detection: NotGivenOr[SessionTurnDetection] = NOT_GIVEN,
        noise_reduction_type: NotGivenOr[NoiseReductionType] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: AsyncAzureADTokenProvider | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
        use_realtime: NotGivenOr[bool] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
        vad: NotGivenOr[VAD | None] = NOT_GIVEN,
    ) -> STT:
        """
        Create a new instance of Azure OpenAI STT.

        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `AZURE_OPENAI_API_KEY`
        - `organization` from `OPENAI_ORG_ID`
        - `project` from `OPENAI_PROJECT_ID`
        - `azure_ad_token` from `AZURE_OPENAI_AD_TOKEN`
        - `api_version` from `OPENAI_API_VERSION`
        - `azure_endpoint` from `AZURE_OPENAI_ENDPOINT`
        """  # noqa: E501

        azure_client = openai.AsyncAzureOpenAI(
            max_retries=0,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            organization=organization,
            project=project,
            base_url=base_url,
            timeout=timeout
            if timeout
            else httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
        )  # type: ignore

        return STT(
            language=language,
            detect_language=detect_language,
            model=model,
            prompt=prompt,
            keywords=keywords,
            turn_detection=turn_detection,
            noise_reduction_type=noise_reduction_type,
            temperature=temperature,
            client=azure_client,
            use_realtime=use_realtime,
            vad=vad,
        )

    @staticmethod
    def with_ovhcloud(
        *,
        model: str = "whisper-large-v3-turbo",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
        client: openai.AsyncClient | None = None,
        language: str | list[str] = "en",
        detect_language: bool = False,
        prompt: NotGivenOr[str] = NOT_GIVEN,
    ) -> STT:
        """
        Create a new instance of OVHcloud AI Endpoints STT.

        ``api_key`` must be set to your OVHcloud AI Endpoints API key, either using the argument or by setting
        the ``OVHCLOUD_API_KEY`` environmental variable.
        """
        ovhcloud_api_key = api_key if is_given(api_key) else os.environ.get("OVHCLOUD_API_KEY")
        if not ovhcloud_api_key:
            raise ValueError("OVHcloud AI Endpoints API key is required")

        return STT(
            model=model,
            api_key=ovhcloud_api_key,
            base_url=base_url,
            client=client,
            language=language,
            detect_language=detect_language,
            prompt=prompt,
            use_realtime=False,
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str | list[str]] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        opts = dataclasses.replace(self._opts)
        if is_given(language):
            opts.languages = _as_languages(language)
            _validate_context(opts.model, opts.languages, opts.keywords)
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            opts=opts,
            options_factory=lambda: dataclasses.replace(self._opts),
            vad_instance=self._vad,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[str | list[str]] = NOT_GIVEN,
        detect_language: NotGivenOr[bool] = NOT_GIVEN,
        prompt: NotGivenOr[str] = NOT_GIVEN,
        keywords: NotGivenOr[list[str]] = NOT_GIVEN,
        turn_detection: NotGivenOr[SessionTurnDetection] = NOT_GIVEN,
        noise_reduction_type: NotGivenOr[NoiseReductionType] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """
        Update the options for the speech stream. Open streams apply the change in place;
        only a new `model` reconnects them.

        Args:
            language: The language to transcribe in, or a list for gpt-transcribe and
                gpt-live-transcribe. An empty list detects the language.
            detect_language: Whether to detect the language. Turning it off falls back to the
                last language asked for.
            model: The model to use for transcription.
            prompt: Optional free-form description of the audio.
            keywords: Literal terms to expect. Only for gpt-transcribe and gpt-live-transcribe.
            turn_detection: When using realtime, this controls how model detects the user is done speaking.
            noise_reduction_type: Type of noise reduction to apply. "near_field" or "far_field"
            temperature: Sampling temperature between 0 and 1. Not supported for realtime transcription.
        """  # noqa: E501
        # resolve first: an unsupported combination must raise before anything is applied
        resolved_model = model if is_given(model) else self._opts.model
        if is_given(language):
            languages = _as_languages(language)
        elif detect_language:
            languages = []
        elif detect_language is False and not self._opts.languages:
            logger.warning(
                "detect_language=False names no language, falling back to %s; "
                "pass `language` to transcribe in another",
                self._specified_languages,
            )
            languages = self._specified_languages
        else:
            languages = self._opts.languages
        user_keywords = list(keywords) if is_given(keywords) else self._user_keywords
        _validate_context(resolved_model, languages, user_keywords)
        if not is_given(language):
            # a stream keeps its own language, which the new model may not take
            for stream in self._streams:
                stream.validate_model(resolved_model, user_keywords)

        # capabilities are fixed at construction: a session wires the STT by them once
        if _supports_context_hints(resolved_model) != self.capabilities.keyterms:
            raise ValueError(
                f"{resolved_model} {'takes' if not self.capabilities.keyterms else 'rejects'} "
                "keyterms, which this STT was created without; create a new STT for it"
            )
        if is_given(model) and _is_realtime_only(model):
            if not self.capabilities.streaming:
                raise ValueError(
                    f"{resolved_model} is served only over the realtime API, and this STT was "
                    "created for the transcriptions endpoint; pass `use_realtime=True` to the "
                    "constructor to reach it"
                )
            if self._vad is None and not self._vad_opted_out:
                raise ValueError(
                    f"{resolved_model} has no server-side endpointing, so it needs a `vad` to "
                    "commit the audio buffer; pass one to the constructor, or pass `vad=None` "
                    "to drive `input_audio_buffer.commit` yourself"
                )

        languages_changed = languages != self._opts.languages
        # a stream keeps a language of its own unless this call names or moves the language
        languages_given = is_given(language) or languages_changed
        self._opts.model = resolved_model
        self._opts.languages = languages
        if languages:
            self._specified_languages = languages
        self._user_keywords = user_keywords
        # detected keyterms must not survive a switch to a model that rejects keywords
        self._opts.keywords = (
            list(dict.fromkeys([*user_keywords, *self._session_keyterms]))
            if self.capabilities.keyterms
            else []
        )
        if is_given(detect_language):
            self._opts.detect_language = detect_language
        if is_given(prompt):
            self._opts.prompt = prompt
        if is_given(turn_detection):
            self._opts.turn_detection = _as_turn_detection(turn_detection)
        if is_given(noise_reduction_type):
            self._opts.noise_reduction_type = noise_reduction_type
        if is_given(temperature):
            if self.capabilities.streaming:
                logger.warning(
                    "temperature is not supported for realtime transcription; "
                    "ignoring the provided value"
                )
            else:
                self._opts.temperature = temperature

        for stream in self._streams:
            stream.update_options(language=languages if languages_given else NOT_GIVEN)

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        if not self.capabilities.keyterms:
            super()._update_session_keyterms(keyterms)
            return
        if keyterms == self._session_keyterms:
            return
        self._session_keyterms = list(keyterms)
        self._opts.keywords = list(dict.fromkeys([*self._user_keywords, *keyterms]))
        for stream in self._streams:
            stream.update_options()

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str | list[str]] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        timeout = http_timeout(conn_options)
        if is_given(language):
            languages = _as_languages(language)
            _validate_context(self._opts.model, languages, self._opts.keywords)
            self._opts.languages = languages

        try:
            data = rtc.combine_audio_frames(buffer).to_wav_bytes()

            format = "json"
            if self._opts.model == "whisper-1":
                # verbose_json returns language and other details, only supported for whisper-1
                format = "verbose_json"

            transcription = _transcription(self._opts)
            resp = await self._client.with_options(max_retries=0).audio.transcriptions.create(
                file=(
                    "file.wav",
                    data,
                    "audio/wav",
                ),
                model=self._opts.model,  # type: ignore
                language=transcription.language or openai.omit,
                languages=transcription.languages or openai.omit,
                keywords=transcription.keywords or openai.omit,
                prompt=transcription.prompt or openai.omit,
                response_format=format,
                temperature=self._opts.temperature
                if is_given(self._opts.temperature)
                else openai.omit,
                timeout=timeout,
            )

            # the detected language beats the hint, dominant code first; an empty list keeps it
            sd = stt.SpeechData(text=resp.text, language=_transcript_language(self._opts.languages))
            if isinstance(resp, TranscriptionVerbose) and resp.language:
                sd.language = LanguageCode(resp.language)
            elif isinstance(resp, Transcription) and resp.languages:
                sd.language = LanguageCode(resp.languages[0].code)

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[sd],
            )

        except openai.APITimeoutError:
            raise APITimeoutError() from None
        except openai.APIStatusError as e:
            raise APIStatusError(
                e.message, status_code=e.status_code, request_id=e.request_id, body=e.body
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


class _SpeechStarted(TypedDict):
    type: Literal["input_audio_buffer.speech_started"]
    item_id: NotRequired[str]
    audio_start_ms: NotRequired[int]


class _SpeechStopped(TypedDict):
    type: Literal["input_audio_buffer.speech_stopped"]
    item_id: NotRequired[str]
    audio_end_ms: NotRequired[int]


class _Delta(TypedDict):
    type: Literal["conversation.item.input_audio_transcription.delta"]
    item_id: NotRequired[str | None]
    delta: NotRequired[str | None]


class _DetectedLanguage(TypedDict):
    code: NotRequired[str | None]


class _InputTokenDetails(TypedDict):
    audio_tokens: NotRequired[int | None]


class _Usage(TypedDict):
    """``{"type": "tokens", ...}`` or ``{"type": "duration", "seconds": N}``."""

    type: NotRequired[str]
    input_tokens: NotRequired[int]
    output_tokens: NotRequired[int]
    input_token_details: NotRequired[_InputTokenDetails | None]
    seconds: NotRequired[float]


class _Completed(TypedDict):
    type: Literal["conversation.item.input_audio_transcription.completed"]
    item_id: NotRequired[str]
    transcript: NotRequired[str | None]
    languages: NotRequired[list[_DetectedLanguage] | None]
    usage: NotRequired[_Usage | None]


class _ErrorBody(TypedDict):
    code: NotRequired[str | None]
    type: NotRequired[str | None]
    message: NotRequired[str | None]


class _Error(TypedDict):
    type: Literal["error"]
    error: NotRequired[_ErrorBody]


_Message = Annotated[
    _SpeechStarted | _SpeechStopped | _Delta | _Completed | _Error,
    Field(discriminator="type"),
]


class _Append(TypedDict):
    type: Literal["input_audio_buffer.append"]
    audio: str


class _Commit(TypedDict):
    type: Literal["input_audio_buffer.commit"]


class _SessionUpdate(TypedDict):
    type: Literal["session.update"]
    session: dict[str, Any]


_Command = _Append | _Commit | _SessionUpdate


class SpeechStream(stt.SocketStream[_Message, _Command]):
    # forward audio to OAI in chunks of 50ms
    frame_ms = 50
    # the Realtime API ends a session after 15 minutes: rotate before
    max_session_duration = _max_session_duration
    # the server's (or the local VAD's) speech_started/stopped bound the speech
    provider_vad = True

    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        opts: _STTOptions,
        options_factory: Callable[[], _STTOptions],
        vad_instance: VAD | None = None,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        # this stream's own options, and the STT's to refresh them from
        self._opts = opts
        self._options_factory = options_factory
        self._stt: STT = stt
        self._language = _transcript_language(opts.languages)
        self._vad = vad_instance
        self._vad_stream: VADStream | None = None
        # whether the server's or the local VAD's speech is open. The transcript of a speech
        # comes after its end, so the end waits for it: the final, or the next speech's start,
        # ends the speech
        self._provider_speaking = False
        self._text = ""
        self._item_id = ""
        self._interim_at = 0.0
        self._item_timing: dict[str, tuple[int, int | None]] = {}

    def validate_model(self, model: str, keywords: list[str]) -> None:
        """Validate a prospective model against this stream's independent language."""
        _validate_context(model, self._opts.languages, keywords)

    def update_options(self, *, language: NotGivenOr[str | list[str]] = NOT_GIVEN) -> None:
        """Set the language for this stream alone, and take the rest from the STT."""
        # only a named language moves this stream off its own
        opts = dataclasses.replace(self._options_factory(), languages=self._opts.languages)
        if is_given(language):
            opts.languages = _as_languages(language)
            _validate_context(opts.model, opts.languages, opts.keywords)
        # a session.update can set a language but never clear one, and gateways route on the
        # ?model= in the URL
        cleared_language = bool(self._opts.languages) and not opts.languages
        rebuild = opts.model != self._opts.model or cleared_language
        self._opts = opts
        self._language = _transcript_language(opts.languages)
        if rebuild:
            self.reconnect("options changed")
        else:
            self.send_command(self.setup())

    async def _run(self) -> None:
        if self._vad is None:
            await super()._run()
            return
        vad_stream = self._vad_stream = self._vad.stream()
        try:
            async with utils.aio.TaskSet() as tasks:
                await tasks.first_completed(super()._run(), self._listen(vad_stream))
        finally:
            self._vad_stream = None
            await vad_stream.aclose()

    async def _listen(self, vad_stream: VADStream) -> None:
        async for ev in vad_stream:
            if ev.type == VADEventType.START_OF_SPEECH:
                self._provider_speech(started=True)
            elif ev.type == VADEventType.END_OF_SPEECH:
                self._provider_speech(started=False)
                # a server-endpointed model closes the segment itself; this would cut it twice
                if _is_realtime_only(self._opts.model):
                    self.send_command(_Commit(type="input_audio_buffer.commit"))

    def _provider_speech(self, *, started: bool) -> None:
        self._provider_speaking = started
        if started:
            self.report(stt.SpeechEnded())
            self.report(stt.SpeechStarted())

    async def connect(self, conn_options: APIConnectOptions) -> http.SocketTarget:
        self._text = ""
        self._item_id = ""
        self._interim_at = 0.0
        self._item_timing.clear()
        query_params: dict[str, str] = {"intent": "transcription"}
        client = self._stt._client
        # OpenAI's native realtime endpoint treats ?model= as selecting a conversation session
        # and rejects the transcription-mode session.update with invalid_model: the model is
        # conveyed via audio.input.transcription.model instead. Gateways need the model on the
        # upgrade URL to route the connection before the first frame.
        if urlparse(str(client.base_url)).hostname != "api.openai.com":
            query_params["model"] = self._opts.model
        url = f"{str(client.base_url).rstrip('/')}/realtime?{urlencode(query_params)}"
        if url.startswith("http"):
            url = url.replace("http", "ws", 1)
        return http.SocketTarget(
            url,
            headers={
                "User-Agent": "LiveKit Agents",
                "Authorization": f"Bearer {client.api_key}",
            },
        )

    def setup(self) -> _SessionUpdate:
        event = _session_update(self._opts)
        return _SessionUpdate(
            type="session.update",
            session=event.session.model_dump(by_alias=True, exclude_unset=True),
        )

    def encode(
        self, item: rtc.AudioFrame | stt.RecognizeStream.Flush | http.EndOfInput
    ) -> http.Encoded[_Command]:
        if not isinstance(item, rtc.AudioFrame):
            return None
        if self._vad_stream is not None:
            self._vad_stream.push_frame(item)
        return _Append(
            type="input_audio_buffer.append",
            audio=base64.b64encode(item.data.tobytes()).decode("utf-8"),
        )

    def on_message(
        self, message: _SpeechStarted | _SpeechStopped | _Delta | _Completed | _Error
    ) -> stt.Report | list[stt.Report] | None:
        if message["type"] == "input_audio_buffer.speech_started":
            item_id = self._item_id = message.get("item_id", "")
            self._item_timing[item_id] = (message.get("audio_start_ms", 0), None)
            if self._vad is None:
                self._provider_speech(started=True)
        elif message["type"] == "input_audio_buffer.speech_stopped":
            item_id = message.get("item_id", "")
            if (timing := self._item_timing.get(item_id)) is not None:
                self._item_timing[item_id] = (timing[0], message.get("audio_end_ms", 0))
            if self._vad is None:
                self._provider_speech(started=False)
        elif message["type"] == "conversation.item.input_audio_transcription.delta":
            self._item_id = message.get("item_id") or self._item_id
            if delta := message.get("delta"):
                self._text += delta
                # interims at most every _delta_transcript_interval
                if time.time() - self._interim_at > _delta_transcript_interval:
                    self._interim_at = time.time()
                    return stt.Transcript(self._text, self._language, request_id=self._item_id)
        elif message["type"] == "conversation.item.input_audio_transcription.completed":
            self._text = ""
            self._completed(message)
        else:
            error = message.get("error", {})
            raise APIError(
                "OpenAI Realtime STT error",
                kind=error_kind(error.get("code") or error.get("type")),
                body=error,
                detail=error.get("message"),
            )
        return None

    def _completed(self, message: _Completed) -> None:
        item_id = message.get("item_id", "")
        if transcript := message.get("transcript"):
            # what the model detected beats the hint that was sent
            detected = [
                LanguageCode(code)
                for lang in message.get("languages") or []
                if (code := lang.get("code"))
            ]
            self.report(
                stt.Transcript(
                    transcript,
                    detected[0] if detected else self._language,
                    final=True,
                    request_id=item_id,
                )
            )
        if not self._provider_speaking:
            self.report(stt.SpeechEnded())

        audio_duration: float | None = None
        start_ms, end_ms = self._item_timing.pop(item_id, (0, None))
        if end_ms is not None and end_ms > start_ms:
            audio_duration = (end_ms - start_ms) / 1000.0
        usage = message.get("usage") or _Usage()
        tokens = TokenUsage()
        if usage.get("type") == "tokens":
            tokens = TokenUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                input_audio_tokens=(usage.get("input_token_details") or {}).get("audio_tokens"),
            )
        elif usage.get("type") == "duration":
            audio_duration = usage.get("seconds", audio_duration)
        self.send(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                recognition_usage=stt.RecognitionUsage(
                    audio_duration=audio_duration, tokens=tokens
                ),
            )
        )
