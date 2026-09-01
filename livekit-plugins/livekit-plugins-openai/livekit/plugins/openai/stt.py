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
import base64
import contextlib
import dataclasses
import json
import os
import time
import weakref
from dataclasses import dataclass, field
from urllib.parse import urlencode, urlparse

import aiohttp
import httpx
from pydantic import TypeAdapter

import openai
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    inference,
    stt,
    utils,
    vad,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given
from openai.types.audio import Transcription, TranscriptionVerbose
from openai.types.beta.realtime.transcription_session_update_param import (
    SessionTurnDetection,
)
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
from .utils import AsyncAzureADTokenProvider

# OpenAI Realtime API has a timeout of 15 mins, we'll attempt to restart the session
# before that timeout is reached
_max_session_duration = 10 * 60
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
    keywords: list[str] = field(default_factory=list)
    prompt: NotGivenOr[str] = NOT_GIVEN
    noise_reduction_type: NotGivenOr[str] = NOT_GIVEN
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
        noise_reduction_type: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        client: openai.AsyncClient | None = None,
        use_realtime: NotGivenOr[bool] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
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
        self._session: aiohttp.ClientSession | None = None
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            max_session_duration=_max_session_duration,
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return self._client._base_url.netloc.decode("utf-8")

    @staticmethod
    def with_azure(
        *,
        language: str | list[str] = "en",
        detect_language: bool = False,
        model: STTModels | str = "gpt-4o-mini-transcribe",
        prompt: NotGivenOr[str] = NOT_GIVEN,
        keywords: NotGivenOr[list[str]] = NOT_GIVEN,
        turn_detection: NotGivenOr[SessionTurnDetection] = NOT_GIVEN,
        noise_reduction_type: NotGivenOr[str] = NOT_GIVEN,
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
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
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
            pool=self._pool,
            conn_options=conn_options,
            opts=opts,
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
        noise_reduction_type: NotGivenOr[str] = NOT_GIVEN,
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
                _validate_context(resolved_model, stream._opts.languages, user_keywords)

        # the transport is fixed at construction: AgentSession wraps a non-streaming STT once
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
        model_changed = resolved_model != self._opts.model
        # a stream keeps a language of its own unless this call names or moves the language
        languages_given = is_given(language) or languages_changed
        self._opts.model = resolved_model
        self._capabilities.keyterms = _supports_context_hints(resolved_model)
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

        if model_changed:
            # every stream has dropped its own socket by now, so this reaches the idle ones the
            # pool holds between two speech sessions
            self._pool.invalidate()

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

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        query_params: dict[str, str] = {
            "intent": "transcription",
        }
        # OpenAI's native realtime endpoint treats ?model= as selecting a
        # conversation session and rejects the transcription-mode
        # session.update with invalid_model — the model is conveyed via
        # audio.input.transcription.model instead. Gateways need the model
        # on the upgrade URL to route the connection before the first frame.
        if urlparse(str(self._client.base_url)).hostname != "api.openai.com":
            query_params["model"] = self._opts.model
        headers = {
            "User-Agent": "LiveKit Agents",
            "Authorization": f"Bearer {self._client.api_key}",
        }
        url = f"{str(self._client.base_url).rstrip('/')}/realtime?{urlencode(query_params)}"
        if url.startswith("http"):
            url = url.replace("http", "ws", 1)

        session = self._ensure_session()
        # the config is sent once the stream acquires the connection, since the pool also
        # hands back sockets it opened earlier
        return await asyncio.wait_for(session.ws_connect(url, headers=headers), timeout)

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    async def aclose(self) -> None:
        """Close the websocket connection pool this STT owns.

        Without this the pool's ``close_cb`` never runs for pooled sockets, so
        they stay open after the STT is closed.
        """
        await self._pool.aclose()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str | list[str]] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
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
            resp = await self._client.audio.transcriptions.create(
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
                timeout=httpx.Timeout(30, connect=conn_options.timeout),
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


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        pool: utils.ConnectionPool[aiohttp.ClientWebSocketResponse],
        opts: _STTOptions,
        vad_instance: vad.VAD | None = None,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)

        self._pool = pool
        # this stream's own options, and the STT's to refresh them from
        self._opts = opts
        self._stt: STT = stt
        self._language = _transcript_language(opts.languages)
        self._request_id = ""
        self._reconnect_event = asyncio.Event()
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._update_task: asyncio.Task[None] | None = None
        # serializes the config a connection is set up with against any later change to it
        self._config_lock = asyncio.Lock()
        self._vad = vad_instance
        self._speaking = False

    def update_options(self, *, language: NotGivenOr[str | list[str]] = NOT_GIVEN) -> None:
        """Set the language for this stream alone, and take the rest from the STT."""
        # only a named language moves this stream off its own
        opts = dataclasses.replace(self._stt._opts, languages=self._opts.languages)
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
            # only this stream's own socket: invalidating the pool would close the ones the
            # other streams are still reading from
            if self._ws is not None:
                self._pool.remove(self._ws)
            self._reconnect_event.set()
        else:
            self._update_task = asyncio.create_task(
                self._send_update(_session_update(opts), old_task=self._update_task)
            )

    async def _send_update(
        self, event: SessionUpdateEvent, *, old_task: asyncio.Task[None] | None
    ) -> None:
        if old_task is not None:
            with contextlib.suppress(Exception):
                await old_task

        ws = self._ws
        if ws is None:
            return  # the next connection is configured when the stream acquires it

        try:
            async with self._config_lock:
                await ws.send_json(event.model_dump(by_alias=True, exclude_unset=True))
        except Exception:
            # a dropped update is not worth failing the stream over
            logger.warning("failed to update the transcription session", exc_info=True)

    async def aclose(self) -> None:
        if self._update_task is not None:
            await utils.aio.gracefully_cancel(self._update_task)

        await super().aclose()

    def _start_speaking(self) -> None:
        if self._speaking:
            return
        self._speaking = True
        self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))

    def _stop_speaking(self) -> None:
        if not self._speaking:
            return
        self._speaking = False
        self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

    @utils.log_exceptions(logger=logger)
    async def _run(self) -> None:
        closing_ws = False

        @utils.log_exceptions(logger=logger)
        async def send_task(
            ws: aiohttp.ClientWebSocketResponse, vad_stream: vad.VADStream | None
        ) -> None:
            nonlocal closing_ws

            # forward audio to OAI in chunks of 50ms
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=SAMPLE_RATE // 20,
            )

            try:
                async for data in self._input_ch:
                    frames: list[rtc.AudioFrame] = []
                    if isinstance(data, rtc.AudioFrame):
                        if vad_stream is not None:
                            vad_stream.push_frame(data)
                        frames.extend(audio_bstream.write(data.data.tobytes()))
                    elif isinstance(data, self._FlushSentinel):
                        frames.extend(audio_bstream.flush())

                    for frame in frames:
                        encoded_frame = {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(frame.data.tobytes()).decode("utf-8"),
                        }
                        await ws.send_json(encoded_frame)
            except (aiohttp.ClientError, ConnectionError) as e:
                if closing_ws:
                    return
                raise APIConnectionError(
                    "OpenAI Realtime STT connection closed unexpectedly"
                ) from e
            finally:
                if vad_stream is not None:
                    vad_stream.end_input()

            closing_ws = True

        @utils.log_exceptions(logger=logger)
        async def vad_task(ws: aiohttp.ClientWebSocketResponse, vad_stream: vad.VADStream) -> None:
            try:
                async for ev in vad_stream:
                    if ev.type == vad.VADEventType.START_OF_SPEECH:
                        self._start_speaking()
                    elif ev.type == vad.VADEventType.END_OF_SPEECH:
                        self._stop_speaking()
                        # a server-endpointed model closes the segment itself; this would cut it twice
                        if _is_realtime_only(self._opts.model):
                            await ws.send_json({"type": "input_audio_buffer.commit"})
            except (aiohttp.ClientError, ConnectionError) as e:
                if closing_ws:
                    return
                raise APIConnectionError(
                    "OpenAI Realtime STT connection closed unexpectedly"
                ) from e

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            current_text = ""
            current_item_id = ""
            last_interim_at: float = 0
            connected_at = time.time()
            item_audio_timing: dict[str, dict[str, int]] = {}
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws:  # close is expected, see SpeechStream.aclose
                        return

                    # this will trigger a reconnection, see the _run loop
                    raise APIStatusError(
                        message="OpenAI Realtime STT connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected OpenAI message type %s", msg.type)
                    continue

                try:
                    data = json.loads(msg.data)
                    msg_type = data.get("type")
                    if msg_type == "input_audio_buffer.speech_started":
                        item_id = data.get("item_id", "")
                        current_item_id = item_id
                        audio_start_ms = data.get("audio_start_ms", 0)
                        item_audio_timing[item_id] = {"start_ms": audio_start_ms}
                        if self._vad is None:
                            self._start_speaking()

                    elif msg_type == "input_audio_buffer.speech_stopped":
                        item_id = data.get("item_id", "")
                        audio_end_ms = data.get("audio_end_ms", 0)
                        if item_id in item_audio_timing:
                            item_audio_timing[item_id]["end_ms"] = audio_end_ms
                        if self._vad is None:
                            self._stop_speaking()

                    elif msg_type == "conversation.item.input_audio_transcription.delta":
                        delta = data.get("delta", "")
                        item_id = data.get("item_id", "") or current_item_id
                        if item_id:
                            current_item_id = item_id
                        if delta:
                            current_text += delta
                            if time.time() - last_interim_at > _delta_transcript_interval:
                                self._event_ch.send_nowait(
                                    stt.SpeechEvent(
                                        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                        request_id=current_item_id,
                                        alternatives=[
                                            stt.SpeechData(
                                                text=current_text,
                                                language=self._language,
                                            )
                                        ],
                                    )
                                )
                                last_interim_at = time.time()

                    elif msg_type == "conversation.item.input_audio_transcription.completed":
                        current_text = ""
                        transcript = data.get("transcript", "")
                        item_id = data.get("item_id", "")
                        # what the model detected beats the hint that was sent
                        detected = [
                            LanguageCode(lang["code"])
                            for lang in data.get("languages") or []
                            if lang.get("code")
                        ]

                        if transcript:
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(
                                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                    request_id=item_id,
                                    alternatives=[
                                        stt.SpeechData(
                                            text=transcript,
                                            language=detected[0] if detected else self._language,
                                        )
                                    ],
                                )
                            )

                        audio_duration = 0.0
                        if item_id in item_audio_timing:
                            timing = item_audio_timing[item_id]
                            start_ms = timing.get("start_ms", 0)
                            end_ms = timing.get("end_ms", 0)
                            if end_ms > start_ms:
                                audio_duration = (end_ms - start_ms) / 1000.0
                            del item_audio_timing[item_id]

                        # extract token usage if available
                        usage = data.get("usage", {})
                        input_tokens = usage.get("input_tokens", 0)
                        output_tokens = usage.get("output_tokens", 0)

                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.RECOGNITION_USAGE,
                                alternatives=[],
                                recognition_usage=stt.RecognitionUsage(
                                    audio_duration=audio_duration,
                                    input_tokens=input_tokens,
                                    output_tokens=output_tokens,
                                ),
                            )
                        )

                        # restart session if needed
                        if time.time() - connected_at > _max_session_duration:
                            logger.info("resetting Realtime STT session due to timeout")
                            self._pool.remove(ws)
                            self._reconnect_event.set()
                            return
                    elif msg_type == "error":
                        error_body = data.get("error", {})
                        raise APIError(
                            message=f"OpenAI Realtime STT error: {error_body.get('message', 'Unknown error')}",
                            body=error_body,
                            retryable=False,
                        )

                except APIError:
                    raise
                except Exception:
                    logger.exception("failed to process OpenAI message")

        while True:
            closing_ws = False  # reset the flag
            # a segment left open across the reconnect gap would fuse into the next utterance
            self._stop_speaking()
            async with self._pool.connection(timeout=self._conn_options.timeout) as ws:
                if self._pool.last_connection_reused and not self._opts.languages:
                    # detecting the language needs a session that was never given one, and a
                    # pooled socket carries whatever the stream before it set
                    self._pool.remove(ws)
                    continue
                self._report_connection_acquired(
                    self._pool.last_acquire_time, self._pool.last_connection_reused
                )
                # published before the config is sent, so a change made during the send lands
                # after it rather than finding no live connection
                self._ws = ws
                # a pooled connection may carry older options; the lock is taken before the
                # first await, so no queued update overtakes this one
                try:
                    async with self._config_lock:
                        await ws.send_json(
                            _session_update(self._opts).model_dump(
                                by_alias=True, exclude_unset=True
                            )
                        )
                except (aiohttp.ClientError, ConnectionError) as e:
                    # the retry gets its own connection; no update should reach this one
                    self._ws = None
                    # a pooled socket can die while idle; keep that retryable
                    raise APIConnectionError(
                        "OpenAI Realtime STT connection closed unexpectedly"
                    ) from e
                vad_stream = self._vad.stream() if self._vad is not None else None
                tasks = [
                    asyncio.create_task(send_task(ws, vad_stream)),
                    asyncio.create_task(recv_task(ws)),
                ]
                if vad_stream is not None:
                    tasks.append(asyncio.create_task(vad_task(ws, vad_stream)))
                tasks_group = asyncio.gather(*tasks)
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())
                try:
                    done, _ = await asyncio.wait(
                        (tasks_group, wait_reconnect_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # propagate exceptions from completed tasks
                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    self._ws = None
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
                    tasks_group.cancel()
                    tasks_group.exception()  # retrieve the exception
                    if vad_stream is not None:
                        await vad_stream.aclose()
