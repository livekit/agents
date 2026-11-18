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

import os
import weakref
from collections import Counter
from dataclasses import dataclass, field
from typing import Annotated, Literal, NotRequired, TypeAlias, TypedDict, assert_never

from pydantic import Field

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    LanguageCode,
    http,
    stt,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TranscriptChunk

from ._utils import to_deepgram_url
from .log import logger
from .models import DeepgramLanguages, DeepgramModels, FluxModels, TranscriptionResponse

_NOVA_URL = "https://api.deepgram.com/v1/listen"
_FLUX_URL = "wss://api.deepgram.com/v2/listen"
# Flux defaults its end-of-turn threshold to 0.7; an eager threshold may not exceed it
_DEFAULT_EOT_THRESHOLD = 0.7

_QueryValue: TypeAlias = str | float | bool | list[str]


def _is_flux(model: str) -> bool:
    """Flux models are served on /v2/listen only, Nova and older ones on /v1/listen."""
    return model.startswith("flux-")


@dataclass
class _Options:
    """What both model families take."""

    model: str
    language: LanguageCode
    endpoint_url: str
    sample_rate: int = 16000
    keyterm: list[str] = field(default_factory=list[str])
    tags: list[str] = field(default_factory=list[str])
    redact: list[str] = field(default_factory=list[str])
    numerals: bool = False
    profanity_filter: bool = False
    mip_opt_out: bool = False


@dataclass
class _NovaOptions(_Options):
    detect_language: bool = False
    interim_results: bool = True
    punctuate: bool = True
    smart_format: bool = False
    no_delay: bool = True
    endpointing_ms: int = 25
    enable_diarization: bool = False
    # filler words improve the turn detector's accuracy
    filler_words: bool = True
    keywords: list[tuple[str, float]] = field(default_factory=list[tuple[str, float]])
    vad_events: bool = True
    utterance_end_ms: int | None = None
    dictation: bool = False
    replace: dict[str, str] | None = None
    search: list[str] | None = None


@dataclass
class _FluxOptions(_Options):
    eager_eot_threshold: float | None = None
    eot_threshold: float | None = None
    eot_timeout_ms: int | None = None
    language_hint: list[str] = field(default_factory=list[str])


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: DeepgramModels | FluxModels | str = "nova-3",
        language: NotGivenOr[DeepgramLanguages | str] = NOT_GIVEN,
        sample_rate: int = 16000,
        keyterm: NotGivenOr[str | list[str]] = NOT_GIVEN,
        tags: NotGivenOr[list[str]] = NOT_GIVEN,
        numerals: bool = False,
        profanity_filter: bool = False,
        redact: NotGivenOr[str | list[str]] = NOT_GIVEN,
        mip_opt_out: bool = False,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: http.HttpSession | None = None,
        detect_language: NotGivenOr[bool] = NOT_GIVEN,
        interim_results: NotGivenOr[bool] = NOT_GIVEN,
        punctuate: NotGivenOr[bool] = NOT_GIVEN,
        smart_format: NotGivenOr[bool] = NOT_GIVEN,
        no_delay: NotGivenOr[bool] = NOT_GIVEN,
        endpointing_ms: NotGivenOr[int] = NOT_GIVEN,
        enable_diarization: NotGivenOr[bool] = NOT_GIVEN,
        filler_words: NotGivenOr[bool] = NOT_GIVEN,
        keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN,
        vad_events: NotGivenOr[bool] = NOT_GIVEN,
        utterance_end_ms: NotGivenOr[int | None] = NOT_GIVEN,
        dictation: NotGivenOr[bool] = NOT_GIVEN,
        replace: NotGivenOr[dict[str, str] | None] = NOT_GIVEN,
        search: NotGivenOr[list[str] | None] = NOT_GIVEN,
        eager_eot_threshold: NotGivenOr[float] = NOT_GIVEN,
        eot_threshold: NotGivenOr[float] = NOT_GIVEN,
        eot_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        language_hint: NotGivenOr[list[str]] = NOT_GIVEN,
    ) -> None:
        """Create a new instance of Deepgram STT.

        A `flux-*` model streams on Flux (`/v2/listen`), which detects turns itself; any other
        model uses `/v1/listen`. Each family takes the shared options and its own; an option of
        the other family raises `ValueError`.

        Args:
            model: The Deepgram model. Defaults to "nova-3".
            language: The language code. Defaults to "en-US", "en" for Flux.
            sample_rate: The sample rate of the audio in Hz. Defaults to 16000.
            keyterm: Key terms to improve recognition accuracy. Nova-3 and Flux only.
            tags: Tags added to the requests for usage reporting.
            numerals: Whether to convert spoken numbers into numerals. Defaults to False.
            profanity_filter: Whether to filter profanity. Defaults to False.
            redact: Values to redact: "pci", "numbers", "ssn" or "true" (everything). Flux takes
                only "numbers" and "aggressive_numbers".
                See https://developers.deepgram.com/docs/redaction
            mip_opt_out: Whether to opt out of the model improvement program. Defaults to False.
            api_key: Your Deepgram API key, else the DEEPGRAM_API_KEY environment variable.
            base_url: The full endpoint URL. Defaults to "https://api.deepgram.com/v1/listen",
                "wss://api.deepgram.com/v2/listen" for Flux.
            http_session: Optional HttpSession for batch requests.
            detect_language: Nova: detect the language (batch recognition only). Defaults to
                False.
            interim_results: Nova: return interim transcripts. Defaults to True.
            punctuate: Nova: punctuate the transcript. Defaults to True.
            smart_format: Nova: format numbers, dates and the like. Defaults to False.
            no_delay: Nova: smart formatting does not wait for a sequence to complete. Defaults
                to True.
            endpointing_ms: Nova: milliseconds of silence that end a speech, 0 to disable.
                Defaults to 25.
            enable_diarization: Nova: label speakers. Defaults to False.
            filler_words: Nova: keep filler words (um, uh). Defaults to True.
            keywords: Nova: (keyword, boost) pairs; not for Nova-3, which takes `keyterm`.
            vad_events: Nova: send SpeechStarted when speech is detected. Defaults to True.
            utterance_end_ms: Nova: milliseconds of silence after the last word that send
                UtteranceEnd. Requires interim results.
                See https://developers.deepgram.com/docs/understand-endpointing-interim-results
            dictation: Nova: convert spoken punctuation commands into marks. Defaults to False.
            replace: Nova: terms to replace in the transcript, original to replacement.
            search: Nova: terms to search for in the transcript.
            eager_eot_threshold: Flux: the eager end-of-turn threshold (0.3-0.9) that enables
                preemptive generation. Disabled by default.
            eot_threshold: Flux: the end-of-turn threshold (0.5-0.9). Defaults to 0.7.
            eot_timeout_ms: Flux: the end-of-turn timeout. Defaults to 3000.
            language_hint: Flux: languages to bias `flux-general-multi` toward.

        Raises:
            ValueError: If no API key is given or found, or an option of the other model
                family is given.
        """
        deepgram_api_key = api_key if is_given(api_key) else os.environ.get("DEEPGRAM_API_KEY")
        if not deepgram_api_key:
            raise ValueError(
                "Deepgram API key is required, either as argument or set"
                " DEEPGRAM_API_KEY environment variable"
            )
        self._api_key = deepgram_api_key
        self._opts: _NovaOptions | _FluxOptions = (
            _FluxOptions(
                model=model,
                language=LanguageCode("en"),
                endpoint_url=base_url if is_given(base_url) else _FLUX_URL,
            )
            if _is_flux(model)
            else _NovaOptions(
                model=model,
                language=LanguageCode("en-US"),
                endpoint_url=base_url if is_given(base_url) else _NOVA_URL,
            )
        )
        # user keyterms; _opts.keyterm holds the effective set (user + session)
        self._user_keyterm: list[str] = []
        self._session_keyterms: list[str] = []
        self._streams = weakref.WeakSet[_NovaStream | _FluxStream]()
        self._session = http_session
        self.update_options(
            model=model,
            language=language,
            sample_rate=sample_rate,
            keyterm=keyterm,
            tags=tags,
            numerals=numerals,
            profanity_filter=profanity_filter,
            redact=redact,
            mip_opt_out=mip_opt_out,
            detect_language=detect_language,
            interim_results=interim_results,
            punctuate=punctuate,
            smart_format=smart_format,
            no_delay=no_delay,
            endpointing_ms=endpointing_ms,
            enable_diarization=enable_diarization,
            filler_words=filler_words,
            keywords=keywords,
            vad_events=vad_events,
            utterance_end_ms=utterance_end_ms,
            dictation=dictation,
            replace=replace,
            search=search,
            eager_eot_threshold=eager_eot_threshold,
            eot_threshold=eot_threshold,
            eot_timeout_ms=eot_timeout_ms,
            language_hint=language_hint,
        )
        opts = self._opts
        nova = opts if isinstance(opts, _NovaOptions) else None
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=nova.interim_results if nova else True,
                diarization=nova.enable_diarization if nova else False,
                aligned_transcript="word",
                offline_recognize=nova is not None,
                keyterms=True,
            )
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Deepgram"

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[DeepgramLanguages | str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        opts = self._opts
        if isinstance(opts, _FluxOptions):
            raise NotImplementedError(f"{opts.model} only streams: use stream()")

        query: dict[str, _QueryValue] = {
            "model": opts.model,
            "punctuate": opts.punctuate,
            "detect_language": opts.detect_language,
            "smart_format": opts.smart_format,
            "profanity_filter": opts.profanity_filter,
            "numerals": opts.numerals,
            "mip_opt_out": opts.mip_opt_out,
        }
        if opts.keyterm:
            query["keyterm"] = opts.keyterm
        if opts.redact:
            query["redact"] = opts.redact
        if opts.enable_diarization:
            logger.warning("speaker diarization is not supported in non-streaming mode, ignoring")
        recognized = (
            None if opts.detect_language else (language if is_given(language) else opts.language)
        )
        if recognized:
            query["language"] = recognized

        try:
            async with self._ensure_session().post(
                to_deepgram_url(query, opts.endpoint_url, websocket=False, keywords=opts.keywords),
                data=rtc.combine_audio_frames(buffer).to_wav_bytes(),
                headers={
                    "Authorization": f"Token {self._api_key}",
                    "Accept": "application/json",
                    "Content-Type": "audio/wav",
                },
                conn_options=conn_options,
            ) as res:
                return prerecorded_transcription_to_speech_event(
                    recognized,
                    TranscriptionResponse.model_validate_json(await res.read()),
                    use_punctuated_word=opts.punctuate or opts.smart_format,
                )
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e

    def stream(
        self,
        *,
        language: NotGivenOr[DeepgramLanguages | str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        opts = self._opts
        stream_language = LanguageCode(language) if is_given(language) else opts.language
        stream: _NovaStream | _FluxStream
        if isinstance(opts, _FluxOptions):
            stream = _FluxStream(
                stt=self, opts=opts, language=stream_language, conn_options=conn_options
            )
        else:
            if opts.detect_language:
                raise ValueError(
                    "language detection is not supported in streaming mode, "
                    "please disable it and specify a language"
                )
            stream = _NovaStream(
                stt=self, opts=opts, language=stream_language, conn_options=conn_options
            )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        model: NotGivenOr[DeepgramModels | FluxModels | str] = NOT_GIVEN,
        language: NotGivenOr[DeepgramLanguages | str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        keyterm: NotGivenOr[str | list[str]] = NOT_GIVEN,
        tags: NotGivenOr[list[str]] = NOT_GIVEN,
        numerals: NotGivenOr[bool] = NOT_GIVEN,
        profanity_filter: NotGivenOr[bool] = NOT_GIVEN,
        redact: NotGivenOr[str | list[str]] = NOT_GIVEN,
        mip_opt_out: NotGivenOr[bool] = NOT_GIVEN,
        endpoint_url: NotGivenOr[str] = NOT_GIVEN,
        detect_language: NotGivenOr[bool] = NOT_GIVEN,
        interim_results: NotGivenOr[bool] = NOT_GIVEN,
        punctuate: NotGivenOr[bool] = NOT_GIVEN,
        smart_format: NotGivenOr[bool] = NOT_GIVEN,
        no_delay: NotGivenOr[bool] = NOT_GIVEN,
        endpointing_ms: NotGivenOr[int] = NOT_GIVEN,
        enable_diarization: NotGivenOr[bool] = NOT_GIVEN,
        filler_words: NotGivenOr[bool] = NOT_GIVEN,
        keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN,
        vad_events: NotGivenOr[bool] = NOT_GIVEN,
        utterance_end_ms: NotGivenOr[int | None] = NOT_GIVEN,
        dictation: NotGivenOr[bool] = NOT_GIVEN,
        replace: NotGivenOr[dict[str, str] | None] = NOT_GIVEN,
        search: NotGivenOr[list[str] | None] = NOT_GIVEN,
        eager_eot_threshold: NotGivenOr[float] = NOT_GIVEN,
        eot_threshold: NotGivenOr[float] = NOT_GIVEN,
        eot_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        language_hint: NotGivenOr[list[str]] = NOT_GIVEN,
    ) -> None:
        """Change the options; open streams reconnect with them, except Flux's thresholds,
        keyterms and language hints, which apply in band. The model can change only within its
        family, and an option of the other family raises `ValueError`. The sample rate applies
        to new streams."""
        opts = self._opts
        resolved_model = model if is_given(model) else opts.model
        if _is_flux(resolved_model) != isinstance(opts, _FluxOptions):
            raise ValueError(
                f"{resolved_model} is served on another endpoint than {opts.model}: "
                "create a new STT for it"
            )
        shared_options = {
            "model": is_given(model),
            "language": is_given(language),
            "sample_rate": is_given(sample_rate),
            "keyterm": is_given(keyterm),
            "tags": is_given(tags),
            "numerals": is_given(numerals),
            "profanity_filter": is_given(profanity_filter),
            "redact": is_given(redact),
            "mip_opt_out": is_given(mip_opt_out),
            "endpoint_url": is_given(endpoint_url),
        }
        nova_options = {
            "detect_language": is_given(detect_language),
            "interim_results": is_given(interim_results),
            "punctuate": is_given(punctuate),
            "smart_format": is_given(smart_format),
            "no_delay": is_given(no_delay),
            "endpointing_ms": is_given(endpointing_ms),
            "enable_diarization": is_given(enable_diarization),
            "filler_words": is_given(filler_words),
            "keywords": is_given(keywords),
            "vad_events": is_given(vad_events),
            "utterance_end_ms": is_given(utterance_end_ms),
            "dictation": is_given(dictation),
            "replace": is_given(replace),
            "search": is_given(search),
        }
        flux_options = {
            "eager_eot_threshold": is_given(eager_eot_threshold),
            "eot_threshold": is_given(eot_threshold),
            "eot_timeout_ms": is_given(eot_timeout_ms),
            "language_hint": is_given(language_hint),
        }
        if isinstance(opts, _FluxOptions):
            _only_for("Nova", resolved_model, nova_options)
            eager = (
                eager_eot_threshold if is_given(eager_eot_threshold) else opts.eager_eot_threshold
            )
            eot = eot_threshold if is_given(eot_threshold) else opts.eot_threshold
            eot = _DEFAULT_EOT_THRESHOLD if eot is None else eot
            if eager is not None and eager > eot:
                raise ValueError(
                    f"eager_eot_threshold ({eager}) must be less than or equal to "
                    f"eot_threshold ({eot}); increase eot_threshold (max 0.9) to use a higher "
                    "eager value"
                )
        else:
            _only_for("Flux", resolved_model, flux_options)
            _validate_keyterm(resolved_model, keyterm, keywords)
        if is_given(tags):
            _validate_tags(tags)

        if is_given(language):
            opts.language = LanguageCode(language)
        if is_given(model):
            opts.model = (
                model if isinstance(opts, _FluxOptions) else _validate_model(model, opts.language)
            )
        if is_given(sample_rate):
            opts.sample_rate = sample_rate
        if is_given(keyterm):
            self._user_keyterm = [keyterm] if isinstance(keyterm, str) else list(keyterm)
            opts.keyterm = list(dict.fromkeys([*self._user_keyterm, *self._session_keyterms]))
        if is_given(tags):
            opts.tags = tags
        if is_given(numerals):
            opts.numerals = numerals
        if is_given(profanity_filter):
            opts.profanity_filter = profanity_filter
        if is_given(redact):
            opts.redact = [redact] if isinstance(redact, str) else list(redact)
        if is_given(mip_opt_out):
            opts.mip_opt_out = mip_opt_out
        if is_given(endpoint_url):
            opts.endpoint_url = endpoint_url

        configure: _Configure | None = None
        if isinstance(opts, _FluxOptions):
            configure = _Configure(type="Configure")
            thresholds = _Thresholds()
            if is_given(eager_eot_threshold):
                opts.eager_eot_threshold = thresholds["eager_eot_threshold"] = eager_eot_threshold
            if is_given(eot_threshold):
                opts.eot_threshold = thresholds["eot_threshold"] = eot_threshold
            if is_given(eot_timeout_ms):
                opts.eot_timeout_ms = thresholds["eot_timeout_ms"] = eot_timeout_ms
            if thresholds:
                configure["thresholds"] = thresholds
            if is_given(keyterm):
                # keyterms replaces the whole list: the full effective set
                configure["keyterms"] = list(opts.keyterm)
            if is_given(language_hint):
                opts.language_hint = configure["language_hints"] = language_hint
                if language_hint and opts.model != "flux-general-multi":
                    logger.warning(
                        "`language_hint` is only supported by `flux-general-multi` and will be "
                        "ignored for model '%s'",
                        opts.model,
                    )
        else:
            if is_given(detect_language):
                opts.detect_language = detect_language
            if is_given(interim_results):
                opts.interim_results = interim_results
            if is_given(punctuate):
                opts.punctuate = punctuate
            if is_given(smart_format):
                opts.smart_format = smart_format
            if is_given(no_delay):
                opts.no_delay = no_delay
            if is_given(endpointing_ms):
                opts.endpointing_ms = endpointing_ms
            if is_given(enable_diarization):
                opts.enable_diarization = enable_diarization
            if is_given(filler_words):
                opts.filler_words = filler_words
            if is_given(keywords):
                opts.keywords = keywords
            if is_given(vad_events):
                opts.vad_events = vad_events
            if is_given(utterance_end_ms):
                opts.utterance_end_ms = utterance_end_ms
            if is_given(dictation):
                opts.dictation = dictation
            if is_given(replace):
                opts.replace = replace
            if is_given(search):
                opts.search = search
        # a stream keeps the sample rate its audio is resampled to; Flux labels transcripts
        # with the language and tunes keyterms in band
        in_band = {"sample_rate", "language", "keyterm"} if configure else {"sample_rate"}
        reconnect = any(nova_options.values()) or any(
            is_set for name, is_set in shared_options.items() if name not in in_band
        )
        for stream in self._streams:
            if is_given(language):
                stream.language = LanguageCode(language)
            if reconnect:
                stream.options_changed()
            elif isinstance(stream, _FluxStream) and configure and len(configure) > 1:
                stream.send_now(configure)

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        if keyterms == self._session_keyterms:
            return
        self._session_keyterms = list(keyterms)
        merged = list(dict.fromkeys([*self._user_keyterm, *keyterms]))
        self._opts.keyterm = merged
        for stream in self._streams:
            stream.keyterms_changed()


def _only_for(family: str, model: str, given: dict[str, bool]) -> None:
    if names := [name for name, is_set in given.items() if is_set]:
        raise ValueError(f"{', '.join(names)} only apply to {family} models, not {model}")


def _shared_query(opts: _Options) -> dict[str, _QueryValue]:
    query: dict[str, _QueryValue] = {
        "model": opts.model,
        "encoding": "linear16",
        "mip_opt_out": opts.mip_opt_out,
    }
    if opts.keyterm:
        query["keyterm"] = opts.keyterm
    if opts.tags:
        query["tag"] = opts.tags
    if opts.redact:
        query["redact"] = opts.redact
    return query


def _target(query: dict[str, _QueryValue], opts: _Options, api_key: str) -> stt.WebSocketTarget:
    return stt.WebSocketTarget(
        to_deepgram_url(
            query,
            base_url=opts.endpoint_url,
            websocket=True,
            keywords=opts.keywords if isinstance(opts, _NovaOptions) else None,
            replace=opts.replace if isinstance(opts, _NovaOptions) else None,
        ),
        headers={"Authorization": f"Token {api_key}"},
        request_id_header="dg-request-id",
    )


# -- Nova (/v1/listen)


class _Word(TypedDict):
    word: str
    start: float
    end: float
    punctuated_word: NotRequired[str]
    confidence: NotRequired[float]
    speaker: NotRequired[int]


class _Alternative(TypedDict):
    transcript: str
    confidence: float
    words: NotRequired[list[_Word]]
    languages: NotRequired[list[str]]


class _Channel(TypedDict):
    alternatives: list[_Alternative]


class _Metadata(TypedDict):
    request_id: str


class _Results(TypedDict):
    type: Literal["Results"]
    channel: _Channel
    is_final: NotRequired[bool]
    speech_final: NotRequired[bool]
    metadata: NotRequired[_Metadata]


class _SpeechStarted(TypedDict):
    type: Literal["SpeechStarted"]


class _UtteranceEnd(TypedDict):
    type: Literal["UtteranceEnd"]


_Message = Annotated[_Results | _SpeechStarted | _UtteranceEnd, Field(discriminator="type")]


class _Out(TypedDict):
    type: Literal["KeepAlive", "CloseStream", "Finalize"]


class _NovaStream(stt.WebSocketStream[_Message, _Out]):
    """A recognition on `/v1/listen`."""

    # https://developers.deepgram.com/reference/listen-live#stream-keepalive
    keepalive = stt.Keepalive(every=5.0, frame=_Out(type="KeepAlive"))
    frame_ms = 50
    # SpeechStarted, speech_final and UtteranceEnd are Deepgram's own speech boundaries
    provider_vad = True

    def __init__(
        self,
        *,
        stt: STT,
        opts: _NovaOptions,
        language: LanguageCode,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._api_key = stt._api_key
        self._sample_rate = opts.sample_rate
        self.language = language
        # session keyterms changed while the user spoke: applied once the speech ended
        self._keyterms_pending = False

    def options_changed(self) -> None:
        self._keyterms_pending = False
        # the options are read at each connection
        self.reconnect("options changed")

    def keyterms_changed(self) -> None:
        """A new connection ends the speech: wait until the user is not speaking."""
        self._keyterms_pending = True
        self._apply_pending_keyterms()

    def _apply_pending_keyterms(self) -> None:
        if self._keyterms_pending and not self.speaking:
            self.options_changed()

    async def connect(self, conn_options: APIConnectOptions) -> stt.WebSocketTarget:
        opts = self._opts
        query = _shared_query(opts) | {
            "language": self.language,
            "sample_rate": self._sample_rate,
            "channels": 1,
            "punctuate": opts.punctuate,
            "smart_format": opts.smart_format,
            "no_delay": opts.no_delay,
            "interim_results": opts.interim_results,
            "vad_events": opts.vad_events,
            "endpointing": False if opts.endpointing_ms == 0 else opts.endpointing_ms,
            "filler_words": opts.filler_words,
            "profanity_filter": opts.profanity_filter,
            "numerals": opts.numerals,
        }
        if opts.enable_diarization:
            query["diarize"] = True
        if opts.utterance_end_ms is not None:
            query["utterance_end_ms"] = opts.utterance_end_ms
        if opts.dictation:
            query["dictation"] = True
        if opts.search:
            query["search"] = opts.search
        return _target(query, opts, self._api_key)

    def encode(self, item: rtc.AudioFrame | stt.Flush | stt.EndOfAudio) -> bytes | _Out:
        match item:
            case rtc.AudioFrame():
                return bytes(item.data)
            case stt.Flush():
                return _Out(type="Finalize")
            case stt.EndOfAudio():
                return _Out(type="CloseStream")
            case _:
                assert_never(item)

    # see https://developers.deepgram.com/docs/understand-endpointing-interim-results for the
    # message types and how speech_final and UtteranceEnd end the speech
    def on_message(self, message: _Results | _SpeechStarted | _UtteranceEnd) -> None:
        if message["type"] == "SpeechStarted":
            self.report(stt.SpeechStarted())
        elif message["type"] == "UtteranceEnd":
            # sent once utterance_end_ms of silence followed the last word
            self.report(stt.SpeechEnded())
        elif message["type"] == "Results":
            self.report(self._transcript(message))
            if message.get("speech_final", False):
                self.report(stt.SpeechEnded())
        else:
            self._skip_unknown_message(message)
        self._apply_pending_keyterms()

    def _transcript(self, message: _Results) -> stt.Transcript:
        final = message.get("is_final", False)
        request_id = message.get("metadata", {"request_id": ""})["request_id"]
        alternatives = message["channel"]["alternatives"]
        if not alternatives:
            return stt.Transcript("", self.language, final=final, request_id=request_id)
        alt = alternatives[0]
        words = alt.get("words", [])
        offset = self.start_time_offset
        punctuated = self._opts.punctuate or self._opts.smart_format
        # interim results carry no reliable speaker
        speakers = [w["speaker"] for w in words if "speaker" in w] if final else []
        speaker = Counter(speakers).most_common(1)[0][0] if speakers else None
        languages = alt.get("languages", [])
        return stt.Transcript(
            alt["transcript"],
            LanguageCode(languages[0] if self.language == "multi" and languages else self.language),
            final=final,
            request_id=request_id,
            start_time=(words[0]["start"] if words else 0) + offset,
            end_time=(words[-1]["end"] if words else 0) + offset,
            confidence=alt["confidence"],
            speaker_id=f"S{speaker}" if speaker is not None else None,
            words=[
                TranscriptChunk(
                    text=_word_text(
                        w["word"], w.get("punctuated_word"), use_punctuated_word=punctuated
                    ),
                    start_time=w["start"] + offset,
                    end_time=w["end"] + offset,
                    confidence=w.get("confidence"),
                    start_time_offset=offset,
                )
                for w in words
            ]
            or None,
        )


# -- Flux (/v2/listen)


class _FluxWord(TypedDict):
    word: str
    start: float
    end: float
    confidence: NotRequired[float]


class _TurnInfo(TypedDict):
    type: Literal["TurnInfo"]
    # StartOfTurn, Update, EagerEndOfTurn, TurnResumed or EndOfTurn
    event: str
    transcript: NotRequired[str]
    words: NotRequired[list[_FluxWord]]
    languages: NotRequired[list[str]]
    audio_window_start: NotRequired[float]
    audio_window_end: NotRequired[float]
    request_id: NotRequired[str]


class _ConfigureSuccess(TypedDict):
    type: Literal["ConfigureSuccess"]


class _ConfigureFailure(TypedDict):
    type: Literal["ConfigureFailure"]


class _Error(TypedDict):
    type: Literal["Error"]
    description: NotRequired[str]


_FluxMessage = Annotated[
    _TurnInfo | _ConfigureSuccess | _ConfigureFailure | _Error, Field(discriminator="type")
]


class _CloseStream(TypedDict):
    type: Literal["CloseStream"]


class _Thresholds(TypedDict, total=False):
    eager_eot_threshold: float
    eot_threshold: float
    eot_timeout_ms: int


class _Configure(TypedDict):
    # https://developers.deepgram.com/docs/flux/configure: only the changed fields
    type: Literal["Configure"]
    thresholds: NotRequired[_Thresholds]
    keyterms: NotRequired[list[str]]
    language_hints: NotRequired[list[str]]


class _FluxStream(stt.WebSocketStream[_FluxMessage, _CloseStream | _Configure]):
    """A recognition on `/v2/listen`."""

    frame_ms = 50
    # Flux detects turns itself: StartOfTurn and EndOfTurn
    provider_vad = True

    def __init__(
        self,
        *,
        stt: STT,
        opts: _FluxOptions,
        language: LanguageCode,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._api_key = stt._api_key
        self._sample_rate = opts.sample_rate
        self.language = language
        self._request_id = ""

    def options_changed(self) -> None:
        # the options are read at each connection
        self.reconnect("options changed")

    def keyterms_changed(self) -> None:
        # tuned in band, safe mid-utterance
        self.send_now(_Configure(type="Configure", keyterms=list(self._opts.keyterm)))

    async def connect(self, conn_options: APIConnectOptions) -> stt.WebSocketTarget:
        opts = self._opts
        query = _shared_query(opts) | {"sample_rate": self._sample_rate}
        if opts.eager_eot_threshold is not None:
            query["eager_eot_threshold"] = opts.eager_eot_threshold
        if opts.eot_threshold is not None:
            query["eot_threshold"] = opts.eot_threshold
        if opts.eot_timeout_ms is not None:
            query["eot_timeout_ms"] = opts.eot_timeout_ms
        if opts.language_hint:
            query["language_hint"] = opts.language_hint
        if opts.numerals:
            query["numerals"] = True
        if opts.profanity_filter:
            query["profanity_filter"] = True
        return _target(query, opts, self._api_key)

    def encode(
        self, item: rtc.AudioFrame | stt.Flush | stt.EndOfAudio
    ) -> bytes | _CloseStream | None:
        match item:
            case rtc.AudioFrame():
                return bytes(item.data)
            case stt.Flush():
                # Flux finds the turns itself: a flush only sends the partial chunk, which the
                # base does
                return None
            case stt.EndOfAudio():
                return _CloseStream(type="CloseStream")
            case _:
                assert_never(item)

    def on_message(
        self, message: _TurnInfo | _ConfigureSuccess | _ConfigureFailure | _Error
    ) -> None:
        if message["type"] == "ConfigureSuccess":
            logger.debug("deepgram applied Configure update", extra={"lk.pii.data": message})
        elif message["type"] == "ConfigureFailure":
            logger.warning("deepgram rejected Configure update", extra={"lk.pii.data": message})
        elif message["type"] == "Error":
            raise APIError(
                "deepgram returned an error",
                kind="invalid_request",
                detail=message.get("description"),
            )
        elif message["type"] == "TurnInfo":
            self._turn_info(message)
        else:
            self._skip_unknown_message(message)

    def _turn_info(self, message: _TurnInfo) -> None:
        if request_id := message.get("request_id"):
            self._request_id = request_id
        event = message["event"]
        if event == "StartOfTurn":
            self.report(stt.SpeechStarted())
            self._report_transcript(message)
        elif event in ("Update", "TurnResumed"):
            # an interim also aborts an eager end of turn
            self._report_transcript(message)
        elif event == "EagerEndOfTurn":
            # a pause: preflight for preemptive generation, the turn still open
            self._report_transcript(message, preflight=True)
        elif event == "EndOfTurn":
            self._report_transcript(message, final=True)
            self.report(stt.SpeechEnded())

    def _report_transcript(
        self, message: _TurnInfo, *, final: bool = False, preflight: bool = False
    ) -> None:
        words = message.get("words", [])
        if not words:
            return
        offset = self.start_time_offset
        languages = message.get("languages", [])
        self.report(
            stt.Transcript(
                message.get("transcript", ""),
                LanguageCode(languages[0]) if languages else self.language,
                final=final,
                preflight=preflight,
                request_id=self._request_id,
                start_time=message.get("audio_window_start", 0.0) + offset,
                end_time=message.get("audio_window_end", 0.0) + offset,
                confidence=sum(w.get("confidence", 0.0) for w in words) / len(words),
                source_languages=[LanguageCode(lang) for lang in languages] or None,
                words=[
                    TranscriptChunk(
                        text=w["word"],
                        start_time=w["start"] + offset,
                        end_time=w["end"] + offset,
                        confidence=w.get("confidence", 0.0),
                        start_time_offset=offset,
                    )
                    for w in words
                ],
            )
        )


def _word_text(word: str, punctuated_word: str | None, *, use_punctuated_word: bool) -> str:
    """The word form matching the transcript asked for: `punctuated_word` when the request
    formats the transcript (`punctuate` or `smart_format`), else `word`."""
    return (punctuated_word or word) if use_punctuated_word else word


def prerecorded_transcription_to_speech_event(
    language: str | None,  # None when 'detect_language' is enabled
    response: TranscriptionResponse,
    *,
    use_punctuated_word: bool = True,
) -> stt.SpeechEvent:
    # one channel only
    channel = response.results.channels[0]
    # the detected language, when detection is on:
    # https://developers.deepgram.com/docs/language-detection
    detected_language = LanguageCode(channel.detected_language)

    return stt.SpeechEvent(
        request_id=response.metadata.request_id,
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            stt.SpeechData(
                language=LanguageCode(language or detected_language),
                start_time=alt.words[0].start if alt.words else 0,
                end_time=alt.words[-1].end if alt.words else 0,
                confidence=alt.confidence,
                text=alt.transcript,
                words=[
                    TranscriptChunk(
                        text=_word_text(
                            word.word, word.punctuated_word, use_punctuated_word=use_punctuated_word
                        ),
                        start_time=word.start,
                        end_time=word.end,
                        confidence=word.confidence,
                    )
                    for word in alt.words
                ],
            )
            for alt in channel.alternatives
        ],
    )


def _validate_model(model: str, language: str) -> str:
    en_only_models = {
        "nova-2-meeting",
        "nova-2-phonecall",
        "nova-2-finance",
        "nova-2-conversationalai",
        "nova-2-voicemail",
        "nova-2-video",
        "nova-2-medical",
        "nova-2-drivethru",
        "nova-2-automotive",
    }
    if language not in ("en-US", "en") and model in en_only_models:
        logger.warning(
            f"{model} does not support language {language}, falling back to nova-2-general"
        )
        return "nova-2-general"
    return model


def _validate_tags(tags: list[str]) -> None:
    if any(len(tag) > 128 for tag in tags):
        raise ValueError("tag must be no more than 128 characters")


def _validate_keyterm(
    model: str,
    keyterm: NotGivenOr[str | list[str]],
    keywords: NotGivenOr[list[tuple[str, float]]],
) -> None:
    """Keyterms are Nova-3's, keywords the older models'.
    See https://developers.deepgram.com/docs/keyterm and https://developers.deepgram.com/docs/keywords
    """
    if model.startswith("nova-3") and is_given(keywords):
        raise ValueError(
            "Keywords is only available for use with Nova-2, Nova-1, Enhanced, and "
            "Base speech to text models. For Nova-3, use Keyterm Prompting."
        )
    if is_given(keyterm) and not model.startswith("nova-3"):
        raise ValueError(
            "Keyterm Prompting is only available for transcription using the Nova-3 Model. "
            "To boost recognition of keywords using another model, use the Keywords feature."
        )
