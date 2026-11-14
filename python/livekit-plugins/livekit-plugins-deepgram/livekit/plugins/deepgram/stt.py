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

import dataclasses
import os
import weakref
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Literal, NotRequired, TypedDict, assert_never

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
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TranscriptChunk

from ._utils import to_deepgram_url
from .log import logger
from .models import DeepgramLanguages, DeepgramModels, TranscriptionResponse


@dataclass
class STTOptions:
    language: LanguageCode | None
    detect_language: bool
    interim_results: bool
    punctuate: bool
    model: DeepgramModels | str
    smart_format: bool
    no_delay: bool
    endpointing_ms: int
    enable_diarization: bool
    filler_words: bool
    sample_rate: int
    num_channels: int
    keywords: list[tuple[str, float]]
    keyterm: str | Sequence[str]
    profanity_filter: bool
    redact: str | list[str]
    endpoint_url: str
    vad_events: bool = True
    numerals: bool = False
    mip_opt_out: bool = False
    tags: NotGivenOr[list[str]] = NOT_GIVEN
    utterance_end_ms: int | None = None
    dictation: bool = False
    replace: dict[str, str] | None = None
    search: list[str] | None = None


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: DeepgramModels | str = "nova-3",
        language: DeepgramLanguages | str = "en-US",
        detect_language: bool = False,
        interim_results: bool = True,
        punctuate: bool = True,
        smart_format: bool = False,
        sample_rate: int = 16000,
        no_delay: bool = True,
        endpointing_ms: int = 25,
        enable_diarization: bool = False,
        # enable filler words by default to improve turn detector accuracy
        filler_words: bool = True,
        keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN,
        keyterm: NotGivenOr[str | list[str]] = NOT_GIVEN,
        tags: NotGivenOr[list[str]] = NOT_GIVEN,
        profanity_filter: bool = False,
        redact: NotGivenOr[str | list[str]] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        http_session: http.HttpSession | None = None,
        base_url: str = "https://api.deepgram.com/v1/listen",
        numerals: bool = False,
        mip_opt_out: bool = False,
        vad_events: bool = True,
        utterance_end_ms: int | None = None,
        dictation: bool = False,
        replace: dict[str, str] | None = None,
        search: list[str] | None = None,
    ) -> None:
        """Create a new instance of Deepgram STT.

        Args:
            model: The Deepgram model to use for speech recognition. Defaults to "nova-3".
            language: The language code for recognition. Defaults to "en-US".
            detect_language: Whether to enable automatic language detection. Defaults to False.
            interim_results: Whether to return interim (non-final) transcription results. Defaults to True.
            punctuate: Whether to add punctuations to the transcription. Defaults to True. Turn detector will work better with punctuations.
            smart_format: Whether to apply smart formatting to numbers, dates, etc. Defaults to False.
            sample_rate: The sample rate of the audio in Hz. Defaults to 16000.
            no_delay: When smart_format is used, ensures it does not wait for sequence to be complete before returning results. Defaults to True.
            endpointing_ms: Time in milliseconds of silence to consider end of speech. Set to 0 to disable. Defaults to 25.
            filler_words: Whether to include filler words (um, uh, etc.) in transcription. Defaults to True.
            keywords: List of tuples containing keywords and their boost values for improved recognition.
                     Each tuple should be (keyword: str, boost: float). Defaults to None.
                     `keywords` does not work with Nova-3 models. Use `keyterm` instead.
            keyterm: str or list of str of key terms to improve recognition accuracy. Defaults to None.
                     `keyterm` is only supported by Nova-3 models.
            tags: List of tags to add to the requests for usage reporting. Defaults to NOT_GIVEN.
            profanity_filter: Whether to filter profanity from the transcription. Defaults to False.
            redact: Redact sensitive information from the transcription. Accepts a single value or
                list of values. Supported values: "pci", "numbers", "ssn", "true" (redact all).
                See https://developers.deepgram.com/docs/redaction for details.
            api_key: Your Deepgram API key. If not provided, will look for DEEPGRAM_API_KEY environment variable.
            http_session: Optional HttpSession to use for requests.
            base_url: The base URL for Deepgram API. Defaults to "https://api.deepgram.com/v1/listen".
            numerals: Whether to include numerals in the transcription. Defaults to False.
            mip_opt_out: Whether to take part in the model improvement program
            vad_events: Whether to enable VAD (Voice Activity Detection) events.
                       When enabled, SpeechStarted events are sent when speech is detected. Defaults to True.
            utterance_end_ms: Duration of silence in milliseconds to detect the end of an utterance
                             and emit an UtteranceEnd event. Requires interim_results=True.
                             See https://developers.deepgram.com/docs/understand-endpointing-interim-results
            dictation: Whether to enable dictation mode which converts spoken punctuation commands
                      (e.g. "comma", "period") into punctuation marks. Defaults to False.
                      See https://developers.deepgram.com/reference/speech-to-text/listen-streaming#query-dictation
            replace: Dictionary of terms to replace in the transcript, where keys are the original
                    terms and values are the replacements (e.g. {"hello": "hi"}).
                    See https://developers.deepgram.com/reference/speech-to-text/listen-streaming#query-replace
            search: List of terms to search for in the transcript. Matched terms are returned with
                   confidence scores in the response.
                   See https://developers.deepgram.com/reference/speech-to-text/listen-streaming#query-search

        Raises:
            ValueError: If no API key is provided or found in environment variables.

        Note:
            The api_key must be set either through the constructor argument or by setting
            the DEEPGRAM_API_KEY environmental variable.
        """  # noqa: E501

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=interim_results,
                diarization=enable_diarization,
                aligned_transcript="word",
                keyterms=True,
            )
        )

        deepgram_api_key = api_key if is_given(api_key) else os.environ.get("DEEPGRAM_API_KEY")
        if not deepgram_api_key:
            raise ValueError(
                "Deepgram API key is required, either as argument or set"
                " DEEPGRAM_API_KEY environment variable"
            )
        self._api_key = deepgram_api_key

        model = _validate_model(model, language)
        _validate_keyterm(model, language, keyterm, keywords)

        self._opts = STTOptions(
            language=LanguageCode(language) if language else None,
            detect_language=detect_language,
            interim_results=interim_results,
            punctuate=punctuate,
            model=model,
            smart_format=smart_format,
            no_delay=no_delay,
            endpointing_ms=endpointing_ms,
            enable_diarization=enable_diarization,
            filler_words=filler_words,
            sample_rate=sample_rate,
            num_channels=1,
            keywords=keywords if is_given(keywords) else [],
            keyterm=([keyterm] if isinstance(keyterm, str) else list(keyterm))
            if is_given(keyterm)
            else [],
            profanity_filter=profanity_filter,
            redact=redact if is_given(redact) else [],
            numerals=numerals,
            mip_opt_out=mip_opt_out,
            vad_events=vad_events,
            tags=_validate_tags(tags) if is_given(tags) else [],
            endpoint_url=base_url,
            utterance_end_ms=utterance_end_ms,
            dictation=dictation,
            replace=replace,
            search=search,
        )
        # user keyterms; _opts.keyterm holds the effective set (user + session)
        self._user_keyterm: list[str] = list(self._opts.keyterm)
        self._session_keyterms: list[str] = []
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

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
        config = self._sanitize_options(language=language)

        recognize_config: dict[str, object] = {
            "model": str(config.model),
            "punctuate": config.punctuate,
            "detect_language": config.detect_language,
            "smart_format": config.smart_format,
            "profanity_filter": config.profanity_filter,
            "numerals": config.numerals,
            "mip_opt_out": config.mip_opt_out,
        }
        if self._opts.keyterm:
            recognize_config["keyterm"] = self._opts.keyterm
        if config.redact:
            recognize_config["redact"] = config.redact
        if config.enable_diarization:
            logger.warning("speaker diarization is not supported in non-streaming mode, ignoring")

        if config.language:
            recognize_config["language"] = config.language

        try:
            async with self._ensure_session().post(
                to_deepgram_url(
                    recognize_config,
                    self._opts.endpoint_url,
                    websocket=False,
                    keywords=self._opts.keywords,
                ),
                data=rtc.combine_audio_frames(buffer).to_wav_bytes(),
                headers={
                    "Authorization": f"Token {self._api_key}",
                    "Accept": "application/json",
                    "Content-Type": "audio/wav",
                },
                conn_options=conn_options,
            ) as res:
                return prerecorded_transcription_to_speech_event(
                    config.language,
                    await res.json(),
                    use_punctuated_word=config.punctuate or config.smart_format,
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
    ) -> SpeechStream:
        config = self._sanitize_options(language=language)
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            opts=config,
            api_key=self._api_key,
            base_url=self._opts.endpoint_url,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        language: NotGivenOr[DeepgramLanguages | str] = NOT_GIVEN,
        model: NotGivenOr[DeepgramModels | str] = NOT_GIVEN,
        interim_results: NotGivenOr[bool] = NOT_GIVEN,
        punctuate: NotGivenOr[bool] = NOT_GIVEN,
        smart_format: NotGivenOr[bool] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        no_delay: NotGivenOr[bool] = NOT_GIVEN,
        endpointing_ms: NotGivenOr[int] = NOT_GIVEN,
        enable_diarization: NotGivenOr[bool] = NOT_GIVEN,
        filler_words: NotGivenOr[bool] = NOT_GIVEN,
        keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN,
        keyterm: NotGivenOr[str | list[str]] = NOT_GIVEN,
        profanity_filter: NotGivenOr[bool] = NOT_GIVEN,
        redact: NotGivenOr[str | list[str]] = NOT_GIVEN,
        numerals: NotGivenOr[bool] = NOT_GIVEN,
        mip_opt_out: NotGivenOr[bool] = NOT_GIVEN,
        vad_events: NotGivenOr[bool] = NOT_GIVEN,
        tags: NotGivenOr[list[str]] = NOT_GIVEN,
        endpoint_url: NotGivenOr[str] = NOT_GIVEN,
        utterance_end_ms: NotGivenOr[int | None] = NOT_GIVEN,
        dictation: NotGivenOr[bool] = NOT_GIVEN,
        replace: NotGivenOr[dict[str, str] | None] = NOT_GIVEN,
        search: NotGivenOr[list[str] | None] = NOT_GIVEN,
    ) -> None:
        if is_given(language):
            self._opts.language = LanguageCode(language)
        if is_given(model):
            self._opts.model = _validate_model(
                model, language if is_given(language) else (self._opts.language or NOT_GIVEN)
            )
        if is_given(interim_results):
            self._opts.interim_results = interim_results
        if is_given(punctuate):
            self._opts.punctuate = punctuate
        if is_given(smart_format):
            self._opts.smart_format = smart_format
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if is_given(no_delay):
            self._opts.no_delay = no_delay
        if is_given(endpointing_ms):
            self._opts.endpointing_ms = endpointing_ms
        if is_given(enable_diarization):
            self._opts.enable_diarization = enable_diarization
        if is_given(filler_words):
            self._opts.filler_words = filler_words
        if is_given(keywords):
            self._opts.keywords = keywords
        if is_given(keyterm):
            self._user_keyterm = [keyterm] if isinstance(keyterm, str) else list(keyterm)
            keyterm = list(dict.fromkeys([*self._user_keyterm, *self._session_keyterms]))
            self._opts.keyterm = keyterm
        if is_given(profanity_filter):
            self._opts.profanity_filter = profanity_filter
        if is_given(redact):
            self._opts.redact = redact
        if is_given(numerals):
            self._opts.numerals = numerals
        if is_given(mip_opt_out):
            self._opts.mip_opt_out = mip_opt_out
        if is_given(vad_events):
            self._opts.vad_events = vad_events
        if is_given(tags):
            self._opts.tags = _validate_tags(tags)
        if is_given(endpoint_url):
            self._opts.endpoint_url = endpoint_url
        if is_given(utterance_end_ms):
            self._opts.utterance_end_ms = utterance_end_ms
        if is_given(dictation):
            self._opts.dictation = dictation
        if is_given(replace):
            self._opts.replace = replace
        if is_given(search):
            self._opts.search = search

        for stream in self._streams:
            stream.update_options(
                language=language,
                model=model,
                interim_results=interim_results,
                punctuate=punctuate,
                smart_format=smart_format,
                sample_rate=sample_rate,
                no_delay=no_delay,
                endpointing_ms=endpointing_ms,
                filler_words=filler_words,
                keywords=keywords,
                keyterm=keyterm,
                profanity_filter=profanity_filter,
                redact=redact,
                numerals=numerals,
                mip_opt_out=mip_opt_out,
                vad_events=vad_events,
                endpoint_url=endpoint_url,
                utterance_end_ms=utterance_end_ms,
                dictation=dictation,
                replace=replace,
                search=search,
            )

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        if keyterms == self._session_keyterms:
            return
        self._session_keyterms = list(keyterms)
        merged = list(dict.fromkeys([*self._user_keyterm, *keyterms]))
        self._opts.keyterm = merged
        for stream in self._streams:
            stream.update_session_keyterms(merged)

    def _sanitize_options(
        self, *, language: NotGivenOr[DeepgramLanguages | str] = NOT_GIVEN
    ) -> STTOptions:
        config = dataclasses.replace(self._opts)
        if is_given(language):
            config.language = LanguageCode(language)

        if config.detect_language:
            config.language = None

        return config


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


class SpeechStream(stt.WebSocketStream[_Message, _Out]):
    # https://developers.deepgram.com/reference/listen-live#stream-keepalive
    keepalive = stt.Keepalive(every=5.0, frame=_Out(type="KeepAlive"))
    frame_ms = 50
    # SpeechStarted, speech_final and UtteranceEnd are Deepgram's own speech boundaries
    provider_vad = True

    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        base_url: str,
    ) -> None:
        if opts.detect_language or opts.language is None:
            raise ValueError(
                "language detection is not supported in streaming mode, "
                "please disable it and specify a language"
            )

        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._api_key = api_key
        self._opts.endpoint_url = base_url
        # keyterms set while the user is speaking; applied once the speech ended (latest wins)
        self._pending_keyterm: list[str] | None = None

    def update_options(
        self,
        *,
        language: NotGivenOr[DeepgramLanguages | str] = NOT_GIVEN,
        model: NotGivenOr[DeepgramModels | str] = NOT_GIVEN,
        interim_results: NotGivenOr[bool] = NOT_GIVEN,
        punctuate: NotGivenOr[bool] = NOT_GIVEN,
        smart_format: NotGivenOr[bool] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        no_delay: NotGivenOr[bool] = NOT_GIVEN,
        endpointing_ms: NotGivenOr[int] = NOT_GIVEN,
        enable_diarization: NotGivenOr[bool] = NOT_GIVEN,
        filler_words: NotGivenOr[bool] = NOT_GIVEN,
        keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN,
        keyterm: NotGivenOr[str | list[str]] = NOT_GIVEN,
        profanity_filter: NotGivenOr[bool] = NOT_GIVEN,
        redact: NotGivenOr[str | list[str]] = NOT_GIVEN,
        numerals: NotGivenOr[bool] = NOT_GIVEN,
        mip_opt_out: NotGivenOr[bool] = NOT_GIVEN,
        vad_events: NotGivenOr[bool] = NOT_GIVEN,
        tags: NotGivenOr[list[str]] = NOT_GIVEN,
        endpoint_url: NotGivenOr[str] = NOT_GIVEN,
        utterance_end_ms: NotGivenOr[int | None] = NOT_GIVEN,
        dictation: NotGivenOr[bool] = NOT_GIVEN,
        replace: NotGivenOr[dict[str, str] | None] = NOT_GIVEN,
        search: NotGivenOr[list[str] | None] = NOT_GIVEN,
    ) -> None:
        if is_given(language):
            self._opts.language = LanguageCode(language)
        if is_given(model):
            self._opts.model = _validate_model(
                model, language if is_given(language) else (self._opts.language or NOT_GIVEN)
            )
        if is_given(interim_results):
            self._opts.interim_results = interim_results
        if is_given(punctuate):
            self._opts.punctuate = punctuate
        if is_given(smart_format):
            self._opts.smart_format = smart_format
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if is_given(no_delay):
            self._opts.no_delay = no_delay
        if is_given(endpointing_ms):
            self._opts.endpointing_ms = endpointing_ms
        if is_given(enable_diarization):
            self._opts.enable_diarization = enable_diarization
        if is_given(filler_words):
            self._opts.filler_words = filler_words
        if is_given(keywords):
            self._opts.keywords = keywords
        if is_given(keyterm):
            self._opts.keyterm = keyterm
            self._pending_keyterm = None
        if is_given(profanity_filter):
            self._opts.profanity_filter = profanity_filter
        if is_given(redact):
            self._opts.redact = redact
        if is_given(numerals):
            self._opts.numerals = numerals
        if is_given(mip_opt_out):
            self._opts.mip_opt_out = mip_opt_out
        if is_given(vad_events):
            self._opts.vad_events = vad_events
        if is_given(tags):
            self._opts.tags = _validate_tags(tags)
        if is_given(endpoint_url):
            self._opts.endpoint_url = endpoint_url
        if is_given(utterance_end_ms):
            self._opts.utterance_end_ms = utterance_end_ms
        if is_given(dictation):
            self._opts.dictation = dictation
        if is_given(replace):
            self._opts.replace = replace
        if is_given(search):
            self._opts.search = search

        # the options are read at each connection
        self.reconnect("options changed")

    def update_session_keyterms(self, keyterms: list[str]) -> None:
        """Apply session keyterms once the user is not speaking: a new connection ends the
        speech."""
        self._pending_keyterm = keyterms
        self._apply_pending_keyterm()

    def _apply_pending_keyterm(self) -> None:
        if self._pending_keyterm is not None and not self.speaking:
            self.update_options(keyterm=self._pending_keyterm)

    async def connect(self, conn_options: APIConnectOptions) -> stt.WebSocketTarget:
        live_config: dict[str, object] = {
            "model": self._opts.model,
            "punctuate": self._opts.punctuate,
            "smart_format": self._opts.smart_format,
            "no_delay": self._opts.no_delay,
            "interim_results": self._opts.interim_results,
            "encoding": "linear16",
            "vad_events": self._opts.vad_events,
            "sample_rate": self._opts.sample_rate,
            "channels": self._opts.num_channels,
            "endpointing": False if self._opts.endpointing_ms == 0 else self._opts.endpointing_ms,
            "filler_words": self._opts.filler_words,
            "profanity_filter": self._opts.profanity_filter,
            "numerals": self._opts.numerals,
            "mip_opt_out": self._opts.mip_opt_out,
        }
        if self._opts.enable_diarization:
            live_config["diarize"] = True
        if self._opts.keyterm:
            live_config["keyterm"] = self._opts.keyterm
        if self._opts.utterance_end_ms is not None:
            live_config["utterance_end_ms"] = self._opts.utterance_end_ms
        if self._opts.dictation:
            live_config["dictation"] = True
        if self._opts.search:
            live_config["search"] = self._opts.search

        if self._opts.language:
            live_config["language"] = self._opts.language

        if self._opts.redact:
            live_config["redact"] = self._opts.redact
        if self._opts.tags:
            live_config["tag"] = self._opts.tags

        return stt.WebSocketTarget(
            to_deepgram_url(
                live_config,
                base_url=self._opts.endpoint_url,
                websocket=True,
                keywords=self._opts.keywords,
                replace=self._opts.replace,
            ),
            headers={"Authorization": f"Token {self._api_key}"},
            request_id_header="dg-request-id",
        )

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
            assert_never(message)
        self._apply_pending_keyterm()

    def _transcript(self, message: _Results) -> stt.Transcript:
        assert self._opts.language is not None
        final = message.get("is_final", False)
        request_id = message.get("metadata", {"request_id": ""})["request_id"]
        alternatives = message["channel"]["alternatives"]
        if not alternatives:
            return stt.Transcript("", self._opts.language, final=final, request_id=request_id)
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
            LanguageCode(
                languages[0]
                if self._opts.language == "multi" and languages
                else self._opts.language
            ),
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


def _word_text(word: str, punctuated_word: str | None, *, use_punctuated_word: bool) -> str:
    """Return the word form matching the transcript the caller asked Deepgram for.

    Deepgram returns `word` (lowercase, unpunctuated) and, when the request formats the
    transcript, `punctuated_word` alongside it. `SpeechData.text` comes from
    `alt.transcript`, so the word list follows the same formatting to stay consistent
    with the text it belongs to.

    Both `punctuate` and `smart_format` produce a formatted transcript, so callers pass
    `punctuate or smart_format` rather than either alone. Falls back to `word` if the key
    is missing, which is what Deepgram sends when neither option is set.
    """
    return (punctuated_word or word) if use_punctuated_word else word


def prerecorded_transcription_to_speech_event(
    language: str | None,  # language should be None when 'detect_language' is enabled
    data: object,
    *,
    use_punctuated_word: bool = True,
) -> stt.SpeechEvent:
    # We only support one channel for now
    response = TranscriptionResponse.model_validate(data)
    request_id = response.metadata.request_id
    channel = response.results.channels[0]
    dg_alts = channel.alternatives

    # Use the detected language if enabled
    # https://developers.deepgram.com/docs/language-detection
    detected_language = LanguageCode(channel.detected_language)

    return stt.SpeechEvent(
        request_id=request_id,
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
                        confidence=word.confidence if word.confidence is not None else None,
                    )
                    for word in alt.words
                ],
            )
            for alt in dg_alts
        ],
    )


def _validate_model(
    model: DeepgramModels | str, language: NotGivenOr[DeepgramLanguages | str]
) -> DeepgramModels | str:
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
    if is_given(language) and language not in ("en-US", "en") and model in en_only_models:
        logger.warning(
            f"{model} does not support language {language}, falling back to nova-2-general"
        )
        return "nova-2-general"
    return model


def _validate_tags(tags: list[str]) -> list[str]:
    for tag in tags:
        if len(tag) > 128:
            raise ValueError("tag must be no more than 128 characters")
    return tags


def _validate_keyterm(
    model: DeepgramModels | str,
    language: NotGivenOr[DeepgramLanguages | str],
    keyterm: NotGivenOr[str | list[str]],
    keywords: NotGivenOr[list[tuple[str, float]]],
) -> None:
    """
    Validating keyterm and keywords for model compatibility.
    See: https://developers.deepgram.com/docs/keyterm and https://developers.deepgram.com/docs/keywords
    """
    if model.startswith("nova-3") and is_given(keywords):
        raise ValueError(
            "Keywords is only available for use with Nova-2, Nova-1, Enhanced, and "
            "Base speech to text models. For Nova-3, use Keyterm Prompting."
        )

    if is_given(keyterm) and (not model.startswith("nova-3")):
        raise ValueError(
            "Keyterm Prompting is only available for transcription using the Nova-3 Model. "
            "To boost recognition of keywords using another model, use the Keywords feature."
        )
