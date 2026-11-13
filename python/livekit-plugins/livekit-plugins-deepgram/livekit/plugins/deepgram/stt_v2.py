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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from pydantic import Field

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
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
from .models import V2Models

FluxRedaction = Literal["numbers", "aggressive_numbers"]


@dataclass
class STTOptions:
    model: V2Models | str
    sample_rate: int
    keyterm: str | Sequence[str]
    endpoint_url: str
    language: str = "en"
    eager_eot_threshold: NotGivenOr[float] = NOT_GIVEN
    eot_threshold: NotGivenOr[float] = NOT_GIVEN
    eot_timeout_ms: NotGivenOr[int] = NOT_GIVEN
    mip_opt_out: bool = False
    numerals: bool = False
    profanity_filter: bool = False
    redact: NotGivenOr[FluxRedaction] = NOT_GIVEN
    tags: NotGivenOr[list[str]] = NOT_GIVEN
    language_hint: NotGivenOr[list[str]] = NOT_GIVEN


class STTv2(stt.STT):
    def __init__(
        self,
        *,
        model: V2Models | str = "flux-general-en",
        sample_rate: int = 16000,
        eager_eot_threshold: NotGivenOr[float] = NOT_GIVEN,
        eot_threshold: NotGivenOr[float] = NOT_GIVEN,
        eot_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        keyterm: NotGivenOr[str | list[str]] = NOT_GIVEN,
        tags: NotGivenOr[list[str]] = NOT_GIVEN,
        language_hint: NotGivenOr[list[str]] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = "wss://api.deepgram.com/v2/listen",
        mip_opt_out: bool = False,
        numerals: bool = False,
        profanity_filter: bool = False,
        redact: NotGivenOr[FluxRedaction] = NOT_GIVEN,
    ) -> None:
        """Create a new instance of Deepgram STT.

        Args:
            model: The Deepgram model to use for speech recognition. Defaults to "flux-general-en".
            sample_rate: The sample rate of the audio in Hz. Defaults to 16000.
            eager_eot_threshold: The threshold for eager end of turn to enable preemptive generation. Disabled by default. Set to 0.3-0.9 to enable preemptive generation.
            eot_threshold: The threshold for end of speech detection, ranges 0.5-0.9. Defaults to 0.7. If using eager_eot_threshold, set this higher to allow a higher eager value.
            eot_timeout_ms: The timeout for end of speech detection. Defaults to 3000.
            keyterm: str or list of str of key terms to improve recognition accuracy. Defaults to None.
            tags: List of tags to add to the requests for usage reporting. Defaults to NOT_GIVEN.
            language_hint: List of str of language hints to bias the model for improved accuracy. Only usable with `flux-general-multi`. Defaults to NOT_GIVEN.
            api_key: Your Deepgram API key. If not provided, will look for DEEPGRAM_API_KEY environment variable.
            base_url: The base URL for Deepgram API. Defaults to "https://api.deepgram.com/v1/listen".
            mip_opt_out: Whether to take part in the model improvement program
            numerals: Whether to convert spoken numbers into numerical formats. Applied at connection time; Flux does not support toggling it mid-stream. Defaults to False.
            profanity_filter: Whether to filter profanity from the transcription. Applied at connection time. Defaults to False.
            redact: Redact numbers from the transcription, "numbers" or "aggressive_numbers". Flux does not support entity redaction (pci, pii, ...). Applied at connection time. Defaults to NOT_GIVEN.

        Raises:
            ValueError: If no API key is provided or found in environment variables.

        Note:
            The api_key must be set either through the constructor argument or by setting
            the DEEPGRAM_API_KEY environmental variable.
        """  # noqa: E501

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript="word",
                offline_recognize=False,
                keyterms=True,
            )
        )

        deepgram_api_key = api_key if is_given(api_key) else os.environ.get("DEEPGRAM_API_KEY")
        if not deepgram_api_key:
            raise ValueError("Deepgram API key is required")
        self._api_key = deepgram_api_key

        if is_given(eager_eot_threshold):
            effective_eot = eot_threshold if is_given(eot_threshold) else 0.7
            if eager_eot_threshold > effective_eot:
                raise ValueError(
                    f"eager_eot_threshold ({eager_eot_threshold}) must be less than or equal to eot_threshold "
                    f"({effective_eot}); increase eot_threshold (max 0.9) to use a higher eager value"
                )
        if language_hint and model != "flux-general-multi":
            logger.warning(
                "`language_hint` is only supported by `flux-general-multi` and will be ignored for model '%s'",
                model,
            )

        self._opts = STTOptions(
            model=model,
            sample_rate=sample_rate,
            keyterm=([keyterm] if isinstance(keyterm, str) else list(keyterm))
            if is_given(keyterm)
            else [],
            mip_opt_out=mip_opt_out,
            numerals=numerals,
            profanity_filter=profanity_filter,
            redact=redact,
            tags=_validate_tags(tags) if is_given(tags) else [],
            language_hint=language_hint if is_given(language_hint) else [],
            eager_eot_threshold=eager_eot_threshold,
            eot_threshold=eot_threshold,
            eot_timeout_ms=eot_timeout_ms,
            endpoint_url=base_url,
        )
        # user keyterms; _opts.keyterm holds the effective set (user + session)
        self._user_keyterm: list[str] = list(self._opts.keyterm)
        self._session_keyterms: list[str] = []
        self._streams = weakref.WeakSet[SpeechStreamv2]()

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        raise NotImplementedError(
            "V2 API does not support non-streaming recognize. Use with a StreamAdapter"
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Deepgram"

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStreamv2:
        stream = SpeechStreamv2(
            stt=self,
            conn_options=conn_options,
            opts=self._opts,
            api_key=self._api_key,
            base_url=self._opts.endpoint_url,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        model: NotGivenOr[V2Models | str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        eager_eot_threshold: NotGivenOr[float] = NOT_GIVEN,
        eot_threshold: NotGivenOr[float] = NOT_GIVEN,
        eot_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        keyterm: NotGivenOr[str | list[str]] = NOT_GIVEN,
        mip_opt_out: NotGivenOr[bool] = NOT_GIVEN,
        numerals: NotGivenOr[bool] = NOT_GIVEN,
        profanity_filter: NotGivenOr[bool] = NOT_GIVEN,
        redact: NotGivenOr[FluxRedaction] = NOT_GIVEN,
        tags: NotGivenOr[list[str]] = NOT_GIVEN,
        language_hint: NotGivenOr[list[str]] = NOT_GIVEN,
        endpoint_url: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        effective_eager = (
            eager_eot_threshold if is_given(eager_eot_threshold) else self._opts.eager_eot_threshold
        )
        effective_eot = (
            eot_threshold
            if is_given(eot_threshold)
            else (self._opts.eot_threshold if is_given(self._opts.eot_threshold) else 0.7)
        )
        if is_given(effective_eager) and effective_eager > effective_eot:
            raise ValueError(
                f"eager_eot_threshold ({effective_eager}) must be less than or equal to eot_threshold ({effective_eot})"
            )
        if is_given(model):
            self._opts.model = model
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if is_given(eot_threshold):
            self._opts.eot_threshold = eot_threshold
        if is_given(eot_timeout_ms):
            self._opts.eot_timeout_ms = eot_timeout_ms
        if is_given(keyterm):
            self._user_keyterm = [keyterm] if isinstance(keyterm, str) else list(keyterm)
            keyterm = list(dict.fromkeys([*self._user_keyterm, *self._session_keyterms]))
            self._opts.keyterm = keyterm
        if is_given(mip_opt_out):
            self._opts.mip_opt_out = mip_opt_out
        if is_given(numerals):
            self._opts.numerals = numerals
        if is_given(profanity_filter):
            self._opts.profanity_filter = profanity_filter
        if is_given(redact):
            self._opts.redact = redact
        if is_given(tags):
            self._opts.tags = _validate_tags(tags)
        if is_given(language_hint):
            self._opts.language_hint = language_hint
            if language_hint and self._opts.model != "flux-general-multi":
                logger.warning(
                    "`language_hint` is only supported by `flux-general-multi` and will be ignored for model '%s'",
                    self._opts.model,
                )
        if is_given(endpoint_url):
            self._opts.endpoint_url = endpoint_url
        if is_given(eager_eot_threshold):
            self._opts.eager_eot_threshold = eager_eot_threshold

        for stream in self._streams:
            stream.update_options(
                model=model,
                sample_rate=sample_rate,
                eot_threshold=eot_threshold,
                eot_timeout_ms=eot_timeout_ms,
                keyterm=keyterm,
                mip_opt_out=mip_opt_out,
                numerals=numerals,
                profanity_filter=profanity_filter,
                redact=redact,
                endpoint_url=endpoint_url,
                tags=tags,
                language_hint=language_hint,
                eager_eot_threshold=eager_eot_threshold,
            )

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        if keyterms == self._session_keyterms:
            return
        self._session_keyterms = list(keyterms)
        merged = list(dict.fromkeys([*self._user_keyterm, *keyterms]))
        self._opts.keyterm = merged
        for stream in self._streams:
            # tuned in-band, safe to apply mid-utterance
            stream.update_options(keyterm=merged)


class _Word(TypedDict):
    word: str
    start: float
    end: float
    confidence: NotRequired[float]


class _TurnInfo(TypedDict):
    type: Literal["TurnInfo"]
    event: str
    transcript: NotRequired[str]
    words: NotRequired[list[_Word]]
    languages: NotRequired[list[str]]
    audio_window_start: NotRequired[float]
    audio_window_end: NotRequired[float]
    request_id: NotRequired[str]


class _ConfigureResult(TypedDict):
    type: Literal["ConfigureSuccess", "ConfigureFailure"]


class _Error(TypedDict):
    type: Literal["Error"]
    description: NotRequired[str]


_Message = Annotated[_TurnInfo | _ConfigureResult | _Error, Field(discriminator="type")]


class _CloseStream(TypedDict):
    type: Literal["CloseStream"]


class _Configure(TypedDict):
    # https://developers.deepgram.com/docs/flux/configure: only the changed fields
    type: Literal["Configure"]
    thresholds: NotRequired[dict[str, float | int]]
    keyterms: NotRequired[list[str]]
    language_hints: NotRequired[list[str]]


class SpeechStreamv2(stt.SocketStream[_Message, _CloseStream | _Configure]):
    goodbye = _CloseStream(type="CloseStream")
    frame_ms = 50
    # Flux detects turns itself: StartOfTurn and EndOfTurn
    provider_vad = True

    def __init__(
        self,
        *,
        stt: STTv2,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        base_url: str,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._api_key = api_key
        self._opts.endpoint_url = base_url
        self._request_id = ""

    def update_options(
        self,
        *,
        model: NotGivenOr[V2Models | str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        eot_threshold: NotGivenOr[float] = NOT_GIVEN,
        eot_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        keyterm: NotGivenOr[str | list[str]] = NOT_GIVEN,
        mip_opt_out: NotGivenOr[bool] = NOT_GIVEN,
        numerals: NotGivenOr[bool] = NOT_GIVEN,
        profanity_filter: NotGivenOr[bool] = NOT_GIVEN,
        redact: NotGivenOr[FluxRedaction] = NOT_GIVEN,
        tags: NotGivenOr[list[str]] = NOT_GIVEN,
        language_hint: NotGivenOr[list[str]] = NOT_GIVEN,
        endpoint_url: NotGivenOr[str] = NOT_GIVEN,
        eager_eot_threshold: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if is_given(eot_threshold):
            self._opts.eot_threshold = eot_threshold
        if is_given(eot_timeout_ms):
            self._opts.eot_timeout_ms = eot_timeout_ms
        if is_given(keyterm):
            self._opts.keyterm = keyterm
        if is_given(mip_opt_out):
            self._opts.mip_opt_out = mip_opt_out
        if is_given(numerals):
            self._opts.numerals = numerals
        if is_given(profanity_filter):
            self._opts.profanity_filter = profanity_filter
        if is_given(redact):
            self._opts.redact = redact
        if is_given(tags):
            self._opts.tags = _validate_tags(tags)
        if is_given(language_hint):
            self._opts.language_hint = language_hint
        if is_given(endpoint_url):
            self._opts.endpoint_url = endpoint_url
        if is_given(eager_eot_threshold):
            self._opts.eager_eot_threshold = eager_eot_threshold

        # these only take effect on a fresh connection: Flux does not support
        # toggling numerals, profanity_filter, or redact through Configure
        # https://developers.deepgram.com/docs/numerals
        needs_reconnect = any(
            is_given(opt)
            for opt in (
                model,
                sample_rate,
                mip_opt_out,
                numerals,
                profanity_filter,
                redact,
                tags,
                endpoint_url,
            )
        )
        if needs_reconnect:
            # the next connection carries the latest options
            self.reconnect("options changed")
            return

        configure = _Configure(type="Configure")
        thresholds: dict[str, float | int] = {}
        if is_given(eager_eot_threshold):
            thresholds["eager_eot_threshold"] = eager_eot_threshold
        if is_given(eot_threshold):
            thresholds["eot_threshold"] = eot_threshold
        if is_given(eot_timeout_ms):
            thresholds["eot_timeout_ms"] = eot_timeout_ms
        if thresholds:
            configure["thresholds"] = thresholds
        if is_given(keyterm):
            # keyterms replaces the whole list, so send the full effective set
            configure["keyterms"] = list(self._opts.keyterm)
        if is_given(language_hint):
            configure["language_hints"] = language_hint
        if len(configure) > 1:
            self.send_command(configure)

    def _live_config(self) -> dict[str, Any]:
        live_config: dict[str, Any] = {
            "model": self._opts.model,
            "sample_rate": self._opts.sample_rate,
            "encoding": "linear16",
            "mip_opt_out": self._opts.mip_opt_out,
        }

        if self._opts.eager_eot_threshold:
            live_config["eager_eot_threshold"] = self._opts.eager_eot_threshold

        if self._opts.eot_threshold:
            live_config["eot_threshold"] = self._opts.eot_threshold

        if self._opts.eot_timeout_ms:
            live_config["eot_timeout_ms"] = self._opts.eot_timeout_ms

        if self._opts.keyterm:
            live_config["keyterm"] = self._opts.keyterm

        if self._opts.tags:
            live_config["tag"] = self._opts.tags

        if self._opts.language_hint:
            live_config["language_hint"] = self._opts.language_hint

        if self._opts.numerals:
            live_config["numerals"] = self._opts.numerals

        if self._opts.profanity_filter:
            live_config["profanity_filter"] = self._opts.profanity_filter

        if is_given(self._opts.redact):
            live_config["redact"] = self._opts.redact

        return live_config

    async def connect(self, conn_options: APIConnectOptions) -> http.SocketTarget:
        return http.SocketTarget(
            to_deepgram_url(self._live_config(), base_url=self._opts.endpoint_url, websocket=True),
            headers={"Authorization": f"Token {self._api_key}"},
            request_id_header="dg-request-id",
        )

    def encode(
        self, item: rtc.AudioFrame | stt.RecognizeStream.Flush | http.EndOfInput
    ) -> http.Encoded[_CloseStream | _Configure]:
        # Flux finds the turns itself: a flush only sends the partial chunk, which the base does
        return bytes(item.data) if isinstance(item, rtc.AudioFrame) else None

    def on_message(
        self, message: _TurnInfo | _ConfigureResult | _Error
    ) -> stt.Report | list[stt.Report] | None:
        if message["type"] == "ConfigureSuccess":
            logger.debug("deepgram applied Configure update", extra={"lk.pii.data": message})
            return None
        if message["type"] == "ConfigureFailure":
            logger.warning("deepgram rejected Configure update", extra={"lk.pii.data": message})
            return None
        if message["type"] == "Error":
            raise APIError(
                "deepgram returned an error",
                kind="invalid_request",
                detail=message.get("description"),
            )

        if request_id := message.get("request_id"):
            self._request_id = request_id
        event = message["event"]
        if event == "StartOfTurn":
            return [stt.SpeechStarted(), *self._transcript(message)]
        if event in ("Update", "TurnResumed"):
            # an interim also aborts an eager end of turn
            return self._transcript(message)
        if event == "EagerEndOfTurn":
            # a pause: preflight for preemptive generation, the turn still open
            return self._transcript(message, preflight=True)
        if event == "EndOfTurn":
            return [*self._transcript(message, final=True), stt.SpeechEnded()]
        return None

    def _transcript(
        self, message: _TurnInfo, *, final: bool = False, preflight: bool = False
    ) -> list[stt.Report]:
        words = message.get("words", [])
        if not words:
            return []
        offset = self.start_time_offset
        languages = message.get("languages", [])
        return [
            stt.Transcript(
                message.get("transcript", ""),
                LanguageCode(languages[0] if languages else self._opts.language),
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
        ]


def _validate_tags(tags: list[str]) -> list[str]:
    for tag in tags:
        if len(tag) > 128:
            raise ValueError("tag must be no more than 128 characters")
    return tags
