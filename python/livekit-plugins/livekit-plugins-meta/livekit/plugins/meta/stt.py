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
import os
import weakref
from typing import ClassVar, Literal, Never, NotRequired, TypedDict, assert_never, final
from urllib.parse import urlparse

from pydantic import JsonValue, TypeAdapter, ValidationError

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIError,
    LanguageCode,
    stt,
)
from livekit.agents.stt import WebSocketPacing
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

DEFAULT_URL = "wss://api.meta.ai/v1/asr/realtime"
DEFAULT_MODEL = "muse-voice-transcribe-1.0"
_SAMPLE_RATE = 24_000
_LOOPBACK = ("localhost", "127.0.0.1", "::1")
_SUPPORTED_LANGUAGES = (
    "Arabic",
    "Bengali",
    "Dutch",
    "English",
    "French",
    "German",
    "Hebrew",
    "Hindi",
    "Indonesian",
    "Italian",
    "Japanese",
    "Kannada",
    "Korean",
    "Malay",
    "Mandarin Chinese",
    "Marathi",
    "Polish",
    "Portuguese",
    "Spanish",
    "Tagalog",
    "Tamil",
    "Telugu",
    "Thai",
    "Turkish",
    "Vietnamese",
)
_LANGUAGE_NAMES = {language.casefold(): language for language in _SUPPORTED_LANGUAGES}
_LANGUAGE_CODES = {
    "ar": "Arabic",
    "bn": "Bengali",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fil": "Tagalog",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "iw": "Hebrew",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "ms": "Malay",
    "mr": "Marathi",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "zh": "Mandarin Chinese",
}


def _normalize_hints(values: list[str] | None, *, name: str) -> list[str]:
    normalized: list[str] = []
    for value in values or ():
        hint = value.strip()
        if not hint:
            raise ValueError(f"{name} entries must be non-empty")
        if hint not in normalized:
            normalized.append(hint)
    return normalized


def _normalize_access_token(api_key: str) -> str:
    parts = api_key.split(None, 1)
    if parts and parts[0].casefold() == "bearer":
        if len(parts) != 2 or not parts[1].strip():
            raise ValueError("Meta Model API key must include a token after Bearer")
        return f"Bearer {parts[1].strip()}"
    return f"Bearer {api_key}"


def _normalize_language_bias(values: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for value in values or ():
        documented_name = _LANGUAGE_NAMES.get(value.strip().casefold())
        if documented_name is None:
            supported = ", ".join(_SUPPORTED_LANGUAGES)
            raise ValueError(f"unsupported language_bias entry {value!r}; supported: {supported}")
        if documented_name not in normalized:
            normalized.append(documented_name)
    return normalized


def _normalize_language_hint(language: str) -> str:
    value = language.strip()
    if not value:
        raise ValueError("language must be non-empty")

    documented_name = _LANGUAGE_NAMES.get(value.casefold())
    if documented_name is not None:
        return documented_name

    primary = value.replace("_", "-").split("-", 1)[0].casefold()
    mapped_name = _LANGUAGE_CODES.get(primary)
    if mapped_name is None:
        supported = ", ".join(_SUPPORTED_LANGUAGES)
        raise ValueError(f"unsupported Muse Voice language {language!r}; supported: {supported}")
    return mapped_name


class STT(stt.STT[Never]):
    """Streaming speech recognition with Meta Muse Voice Transcribe."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        url: str = DEFAULT_URL,
        keywords: list[str] | None = None,
        language_bias: list[str] | None = None,
    ) -> None:
        """Create a Meta Muse streaming STT provider.

        Args:
            api_key: Meta Model API key. Falls back to ``MODEL_API_KEY``, then
                ``META_API_KEY``.
            model: Muse Voice Transcribe model identifier.
            url: Realtime Muse ASR WebSocket endpoint: ``wss://``, or ``ws://`` on this machine.
            keywords: Static recognition keywords sent when each stream starts.
            language_bias: Static supported language names sent when each stream starts.
        """
        if api_key is not None:
            resolved_key = api_key.strip()
        else:
            resolved_key = (
                os.getenv("MODEL_API_KEY", "").strip() or os.getenv("META_API_KEY", "").strip()
            )
        if not resolved_key:
            raise ValueError(
                "Meta Model API key is required. Pass api_key or set MODEL_API_KEY or META_API_KEY"
            )
        if not model.strip():
            raise ValueError("model must be non-empty")

        # the access token goes in the handshake message: never in clear off this machine
        parsed_url = urlparse(url)
        if (
            not (
                parsed_url.scheme == "wss"
                or (parsed_url.scheme == "ws" and parsed_url.hostname in _LOOPBACK)
            )
            or not parsed_url.hostname
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.fragment
        ):
            raise ValueError(
                "url must be an absolute wss:// URL (ws:// on this machine) without credentials "
                "or a fragment"
            )

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=False,
                aligned_transcript=False,
                offline_recognize=False,
                keyterms=False,
                # usage events carry the audio the server reports as processed
                reports_usage=True,
                # each turn's speechComplete is its final, empty for noise, and may follow
                # its speechEnd
                transcribes_after_speech_ends=True,
            )
        )
        self._api_key = _normalize_access_token(resolved_key)
        self._model = model.strip()
        self._url = url
        self._keywords = _normalize_hints(keywords, name="keywords")
        self._language_bias = _normalize_language_bias(language_bias)
        self._streams: weakref.WeakSet[SpeechStream] = weakref.WeakSet()
        self._closed = False

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "Meta"

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise APIError(
            "Meta Muse Voice Transcribe supports streaming recognition only",
            kind="invalid_request",
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        if self._closed:
            raise RuntimeError("Meta STT is closed")

        language_bias = list(self._language_bias)
        if is_given(language):
            language_hint = _normalize_language_hint(str(language))
            if language_hint not in language_bias:
                language_bias.append(language_hint)

        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            api_key=self._api_key,
            model=self._model,
            url=self._url,
            keywords=list(self._keywords),
            language_bias=language_bias,
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        self._closed = True
        streams = tuple(self._streams)
        if streams:
            await asyncio.gather(*(stream.aclose() for stream in streams), return_exceptions=True)


@final
class _SessionStarted(TypedDict):
    """The handshake's acknowledgement, the one message Muse does not tag."""

    type: NotRequired[Literal[None]]
    sessionId: str


@final
class _Error(TypedDict):
    type: Literal["error"]


@final
class _SpeechStart(TypedDict):
    type: Literal["speechStart"]
    turnId: str | int


@final
class _Transcript(TypedDict):
    type: Literal["transcript"]
    transcript: str
    turnId: NotRequired[str | int]


@final
class _SpeechEnd(TypedDict):
    type: Literal["speechEnd"]
    turnId: str | int


@final
class _SpeechComplete(TypedDict):
    type: Literal["speechComplete"]
    transcript: str
    turnId: str | int


@final
class _AudioProgress(TypedDict):
    type: Literal["audioProgress"]
    audioProcessedMs: float


_Message = (
    _SessionStarted
    | _Error
    | _SpeechStart
    | _Transcript
    | _SpeechEnd
    | _SpeechComplete
    | _AudioProgress
)


_OBJECT: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])


class _Authorization(TypedDict):
    accessToken: str


class _Handshake(TypedDict):
    mode: Literal["ENDPOINTING"]
    authorization: _Authorization
    audioEncoding: Literal["PCM_24KHZ"]
    model: str
    partialMode: Literal["CUMULATIVE"]
    emitAudioProgress: bool
    keywords: NotRequired[list[str]]
    languageBias: NotRequired[list[str]]


class _EndStream(TypedDict):
    type: Literal["endStream"]


class SpeechStream(stt.WebSocketStream[_Message, _Handshake | _EndStream]):
    completion: ClassVar[stt.Completion] = "peer_closed"
    # Muse reads audio no faster than it plays, in 80 ms packets
    frame_ms = 80
    awaits_ready = True
    # speechStart and speechEnd are Muse's own turn boundaries
    provider_vad = True

    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        api_key: str,
        model: str,
        url: str,
        keywords: list[str],
        language_bias: list[str],
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=_SAMPLE_RATE)
        # real-time pacing per stream: one shared would pace every stream together
        self.pacing = WebSocketPacing(real_time=True)
        self._api_key = api_key
        self._model = model
        self._url = url
        self._keywords = keywords
        self._language_bias = language_bias
        self._session_id = ""
        self._processed_ms = 0.0

    async def connect(self, conn_options: APIConnectOptions) -> stt.WebSocketTarget:
        self._processed_ms = 0.0
        return stt.WebSocketTarget(self._url)

    def setup(self) -> _Handshake:
        handshake = _Handshake(
            mode="ENDPOINTING",
            authorization=_Authorization(accessToken=self._api_key),
            audioEncoding="PCM_24KHZ",
            model=self._model,
            partialMode="CUMULATIVE",
            emitAudioProgress=True,
        )
        if self._keywords:
            handshake["keywords"] = self._keywords
        if self._language_bias:
            handshake["languageBias"] = self._language_bias
        return handshake

    def encode(
        self, item: rtc.AudioFrame | stt.Flush | stt.EndOfAudio
    ) -> bytes | _EndStream | None:
        match item:
            case rtc.AudioFrame():
                if item.num_channels != 1:
                    raise APIError(
                        "Meta Muse realtime ASR requires mono audio", kind="invalid_request"
                    )
                return bytes(item.data)
            case stt.Flush():
                return None
            case stt.EndOfAudio():
                return _EndStream(type="endStream")
            case _:
                assert_never(item)

    def decode(self, frame: bytes | str) -> _Message | None:
        try:
            # an untagged message is the handshake's acknowledgement only when it names the
            # session: another (one Muse added since) is skipped
            if isinstance(frame, str):
                fields = _OBJECT.validate_json(frame)
                if "type" not in fields and "sessionId" not in fields:
                    stt.log_skipped("Meta Muse", f"untagged: {','.join(sorted(fields))}")
                    return None
            return super().decode(frame)
        except (APIError, ValidationError):
            # the frame may hold the customer's words: never attached
            raise APIError("Meta Muse realtime ASR protocol error", kind="protocol") from None

    def on_message(self, message: _Message) -> None:
        if "sessionId" in message:
            self._session_id = message["sessionId"]
            self.report(stt.Ready())
        elif message["type"] == "error":
            # its message can hold the customer's words: never attached
            raise APIError("Meta Muse realtime ASR error", kind="invalid_request")
        elif message["type"] == "speechStart":
            self.report(stt.SpeechStarted())
        elif message["type"] == "transcript":
            if message["transcript"]:
                self.report(self._transcript(message["transcript"], message.get("turnId")))
        elif message["type"] == "speechComplete":
            self.report(self._transcript(message["transcript"], message["turnId"], final=True))
        elif message["type"] == "speechEnd":
            self.report(stt.SpeechEnded())
        elif message["type"] == "audioProgress":
            processed = message["audioProcessedMs"]
            if processed > self._processed_ms:
                self.send(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.RECOGNITION_USAGE,
                        request_id=self._session_id,
                        recognition_usage=stt.RecognitionUsage(
                            audio_duration=(processed - self._processed_ms) / 1000
                        ),
                    )
                )
                self._processed_ms = processed
        else:
            assert_never(message)

    def _transcript(
        self, text: str, turn_id: str | int | None, *, final: bool = False
    ) -> stt.Transcript:
        return stt.Transcript(
            text,
            LanguageCode(""),
            final=final,
            request_id=str(turn_id) if turn_id is not None else "",
        )
