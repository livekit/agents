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
from dataclasses import dataclass, replace
from typing import Annotated, Literal, NotRequired, TypedDict, assert_never

from pydantic import Field, JsonValue, TypeAdapter, ValidationError

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIError,
    APIStatusError,
    LanguageCode,
    stt,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .connection import PluginEvent, bridge_endpoint
from .gateway_adapter import (
    build_external_tracking_headers,
    build_stt_init_payload,
    extract_error_status,
    normalize_region_override,
    normalize_world_part_override,
)

# Only pcm_s16le is supported: LiveKit AudioFrames carry 16-bit PCM and the
# plugin sends those bytes verbatim (no transcoding is performed).
STTEncoding = Literal["pcm_s16le"]
DEFAULT_BUFFER_SIZE_SECONDS = 0.064


@dataclass
class _Options:
    model: str
    endpoint: str
    api_key: str
    headers: dict[str, str]
    model_options: dict[str, JsonValue]
    sample_rate: int
    encoding: STTEncoding
    enable_partial_transcripts: bool
    vad_threshold: float
    vad_min_silence_duration_ms: int
    vad_speech_pad_ms: int
    enable_diarization: bool
    min_speakers: int | None
    max_speakers: int | None
    language: str


# The SLNG bridge's socket: its init, then the audio, a finalize at each end of segment and a
# close at the end. The bridge relays each provider's results, in its own shape (``*_transcript``)
# or Deepgram's (``Results``); a message of any other type is not read.


class _Word(TypedDict):
    start: NotRequired[float | None]
    end: NotRequired[float | None]


class _TranscriptBody(TypedDict):
    transcript: NotRequired[str | None]
    confidence: NotRequired[float | None]
    words: NotRequired[list[_Word] | None]
    language: NotRequired[str | None]


class _Partial(_TranscriptBody):
    type: Literal["partial_transcript"]


class _Final(_TranscriptBody):
    type: Literal["final_transcript"]


class _Channel(TypedDict):
    alternatives: NotRequired[list[_TranscriptBody] | None]


class _Results(TypedDict):
    type: Literal["Results"]
    is_final: NotRequired[bool | str | None]
    channel: NotRequired[_Channel | None]
    language: NotRequired[str | None]


class _Ready(TypedDict):
    type: Literal["ready"]
    slng_request_id: NotRequired[str | None]
    session_id: NotRequired[str | None]


class _ErrorData(TypedDict):
    message: NotRequired[JsonValue]
    description: NotRequired[JsonValue]
    error: NotRequired[JsonValue]
    status_code: NotRequired[int | str]
    code: NotRequired[int | str]
    err_code: NotRequired[int | str]


class _ErrorBody(_ErrorData):
    data: NotRequired[_ErrorData | None]


class _Error(_ErrorBody):
    type: Literal["error"]


class _ErrorTitled(_ErrorBody):
    type: Literal["Error"]


_Message = _Ready | _Results | _Partial | _Final | _Error | _ErrorTitled
_MESSAGE: TypeAdapter[_Message] = TypeAdapter(
    Annotated[
        _Ready | _Results | _Partial | _Final | _Error | _ErrorTitled,
        Field(discriminator="type"),
    ]
)
_READ = {"ready", "Results", "partial_transcript", "final_transcript", "error", "Error"}


_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


class _Control(TypedDict):
    type: Literal["finalize", "close", "keepalive"]


_Command = dict[str, JsonValue] | _Control


class STT(stt.STT[Literal["slng_event"]]):
    """SLNG STT over one Unmute Bridge model. For failover across models or regions, wrap
    several in ``stt.FallbackAdapter``."""

    def __bool__(self) -> bool:
        # LiveKit Agents code may use truthiness checks like `stt or None`.
        # Some EventEmitter-style bases can be falsy when they have no listeners,
        # which would unintentionally disable STT.
        return True

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        provider_api_key: str | None = None,
        slng_base_url: str = "api.slng.ai",
        region_override: str | list[str] | None = None,
        world_part_override: str | None = None,
        external_agent_id: str | None = None,
        external_session_id: str | None = None,
        sample_rate: int = 16000,
        encoding: STTEncoding = "pcm_s16le",
        buffer_size_seconds: float = DEFAULT_BUFFER_SIZE_SECONDS,
        enable_partial_transcripts: bool = True,
        vad_threshold: float = 0.5,
        vad_min_silence_duration_ms: int = 300,
        vad_speech_pad_ms: int = 30,
        enable_diarization: bool = False,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        language: str = "en",
        extra_headers: dict[str, str] | None = None,
        **model_options: JsonValue,
    ) -> None:
        """
        Initialize SLNG STT.

        Args:
            model: Unmute Bridge model identifier, e.g. "deepgram/nova:3".
            api_key: SLNG API key. Falls back to the ``SLNG_API_KEY`` env var.
            provider_api_key: Optional BYOK provider credential, sent as the
                ``X-Slng-Provider-Key`` header (external providers only).
            slng_base_url: Gateway host. Defaults to "api.slng.ai".
            region_override: Optional gateway region override, sent as the
                ``X-Region-Override`` header. Accepts a single region or a list
                of preferred regions in priority order.
            world_part_override: Optional gateway world-part override, sent as
                the ``X-World-Part-Override`` header. Constrains routing to a
                broad geographic zone (for example "eu", "na", "ap") when an
                exact region is not required. ``region_override`` takes
                precedence when both are set.
            external_agent_id: Optional tracking ID attached to usage events as
                the ``X-SLNG-Agent-Id`` header (max 128 chars).
            external_session_id: Optional tracking ID attached to usage events as
                the ``X-SLNG-Session-Id`` header (max 128 chars).
            sample_rate: Audio sample rate (default: 16000)
            encoding: Audio encoding; only pcm_s16le, sent as LiveKit's frames carry it.
            buffer_size_seconds: Duration of each audio frame sent.
            enable_partial_transcripts: Enable interim results
            vad_threshold: Voice activity detection threshold
            vad_min_silence_duration_ms: Min silence duration for VAD
            vad_speech_pad_ms: Speech padding for VAD
            enable_diarization: Enable speaker identification
            min_speakers: Minimum speakers for diarization
            max_speakers: Maximum speakers for diarization
            language: Language code (default: "en")
            extra_headers: Headers sent with each connection.
            **model_options: Model-specific options (e.g., whisper_params={"task": "translate"})
        """
        resolved_key = api_key or os.environ.get("SLNG_API_KEY")
        if not resolved_key:
            raise ValueError("api_key is required, or set the SLNG_API_KEY environment variable")
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript="chunk",
                offline_recognize=False,
            ),
        )
        headers = dict(extra_headers or {})
        # route toward preferred gateway regions
        if region_header := normalize_region_override(region_override):
            headers.setdefault("X-Region-Override", region_header)
        if world_part_header := normalize_world_part_override(world_part_override):
            headers.setdefault("X-World-Part-Override", world_part_header)
        headers.update(
            build_external_tracking_headers(
                external_agent_id=external_agent_id,
                external_session_id=external_session_id,
            )
        )
        if provider_api_key is not None:
            if not (byok_key := provider_api_key.strip()):
                raise ValueError("provider_api_key must not be empty")
            headers["X-Slng-Provider-Key"] = byok_key
        self._frame_ms = max(1, round(buffer_size_seconds * 1000))
        self._opts = _Options(
            model=model,
            endpoint=bridge_endpoint(slng_base_url, "stt", model),
            api_key=resolved_key,
            headers=headers,
            model_options=dict(model_options),
            sample_rate=sample_rate,
            encoding=encoding,
            enable_partial_transcripts=enable_partial_transcripts,
            vad_threshold=vad_threshold,
            vad_min_silence_duration_ms=vad_min_silence_duration_ms,
            vad_speech_pad_ms=vad_speech_pad_ms,
            enable_diarization=enable_diarization,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            language=language,
        )
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "SLNG"

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError(
            "SLNG STT recognize() is not supported: the Unmute Bridge is "
            "WebSocket-only. Use stream() instead."
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        opts = replace(self._opts, language=language) if is_given(language) else self._opts
        stream = SpeechStream(stt=self, opts=replace(opts), conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        enable_partial_transcripts: NotGivenOr[bool] = NOT_GIVEN,
        enable_diarization: NotGivenOr[bool] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        vad_min_silence_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_speech_pad_ms: NotGivenOr[int] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Update the options; each open stream reconnects with them."""
        for opts in (self._opts, *(stream._opts for stream in self._streams)):
            if is_given(enable_partial_transcripts):
                opts.enable_partial_transcripts = enable_partial_transcripts
            if is_given(enable_diarization):
                opts.enable_diarization = enable_diarization
            if is_given(vad_threshold):
                opts.vad_threshold = vad_threshold
            if is_given(vad_min_silence_duration_ms):
                opts.vad_min_silence_duration_ms = vad_min_silence_duration_ms
            if is_given(vad_speech_pad_ms):
                opts.vad_speech_pad_ms = vad_speech_pad_ms
            if is_given(language):
                opts.language = language
        for stream in self._streams:
            stream.reconnect("options changed")


class SpeechStream(stt.WebSocketStream[_Message, _Command]):
    """One recognition on the bridge: its init, then the audio; a flush sends a finalize, the
    end of the audio a close."""

    keepalive = stt.Keepalive(every=5.0, frame=_Control(type="keepalive"))
    # nothing but the init goes until the bridge says it is ready
    awaits_ready = True

    def __init__(self, *, stt: STT, opts: _Options, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self.frame_ms = stt._frame_ms
        self._slng = stt
        self._opts = opts
        self._gateway_request = ""

    async def connect(self, conn_options: APIConnectOptions) -> stt.WebSocketTarget:
        return stt.WebSocketTarget(
            self._opts.endpoint,
            headers={
                "Authorization": f"Bearer {self._opts.api_key}",
                "X-API-Key": self._opts.api_key,
                **self._opts.headers,
            },
        )

    def setup(self) -> dict[str, JsonValue]:
        opts = self._opts
        return build_stt_init_payload(
            model=opts.model,
            language=opts.language,
            sample_rate=opts.sample_rate,
            encoding=opts.encoding,
            vad_threshold=opts.vad_threshold,
            vad_min_silence_duration_ms=opts.vad_min_silence_duration_ms,
            vad_speech_pad_ms=opts.vad_speech_pad_ms,
            enable_diarization=opts.enable_diarization,
            enable_partial_transcripts=opts.enable_partial_transcripts,
            min_speakers=opts.min_speakers,
            max_speakers=opts.max_speakers,
            model_options=opts.model_options,
        )

    def decode(self, frame: bytes | str) -> _Message | None:
        if isinstance(frame, bytes):
            stt.log_skipped("SLNG STT", "a binary frame")
            return None
        try:
            value = _JSON.validate_json(frame)
            kind = value.get("type") if isinstance(value, dict) else None
            if not isinstance(kind, str) or kind not in _READ:
                # a message type the bridge added since, or a frame that is not one: skipped
                shown = f"type={kind!r}" if isinstance(kind, str) else "an untyped frame"
                stt.log_skipped("SLNG STT", shown)
                return None
            return _MESSAGE.validate_python(value)
        except ValidationError as e:
            raise stt.protocol_error(self, f"{e}; frame: {frame}") from e

    def encode(self, item: rtc.AudioFrame | stt.Flush | stt.EndOfAudio) -> bytes | _Control:
        match item:
            case rtc.AudioFrame():
                return bytes(item.data)
            case stt.Flush():
                # the provider returns the segment's final now, not at its own endpointing
                return _Control(type="finalize")
            case stt.EndOfAudio():
                return _Control(type="close")
            case _:
                assert_never(item)

    def on_message(self, message: _Message) -> None:
        if message["type"] == "ready":
            self._gateway_request = message.get("slng_request_id") or ""
            self.report(stt.Ready())
            self._slng.emit(
                "slng_event",
                PluginEvent(
                    name="gateway.session",
                    component="stt",
                    data={
                        "gateway_request_id": message.get("slng_request_id"),
                        "gateway_session_id": message.get("session_id"),
                    },
                ),
            )
        elif message["type"] == "Results":
            alternatives = (message.get("channel") or _Channel()).get("alternatives") or []
            alternative = alternatives[0] if alternatives else _TranscriptBody()
            self._transcript(
                alternative,
                final=_is_true(message.get("is_final")),
                language=message.get("language") or alternative.get("language"),
            )
        elif message["type"] == "partial_transcript":
            self._transcript(message, final=False, language=message.get("language"))
        elif message["type"] == "final_transcript":
            self._transcript(message, final=True, language=message.get("language"))
        elif message["type"] == "error" or message["type"] == "Error":
            raise _error(message)
        else:
            assert_never(message)

    def _transcript(self, body: _TranscriptBody, *, final: bool, language: str | None) -> None:
        text = (body.get("transcript") or "").strip()
        if not final and (not text or not self._opts.enable_partial_transcripts):
            return
        words = body.get("words") or []
        offset = self.start_time_offset
        # an empty final retracts the interim it follows
        self.report(
            stt.Transcript(
                text,
                LanguageCode(language or self._opts.language),
                final=final,
                request_id=self._gateway_request,
                start_time=((words[0].get("start") or 0.0) if final and words else 0.0) + offset,
                end_time=((words[-1].get("end") or 0.0) if final and words else 0.0) + offset,
                confidence=body.get("confidence") or 0.0,
            )
        )


def _is_true(value: bool | str | None) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1")
    return value is True


def _text(value: JsonValue | None) -> str | None:
    """A message given as text, or nested one level as ``message``/``description``/``error``."""
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        for key in ("message", "description", "error"):
            if isinstance(nested := value.get(key), str) and nested:
                return nested
    return None


def _error(message: _Error | _ErrorTitled) -> APIError:
    data = message.get("data") or _ErrorData()
    detail = next(
        (
            text
            for value in (
                data.get("message"),
                data.get("description"),
                data.get("error"),
                message.get("message"),
                message.get("description"),
                message.get("error"),
            )
            if (text := _text(value)) is not None
        ),
        "Unknown error",
    )
    if (status := extract_error_status(message)) is None:
        # an error frame without a status refused the request
        return APIError("SLNG STT error", kind="invalid_request", detail=detail)
    return APIStatusError("SLNG STT error", status_code=status, detail=detail)
