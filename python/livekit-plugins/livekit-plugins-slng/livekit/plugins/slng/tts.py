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
import binascii
import os
from dataclasses import dataclass, replace
from typing import Literal, TypedDict, assert_never

from pydantic import JsonValue, TypeAdapter, ValidationError

from livekit.agents import APIConnectOptions, APIError, APIStatusError, tokenize, tts
from livekit.agents.metrics import TTSMetrics
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .connection import (
    PluginEvent,
    TTSConnectionConfig,
    bridge_endpoint,
    bridge_model,
)
from .gateway_adapter import (
    build_external_tracking_headers,
    build_tts_init_payload,
    extract_error_status,
    normalize_region_override,
    normalize_world_part_override,
)

NUM_CHANNELS = 1

# ElevenLabs generates one audio fragment per incoming text frame when
# auto_mode is enabled, so single-word frames produce fragmented (choppy,
# slow-to-complete) audio. Batch words into phrases before sending: flush on
# clause/sentence punctuation or once the buffer reaches this many characters.
_PHRASE_FLUSH_SUFFIXES = (".", "!", "?", ",", ";", ":")
# message types that end a segment, and the event names that do, or fail it
_AUDIO_TYPES = {"audio_chunk", "Audio", "audio", "chunk"}
_END_TYPES = {"audio_end", "end", "flushed", "Flushed", "done"}
_END_EVENTS = {"complete", "completed", "done", "end", "final"}
_ERROR_EVENTS = {"error", "failed"}


def _contains_letter(text: str) -> bool:
    """True if the text has at least one alphabetic character in any script.

    Some providers (notably Sarvam Bulbul) reject a per-token WebSocket frame
    that carries no allowed-language character, so such tokens must be merged
    into a neighbouring word rather than sent on their own.
    """
    return any(ch.isalpha() for ch in text)


@dataclass
class _TTSOptions:
    model_endpoint: str
    model: str
    voice: str
    language: str
    sample_rate: int
    encoding: Literal["linear16"]
    speed: float
    api_key: str
    model_options: dict[str, JsonValue]
    extra_headers: dict[str, str]
    runtime_init: dict[str, JsonValue] | None
    text_chunking: Literal["auto", "word", "phrase"]
    phrase_max_chars: int


# The SLNG bridge's socket: one segment a connection, its init, its text and a flush. The
# bridge relays each provider's own messages, tagged by `type` in several spellings or not
# tagged at all (ElevenLabs passes `audio` and `isFinal` through), so one message type declares
# every field read.


class _Data(TypedDict, total=False):
    audio: str
    event: str
    event_type: str
    message: str
    description: str
    error: str


class _Message(TypedDict, total=False):
    type: str
    data: str | _Data | None
    audio: str | None
    isFinal: bool | int | str | None
    error: JsonValue
    message: str
    description: str
    slng_request_id: str
    session_id: str


_MESSAGE = TypeAdapter(_Message)


class _Text(TypedDict):
    type: Literal["text"]
    text: str


class _Flush(TypedDict):
    type: Literal["flush"]


_Command = dict[str, JsonValue] | _Text | _Flush


class _Connection(tts.TTS[Literal["slng_event"]]):
    """One SLNG bridge connection; ``TTS`` chains them."""

    def __bool__(self) -> bool:
        # LiveKit Agents code may use truthiness checks like `tts or None`.
        # Some EventEmitter-style bases can be falsy when they have no listeners,
        # which would unintentionally disable TTS.
        return True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        connections: list[str | TTSConnectionConfig] | None = None,
        provider_api_key: str | None = None,
        voice: str,
        slng_base_url: str = "api.slng.ai",
        region_override: str | list[str] | None = None,
        world_part_override: str | None = None,
        external_agent_id: str | None = None,
        external_session_id: str | None = None,
        language: str = "en",
        sample_rate: int = 24000,
        speed: float = 1.0,
        extra_headers: dict[str, str] | None = None,
        # Advanced / optional. Used by integrations that drive the session
        # themselves; a typical client can ignore these.
        runtime_init: dict[str, JsonValue] | None = None,
        text_chunking: Literal["auto", "word", "phrase"] = "auto",
        phrase_max_chars: int = 60,
        warm_standby: bool = False,
        **model_options: JsonValue,
    ) -> None:
        """
        Create a new instance of SLNG TTS.

        Args:
            api_key (str): SLNG API key. Falls back to the ``SLNG_API_KEY`` env var.
            model: Unmute Bridge model identifier, e.g. "deepgram/aura:2".
            connections: Ordered model, endpoint, typed config, or SLNG TTS candidates.
            provider_api_key: Optional BYOK provider credential, sent as the
                ``X-Slng-Provider-Key`` header (external providers only).
            slng_base_url (str): Gateway host. Defaults to "api.slng.ai".
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
            voice: Required voice identifier.
            language (str): Language code. Defaults to "en".
            sample_rate (int): Sample rate of audio. Defaults to 24000.
            warm_standby (bool): Keep the next segment's bridge connection open and initialized
                ahead, so a segment pays no connect: one more connection is held open.
        """
        if {"first_audio_timeout_s", "fallback_recovery_cooldown_s"} & model_options.keys():
            raise TypeError(
                "Configure timeouts with APIConnectOptions and fallback with tts.FallbackAdapter"
            )
        resolved_key = api_key or os.environ.get("SLNG_API_KEY")
        if not resolved_key:
            raise ValueError("api_key is required, or set the SLNG_API_KEY environment variable")

        if not voice.strip():
            raise ValueError("voice is required")
        if text_chunking not in {"auto", "word", "phrase"}:
            raise ValueError("text_chunking must be 'auto', 'word', or 'phrase'")
        if phrase_max_chars <= 0:
            raise ValueError("phrase_max_chars must be positive")
        if model is not None and connections:
            raise ValueError("use model or connections, not both")

        raw_connections: list[str | TTSConnectionConfig]
        if connections:
            raw_connections = connections
        elif model is not None:
            raw_connections = [model]
        else:
            raise ValueError("model or connections is required")

        primary = raw_connections[0]
        if isinstance(primary, TTSConnectionConfig):
            resolved_model_endpoint = primary.endpoint
            endpoint_model = bridge_model(primary.endpoint, "tts")
            if primary.model is not None and primary.model != endpoint_model:
                raise ValueError("TTS connection model must match its endpoint")
            model = endpoint_model
            voice = primary.voice or voice
        else:
            resolved_model_endpoint = (
                primary if "://" in primary else bridge_endpoint(slng_base_url, "tts", primary)
            )
            model = bridge_model(resolved_model_endpoint, "tts")
            primary = TTSConnectionConfig(
                endpoint=resolved_model_endpoint,
                model=model,
                voice=voice,
            )

        headers = dict(extra_headers or {})
        region_override_header = normalize_region_override(region_override)
        if region_override_header:
            headers.setdefault("X-Region-Override", region_override_header)
        world_part_header = normalize_world_part_override(world_part_override)
        if world_part_header:
            headers.setdefault("X-World-Part-Override", world_part_header)
        headers.update(primary.headers)
        headers.update(
            build_external_tracking_headers(
                external_agent_id=external_agent_id,
                external_session_id=external_session_id,
            )
        )
        if provider_api_key is not None:
            byok_key = provider_api_key.strip()
            if not byok_key:
                raise ValueError("provider_api_key must not be empty")
            headers["X-Slng-Provider-Key"] = byok_key

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, text_unit="words"),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        self._opts = _TTSOptions(
            model_endpoint=resolved_model_endpoint,
            model=model,
            voice=voice,
            language=language,
            sample_rate=sample_rate,
            # LiveKit expects raw PCM. Some SLNG models default to MP3 unless explicitly requested.
            encoding="linear16",
            speed=speed,
            api_key=resolved_key,
            model_options=dict(model_options),
            extra_headers=headers,
            runtime_init=(
                dict(primary.init)
                if primary.init is not None
                else dict(runtime_init)
                if runtime_init is not None
                else None
            ),
            text_chunking=text_chunking,
            phrase_max_chars=phrase_max_chars,
        )
        # Set by the chain: a connection's voice is only updated at runtime when it was inherited
        # from the chain-level default rather than set explicitly per connection.
        self._inherits_voice = False
        # each segment's connection, initialized ahead: one session a connection, never reused
        self._standby = _Standby(self._opts) if warm_standby else None

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "slng"

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            voice (str): Voice to use.
            language (str): Language code.
            speed (float): Playback speed multiplier.
            sentence_tokenizer (SentenceTokenizer): What splits the text into the chunks the TTS
                is sent.
            pacing (SentenceStreamPacer | None): A ``SentenceStreamPacer``, or ``None`` to send each
                chunk as soon as it is split.
        """
        with self._updating_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing):
            _check_voice(voice)
        if is_given(voice):
            self._opts.voice = voice
        if is_given(language):
            self._opts.language = language
        if is_given(speed):
            self._opts.speed = speed
        if self._standby is not None:
            # a standby was initialized with the options before
            self._standby.invalidate()

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(chunk, conn_options=conn_options)

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, opts=replace(self._opts), conn_options=conn_options)
        stream.multiplexer = self._standby
        return stream

    def prewarm(self) -> None:
        if self._standby is not None:
            self._standby.prewarm()

    async def aclose(self) -> None:
        await super().aclose()
        if self._standby is not None:
            await self._standby.aclose()


class TTS(tts.FallbackAdapter[Literal["slng_event"]]):
    """SLNG TTS over one or more bridge connections, in order of preference: a failed one is
    skipped until it recovers. Its events (``error``, ``tts_availability_changed``,
    ``metrics_collected``, ``slng_event``) come from this object."""

    def __bool__(self) -> bool:
        # LiveKit Agents code may use truthiness checks like `tts or None`.
        return True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        connections: list[str | TTSConnectionConfig] | None = None,
        provider_api_key: str | None = None,
        voice: str,
        slng_base_url: str = "api.slng.ai",
        region_override: str | list[str] | None = None,
        world_part_override: str | None = None,
        external_agent_id: str | None = None,
        external_session_id: str | None = None,
        language: str = "en",
        sample_rate: int = 24000,
        speed: float = 1.0,
        extra_headers: dict[str, str] | None = None,
        runtime_init: dict[str, JsonValue] | None = None,
        text_chunking: Literal["auto", "word", "phrase"] = "auto",
        phrase_max_chars: int = 60,
        warm_standby: bool = False,
        **model_options: JsonValue,
    ) -> None:
        """
        Create a new instance of SLNG TTS.

        Args:
            api_key (str): SLNG API key. Falls back to the ``SLNG_API_KEY`` env var.
            model: Unmute Bridge model identifier, e.g. "deepgram/aura:2".
            connections: Ordered model, endpoint, typed config, or SLNG TTS candidates.
            provider_api_key: Optional BYOK provider credential, sent as the
                ``X-Slng-Provider-Key`` header (external providers only).
            slng_base_url (str): Gateway host. Defaults to "api.slng.ai".
            region_override: Optional gateway region override, sent as the
                ``X-Region-Override`` header. Accepts a single region or a list
                of preferred regions in priority order.
            world_part_override: Optional gateway world-part override, sent as
                the ``X-World-Part-Override`` header.
            external_agent_id: Optional tracking ID attached to usage events as
                the ``X-SLNG-Agent-Id`` header (max 128 chars).
            external_session_id: Optional tracking ID attached to usage events as
                the ``X-SLNG-Session-Id`` header (max 128 chars).
            voice: Required voice identifier.
            language (str): Language code. Defaults to "en".
            sample_rate (int): Sample rate of audio. Defaults to 24000.
            warm_standby (bool): Keep each connection's next segment connection open and
                initialized ahead, so a segment pays no connect.
        """
        if model is not None and connections:
            raise ValueError("use model or connections, not both")
        raw_connections: list[str | TTSConnectionConfig]
        if connections:
            raw_connections = connections
        elif model is not None:
            raw_connections = [model]
        else:
            raise ValueError("model or connections is required")

        self._connections: list[_Connection] = []
        for connection in raw_connections:
            config = (
                connection
                if isinstance(connection, TTSConnectionConfig)
                else TTSConnectionConfig(
                    endpoint=(
                        connection
                        if "://" in connection
                        else bridge_endpoint(slng_base_url, "tts", connection)
                    )
                )
            )
            candidate = _Connection(
                api_key=api_key,
                model=None,
                connections=[config],
                provider_api_key=provider_api_key,
                voice=config.voice or voice,
                slng_base_url=slng_base_url,
                region_override=region_override,
                world_part_override=world_part_override,
                external_agent_id=external_agent_id,
                external_session_id=external_session_id,
                language=language,
                sample_rate=sample_rate,
                speed=speed,
                extra_headers=extra_headers,
                runtime_init=runtime_init,
                text_chunking=text_chunking,
                phrase_max_chars=phrase_max_chars,
                warm_standby=warm_standby,
                **model_options,
            )
            candidate._inherits_voice = config.voice is None
            self._connections.append(candidate)
        super().__init__(list(self._connections), sample_rate=sample_rate)
        for candidate in self._connections:
            candidate.on("metrics_collected", self._forward_metrics)
            candidate.on("slng_event", self._forward_plugin_event)

    def _forward_metrics(self, metrics: TTSMetrics) -> None:
        self.emit("metrics_collected", metrics)

    def _forward_plugin_event(self, event: PluginEvent) -> None:
        self.emit("slng_event", event)

    @property
    def provider(self) -> str:
        return "slng"

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """Update every connection, so a later failover does not synthesize with
        construction-time settings; a connection with its own voice keeps it.

        Args:
            voice (str): Voice to use.
            language (str): Language code.
            speed (float): Playback speed multiplier.
            sentence_tokenizer (SentenceTokenizer): What splits the text into the chunks the TTS
                is sent.
            pacing (SentenceStreamPacer | None): A ``SentenceStreamPacer``, or ``None`` to send each
                chunk as soon as it is split.
        """
        # every option is checked before any connection applies one; each connection applies
        # the text options all or nothing itself
        _check_voice(voice)
        for candidate in self._connections:
            candidate.update_options(
                voice=voice if is_given(voice) and candidate._inherits_voice else NOT_GIVEN,
                language=language,
                speed=speed,
                sentence_tokenizer=sentence_tokenizer,
                pacing=pacing,
            )

    async def aclose(self) -> None:
        await super().aclose()
        for candidate in self._connections:
            await candidate.aclose()


def _target(opts: _TTSOptions) -> tts.WebSocketTarget:
    return tts.WebSocketTarget(
        opts.model_endpoint,
        headers={
            "Authorization": f"Bearer {opts.api_key}",
            "X-API-Key": opts.api_key,
            **opts.extra_headers,
        },
    )


def _init(opts: _TTSOptions) -> dict[str, JsonValue]:
    if opts.runtime_init is not None:
        return opts.runtime_init
    return build_tts_init_payload(
        model=opts.model,
        voice=opts.voice,
        language=opts.language,
        sample_rate=opts.sample_rate,
        encoding=opts.encoding,
        speed=opts.speed,
        model_options=opts.model_options,
    )


def _decode(protocol: SynthesizeStream | _Standby, frame: bytes | str) -> bytes | _Message | None:
    if isinstance(frame, bytes):
        return frame
    try:
        return _MESSAGE.validate_json(frame)
    except ValidationError as e:
        raise tts.protocol_error(protocol, f"{e}; frame: {frame}") from e


class _Standby(tts.WebSocketMultiplexer[bytes | _Message, _Command]):
    """Bridge connections opened and initialized ahead, one segment's session each: the next
    segment's is open while the one before speaks."""

    routed = False
    reuses_sockets = False
    keep_warm = True

    def __init__(self, opts: _TTSOptions) -> None:
        self._opts = opts

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        return _target(self._opts)

    def setup(self) -> dict[str, JsonValue]:
        return _init(self._opts)

    def decode(self, frame: bytes | str) -> bytes | _Message | None:
        return _decode(self, frame)


class SynthesizeStream(tts.WebSocketStream[bytes | _Message, _Command]):
    """Each segment on its own bridge connection: its init, its text batched into phrases (or
    words), then a flush; the bridge ends it with its end message or by closing."""

    connection_per_segment = True

    def __init__(self, *, tts: _Connection, opts: _TTSOptions, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._connection = tts
        self._opts = opts
        self._segment = ""
        # the text not yet sent, held so a letterless token joins a word
        self._buffer = ""
        self._buffer_has_letter = False

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        return _target(self._opts)

    def setup(self) -> dict[str, JsonValue]:
        return _init(self._opts)

    def decode(self, frame: bytes | str) -> bytes | _Message | None:
        return _decode(self, frame)

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> _Command | None:
        match item:
            case tts.SegmentStart():
                self._segment, self._buffer, self._buffer_has_letter = item.segment_id, "", False
                return None
            case tts.SegmentText():
                return self._batch(item.chunk.text)
            case tts.SegmentEnd():
                # a buffer with no letter at all carries no audio and would be refused
                if self._buffer and self._buffer_has_letter:
                    self.send_now(_Text(type="text", text=self._buffer))
                return _Flush(type="flush")
            case tts.EndOfText():
                return None
            case _:
                assert_never(item)

    def _batch(self, chunk: str) -> _Text | None:
        """The text frame a word completes: a phrase (at punctuation or ``phrase_max_chars``), or
        the word before it, a letterless token kept with a word."""
        piece, letter = f"{chunk} ", _contains_letter(chunk)
        if self._opts.text_chunking in ("auto", "phrase"):
            self._buffer += piece
            self._buffer_has_letter = self._buffer_has_letter or letter
            stripped = self._buffer.rstrip()
            at_boundary = stripped.endswith(_PHRASE_FLUSH_SUFFIXES) or (
                len(stripped) >= self._opts.phrase_max_chars
            )
            return self._take() if at_boundary and self._buffer_has_letter else None
        if letter and self._buffer_has_letter:
            text = self._take()
            self._buffer, self._buffer_has_letter = piece, True
            return text
        self._buffer += piece
        self._buffer_has_letter = self._buffer_has_letter or letter
        return None

    def _take(self) -> _Text:
        text = _Text(type="text", text=self._buffer)
        self._buffer, self._buffer_has_letter = "", False
        return text

    def on_message(self, message: bytes | _Message) -> None:
        if isinstance(message, bytes):
            self.report(tts.SegmentAudio(self._segment, message))
            return
        kind = message.get("type")
        data = message.get("data")
        nested = data if isinstance(data, dict) else _Data()
        if kind == "ready":
            if request_id := message.get("slng_request_id"):
                self.report(tts.ProviderRequestId(request_id))
            self._connection.emit(
                "slng_event",
                PluginEvent(
                    name="gateway.session",
                    component="tts",
                    data={
                        "gateway_request_id": message.get("slng_request_id"),
                        "gateway_session_id": message.get("session_id"),
                    },
                ),
            )
            return
        event = (nested.get("event") or nested.get("event_type") or "").strip().lower()
        failed = kind in ("Error", "error") or (kind == "event" and event in _ERROR_EVENTS)
        if failed or (kind is None and message.get("error") is not None):
            raise _error(message, nested)
        audio = (data if isinstance(data, str) else None) or nested.get("audio")
        audio = audio or message.get("audio")
        if audio and (kind is None or kind in _AUDIO_TYPES or kind in _END_TYPES):
            self.report(tts.SegmentAudio(self._segment, _decode_audio(audio)))
        ended = kind in _END_TYPES or (kind == "event" and event in _END_EVENTS)
        if ended or (kind is None and _is_true(message.get("isFinal"))):
            self.report(tts.SegmentDone(self._segment))


def _check_voice(voice: NotGivenOr[str]) -> None:
    if is_given(voice) and not voice.strip():
        raise ValueError("voice is required")


def _is_true(value: bool | int | str | None) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1")
    return value is True or value == 1


def _decode_audio(audio_b64: str) -> bytes:
    try:
        return base64.b64decode(audio_b64, validate=True)
    except binascii.Error as e:
        raise APIError("SLNG sent malformed audio", kind="protocol") from e


def _error(message: _Message, nested: _Data) -> APIError:
    detail = (
        nested.get("message")
        or nested.get("description")
        or nested.get("error")
        or message.get("message")
        or message.get("description")
    )
    if detail is None and (error := message.get("error")) is not None:
        detail = str(error)
    if (status := extract_error_status(message)) is None:
        # an error frame without a status refused the request
        return APIError("SLNG TTS error", kind="invalid_request", detail=detail)
    return APIStatusError("SLNG TTS error", status_code=status, detail=detail)
