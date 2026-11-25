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
import os
import time
import weakref
from dataclasses import dataclass, field
from typing import NotRequired, TypedDict, assert_never

from livekit.agents import APIConnectOptions, APIError, APIStatusError, tokenize, tts, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

WEBSOCKET_URL = "wss://tts-rt.soniox.com/tts-websocket"
NUM_CHANNELS = 1
DEFAULT_MODEL = "tts-rt-v1-preview"
DEFAULT_LANGUAGE = "en"
DEFAULT_VOICE = "Maya"
DEFAULT_AUDIO_FORMAT = "pcm_s16le"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_SPEED = 1.0
MIN_SPEED = 0.7
MAX_SPEED = 1.3
KEEPALIVE_INTERVAL = 10  # seconds
DEFAULT_STREAM_IDLE_TIMEOUT = 5.0  # seconds
MAX_STREAM_AGE = 90.0  # seconds, under the provider stream's 2-minute lifespan
MODELS_WITHOUT_SILENCE_REDUCTION = frozenset({"tts-rt-v1", "tts-rt-v1-preview"})


def _validate_reduce_silence(model: str, reduce_silence: bool) -> None:
    if reduce_silence and model in MODELS_WITHOUT_SILENCE_REDUCTION:
        raise ValueError(
            f"reduce_silence is not supported on model {model!r}; use a model with "
            "supports_silence_reduction (e.g. tts-rt-v2). "
            "See https://soniox.com/docs/tts/concepts/reduce-silence"
        )


def _audio_format_to_mime_type(audio_format: str) -> str:
    if audio_format.startswith("pcm"):
        return "audio/pcm"
    if audio_format == "mp3":
        return "audio/mpeg"
    return f"audio/{audio_format}"


@dataclass
class _TTSOptions:
    model: str
    language: str
    voice: str
    reduce_silence: bool
    audio_format: str
    sample_rate: int
    bitrate: int | None
    speed: float
    websocket_url: str
    api_key: str
    stream_idle_timeout: float


# Soniox's socket carries provider streams keyed by `stream_id`: each opened by its config, fed
# text, ended by `text_end` and closed by `terminated`. Messages carry no tag; each field read
# is declared.


class _Message(TypedDict, total=False):
    stream_id: str | None
    audio: str | None
    audio_end: bool | None
    terminated: bool | None
    error_code: int | None
    error_message: str | None
    request_id: str | None


class _Config(TypedDict):
    model: str
    language: str
    voice: str
    audio_format: str
    sample_rate: int
    speed: float
    stream_id: str
    bitrate: NotRequired[int]
    reduce_silence: NotRequired[bool]


class _Text(TypedDict):
    stream_id: str
    text: NotRequired[str]
    text_end: NotRequired[bool]


class _Cancel(TypedDict):
    stream_id: str
    cancel: bool


class _KeepAlive(TypedDict):
    keep_alive: bool


_Command = _Config | _Text | _Cancel | _KeepAlive


class TTS(tts.TTS):
    """Text-to-Speech service using Soniox Text-to-Speech API.

    This service connects to Soniox Text-to-Speech API for real-time speech synthesis
    with support for multiple languages, voices, and audio formats.

    For complete API documentation, see: https://soniox.com/docs/api-reference/tts/websocket-api
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        language: str = DEFAULT_LANGUAGE,
        voice: str = DEFAULT_VOICE,
        audio_format: str = DEFAULT_AUDIO_FORMAT,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        bitrate: int | None = None,
        speed: float = DEFAULT_SPEED,
        reduce_silence: bool = False,
        api_key: str | None = None,
        websocket_url: str = WEBSOCKET_URL,
        stream_idle_timeout: float = DEFAULT_STREAM_IDLE_TIMEOUT,
    ) -> None:
        """Initialize instance of Soniox Text-to-Speech API service.

        Args:
            model (str): Soniox TTS model to use. Defaults to "tts-rt-v1-preview".
            language (str): Language code (e.g., "en", "es", "fr"). Defaults to "en".
            voice (str): Voice name (e.g., "Maya", "Adrian"). Defaults to "Maya".
            audio_format (str): Audio format (e.g., "pcm_s16le", "mp3"). Defaults to "pcm_s16le".
            sample_rate (int): Sample rate in Hz. Required for raw audio formats. Defaults to 24000.
            bitrate (int): Codec bitrate in bps for compressed formats. Optional.
            speed (float): Speaking rate. 1.0 is the normal rate; values below 1.0 slow speech
                down and values above 1.0 speed it up. Range is [0.7, 1.3]. Defaults to 1.0.
            reduce_silence (bool): Shorten the pauses between words. Only supported on
                models that report supports_silence_reduction (e.g. tts-rt-v2); the server
                rejects it on other models. Defaults to False.
            api_key (str): Soniox API key. If not provided, will look for SONIOX_API_KEY env variable.
            websocket_url (str): Base WebSocket URL for Soniox TTS API.
            stream_idle_timeout (float): Seconds without a new sentence before the current
                stream is finalized; the next sentence starts a fresh stream. Prevents slow
                LLM gaps from hitting the server's per-stream timeout (observed ~8-18s).
                Defaults to 5.0.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        api_key = api_key or os.environ.get("SONIOX_API_KEY")
        if not api_key:
            raise ValueError("Soniox API key is required. Set SONIOX_API_KEY or provide api_key.")

        if not MIN_SPEED <= speed <= MAX_SPEED:
            raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}, but got {speed}")

        if stream_idle_timeout <= 0:
            raise ValueError(f"stream_idle_timeout must be > 0, but got {stream_idle_timeout}")

        _validate_reduce_silence(model, reduce_silence)

        self._opts = _TTSOptions(
            model=model,
            language=language,
            voice=voice,
            reduce_silence=reduce_silence,
            audio_format=audio_format,
            sample_rate=sample_rate,
            bitrate=bitrate,
            speed=speed,
            websocket_url=websocket_url,
            api_key=api_key,
            stream_idle_timeout=stream_idle_timeout,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._mux = _Mux(self._opts)

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "soniox"

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        reduce_silence: NotGivenOr[bool] = NOT_GIVEN,
        stream_idle_timeout: NotGivenOr[float] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            model: TTS model to use.
            language: Language code to use.
            voice: Voice to use.
            speed: Speaking rate in the range [0.7, 1.3]; 1.0 is the normal rate.
            reduce_silence: Shorten the pauses between words (models with
                supports_silence_reduction only).
            stream_idle_timeout: Idle seconds before the current stream is finalized.
            sentence_tokenizer: What splits the text into the chunks the TTS is sent.
            pacing: A ``SentenceStreamPacer``, or ``None`` to send each chunk as soon as it is
                split.
        """
        with self._updating_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing):
            _validate_reduce_silence(
                model if is_given(model) else self._opts.model,
                reduce_silence if is_given(reduce_silence) else self._opts.reduce_silence,
            )
            if is_given(speed) and not MIN_SPEED <= speed <= MAX_SPEED:
                raise ValueError(
                    f"speed must be between {MIN_SPEED} and {MAX_SPEED}, but got {speed}"
                )
            if is_given(stream_idle_timeout) and stream_idle_timeout <= 0:
                raise ValueError(f"stream_idle_timeout must be > 0, but got {stream_idle_timeout}")
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language
        if is_given(voice):
            self._opts.voice = voice
        if is_given(speed):
            self._opts.speed = speed
        if is_given(reduce_silence):
            self._opts.reduce_silence = reduce_silence
        if is_given(stream_idle_timeout):
            # the streams made next finalize their provider stream after it
            self._opts.stream_idle_timeout = stream_idle_timeout

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(chunk, conn_options=conn_options)

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, opts=self._opts, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        self._mux.prewarm()

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._mux.aclose()


@dataclass
class _Segment:
    """A segment's provider streams: the one taking its text, and those not yet terminated."""

    current: str | None = None
    latest: str | None = None
    opened_at: float = 0.0
    live: set[str] = field(default_factory=set[str])
    spoken: set[str] = field(default_factory=set[str])
    ended: bool = False


class _Mux(tts.WebSocketMultiplexer[_Message, _Command]):
    """One Soniox socket carries every stream's segments, each segment a provider stream (a new
    one after the text idles ``stream_idle_timeout``, before the server times the stream out)."""

    keepalive = tts.Keepalive(every=KEEPALIVE_INTERVAL, frame=_KeepAlive(keep_alive=True))

    def __init__(self, opts: _TTSOptions) -> None:
        self._opts = opts
        # each open segment, and the segment each provider stream speaks for
        self.segments: dict[str, _Segment] = {}
        self.provider_streams: dict[str, str] = {}

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        return tts.WebSocketTarget(
            self._opts.websocket_url, headers={"Authorization": f"Bearer {self._opts.api_key}"}
        )

    def route(self, message: _Message) -> str | None:
        if stream_id := message.get("stream_id"):
            return self.provider_streams.get(stream_id)
        if error_code := message.get("error_code"):
            # an error naming no stream fails every stream on the socket
            raise _error("Soniox TTS connection error", error_code, message)
        return None

    def close_context(self, context_id: str) -> _Cancel | None:
        # the stream left mid-segment: its provider stream taking text is cancelled; one ended
        # by an idle terminates on its own
        segment = self.segments.pop(context_id, None)
        if segment is None:
            return None
        for stream_id in segment.live:
            self.provider_streams.pop(stream_id, None)
        if (latest := segment.latest) is None or latest not in segment.live:
            return None
        return _Cancel(stream_id=latest, cancel=True)


class SynthesizeStream(tts.WebSocketStream[_Message, _Command]):
    """A stream on the shared Soniox socket: each segment is a provider stream, which keeps its
    sentences' prosody continuous; its config goes with its first text, since the server times
    a stream from its config."""

    def __init__(self, *, tts: TTS, opts: _TTSOptions, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self.multiplexer = tts._mux
        self._mux = tts._mux
        self._opts = opts
        self.mime_type = _audio_format_to_mime_type(opts.audio_format)
        # the text stalled this long: its provider stream ends before the server times it out
        self.text_idle = opts.stream_idle_timeout
        self._segment_ids: list[str] = []

    def on_text_idle(self) -> None:
        # the stream's segments still open: a done one left the multiplexer
        self._segment_ids = [i for i in self._segment_ids if i in self._mux.segments]
        for segment_id in self._segment_ids:
            segment = self._mux.segments.get(segment_id)
            if segment is not None and segment.current is not None:
                current, segment.current = segment.current, None
                self.send_now(_Text(stream_id=current, text_end=True))

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> _Command | None:
        match item:
            case tts.SegmentStart():
                self._mux.segments[item.segment_id] = _Segment()
                self._segment_ids.append(item.segment_id)
                return None
            case tts.SegmentText():
                segment = self._mux.segments[item.segment_id]
                if segment.current is not None and time.monotonic() - segment.opened_at > (
                    MAX_STREAM_AGE
                ):
                    self.send_now(_Text(stream_id=segment.current, text_end=True))
                    segment.current = None
                if segment.current is None:
                    segment.current = self._open(item.segment_id, segment)
                # the stream concatenates its text, so each chunk ends with a space
                return _Text(stream_id=segment.current, text=item.chunk.text + " ")
            case tts.SegmentEnd():
                segment = self._mux.segments[item.segment_id]
                segment.ended = True
                if segment.current is not None:
                    current, segment.current = segment.current, None
                    return _Text(stream_id=current, text_end=True)
                if not segment.live:
                    self._done(item.segment_id)
                return None
            case tts.EndOfText():
                return None
            case _:
                assert_never(item)

    def _open(self, segment_id: str, segment: _Segment) -> str:
        stream_id = utils.shortuuid()
        opts = self._opts
        config = _Config(
            model=opts.model,
            language=opts.language,
            voice=opts.voice,
            audio_format=opts.audio_format,
            sample_rate=opts.sample_rate,
            speed=opts.speed,
            stream_id=stream_id,
        )
        if opts.bitrate is not None:
            config["bitrate"] = opts.bitrate
        if opts.reduce_silence:
            config["reduce_silence"] = True
        self.send_now(config)
        segment.opened_at = time.monotonic()
        segment.live.add(stream_id)
        segment.latest = stream_id
        self._mux.provider_streams[stream_id] = segment_id
        return stream_id

    def on_message(self, message: _Message) -> None:
        stream_id = message.get("stream_id") or ""
        segment_id = self._mux.provider_streams.get(stream_id, "")
        if error_code := message.get("error_code"):
            raise _error(f"Soniox TTS stream error (stream_id={stream_id})", error_code, message)
        if audio := message.get("audio"):
            self.report(tts.SegmentAudio(segment_id, base64.b64decode(audio)))
        segment = self._mux.segments.get(segment_id)
        if segment is None:
            return
        if message.get("audio_end"):
            segment.spoken.add(stream_id)
        if message.get("terminated"):
            if stream_id not in segment.spoken:
                # the server aborted on its own: raised, so the turn is not silent
                raise APIError(
                    "Soniox TTS stream terminated without producing audio",
                    kind="server",
                    body=f"stream_id={stream_id}",
                )
            segment.live.discard(stream_id)
            del self._mux.provider_streams[stream_id]
            if segment.ended and not segment.live:
                self._done(segment_id)

    def _done(self, segment_id: str) -> None:
        self._mux.segments.pop(segment_id, None)
        self.report(tts.SegmentDone(segment_id))


def _error(message: str, status: int, body: _Message) -> APIStatusError:
    return APIStatusError(
        message,
        status_code=status,
        request_id=body.get("request_id"),
        body=body,
        detail=body.get("error_message"),
    )
