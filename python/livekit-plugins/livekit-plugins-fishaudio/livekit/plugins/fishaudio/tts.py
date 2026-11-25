from __future__ import annotations

import os
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, NotRequired, TypedDict, assert_never

import msgpack
from pydantic import TypeAdapter, ValidationError

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    http,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import LatencyMode, MP3Bitrate, OpusBitrate, OutputFormat, TTSModels
from .version import __version__

DEFAULT_MODEL: TTSModels = "s2.1-pro"
DEFAULT_VOICE_ID = "933563129e564b19a115bedd57b7406a"
DEFAULT_BASE_URL = "https://api.fish.audio"
NUM_CHANNELS = 1
USER_AGENT = f"livekit-plugins-fishaudio/{__version__}"

# Fish Audio's default sample rate per output format. Opus only supports 48 kHz;
# the other formats default to 24 kHz, which matches the previous plugin default.
_DEFAULT_SAMPLE_RATE: dict[OutputFormat, int] = {
    "opus": 48000,
    "pcm": 24000,
    "wav": 24000,
    "mp3": 32000,
}


@dataclass
class _TTSOptions:
    model: TTSModels | str
    output_format: OutputFormat
    sample_rate: int
    voice_id: NotGivenOr[str]
    base_url: str
    api_key: str
    latency_mode: LatencyMode
    chunk_length: int
    speed: NotGivenOr[float]
    volume: NotGivenOr[float]
    temperature: float
    top_p: float
    mp3_bitrate: MP3Bitrate
    opus_bitrate: OpusBitrate
    normalize: bool
    normalize_loudness: NotGivenOr[bool]
    max_new_tokens: NotGivenOr[int]
    min_chunk_length: NotGivenOr[int]
    condition_on_previous_chunks: NotGivenOr[bool]
    early_stop_threshold: NotGivenOr[float]

    def get_http_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get_ws_url(self, path: str) -> str:
        return f"{self.base_url.replace('http', 'ws', 1)}{path}"


# Fish Audio's live websocket speaks msgpack: a session per segment (start, text and flush, stop),
# answered by audio events and a finish. Each message declares only the fields read.


class _Audio(TypedDict):
    event: Literal["audio"]
    audio: bytes


class _Finish(TypedDict):
    event: Literal["finish"]
    reason: str


_Message = _Audio | _Finish
_MESSAGE: TypeAdapter[_Message] = TypeAdapter(_Message)


class _Prosody(TypedDict, total=False):
    speed: float
    volume: float
    normalize_loudness: bool


class _Request(TypedDict):
    text: str
    chunk_length: int
    format: str
    sample_rate: int
    mp3_bitrate: int
    opus_bitrate: int
    references: list[str]
    reference_id: str | None
    normalize: bool
    latency: str
    prosody: _Prosody | None
    top_p: float
    temperature: float
    features: list[str]
    max_new_tokens: NotRequired[int]
    min_chunk_length: NotRequired[int]
    condition_on_previous_chunks: NotRequired[bool]
    early_stop_threshold: NotRequired[float]


class _Start(TypedDict):
    event: Literal["start"]
    request: _Request


class _Text(TypedDict):
    event: Literal["text"]
    text: str


class _Control(TypedDict):
    event: Literal["flush", "stop"]


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        model: TTSModels | str = DEFAULT_MODEL,
        voice_id: NotGivenOr[str] = DEFAULT_VOICE_ID,
        output_format: OutputFormat = "wav",
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        latency_mode: LatencyMode = "balanced",
        chunk_length: int = 100,
        speed: NotGivenOr[float] = NOT_GIVEN,
        volume: NotGivenOr[float] = NOT_GIVEN,
        temperature: float = 0.7,
        top_p: float = 0.7,
        mp3_bitrate: MP3Bitrate = 64,
        opus_bitrate: OpusBitrate = 64000,
        normalize: bool = True,
        normalize_loudness: NotGivenOr[bool] = NOT_GIVEN,
        max_new_tokens: NotGivenOr[int] = NOT_GIVEN,
        min_chunk_length: NotGivenOr[int] = NOT_GIVEN,
        condition_on_previous_chunks: NotGivenOr[bool] = NOT_GIVEN,
        early_stop_threshold: NotGivenOr[float] = NOT_GIVEN,
        http_session: http.HttpSession | None = None,
    ) -> None:
        """
        Create a new instance of Fish Audio TTS.

        See https://docs.fish.audio/api-reference/endpoint/websocket/tts-live for more details
        on the Fish Audio Live TTS WebSocket API.

        Args:
            api_key (NotGivenOr[str]): Fish Audio API key. Reads ``FISH_API_KEY`` if unset.
            model (TTSModels | str): TTS model to use. Defaults to ``"s2.1-pro"``.
            voice_id (NotGivenOr[str]): Voice model ID. Fish Audio's API refers to this
                as ``reference_id``; it's the same value either way.
            output_format (OutputFormat): Audio output format. Defaults to ``"wav"``.
            sample_rate (int): Audio sample rate in Hz.
            base_url (NotGivenOr[str]): Custom base URL. Defaults to ``https://api.fish.audio``.
            latency_mode (LatencyMode): Streaming latency mode. ``"normal"``, ``"balanced"``,
                or ``"low"``. Defaults to ``"balanced"``.
            chunk_length (int): Upper bound on text Fish buffers before auto-synthesizing
                (100–300). With sentence-level flushing this is only hit by sentences longer
                than ``chunk_length``; otherwise audio is produced when each sentence is
                flushed. Defaults to 100.
            speed (NotGivenOr[float]): Speaking rate multiplier (Fish ``prosody.speed``).
                ``1.0`` is normal; below 1.0 is slower, above is faster. Unset uses the
                voice's natural pace.
            volume (NotGivenOr[float]): Loudness adjustment in decibels (Fish
                ``prosody.volume``). ``0`` is the voice's natural level. Unset leaves it
                unchanged.
            temperature (float): Sampling temperature (0–1). Higher values produce more
                varied, expressive speech; lower values are more stable. Defaults to 0.7.
            top_p (float): Nucleus sampling probability mass (0–1). Defaults to 0.7.
            mp3_bitrate (MP3Bitrate): MP3 bitrate in kbps: 64, 128, or 192. Only used
                when ``output_format`` is ``"mp3"``. Defaults to 64.
            opus_bitrate (OpusBitrate): Opus bitrate in bps: -1000 (auto), 24000, 32000,
                48000, or 64000. Only used when ``output_format`` is ``"opus"``.
                Defaults to 64000.
            normalize (bool): Whether Fish normalizes the input text (numbers, dates,
                abbreviations) before synthesis. Defaults to True.
            normalize_loudness (NotGivenOr[bool]): Whether Fish normalizes the output
                loudness for more consistent volume (Fish ``prosody.normalize_loudness``).
                S2-Pro family only: on the ``s1`` model the option is dropped with a
                warning. Unset uses Fish's server default (True).
            max_new_tokens (NotGivenOr[int]): Maximum audio tokens Fish generates per
                text chunk. Unset uses Fish's server default (1024).
            min_chunk_length (NotGivenOr[int]): Minimum characters before Fish splits
                text into a new chunk (0-100). Unset uses Fish's server default (50).
            condition_on_previous_chunks (NotGivenOr[bool]): Whether Fish uses previous
                audio as context for voice consistency across chunks. Unset uses Fish's
                server default (True).
            early_stop_threshold (NotGivenOr[float]): Early stopping threshold for batch
                processing (0-1). Unset uses Fish's server default (1.0).
            http_session (http.HttpSession | None): Optional HTTP session.
        """
        if is_given(sample_rate):
            if output_format == "opus" and sample_rate != 48000:
                raise ValueError(
                    "Fish Audio only supports 48000 Hz for opus output; "
                    f"got sample_rate={sample_rate}"
                )
            resolved_sample_rate = sample_rate
        else:
            resolved_sample_rate = _DEFAULT_SAMPLE_RATE[output_format]

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=resolved_sample_rate,
            num_channels=NUM_CHANNELS,
        )

        fish_api_key = api_key if is_given(api_key) else os.getenv("FISH_API_KEY")
        if not fish_api_key:
            raise ValueError(
                "Fish Audio API key is required, either as argument or set "
                "FISH_API_KEY environment variable"
            )

        if not 100 <= chunk_length <= 300:
            raise ValueError("chunk_length must be between 100 and 300")
        if not 0 <= temperature <= 1:
            raise ValueError("temperature must be between 0 and 1")
        if not 0 <= top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        if is_given(max_new_tokens) and max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if is_given(min_chunk_length) and not 0 <= min_chunk_length <= 100:
            raise ValueError("min_chunk_length must be between 0 and 100")
        if is_given(early_stop_threshold) and not 0 <= early_stop_threshold <= 1:
            raise ValueError("early_stop_threshold must be between 0 and 1")
        if is_given(normalize_loudness) and model == "s1":
            logger.warning("normalize_loudness is not supported by the s1 model, dropping")
            normalize_loudness = NOT_GIVEN

        self._opts = _TTSOptions(
            model=model,
            output_format=output_format,
            sample_rate=resolved_sample_rate,
            voice_id=voice_id,
            base_url=base_url if is_given(base_url) else DEFAULT_BASE_URL,
            api_key=fish_api_key,
            latency_mode=latency_mode,
            chunk_length=chunk_length,
            speed=speed,
            volume=volume,
            temperature=temperature,
            top_p=top_p,
            mp3_bitrate=mp3_bitrate,
            opus_bitrate=opus_bitrate,
            normalize=normalize,
            normalize_loudness=normalize_loudness,
            max_new_tokens=max_new_tokens,
            min_chunk_length=min_chunk_length,
            condition_on_previous_chunks=condition_on_previous_chunks,
            early_stop_threshold=early_stop_threshold,
        )

        self._mux = _Mux(self._opts)
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def model(self) -> TTSModels | str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "fishaudio"

    @property
    def output_format(self) -> OutputFormat:
        return self._opts.output_format

    @property
    def voice_id(self) -> NotGivenOr[str]:
        return self._opts.voice_id

    @property
    def latency_mode(self) -> LatencyMode:
        return self._opts.latency_mode

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()
        return self._session

    def prewarm(self) -> None:
        self._mux.prewarm()

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        latency_mode: NotGivenOr[LatencyMode] = NOT_GIVEN,
        chunk_length: NotGivenOr[int] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        volume: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        mp3_bitrate: NotGivenOr[MP3Bitrate] = NOT_GIVEN,
        opus_bitrate: NotGivenOr[OpusBitrate] = NOT_GIVEN,
        normalize: NotGivenOr[bool] = NOT_GIVEN,
        normalize_loudness: NotGivenOr[bool] = NOT_GIVEN,
        max_new_tokens: NotGivenOr[int] = NOT_GIVEN,
        min_chunk_length: NotGivenOr[int] = NOT_GIVEN,
        condition_on_previous_chunks: NotGivenOr[bool] = NOT_GIVEN,
        early_stop_threshold: NotGivenOr[float] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """Change the options of the next synthesis; an invalid option is refused and changes
        nothing."""
        with self._updating_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing):
            if is_given(chunk_length) and not 100 <= chunk_length <= 300:
                raise ValueError("chunk_length must be between 100 and 300")
            if is_given(temperature) and not 0 <= temperature <= 1:
                raise ValueError("temperature must be between 0 and 1")
            if is_given(top_p) and not 0 <= top_p <= 1:
                raise ValueError("top_p must be between 0 and 1")
            if is_given(max_new_tokens) and max_new_tokens < 0:
                raise ValueError("max_new_tokens must be non-negative")
            if is_given(min_chunk_length) and not 0 <= min_chunk_length <= 100:
                raise ValueError("min_chunk_length must be between 0 and 100")
            if is_given(early_stop_threshold) and not 0 <= early_stop_threshold <= 1:
                raise ValueError("early_stop_threshold must be between 0 and 1")
        if is_given(model) and model != self._opts.model:
            self._opts.model = model
            # the model is a handshake header, not in the per-request body: a pooled
            # socket keeps the old one; other options ride in each request
            self._mux.invalidate()
            if model == "s1" and is_given(self._opts.normalize_loudness):
                logger.warning("normalize_loudness is not supported by the s1 model, dropping")
                self._opts.normalize_loudness = NOT_GIVEN
        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(latency_mode):
            self._opts.latency_mode = latency_mode
        if is_given(chunk_length):
            self._opts.chunk_length = chunk_length
        if is_given(speed):
            self._opts.speed = speed
        if is_given(volume):
            self._opts.volume = volume
        if is_given(temperature):
            self._opts.temperature = temperature
        if is_given(top_p):
            self._opts.top_p = top_p
        if is_given(mp3_bitrate):
            self._opts.mp3_bitrate = mp3_bitrate
        if is_given(opus_bitrate):
            self._opts.opus_bitrate = opus_bitrate
        if is_given(normalize):
            self._opts.normalize = normalize
        if is_given(normalize_loudness):
            if self._opts.model == "s1":
                logger.warning("normalize_loudness is not supported by the s1 model, dropping")
            else:
                self._opts.normalize_loudness = normalize_loudness
        if is_given(max_new_tokens):
            self._opts.max_new_tokens = max_new_tokens
        if is_given(min_chunk_length):
            self._opts.min_chunk_length = min_chunk_length
        if is_given(condition_on_previous_chunks):
            self._opts.condition_on_previous_chunks = condition_on_previous_chunks
        if is_given(early_stop_threshold):
            self._opts.early_stop_threshold = early_stop_threshold

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> ChunkedStream:
        return ChunkedStream(
            get_session=self._ensure_session,
            opts=replace(self._opts),
            tts=self,
            input_text=chunk.text,
            conn_options=conn_options,
        )

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        stream = SynthesizeStream(opts=replace(self._opts), tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._mux.aclose()


def _pack(message: _Request | _Start | _Text | _Control) -> bytes:
    # msgpack is untyped; packb returns the packed bytes unless autoreset is off
    packed = msgpack.packb(message, use_bin_type=True)
    if not isinstance(packed, bytes):
        raise TypeError(f"msgpack.packb returned {type(packed).__name__}")
    return packed


def _decode(frame: bytes | str) -> _Message | None:
    """A received frame, or None for an event the stream does not read (a log)."""
    if isinstance(frame, str):
        return None
    try:
        value = msgpack.unpackb(frame, raw=False)
    except ValueError as e:
        raise APIError("Fish Audio sent a malformed frame", kind="protocol", detail=str(e)) from e
    event = value.get("event") if isinstance(value, dict) else None
    if event not in ("audio", "finish"):
        # an event the stream does not read (a log, or one Fish added since): named once
        tts.log_skipped("Fish Audio", str(event))
        return None
    try:
        return _MESSAGE.validate_python(value)
    except ValidationError as e:
        raise APIError(
            "Fish Audio sent a message that breaks its protocol", kind="protocol", detail=str(e)
        ) from e


def _build_tts_request(opts: _TTSOptions, *, text: str = "") -> _Request:
    # Send the same field set the upstream Fish Audio Python SDK sends so the
    # server doesn't fall back to its own (larger) defaults — in particular the
    # docs default of `chunk_length=300` produces large bursts that leave audible
    # gaps between Fish's chunk boundaries.
    # `prosody` stays None unless the caller set speed/volume/normalize_loudness,
    # so the default request is byte-for-byte unchanged.
    prosody: _Prosody | None = None
    if is_given(opts.speed) or is_given(opts.volume) or is_given(opts.normalize_loudness):
        prosody = _Prosody()
        if is_given(opts.speed):
            prosody["speed"] = opts.speed
        if is_given(opts.volume):
            prosody["volume"] = opts.volume
        if is_given(opts.normalize_loudness):
            prosody["normalize_loudness"] = opts.normalize_loudness
    request = _Request(
        text=text,
        chunk_length=opts.chunk_length,
        format=opts.output_format,
        sample_rate=opts.sample_rate,
        mp3_bitrate=opts.mp3_bitrate,
        opus_bitrate=opts.opus_bitrate,
        references=[],
        # Fish Audio's wire field is `reference_id`; we expose it as `voice_id` on
        # the plugin for consistency with other TTS plugins.
        reference_id=opts.voice_id if is_given(opts.voice_id) else None,
        normalize=opts.normalize,
        latency=opts.latency_mode,
        prosody=prosody,
        top_p=opts.top_p,
        temperature=opts.temperature,
        # Server-side reliability feature, deliberately not exposed as an option:
        # every synthesis request runs with the quality guard enabled.
        features=["quality-guard"],
    )
    # Generation-tuning fields have no plugin default: they are omitted unless
    # set so Fish's server-side defaults stay in effect.
    if is_given(opts.max_new_tokens):
        request["max_new_tokens"] = opts.max_new_tokens
    if is_given(opts.min_chunk_length):
        request["min_chunk_length"] = opts.min_chunk_length
    if is_given(opts.condition_on_previous_chunks):
        request["condition_on_previous_chunks"] = opts.condition_on_previous_chunks
    if is_given(opts.early_stop_threshold):
        request["early_stop_threshold"] = opts.early_stop_threshold
    return request


class ChunkedStream(tts.ChunkedStream):
    """Synthesize via the Fish Audio HTTP /v1/tts endpoint."""

    def __init__(
        self,
        *,
        get_session: Callable[[], http.HttpSession],
        opts: _TTSOptions,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = opts
        self._get_session = get_session

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        payload = _build_tts_request(self._opts, text=self._input_text)

        try:
            async with self._get_session().post(
                self._opts.get_http_url("/v1/tts"),
                headers={
                    "Authorization": f"Bearer {self._opts.api_key}",
                    "Content-Type": "application/msgpack",
                    "model": self._opts.model,
                },
                data=_pack(payload),
                conn_options=self._conn_options,
            ) as resp:
                await resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type=f"audio/{self._opts.output_format}",
                )

                async for data in resp.iter_bytes():
                    output_emitter.push(data)

                output_emitter.flush()

        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


class _Mux(tts.WebSocketMultiplexer[_Message, bytes]):
    """A pool of live sockets, each carrying one stream at a time; the model is a handshake
    header."""

    routed = False
    max_session_duration = 300

    def __init__(self, opts: _TTSOptions) -> None:
        self._opts = opts

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        return tts.WebSocketTarget(
            self._opts.get_ws_url("/v1/tts/live"),
            headers={
                "Authorization": f"Bearer {self._opts.api_key}",
                "User-Agent": USER_AGENT,
                "model": self._opts.model,
            },
        )

    def decode(self, frame: bytes | str) -> _Message | None:
        return _decode(frame)


class SynthesizeStream(tts.WebSocketStream[_Message, bytes]):
    """Each segment is one Fish session, its text flushed chunk by chunk so Fish synthesizes at
    chunk boundaries instead of waiting for ``chunk_length`` characters."""

    connection_per_segment = True

    def __init__(self, *, opts: _TTSOptions, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self.multiplexer = tts._mux
        self.mime_type = f"audio/{opts.output_format}"
        self._opts = opts
        self._segment_id = ""

    def decode(self, frame: bytes | str) -> _Message | None:
        return _decode(frame)

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> bytes | None:
        match item:
            case tts.SegmentStart():
                self._segment_id = item.segment_id
                return _pack(_Start(event="start", request=_build_tts_request(self._opts)))
            case tts.SegmentText():
                self.send_now(_pack(_Text(event="text", text=f"{item.chunk.text} ")))
                return _pack(_Control(event="flush"))
            case tts.SegmentEnd():
                return _pack(_Control(event="stop"))
            case tts.EndOfText():
                return None
            case _:
                assert_never(item)

    def on_message(self, message: _Message) -> None:
        if message["event"] == "audio":
            if message["audio"]:
                self.report(tts.SegmentAudio(self._segment_id, message["audio"]))
        elif message["event"] == "finish":
            if message["reason"] == "error":
                raise APIError(
                    "Fish Audio TTS reported an error", kind="invalid_request", body=message
                )
            self.report(tts.SegmentDone(self._segment_id))
        else:
            self._skip_unknown_message(message)
