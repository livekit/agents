from __future__ import annotations

import asyncio
import os
import weakref
from dataclasses import dataclass, replace
from typing import Any

import aiohttp
import msgpack

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
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

    def get_http_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get_ws_url(self, path: str) -> str:
        return f"{self.base_url.replace('http', 'ws', 1)}{path}"


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
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
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
            tokenizer (tokenize.SentenceTokenizer): Sentence tokenizer used to detect
                sentence boundaries. Defaults to ``tokenize.blingfire.SentenceTokenizer()``.
            http_session (aiohttp.ClientSession | None): Optional aiohttp session.
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
        )

        self._session = http_session
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )
        # min_sentence_len=1 emits each sentence as soon as the next one starts,
        # rather than batching short sentences together — minimizes TTFB on the
        # first sentence and keeps Fish synthesizing continuously.
        self._sentence_tokenizer = (
            tokenizer
            if is_given(tokenizer)
            else tokenize.blingfire.SentenceTokenizer(min_sentence_len=1)
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def model(self) -> TTSModels | str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "FishAudio"

    @property
    def output_format(self) -> OutputFormat:
        return self._opts.output_format

    @property
    def voice_id(self) -> NotGivenOr[str]:
        return self._opts.voice_id

    @property
    def latency_mode(self) -> LatencyMode:
        return self._opts.latency_mode

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        return await asyncio.wait_for(
            session.ws_connect(
                self._opts.get_ws_url("/v1/tts/live"),
                headers={
                    "Authorization": f"Bearer {self._opts.api_key}",
                    "User-Agent": USER_AGENT,
                    "model": self._opts.model,
                },
                heartbeat=30.0,
            ),
            timeout,
        )

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def prewarm(self) -> None:
        self._pool.prewarm()

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
    ) -> None:
        if is_given(model) and model != self._opts.model:
            self._opts.model = model
            # The model is sent as a connection header at ws-handshake time, not in the
            # per-request body, so a pooled socket keeps the old model. Drop pooled
            # connections so the next stream reconnects with the new model. Other
            # options ride in the per-request body and need no reconnect.
            self._pool.invalidate()
        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(latency_mode):
            self._opts.latency_mode = latency_mode
        if is_given(chunk_length):
            if not 100 <= chunk_length <= 300:
                raise ValueError("chunk_length must be between 100 and 300")
            self._opts.chunk_length = chunk_length
        if is_given(speed):
            self._opts.speed = speed
        if is_given(volume):
            self._opts.volume = volume
        if is_given(temperature):
            if not 0 <= temperature <= 1:
                raise ValueError("temperature must be between 0 and 1")
            self._opts.temperature = temperature
        if is_given(top_p):
            if not 0 <= top_p <= 1:
                raise ValueError("top_p must be between 0 and 1")
            self._opts.top_p = top_p
        if is_given(mp3_bitrate):
            self._opts.mp3_bitrate = mp3_bitrate
        if is_given(opus_bitrate):
            self._opts.opus_bitrate = opus_bitrate
        if is_given(normalize):
            self._opts.normalize = normalize

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._pool.aclose()


def _build_tts_request(opts: _TTSOptions, *, text: str = "") -> dict[str, Any]:
    # Send the same field set the upstream Fish Audio Python SDK sends so the
    # server doesn't fall back to its own (larger) defaults — in particular the
    # docs default of `chunk_length=300` produces large bursts that leave audible
    # gaps between Fish's chunk boundaries.
    # `prosody` stays None unless the caller set speed/volume, so the default
    # request is byte-for-byte unchanged.
    prosody: dict[str, float] | None = None
    if is_given(opts.speed) or is_given(opts.volume):
        prosody = {}
        if is_given(opts.speed):
            prosody["speed"] = opts.speed
        if is_given(opts.volume):
            prosody["volume"] = opts.volume
    return {
        "text": text,
        "chunk_length": opts.chunk_length,
        "format": opts.output_format,
        "sample_rate": opts.sample_rate,
        "mp3_bitrate": opts.mp3_bitrate,
        "opus_bitrate": opts.opus_bitrate,
        "references": [],
        # Fish Audio's wire field is `reference_id`; we expose it as `voice_id` on
        # the plugin for consistency with other TTS plugins.
        "reference_id": opts.voice_id if is_given(opts.voice_id) else None,
        "normalize": opts.normalize,
        "latency": opts.latency_mode,
        "prosody": prosody,
        "top_p": opts.top_p,
        "temperature": opts.temperature,
    }


class ChunkedStream(tts.ChunkedStream):
    """Synthesize via the Fish Audio HTTP /v1/tts endpoint."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        payload = _build_tts_request(self._opts, text=self._input_text)

        try:
            async with self._tts._ensure_session().post(
                self._opts.get_http_url("/v1/tts"),
                headers={
                    "Authorization": f"Bearer {self._opts.api_key}",
                    "Content-Type": "application/msgpack",
                    "model": self._opts.model,
                },
                data=msgpack.packb(payload, use_bin_type=True),
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
            ) as resp:
                resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type=f"audio/{self._opts.output_format}",
                )

                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

                output_emitter.flush()

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Streaming TTS via the Fish Audio /v1/tts/live WebSocket endpoint."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type=f"audio/{self._opts.output_format}",
            stream=True,
        )
        output_emitter.start_segment(segment_id=request_id)

        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                await self._run_ws(ws, output_emitter)

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=request_id, body=None
            ) from None
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            output_emitter.end_segment()

    async def _run_ws(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        output_emitter: tts.AudioEmitter,
    ) -> None:
        # Tokenize incoming text by sentence and flush after each sentence so
        # Fish synthesizes immediately at sentence boundaries instead of waiting
        # for `chunk_length` characters to accumulate. The result is much smoother
        # audio: gaps line up with sentence breaks (where pauses are natural)
        # rather than mid-clause.
        sent_stream = self._tts._sentence_tokenizer.stream()

        async def input_task() -> None:
            try:
                first_token = True
                async for data in self._input_ch:
                    if isinstance(data, self._FlushSentinel):
                        sent_stream.flush()
                        continue
                    if not data:
                        continue
                    if first_token:
                        self._mark_started()
                        first_token = False
                    sent_stream.push_text(data)
            finally:
                sent_stream.end_input()

        async def send_task() -> None:
            start_msg = {"event": "start", "request": _build_tts_request(self._opts)}
            await ws.send_bytes(msgpack.packb(start_msg, use_bin_type=True))

            async for ev in sent_stream:
                sentence = ev.token
                if not sentence:
                    continue
                await ws.send_bytes(
                    msgpack.packb({"event": "text", "text": sentence + " "}, use_bin_type=True)
                )
                await ws.send_bytes(msgpack.packb({"event": "flush"}, use_bin_type=True))

            await ws.send_bytes(msgpack.packb({"event": "stop"}, use_bin_type=True))

        async def recv_task() -> None:
            # No per-receive timeout: Fish has natural inter-sentence gaps that
            # can exceed `_conn_options.timeout` when the LLM is slow. Dead
            # connections are detected by aiohttp's ws heartbeat (see ws_connect).
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Fish Audio websocket connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        request_id=None,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                if msg.type != aiohttp.WSMsgType.BINARY:
                    logger.debug("unexpected Fish Audio message type %s", msg.type)
                    continue

                data = msgpack.unpackb(msg.data, raw=False)
                event = data.get("event")
                if event == "audio":
                    audio = data.get("audio")
                    if audio:
                        output_emitter.push(audio)
                elif event == "finish":
                    reason = data.get("reason")
                    if reason == "error":
                        raise APIStatusError(
                            "Fish Audio TTS reported an error",
                            status_code=-1,
                            request_id=None,
                            body=str(data),
                        )
                    break
                else:
                    logger.debug("unknown Fish Audio event", extra={"lk.pii.data": data})

        tasks = [
            asyncio.create_task(input_task()),
            asyncio.create_task(send_task()),
            asyncio.create_task(recv_task()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await sent_stream.aclose()
            await utils.aio.gracefully_cancel(*tasks)
