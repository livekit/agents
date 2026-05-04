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
from .models import LatencyMode, OutputFormat, TTSModels
from .version import __version__

DEFAULT_MODEL: TTSModels = "s2-pro"
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
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Fish Audio TTS.

        See https://docs.fish.audio/api-reference/endpoint/websocket/tts-live for more details
        on the Fish Audio Live TTS WebSocket API.

        Args:
            api_key (NotGivenOr[str]): Fish Audio API key. Reads ``FISH_API_KEY`` if unset.
            model (TTSModels | str): TTS model to use. Defaults to ``"s2-pro"``.
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

        self._opts = _TTSOptions(
            model=model,
            output_format=output_format,
            sample_rate=resolved_sample_rate,
            voice_id=voice_id,
            base_url=base_url if is_given(base_url) else DEFAULT_BASE_URL,
            api_key=fish_api_key,
            latency_mode=latency_mode,
            chunk_length=chunk_length,
        )

        self._session = http_session
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

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        latency_mode: NotGivenOr[LatencyMode] = NOT_GIVEN,
        chunk_length: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(latency_mode):
            self._opts.latency_mode = latency_mode
        if is_given(chunk_length):
            if not 100 <= chunk_length <= 300:
                raise ValueError("chunk_length must be between 100 and 300")
            self._opts.chunk_length = chunk_length

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


def _build_tts_request(opts: _TTSOptions, *, text: str = "") -> dict[str, Any]:
    # Send the same field set the upstream Fish Audio Python SDK sends so the
    # server doesn't fall back to its own (larger) defaults — in particular the
    # docs default of `chunk_length=300` produces large bursts that leave audible
    # gaps between Fish's chunk boundaries.
    return {
        "text": text,
        "chunk_length": opts.chunk_length,
        "format": opts.output_format,
        "sample_rate": opts.sample_rate,
        "mp3_bitrate": 64,
        "opus_bitrate": 64000,
        "references": [],
        # Fish Audio's wire field is `reference_id`; we expose it as `voice_id` on
        # the plugin for consistency with other TTS plugins.
        "reference_id": opts.voice_id if is_given(opts.voice_id) else None,
        "normalize": True,
        "latency": opts.latency_mode,
        "prosody": None,
        "top_p": 0.7,
        "temperature": 0.7,
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

        ws: aiohttp.ClientWebSocketResponse | None = None
        try:
            ws = await asyncio.wait_for(
                self._tts._ensure_session().ws_connect(
                    self._opts.get_ws_url("/v1/tts/live"),
                    headers={
                        "Authorization": f"Bearer {self._opts.api_key}",
                        "User-Agent": USER_AGENT,
                        "model": self._opts.model,
                    },
                ),
                self._conn_options.timeout,
            )

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
            if ws is not None:
                await ws.close()
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
            # can exceed `_conn_options.timeout` when the LLM is slow.
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
                    logger.debug("unknown Fish Audio event: %s", data)

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
