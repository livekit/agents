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

import asyncio
import base64
import contextlib
import json
import os
import time
import weakref
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.utils.aio.channel import ChanClosed

from .log import logger

WEBSOCKET_URL = "wss://tts-rt.soniox.com/tts-websocket"
NUM_CHANNELS = 1
DEFAULT_MODEL = "tts-rt-v1-preview"
DEFAULT_LANGUAGE = "en"
DEFAULT_VOICE = "Maya"
DEFAULT_AUDIO_FORMAT = "pcm_s16le"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_SPEED = 1.0
# Speaking rate bounds accepted by the Soniox TTS API (values below 1.0 slow
# speech down, above 1.0 speed it up).
MIN_SPEED = 0.7
MAX_SPEED = 1.3
KEEPALIVE_INTERVAL = 10  # seconds
KEEPALIVE_MESSAGE = json.dumps({"keep_alive": True})
# Idle stream_id rotation is opt-in (0 = disabled). Callers who hit Soniox mid-turn 408s
# during LLM stalls should set ~5.0s (under Soniox's observed ~8–18s per-stream timeout).
# See github.com/livekit/agents/issues/6225.
DEFAULT_STREAM_IDLE_TIMEOUT = 0.0


def _audio_format_to_mime_type(audio_format: str) -> str:
    if audio_format.startswith("pcm"):
        return "audio/pcm"
    if audio_format == "mp3":
        return "audio/mpeg"
    return f"audio/{audio_format}"


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
        stream_idle_timeout: float = DEFAULT_STREAM_IDLE_TIMEOUT,
        api_key: str | None = None,
        websocket_url: str = WEBSOCKET_URL,
        http_session: aiohttp.ClientSession | None = None,
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
            stream_idle_timeout (float): Seconds of silence on an open Soniox ``stream_id``
                (no new text and no ``FlushSentinel``) before sending ``text_end`` and
                starting a new ``stream_id`` when text resumes. Prevents Soniox per-stream
                408 request timeouts during LLM stalls. ``0`` disables (default); ``5.0``
                is a typical value under Soniox's observed per-stream timeout.
            api_key (str): Soniox API key. If not provided, will look for SONIOX_API_KEY env variable.
            websocket_url (str): Base WebSocket URL for Soniox TTS API.
            http_session (aiohttp.ClientSession): Optional aiohttp.ClientSession to use for requests.
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
        if stream_idle_timeout < 0:
            raise ValueError(f"stream_idle_timeout must be >= 0, but got {stream_idle_timeout}")

        self._opts = _TTSOptions(
            model=model,
            language=language,
            voice=voice,
            audio_format=audio_format,
            sample_rate=sample_rate,
            bitrate=bitrate,
            speed=speed,
            stream_idle_timeout=stream_idle_timeout,
            websocket_url=websocket_url,
            api_key=api_key,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

        # One persistent connection shared across streams (see _Connection).
        self.__current_connection: _Connection | None = None
        self.__conn_lock = asyncio.Lock()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Soniox"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _current_connection(self, *, timeout: float) -> tuple[_Connection, float, bool]:
        """Return the live connection, creating one if needed.

        Returns ``(connection, acquire_time, reused)`` — matches the
        ``ConnectionPool`` surface used by other plugins' metrics.
        """
        async with self.__conn_lock:
            conn = self.__current_connection
            if conn is not None and conn.is_current and not conn.closed:
                return conn, 0.0, True

            # Discard any stale connection reference so it can drain and close
            # itself once any outstanding streams finish.
            if conn is not None and not conn.closed:
                conn.mark_non_current()

            t0 = time.perf_counter()
            new_conn = _Connection(self._opts, self._ensure_session())
            await asyncio.wait_for(new_conn.connect(), timeout=timeout)
            self.__current_connection = new_conn
            return new_conn, time.perf_counter() - t0, False

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        stream_idle_timeout: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            model: TTS model to use.
            language: Language code to use.
            voice: Voice to use.
            speed: Speaking rate in the range [0.7, 1.3]; 1.0 is the normal rate.
            stream_idle_timeout: Idle seconds before ending the active Soniox stream_id.
                ``0`` disables idle rotation.
        """
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language
        if is_given(voice):
            self._opts.voice = voice
        if is_given(speed):
            if not MIN_SPEED <= speed <= MAX_SPEED:
                raise ValueError(
                    f"speed must be between {MIN_SPEED} and {MAX_SPEED}, but got {speed}"
                )
            self._opts.speed = speed
        if is_given(stream_idle_timeout):
            if stream_idle_timeout < 0:
                raise ValueError(f"stream_idle_timeout must be >= 0, but got {stream_idle_timeout}")
            self._opts.stream_idle_timeout = stream_idle_timeout
            for stream in list(self._streams):
                stream._opts.stream_idle_timeout = stream_idle_timeout

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        """Create a streaming TTS session."""
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        """Pre-warm the persistent connection in the background."""

        async def _task() -> None:
            try:
                await self._current_connection(timeout=20.0)
            except Exception as e:
                logger.debug(f"Soniox TTS prewarm failed: {e}")

        try:
            asyncio.create_task(_task(), name="soniox-tts-prewarm")
        except RuntimeError:
            # No running event loop (e.g. called outside async context) — skip.
            pass

    async def aclose(self) -> None:
        """Close all streams and the persistent connection."""
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()

        if self.__current_connection is not None:
            await self.__current_connection.aclose()
            self.__current_connection = None


class _StreamLifecycle:
    """Owns Soniox ``stream_id`` registration / rotation for one ``SynthesizeStream``.

    The LiveKit AudioEmitter stays on a single segment; this only rotates the Soniox
    multiplexed ``stream_id`` on idle stall or flush.
    """

    def __init__(
        self,
        *,
        stream: SynthesizeStream,
        connection: _Connection,
        output_emitter: tts.AudioEmitter,
        opts: _TTSOptions,
    ) -> None:
        self._stream = stream
        self._connection = connection
        self._output_emitter = output_emitter
        self._opts = opts
        self._loop = asyncio.get_running_loop()
        self._waiter: asyncio.Future[None] = self._loop.create_future()
        self._registered = False
        self._registered_gate = asyncio.Event()

    @property
    def registered(self) -> bool:
        return self._registered

    @property
    def waiter(self) -> asyncio.Future[None] | None:
        """Active completion future, or None between ``end`` and the next ``start``."""
        return self._waiter if self._registered else None

    def start(self) -> None:
        """Allocate and register a new Soniox stream_id."""
        self._stream._stream_id = utils.shortuuid()
        self._waiter = self._loop.create_future()
        self._connection.register_stream(
            self._stream._stream_id,
            self._output_emitter,
            self._waiter,
            opts=self._opts,
        )
        self._registered = True
        self._registered_gate.set()

    async def wait_until_registered(self) -> None:
        """Block until ``start()`` registers a stream_id (no busy-wait)."""
        while not self._registered:
            self._registered_gate.clear()
            await self._registered_gate.wait()

    async def end(self, *, send_text_end: bool) -> None:
        """Finish the current Soniox stream_id (AudioEmitter segment stays open)."""
        if not self._registered:
            return
        if send_text_end:
            self._connection.send_text(self._stream._stream_id, "", text_end=True)
        try:
            await self._waiter
        except (APIStatusError, APIConnectionError, asyncio.TimeoutError):
            raise
        except aiohttp.ClientError as e:
            raise APIConnectionError() from e
        finally:
            self.unregister()

    def unregister(self) -> None:
        if not self._registered:
            return
        self._connection.unregister_stream(self._stream._stream_id)
        self._registered = False

    def send_text(self, text: str, *, text_end: bool = False) -> None:
        self._connection.send_text(self._stream._stream_id, text, text_end=text_end)


class SynthesizeStream(tts.SynthesizeStream):
    """Streaming TTS implementation on a shared _Connection."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        self._stream_id: str = ""
        self._connection: _Connection | None = None
        self._cancelled = asyncio.Event()

    async def aclose(self) -> None:
        """Close the stream, signalling cancel to the server if still active.

        Cancelling only affects this stream's ``stream_id``; the underlying
        WebSocket stays alive for subsequent streams.
        """
        if self._cancelled.is_set():
            await super().aclose()
            return

        self._cancelled.set()
        if self._connection is not None and not self._connection.closed and self._stream_id:
            self._connection.cancel_stream(self._stream_id)
            logger.debug(
                "Sent cancellation for Soniox TTS stream",
                extra={"stream_id": self._stream_id},
            )

        await super().aclose()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Register with the connection, stream text, await segment completion.

        Ends the active Soniox ``stream_id`` (``text_end``) when input goes idle for
        ``stream_idle_timeout`` or when a ``FlushSentinel`` arrives after text was
        sent, then lazily starts a new ``stream_id`` when text resumes. This matches
        Soniox's multiplexed-stream model and avoids per-stream 408 timeouts during
        LLM stalls (see livekit/agents#6225, #6425).

        Soniox ``stream_id`` may rotate on idle/flush, but the AudioEmitter stays on
        a single segment for the life of this ``SynthesizeStream``.
        """
        request_id = utils.shortuuid()

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type=_audio_format_to_mime_type(self._opts.audio_format),
            stream=True,
        )
        output_emitter.start_segment(segment_id=utils.shortuuid())

        try:
            (
                connection,
                self._acquire_time,
                self._connection_reused,
            ) = await self._tts._current_connection(timeout=self._conn_options.timeout)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=request_id, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e

        self._connection = connection
        lifecycle = _StreamLifecycle(
            stream=self,
            connection=connection,
            output_emitter=output_emitter,
            opts=self._opts,
        )
        lifecycle.start()

        async def _input_task() -> None:
            segment_has_text = False
            need_new_stream = False

            while not self._cancelled.is_set():
                # Re-read each iteration so update_options(stream_idle_timeout=...) applies.
                idle_timeout = self._opts.stream_idle_timeout
                try:
                    if segment_has_text and idle_timeout > 0:
                        data = await asyncio.wait_for(self._input_ch.recv(), timeout=idle_timeout)
                    else:
                        data = await self._input_ch.recv()
                except asyncio.TimeoutError:
                    # Input stalled while a stream_id is open — end it before Soniox 408s.
                    await lifecycle.end(send_text_end=True)
                    segment_has_text = False
                    need_new_stream = True
                    continue
                except ChanClosed:
                    break

                if isinstance(data, self._FlushSentinel):
                    if segment_has_text:
                        await lifecycle.end(send_text_end=True)
                        segment_has_text = False
                        need_new_stream = True
                    continue

                if need_new_stream:
                    lifecycle.start()
                    need_new_stream = False

                self._mark_started()
                lifecycle.send_text(data, text_end=False)
                segment_has_text = True

            if not self._cancelled.is_set() and lifecycle.registered:
                # Input finished: end the open stream if any text was (or would be) active.
                await lifecycle.end(send_text_end=True)

        async def _monitor_server() -> None:
            """Surface server errors promptly while input is still open (pre-rotation behavior).

            On main, ``_run`` awaited the stream waiter concurrently with input. With
            stream_id rotation we only ``await`` inside ``lifecycle.end``; without this
            monitor, a mid-turn server error would sit on the Future until flush/idle/end.
            """
            while not self._cancelled.is_set():
                if input_t.done():
                    return

                waiter = lifecycle.waiter
                if waiter is None:
                    # Between end() and the next start() — park until one of them happens.
                    wait_reg = asyncio.create_task(
                        lifecycle.wait_until_registered(),
                        name="soniox-tts-wait-registered",
                    )
                    try:
                        await asyncio.wait(
                            {wait_reg, input_t},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    finally:
                        if not wait_reg.done():
                            wait_reg.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await wait_reg
                    if input_t.done():
                        return
                    continue

                done, _ = await asyncio.wait(
                    {waiter, input_t},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if input_t in done and waiter not in done:
                    return
                if waiter in done:
                    if waiter.cancelled():
                        continue
                    exc = waiter.exception()
                    if exc is not None:
                        raise exc
                    # Successful completion (text_end drained) — watch the next stream_id.
                    continue

        input_t = asyncio.create_task(_input_task(), name="soniox-tts-stream-input")
        monitor_t = asyncio.create_task(_monitor_server(), name="soniox-tts-stream-monitor")

        try:
            done, _ = await asyncio.wait(
                {input_t, monitor_t},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                if task.cancelled():
                    continue
                exc = task.exception()
                if exc is not None:
                    raise exc
        except (APIStatusError, APIConnectionError, asyncio.TimeoutError):
            raise
        except aiohttp.ClientError as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(input_t, monitor_t)
            output_emitter.end_segment()
            lifecycle.unregister()


@dataclass
class _TTSOptions:
    model: str
    language: str
    voice: str
    audio_format: str
    sample_rate: int
    bitrate: int | None
    speed: float
    stream_idle_timeout: float
    websocket_url: str
    api_key: str


@dataclass
class _StartConfig:
    stream_id: str
    opts: _TTSOptions


@dataclass
class _SendText:
    stream_id: str
    text: str
    text_end: bool


@dataclass
class _CancelStream:
    stream_id: str


_OutboundMsg = _StartConfig | _SendText | _CancelStream


@dataclass
class _StreamData:
    emitter: tts.AudioEmitter
    waiter: asyncio.Future[None]
    opts: _TTSOptions
    audio_ended: bool = False
    # Client cancel and server abort produce identical `terminated` messages.
    # This flag is how recv loop tells them apart.
    cancel_sent: bool = False
    config_sent: bool = False


class _Connection:
    """Persistent Soniox TTS WebSocket connection with multi stream support.

    The connection is independent of any single ``SynthesizeStream``: cancelling
    a stream only sends ``{"cancel": true}`` for that ``stream_id`` — the
    WebSocket stays open and remains available for subsequent streams.
    """

    def __init__(self, opts: _TTSOptions, session: aiohttp.ClientSession) -> None:
        self._opts = opts
        self._session = session
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._streams: dict[str, _StreamData] = {}
        self._input_queue = utils.aio.Chan[_OutboundMsg]()
        self._send_task: asyncio.Task[None] | None = None
        self._recv_task: asyncio.Task[None] | None = None
        self._keepalive_task: asyncio.Task[None] | None = None
        self._is_current = True
        self._closed = False

    @property
    def is_current(self) -> bool:
        return self._is_current

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def num_active_streams(self) -> int:
        return len(self._streams)

    def has_stream(self, stream_id: str) -> bool:
        return stream_id in self._streams

    def mark_non_current(self) -> None:
        """Flag this connection to be replaced; self-closes once idle."""
        self._is_current = False
        if not self._streams and not self._closed:
            asyncio.create_task(self.aclose(), name="soniox-tts-conn-drain-close")

    async def connect(self) -> None:
        """Open the WebSocket and start the send/recv/keepalive loops."""
        if self._ws is not None or self._closed:
            return

        try:
            self._ws = await self._session.ws_connect(self._opts.websocket_url)
        except Exception:
            self._closed = True
            self._is_current = False
            raise

        logger.debug("Soniox TTS WebSocket connection established!")
        self._send_task = asyncio.create_task(self._send_loop(), name="soniox-tts-send")
        self._recv_task = asyncio.create_task(self._recv_loop(), name="soniox-tts-recv")
        self._keepalive_task = asyncio.create_task(
            self._keepalive_loop(), name="soniox-tts-keepalive"
        )

    def register_stream(
        self,
        stream_id: str,
        emitter: tts.AudioEmitter,
        waiter: asyncio.Future[None],
        *,
        opts: _TTSOptions,
    ) -> None:
        """Register a new stream and queue its config message."""
        if self._closed:
            if not waiter.done():
                waiter.set_exception(APIConnectionError("Soniox TTS connection is closed"))
            return

        if stream_id in self._streams:
            raise ValueError(f"stream_id {stream_id} already registered")

        # Server starts a per-stream timeout on _StartConfig receipt; we queue it lazily in send_text.
        self._streams[stream_id] = _StreamData(emitter=emitter, waiter=waiter, opts=opts)

    def unregister_stream(self, stream_id: str) -> None:
        self._streams.pop(stream_id, None)
        # If flagged non-current and idle, self-close.
        if not self._is_current and not self._streams and not self._closed:
            asyncio.create_task(self.aclose(), name="soniox-tts-conn-drain-close")

    def send_text(self, stream_id: str, text: str, *, text_end: bool = False) -> None:
        if self._closed or stream_id not in self._streams:
            return

        stream = self._streams[stream_id]

        # Empty text would make the server hallucinate audio; handle the two cases locally.
        if not text:
            if not text_end:
                return
            if not stream.config_sent:
                # Server doesn't know about this stream; resolve locally.
                if not stream.waiter.done():
                    stream.waiter.set_result(None)
                self._streams.pop(stream_id, None)
                return

        if not stream.config_sent:
            stream.config_sent = True
            self._input_queue.send_nowait(_StartConfig(stream_id=stream_id, opts=stream.opts))
        self._input_queue.send_nowait(_SendText(stream_id=stream_id, text=text, text_end=text_end))

    def cancel_stream(self, stream_id: str) -> None:
        if self._closed:
            return
        stream = self._streams.get(stream_id)
        if stream is None:
            return
        if not stream.config_sent:
            # Server doesn't know about this stream; resolve locally.
            if not stream.waiter.done():
                stream.waiter.set_result(None)
            self._streams.pop(stream_id, None)
            return
        self._streams[stream_id].cancel_sent = True
        self._input_queue.send_nowait(_CancelStream(stream_id=stream_id))

    async def _send_loop(self) -> None:
        try:
            async for msg in self._input_queue:
                if self._ws is None or self._ws.closed:
                    break

                if isinstance(msg, _StartConfig):
                    config: dict[str, Any] = {
                        "api_key": msg.opts.api_key,
                        "model": msg.opts.model,
                        "language": msg.opts.language,
                        "voice": msg.opts.voice,
                        "audio_format": msg.opts.audio_format,
                        "sample_rate": msg.opts.sample_rate,
                        "speed": msg.opts.speed,
                        "stream_id": msg.stream_id,
                    }
                    if msg.opts.bitrate is not None:
                        config["bitrate"] = msg.opts.bitrate
                    await self._ws.send_str(json.dumps(config))
                elif isinstance(msg, _SendText):
                    payload: dict[str, Any] = {"stream_id": msg.stream_id}
                    if msg.text:
                        payload["text"] = msg.text
                    if msg.text_end:
                        payload["text_end"] = True
                    await self._ws.send_str(json.dumps(payload))
                elif isinstance(msg, _CancelStream):
                    await self._ws.send_str(
                        json.dumps({"stream_id": msg.stream_id, "cancel": True})
                    )
        except Exception as e:
            logger.warning("Soniox TTS send loop error", exc_info=e)
            self._fail_all(APIConnectionError("Soniox TTS send loop error"))
        finally:
            if not self._closed:
                asyncio.create_task(self.aclose(), name="soniox-tts-conn-fail-close")

    async def _recv_loop(self) -> None:
        try:
            while not self._closed and self._ws is not None and not self._ws.closed:
                msg = await self._ws.receive()

                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    self._fail_all(
                        APIStatusError(
                            "Soniox TTS WebSocket connection closed unexpectedly",
                            status_code=(self._ws.close_code if self._ws else -1) or -1,
                            body=f"{msg.data=} {msg.extra=}",
                        )
                    )
                    break

                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                resp = json.loads(msg.data)
                stream_id = resp.get("stream_id")

                if stream_id is None:
                    # Connection-level message (only errors are expected here).
                    if resp.get("error_code"):
                        logger.error(
                            f"Soniox TTS connection-level error: "
                            f"{resp.get('error_code')} - {resp.get('error_message')}"
                        )
                    continue

                stream = self._streams.get(stream_id)
                if stream is None:
                    logger.debug(f"Ignoring message for unknown stream {stream_id}")
                    continue

                # Per-stream error.
                if resp.get("error_code"):
                    logger.error(
                        f"Soniox TTS error: {resp.get('error_code')} - {resp.get('error_message')}",
                        extra={"stream_id": stream_id},
                    )
                    if not stream.waiter.done():
                        code: int = resp.get("error_code", 500)
                        # 408/429 are transient (timeout, rate limit); 5xx is
                        # server-side. Other 4xx (400/401/402) is config/auth
                        # and retrying won't help.
                        retryable = code in (408, 429) or code >= 500
                        stream.waiter.set_exception(
                            APIStatusError(
                                message=resp.get("error_message", "Unknown error"),
                                status_code=code,
                                body=f"stream_id={stream_id} {msg.data}",
                                retryable=retryable,
                            )
                        )
                    continue

                audio_b64 = resp.get("audio")
                if audio_b64:
                    stream.emitter.push(base64.b64decode(audio_b64))

                if resp.get("audio_end"):
                    stream.audio_ended = True
                    # end_segment() is called from SynthesizeStream._run's finally
                    # block (covers cancel/error paths too).

                if resp.get("terminated"):
                    # Don't clobber an exception already raised on the error_code path.
                    if not stream.waiter.done():
                        # Server aborted on its own - no audio_end, no cancel from us.
                        server_error = not stream.audio_ended and not stream.cancel_sent
                        if server_error:
                            # Raise so the framework retries, otherwise the
                            # turn completes with no audio (silent muteness).
                            logger.warning(
                                "Soniox TTS stream terminated without audio_end",
                                extra={"stream_id": stream_id},
                            )
                            stream.waiter.set_exception(
                                APIStatusError(
                                    message=(
                                        "Soniox TTS stream terminated without producing audio"
                                    ),
                                    body=f"stream_id={stream_id}",
                                    retryable=True,
                                )
                            )
                        else:
                            # Normal end: audio_end seen, or our own cancel landed.
                            stream.waiter.set_result(None)
                    self._streams.pop(stream_id, None)
        except Exception as e:
            logger.warning("Soniox TTS recv loop error", exc_info=e)
            self._fail_all(APIConnectionError("Soniox TTS recv loop error"))
        finally:
            if not self._closed:
                asyncio.create_task(self.aclose(), name="soniox-tts-conn-fail-close")

    async def _keepalive_loop(self) -> None:
        try:
            while not self._closed and self._ws is not None and not self._ws.closed:
                await asyncio.sleep(KEEPALIVE_INTERVAL)
                if self._ws is not None and not self._ws.closed:
                    await self._ws.send_str(KEEPALIVE_MESSAGE)
        except Exception as e:
            logger.warning(f"Soniox TTS keepalive error: {e}")

    def _fail_all(self, err: BaseException) -> None:
        """Fail all registered streams and mark the connection non-current."""
        for stream in list(self._streams.values()):
            if not stream.waiter.done():
                stream.waiter.set_exception(err)
        self._streams.clear()
        self._is_current = False

    async def aclose(self) -> None:
        """Close the connection, failing any remaining streams."""
        if self._closed:
            return
        self._closed = True
        self._is_current = False

        for stream in list(self._streams.values()):
            if not stream.waiter.done():
                stream.waiter.set_exception(APIConnectionError("Soniox TTS connection closed"))
        self._streams.clear()

        self._input_queue.close()

        if self._ws is not None and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception as e:
                logger.debug(f"Error closing Soniox TTS WebSocket: {e}")

        # Cancel all loops except the current task (avoid self-deadlock).
        current = asyncio.current_task()
        to_cancel = [
            t
            for t in (self._send_task, self._recv_task, self._keepalive_task)
            if t is not None and t is not current
        ]
        if to_cancel:
            await utils.aio.gracefully_cancel(*to_cancel)
