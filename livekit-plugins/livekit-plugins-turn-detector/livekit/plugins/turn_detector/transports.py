"""Audio EOT transports: protocol + cloud (WS) + local (ctypes).

The native ``_audio_eot`` library is loaded at module import so its weight
pages are inherited via COW by forked job processes.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import sys
import time
import warnings
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import aiohttp
from google.protobuf.timestamp_pb2 import Timestamp

from livekit import rtc
from livekit.agents import utils
from livekit.agents._exceptions import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    create_api_error_from_http,
)
from livekit.agents.inference._utils import create_access_token, get_inference_headers
from livekit.agents.metrics import AudioEOTMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import APIConnectOptions
from livekit.agents.utils import aio
from livekit.agents.voice.turn import (
    DEFAULT_SAMPLE_RATE,
    TurnDetectorOptions,
    _AudioTurnDetectorStream,
)
from livekit.protocol.agent_pb.agent_inference import (
    AUDIO_ENCODING_PCM_S16LE,
    ClientMessage,
    EotPrediction,
    InferenceStart,
    InferenceStop,
    InputAudio,
    ServerMessage,
    SessionClose,
    SessionCreate,
    SessionFlush,
    SessionSettings,
)

from .log import logger

if TYPE_CHECKING:
    from .audio import AudioTurnDetector


__all__ = [
    "_AudioTurnDetectionTransport",
    "_CloudTransport",
    "_LocalTransport",
]


@runtime_checkable
class _AudioTurnDetectionTransport(Protocol):
    """Transport bound to an `_AudioTurnDetectorStream` via `bind`.
    Implementations call back through `stream._emit_prediction(...)` etc."""

    def bind(self, stream: _AudioTurnDetectorStream) -> None: ...
    async def run(self) -> None: ...
    def on_warmup_start(self, request_id: str) -> None: ...
    async def on_audio_chunk(self, frame: rtc.AudioFrame) -> None: ...
    async def on_flush_sentinel(
        self, sentinel: _AudioTurnDetectorStream._FlushSentinel
    ) -> None: ...
    def on_activate(self) -> None: ...
    def on_inference_stop(self, *, reason: str | None) -> None: ...
    def close_nowait(self) -> None: ...
    def transport_ready(self) -> bool: ...


# Native lib expects up to 1.2 s of 16 kHz s16le PCM per predict.
_CLIENT_BUFFER_SECONDS = 1.2
_BYTES_PER_SECOND = DEFAULT_SAMPLE_RATE * 2
_CLIENT_BUFFER_BYTES = int(_CLIENT_BUFFER_SECONDS * _BYTES_PER_SECOND)

_LIB_BASENAME = "_audio_eot"


def _native_lib_path() -> Path | None:
    pkg_dir = Path(__file__).resolve().parent
    if sys.platform == "darwin":
        ext = ".dylib"
    elif sys.platform == "win32":
        ext = ".dll"
    else:
        ext = ".so"
    candidate = pkg_dir / f"{_LIB_BASENAME}{ext}"
    return candidate if candidate.exists() else None


_lib: ctypes.CDLL | None = None
_lib_load_error: BaseException | None = None


def _load_lib() -> None:
    # Errors are captured (not raised) so importing the package on a host
    # without the .so doesn't crash — only `_predict` callers see the failure.
    global _lib, _lib_load_error

    lib_path = _native_lib_path()
    if lib_path is None:
        _lib_load_error = RuntimeError(
            f"audio EOT native library not found next to livekit.plugins.turn_detector "
            f"(expected `{_LIB_BASENAME}.{{so,dylib,dll}}`). "
            f"Run `make fetch-audio-eot` or set LK_AUDIO_EOT_SOURCES_DIR + "
            f"LK_AUDIO_EOT_WEIGHTS_FILE and rebuild the plugin."
        )
        return

    try:
        lib = ctypes.CDLL(str(lib_path))
        lib.audio_eot_predict.argtypes = (ctypes.POINTER(ctypes.c_int16), ctypes.c_size_t)
        lib.audio_eot_predict.restype = ctypes.c_float
    except OSError as e:
        _lib_load_error = RuntimeError(
            f"failed to load audio EOT native library at {lib_path}: {e}"
        )
        return

    _lib = lib

    # Force lazy table init (FFT twiddles, mel filterbank) here so forkserver
    # children inherit them via COW instead of each paying the init cost.
    try:
        _predict(b"\x00\x00" * 16)
    except Exception as e:
        logger.warning("audio EOT lib warmup predict failed: %s", e)
        _lib_load_error = RuntimeError(f"audio EOT native library failed during warmup: {e}")
        _lib = None


def _predict(pcm_bytes: bytes) -> float:
    if _lib is None:
        raise _lib_load_error or RuntimeError("audio EOT native library not loaded")
    n = len(pcm_bytes) // 2
    buf = (ctypes.c_int16 * n).from_buffer_copy(pcm_bytes[: n * 2])
    return float(_lib.audio_eot_predict(buf, ctypes.c_size_t(n)))


def _lib_available() -> bool:
    return _lib is not None


def _get_lib_load_error() -> BaseException | None:
    return _lib_load_error


_load_lib()

if _lib is None and _lib_load_error is not None:
    # Surface the load failure at import — the deferred RuntimeError at first
    # use is otherwise easy to miss deep inside session bring-up.
    warnings.warn(
        f"livekit.plugins.turn_detector: {_lib_load_error}. "
        f"Local AudioTurnDetector will raise on first use.",
        stacklevel=2,
    )


class _CloudTransport:
    """WebSocket transport for `eot-audio`."""

    def __init__(
        self,
        *,
        detector: AudioTurnDetector,
        opts: TurnDetectorOptions,
        http_session: aiohttp.ClientSession | None,
        conn_options: APIConnectOptions,
    ) -> None:
        self._detector_ref: weakref.ref[AudioTurnDetector] = weakref.ref(detector)
        self._opts = opts
        self._conn_options = conn_options
        self._session_holder = http_session
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._num_retries = 0
        self._held_probability: float | None = None
        # FIFO outbound queue so sync-scheduled sends (e.g. inference_start)
        # aren't overtaken by awaited audio chunks. Allocated per connection.
        self._send_ch: aio.Chan[ClientMessage] | None = None
        self._stream_ref: weakref.ref[_AudioTurnDetectorStream] | None = None

    def bind(self, stream: _AudioTurnDetectorStream) -> None:
        self._stream_ref = weakref.ref(stream)

    def transport_ready(self) -> bool:
        ws = self._ws
        return ws is not None and not ws.closed

    def on_warmup_start(self, request_id: str) -> None:
        stream = self._stream_ref() if self._stream_ref is not None else None
        created_at = Timestamp()
        created_at.GetCurrentTime()
        msg = ClientMessage(
            inference_start=InferenceStart(request_id=request_id), created_at=created_at
        )
        self._send_message(msg)
        if stream is not None:
            stream._active_window_min_client_created_at_ms = msg.created_at.ToMilliseconds()

    def on_inference_stop(self, *, reason: str | None) -> None:
        created_at = Timestamp()
        created_at.GetCurrentTime()
        msg = ClientMessage(inference_stop=InferenceStop(), created_at=created_at)
        self._send_message(msg)

    def on_activate(self) -> None:
        if self._held_probability is not None:
            stream = self._stream_ref() if self._stream_ref is not None else None
            prob = self._held_probability
            self._held_probability = None
            if stream is not None:
                stream._emit_prediction(prob)

    async def on_audio_chunk(self, frame: rtc.AudioFrame) -> None:
        pcm_bytes = bytes(frame.data)
        if not pcm_bytes:
            return
        audio_created_at = Timestamp()
        audio_created_at.GetCurrentTime()
        self._send_message(
            ClientMessage(
                input_audio=InputAudio(
                    audio=pcm_bytes,
                    num_samples=frame.samples_per_channel,
                    created_at=audio_created_at,
                )
            )
        )

    async def on_flush_sentinel(self, sentinel: _AudioTurnDetectorStream._FlushSentinel) -> None:
        # TODO(audio-eot): forward sentinel.keep_tail_ms once SessionFlush
        # supports it server-side (local backend already honors it).
        self._send_message(ClientMessage(session_flush=SessionFlush()))
        self._held_probability = None

    def close_nowait(self) -> None:
        # Detach send_ch + ws ref; the actual ws.close() awaits in run()'s finally.
        if self._send_ch is not None:
            self._send_ch.close()
        self._ws = None

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session_holder is None:
            self._session_holder = utils.http_context.http_session()
        return self._session_holder

    def _build_auth_headers(self) -> dict[str, str]:
        return {
            **get_inference_headers(),
            "Authorization": f"Bearer {create_access_token(self._opts.api_key, self._opts.api_secret)}",
        }

    def _send_message(self, msg: ClientMessage) -> None:
        # Dropped silently when no active connection. Caller may pre-stamp
        # `created_at` (sender preserves it); otherwise sender stamps "now".
        ch = self._send_ch
        if ch is None or ch.closed or self._ws is None or self._ws.closed:
            return
        try:
            ch.send_nowait(msg)
        except aio.ChanClosed:
            pass

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        base_url = self._opts.base_url
        if base_url.startswith(("http://", "https://")):
            base_url = base_url.replace("http", "ws", 1)

        try:
            ws = await asyncio.wait_for(
                self._ensure_session().ws_connect(
                    f"{base_url}/eot",
                    headers=self._build_auth_headers(),
                ),
                self._conn_options.timeout,
            )
            session_create_msg = ClientMessage(
                session_create=SessionCreate(
                    settings=SessionSettings(
                        sample_rate=self._opts.sample_rate,
                        encoding=AUDIO_ENCODING_PCM_S16LE,
                    ),
                )
            )
            created_at = Timestamp()
            created_at.GetCurrentTime()
            session_create_msg.created_at.CopyFrom(created_at)
            await ws.send_bytes(session_create_msg.SerializeToString())
        except aiohttp.ClientResponseError as e:
            raise create_api_error_from_http(e.message, status=e.status) from e
        except asyncio.TimeoutError as e:
            raise APITimeoutError("turn detector connection timed out") from e
        except aiohttp.ClientConnectorError as e:
            raise APIConnectionError("failed to connect to turn detector") from e
        except Exception as e:
            raise APIConnectionError("failed to connect to turn detector") from e
        return ws

    def _process_message(self, msg: ServerMessage) -> None:
        stream = self._stream_ref() if self._stream_ref is not None else None
        if stream is None:
            return

        match msg.WhichOneof("message"):
            case "eot_prediction":
                prediction: EotPrediction = msg.eot_prediction
                request_id = msg.request_id
                if request_id != stream._active_request_id:
                    return

                probability = prediction.probability
                inference_stats = prediction.inference_stats
                window_started_at_ms = stream._active_window_min_client_created_at_ms
                request_sent_at_ms = inference_stats.latest_client_created_at.ToMilliseconds()

                if window_started_at_ms is not None and request_sent_at_ms < window_started_at_ms:
                    # Prediction belongs to a prior active-window — drop it.
                    return

                fut = stream._active_request_fut
                if fut is not None:
                    with contextlib.suppress(asyncio.InvalidStateError):
                        fut.set_result(probability)

                current_time = Timestamp()
                current_time.GetCurrentTime()
                detection_delay_ms = current_time.ToMilliseconds() - request_sent_at_ms
                inference_duration_ms = inference_stats.server_e2e_latency.ToMilliseconds()

                if stream.is_active:
                    stream._emit_prediction(
                        probability, detection_delay=detection_delay_ms / 1000.0
                    )
                else:
                    # WARMING_UP: hold until on_activate replays it.
                    self._held_probability = probability
                stream._active_request_fut = asyncio.Future[float]()

                client_e2e_ms = inference_stats.client_e2e_latency.ToMilliseconds()
                detector = self._detector_ref()
                if detector is not None:
                    detector.emit(
                        "metrics_collected",
                        AudioEOTMetrics(
                            timestamp=time.time(),
                            total_duration=client_e2e_ms / 1000.0,
                            prediction_duration=inference_duration_ms / 1000.0,
                            detection_delay=detection_delay_ms / 1000.0,
                            num_requests=1,
                            metadata=Metadata(
                                model_name=detector.model,
                                model_provider=detector.provider,
                            ),
                        ),
                    )
            case "session_created" | "session_closed" | "inference_started" | "inference_stopped":
                current_time = Timestamp()
                current_time.GetCurrentTime()
                if (
                    transport_latency := current_time.ToMilliseconds()
                    - msg.client_created_at.ToMilliseconds()
                ) > 500 and msg.client_created_at.ToMilliseconds() > 0:
                    logger.warning(
                        "turn detection transport latency is too high: %sms",
                        transport_latency,
                    )

            case "error":
                raise APIStatusError(
                    f"{msg.error.message}",
                    status_code=msg.error.code,
                    request_id=msg.request_id,
                )
            case _:
                logger.warning("unexpected turn detector message: %s", msg.WhichOneof("message"))

    async def run(self) -> None:
        max_retries = self._conn_options.max_retry
        while self._num_retries <= max_retries:
            try:
                return await self._run_transport()
            except APIError as e:
                if max_retries == 0 or not e.retryable:
                    raise

                if self._num_retries == max_retries:
                    raise APIConnectionError(
                        f"failed to connect livekit turn detector after {self._num_retries} attempts",
                    ) from e

                retry_interval = self._conn_options._interval_for_retry(self._num_retries)
                logger.warning(
                    "livekit turn detector connection failed: %s, retrying in %ss",
                    e,
                    retry_interval,
                    extra={"attempt": self._num_retries},
                )
                await asyncio.sleep(retry_interval)
                self._num_retries += 1

    async def _run_transport(self) -> None:
        stream = self._stream_ref() if self._stream_ref is not None else None
        if stream is None:
            return

        closing_ws = False
        send_ch: aio.Chan[ClientMessage] = aio.Chan()
        self._send_ch = send_ch

        @utils.log_exceptions(logger=logger)
        async def drain_audio_task() -> None:
            nonlocal closing_ws
            await stream._drain_audio_channel()
            closing_ws = True
            self._send_message(ClientMessage(session_close=SessionClose()))
            send_ch.close()

        @utils.log_exceptions(logger=logger)
        async def sender_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            async for msg in send_ch:
                if ws.closed:
                    return
                # Preserve caller-stamped `created_at` (e.g. on_warmup_start
                # caches it for the stale-prediction filter). Stamp only if unset.
                if not msg.HasField("created_at"):
                    created_at = Timestamp()
                    created_at.GetCurrentTime()
                    msg.created_at.CopyFrom(created_at)
                try:
                    await ws.send_bytes(msg.SerializeToString())
                except (ConnectionResetError, aiohttp.ClientConnectionError):
                    return

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            while True:
                ws_msg = await ws.receive()
                if ws_msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws or self._ensure_session().closed:
                        return
                    raise APIStatusError(
                        message="turn detector connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{ws_msg.data=} {ws_msg.extra=}",
                    )

                if ws_msg.type != aiohttp.WSMsgType.BINARY:
                    logger.warning("unexpected turn detector message type %s", ws_msg.type)
                    continue

                server_msg = ServerMessage()
                server_msg.ParseFromString(ws_msg.data)
                self._process_message(server_msg)

        ws: aiohttp.ClientWebSocketResponse | None = None
        try:
            ws = await self._connect_ws()
            self._ws = ws
            self._num_retries = 0
            tasks = [
                asyncio.create_task(drain_audio_task()),
                asyncio.create_task(sender_task(ws)),
                asyncio.create_task(recv_task(ws)),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                await aio.gracefully_cancel(*tasks)
        finally:
            send_ch.close()
            if self._send_ch is send_ch:
                self._send_ch = None
            self._ws = None
            if ws is not None:
                await ws.close()


class _LocalTransport:
    """In-process ctypes transport for `eot-audio-mini`."""

    def __init__(self, *, opts: TurnDetectorOptions) -> None:
        self._opts = opts
        self._buf: bytearray = bytearray()
        self._held_probability: float | None = None
        self._stream_ref: weakref.ref[_AudioTurnDetectorStream] | None = None
        self._tasks: set[asyncio.Task[Any]] = set()

    def bind(self, stream: _AudioTurnDetectorStream) -> None:
        self._stream_ref = weakref.ref(stream)

    def transport_ready(self) -> bool:
        return _lib is not None

    def on_warmup_start(self, request_id: str) -> None:
        stream = self._stream_ref() if self._stream_ref is not None else None
        if stream is None:
            return
        snapshot = bytes(self._buf)
        fut = stream._active_request_fut
        task = asyncio.create_task(self._run_predict(request_id, snapshot, fut))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_predict(
        self,
        request_id: str,
        pcm_snapshot: bytes,
        fut: asyncio.Future[float] | None,
    ) -> None:
        stream = self._stream_ref() if self._stream_ref is not None else None
        if stream is None:
            return

        prob = 0.0
        try:
            prob = await asyncio.to_thread(_predict, pcm_snapshot)
        except Exception:
            logger.exception("local audio EOT prediction failed")

        if request_id != stream._active_request_id:
            return

        if fut is not None:
            with contextlib.suppress(asyncio.InvalidStateError):
                fut.set_result(prob)

        if stream.is_active:
            stream._emit_prediction(prob)
        else:
            self._held_probability = prob

    def on_activate(self) -> None:
        if self._held_probability is None:
            return
        stream = self._stream_ref() if self._stream_ref is not None else None
        prob = self._held_probability
        self._held_probability = None
        if stream is not None:
            stream._emit_prediction(prob)

    async def on_audio_chunk(self, frame: rtc.AudioFrame) -> None:
        data = bytes(frame.data)
        if not data:
            return
        self._buf.extend(data)
        overflow = len(self._buf) - _CLIENT_BUFFER_BYTES
        if overflow > 0:
            del self._buf[:overflow]

    async def on_flush_sentinel(self, sentinel: _AudioTurnDetectorStream._FlushSentinel) -> None:
        keep_bytes = sentinel.keep_tail_ms * _BYTES_PER_SECOND // 1000
        if keep_bytes < len(self._buf):
            del self._buf[: len(self._buf) - keep_bytes]
        self._held_probability = None

    def on_inference_stop(self, *, reason: str | None) -> None:
        # In-flight predicts run to completion; `_run_predict` drops stale results.
        return

    def close_nowait(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()

    async def run(self) -> None:
        stream = self._stream_ref() if self._stream_ref is not None else None
        if stream is None:
            return
        await stream._drain_audio_channel()
