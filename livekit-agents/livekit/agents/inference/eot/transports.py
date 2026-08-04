"""Audio EOT transports: cloud (WebSocket) + local (livekit-local-inference)."""

from __future__ import annotations

import asyncio
import time
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import aiohttp
import numpy as np
from google.protobuf.timestamp_pb2 import Timestamp

from livekit import rtc
from livekit.local_inference import EOT as _EOT
from livekit.protocol.agent_pb.agent_inference import (
    AUDIO_ENCODING_PCM_S16LE,
    ClientMessage,
    EotPrediction,
    InferenceStart,
    InputAudio,
    ServerMessage,
    SessionClose,
    SessionCreate,
    SessionFlush,
    SessionSettings,
)

from ... import utils
from ..._exceptions import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    create_api_error_from_http,
)
from ...log import logger
from ...metrics import EOTInferenceMetrics
from ...metrics.base import Metadata
from ...types import APIConnectOptions
from ...utils import aio, is_given
from .._utils import create_access_token, get_inference_headers
from .base import (
    DEFAULT_SAMPLE_RATE,
    TurnDetectorOptions,
    _BaseStreamingTurnDetectorStream,
    _StreamingTurnDetectionTransport,
)

if TYPE_CHECKING:
    from .detector import TurnDetector


__all__ = [
    "_CloudTransport",
    "_CloudTransportOptions",
    "_LocalTransport",
    "_StreamingTurnDetectionTransport",
]


@dataclass
class _CloudTransportOptions:
    """Cloud-WebSocket-specific options. Held separately from
    ``TurnDetectorOptions`` so the local transport doesn't see fields that
    don't apply to it."""

    base_url: str
    api_key: str
    api_secret: str
    conn_options: APIConnectOptions


_CLIENT_BUFFER_SECONDS = 1.2
_CLIENT_BUFFER_SAMPLES = int(_CLIENT_BUFFER_SECONDS * DEFAULT_SAMPLE_RATE)


class _CloudTransport:
    """WebSocket transport for `turn-detector-v1`."""

    def __init__(
        self,
        *,
        detector: TurnDetector,
        opts: TurnDetectorOptions,
        cloud_opts: _CloudTransportOptions,
        http_session: aiohttp.ClientSession | None,
    ) -> None:
        self._detector_ref: weakref.ref[TurnDetector] = weakref.ref(detector)
        self._opts = opts
        self._cloud_opts = cloud_opts
        self._conn_options = cloud_opts.conn_options
        self._session_holder = http_session
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._num_retries = 0

        self._send_ch: aio.Chan[ClientMessage] | None = None
        self._stream_ref: weakref.ref[_BaseStreamingTurnDetectorStream] | None = None

    def attach(self, stream: _BaseStreamingTurnDetectorStream) -> None:
        self._stream_ref = weakref.ref(stream)

    def run_inference(self, request_id: str) -> None:
        self._send_message(ClientMessage(inference_start=InferenceStart(request_id=request_id)))

    def push_frame(self, frame: rtc.AudioFrame) -> None:
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

    def flush(self) -> None:
        self._send_message(ClientMessage(session_flush=SessionFlush()))

    def detach(self) -> None:
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
            "Authorization": f"Bearer {create_access_token(self._cloud_opts.api_key, self._cloud_opts.api_secret)}",
        }

    def _send_message(self, msg: ClientMessage) -> None:
        ch = self._send_ch
        if ch is None or ch.closed or self._ws is None or self._ws.closed:
            return
        try:
            ch.send_nowait(msg)
        except aio.ChanClosed:
            pass

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        base_url = self._cloud_opts.base_url
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
            exc = create_api_error_from_http(e.message, status=e.status)
            exc.retryable = False
            raise exc from e
        except asyncio.TimeoutError as e:
            raise APITimeoutError("turn detector connection timed out", retryable=False) from e
        except Exception as e:
            raise APIConnectionError("failed to connect to turn detector", retryable=False) from e
        return ws

    def _warn_transport_latency(self, msg: ServerMessage) -> None:
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

    def _process_message(self, msg: ServerMessage) -> None:
        stream = self._stream_ref() if self._stream_ref is not None else None
        if stream is None:
            return

        match msg.WhichOneof("message"):
            case "eot_prediction":
                prediction: EotPrediction = msg.eot_prediction
                inference_stats = prediction.inference_stats
                request_sent_at_ms = inference_stats.latest_client_created_at.ToMilliseconds()
                current_time = Timestamp()
                current_time.GetCurrentTime()
                detection_delay_ms = current_time.ToMilliseconds() - request_sent_at_ms
                inference_duration_ms = inference_stats.server_e2e_latency.ToMilliseconds()

                stream._resolve_prediction(
                    msg.request_id,
                    prediction.probability,
                    detection_delay=detection_delay_ms / 1000.0,
                    inference_duration=inference_duration_ms / 1000.0,
                    backchannel_probability=prediction.backchannel_probability,
                )

                client_e2e_ms = inference_stats.client_e2e_latency.ToMilliseconds()
                detector = self._detector_ref()
                if detector is not None:
                    detector.emit(
                        "metrics_collected",
                        EOTInferenceMetrics(
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
            case "session_created":
                self._warn_transport_latency(msg)
                created = msg.session_created
                thresholds = stream._opts.thresholds
                thresholds._update_defaults(
                    dict(created.default_thresholds),
                    created.default_threshold,
                    dict(created.default_backchannel_thresholds),
                    created.default_backchannel_threshold,
                )
                logger.debug(
                    "audio turn detector initialized",
                    extra={
                        "model": thresholds.model,
                        "thresholds": thresholds.thresholds,
                        "default_threshold": thresholds.default_threshold,
                        "overrides": thresholds.overrides
                        if is_given(thresholds.overrides)
                        else None,
                    },
                )

            case "session_closed" | "inference_started" | "inference_stopped":
                self._warn_transport_latency(msg)

            case "error":
                raise APIStatusError(
                    f"{msg.error.message}",
                    status_code=msg.error.code,
                    request_id=msg.request_id,
                    retryable=False,
                )
            case _:
                logger.warning("unexpected turn detector message: %s", msg.WhichOneof("message"))

    async def run(self) -> None:
        max_retries = self._conn_options.max_retry
        while self._num_retries <= max_retries:
            try:
                return await self._run_once()
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

    async def _run_once(self) -> None:
        stream = self._stream_ref() if self._stream_ref is not None else None
        if stream is None:
            return

        closing_ws = False
        send_ch: aio.Chan[ClientMessage] = aio.Chan()
        self._send_ch = send_ch

        async def drain_audio_task() -> None:
            nonlocal closing_ws
            await stream._drain_audio_channel()
            closing_ws = True
            self._send_message(ClientMessage(session_close=SessionClose()))
            send_ch.close()

        async def sender_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            async for msg in send_ch:
                if ws.closed:
                    return
                if not msg.HasField("created_at"):
                    created_at = Timestamp()
                    created_at.GetCurrentTime()
                    msg.created_at.CopyFrom(created_at)
                try:
                    await ws.send_bytes(msg.SerializeToString())
                except (ConnectionResetError, aiohttp.ClientConnectionError):
                    return

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
                        retryable=False,
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
    """In-process ctypes transport for `turn-detector-v1-mini`."""

    def __init__(self, *, opts: TurnDetectorOptions) -> None:
        self._opts = opts
        self._buf = utils.AudioArrayBuffer(
            buffer_size=_CLIENT_BUFFER_SAMPLES, sample_rate=DEFAULT_SAMPLE_RATE
        )
        self._eot = _EOT()
        self._stream_ref: weakref.ref[_BaseStreamingTurnDetectorStream] | None = None
        self._tasks: set[asyncio.Task[Any]] = set()

    def attach(self, stream: _BaseStreamingTurnDetectorStream) -> None:
        self._stream_ref = weakref.ref(stream)

    def run_inference(self, request_id: str) -> None:
        task = asyncio.create_task(self._predict(request_id, self._buf.read()))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _predict(self, request_id: str, pcm_snapshot: np.ndarray) -> None:
        prob = 0.0
        t0 = time.monotonic()
        try:
            prob = float(await asyncio.to_thread(self._eot.predict, pcm_snapshot))
        except Exception:
            logger.exception("local audio EOT prediction failed")
        inference_duration = time.monotonic() - t0

        stream = self._stream_ref() if self._stream_ref is not None else None
        if stream is None:
            return
        stream._resolve_prediction(request_id, prob, inference_duration=inference_duration)

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        self._buf.push_frame(frame)

    def flush(self) -> None:
        if len(self._buf) > 0:
            self._buf.shift(len(self._buf))

    def detach(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()

    async def run(self) -> None:
        stream = self._stream_ref() if self._stream_ref is not None else None
        if stream is None:
            return
        await stream._drain_audio_channel()
