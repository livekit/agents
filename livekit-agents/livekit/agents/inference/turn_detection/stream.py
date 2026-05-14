from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING

import aiohttp
from google.protobuf.timestamp_pb2 import Timestamp

from livekit import rtc
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

from ... import utils
from ..._exceptions import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    create_api_error_from_http,
)
from ...log import logger
from ...metrics import AudioEOTMetrics
from ...metrics.base import Metadata
from ...types import APIConnectOptions
from ...utils import aio
from .._utils import create_access_token, get_inference_headers
from .base import MIN_SILENCE_DURATION_MS, BaseAudioTurnDetectionStream
from .detector import TurnDetectorOptions

if TYPE_CHECKING:
    from .detector import AudioTurnDetector


__all__ = ["AudioTurnDetectionStream", "MIN_SILENCE_DURATION_MS"]


# Turn detection states (cloud transport, identical FSM as the base):
#
# | state        | inference state | results admitted | audio streaming |
# |--------------|-----------------|------------------|-------------------|
# | warming up   | running         | delayed          | yes               | <- VAD silence detected (>200ms)
# | active       | running         | yes              | yes               | <- user not speaking (VAD EOS)
# | not active   | stopped         | no               | yes               | <- user speaking (VAD SOS)
# | flushed      | stopped         | no               | cleared           | <- agent speaking started/ended
#
# stateDiagram-v2
#     state "warming up" as warming_up
#     state "not active" as not_active
#     warming_up --> not_active
#     warming_up --> active
#     active --> not_active
#     not_active --> active
#     active --> flushed
#     flushed --> warming_up


class AudioTurnDetectionStream(BaseAudioTurnDetectionStream):
    """Cloud-backed audio EOT stream.

    Forwards audio over a websocket to the LiveKit inference gateway and
    emits predictions as they arrive.
    """

    def __init__(
        self,
        *,
        detector: AudioTurnDetector,
        opts: TurnDetectorOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        self._conn_options = conn_options
        self._session = detector._ensure_session()
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._num_retries = 0
        self._held_probability: float | None = None
        super().__init__(detector=detector, opts=opts)

    # region: transport hooks

    def _transport_ready(self) -> bool:
        ws = self._ws
        return ws is not None and not ws.closed

    def _on_warmup_start(self, request_id: str) -> None:
        created_at = Timestamp()
        created_at.GetCurrentTime()
        msg = ClientMessage(
            inference_start=InferenceStart(request_id=request_id), created_at=created_at
        )
        self._send_message_sync(msg)
        self._active_window_min_client_created_at_ms = msg.created_at.ToMilliseconds()

    def _on_inference_stop(self, *, reason: str | None) -> None:
        created_at = Timestamp()
        created_at.GetCurrentTime()
        msg = ClientMessage(inference_stop=InferenceStop(), created_at=created_at)
        self._send_message_sync(msg)

    def _on_activate(self) -> None:
        if self._held_probability is not None:
            prob = self._held_probability
            self._held_probability = None
            self._emit_prediction(prob)

    async def _on_audio_chunk(self, frame: rtc.AudioFrame) -> None:
        # `frame.data` is a memoryview over int16 little-endian PCM samples.
        pcm_bytes = bytes(frame.data)
        if not pcm_bytes:
            return
        audio_created_at = Timestamp()
        audio_created_at.GetCurrentTime()
        await self._send_message_async(
            ClientMessage(
                input_audio=InputAudio(
                    audio=pcm_bytes,
                    num_samples=frame.samples_per_channel,
                    created_at=audio_created_at,
                )
            )
        )

    async def _on_flush_sentinel(
        self, sentinel: BaseAudioTurnDetectionStream._FlushSentinel
    ) -> None:
        # TODO(audio-eot): forward sentinel.keep_tail_ms to the gateway once
        # SessionFlush gains a keep_tail_ms field, so cloud preserves the same
        # VAD-onset pre-roll as the local backend.
        await self._send_message_async(ClientMessage(session_flush=SessionFlush()))
        self._held_probability = None

    # endregion

    # region: WS plumbing

    def _build_auth_headers(self) -> dict[str, str]:
        return {
            **get_inference_headers(),
            "Authorization": f"Bearer {create_access_token(self._opts.api_key, self._opts.api_secret)}",
        }

    async def _send_message_async(self, msg: ClientMessage) -> None:
        if self._ws is None or self._ws.closed:
            return
        created_at = Timestamp()
        created_at.GetCurrentTime()
        msg.created_at.CopyFrom(created_at)
        return await self._ws.send_bytes(msg.SerializeToString())

    def _send_message_sync(self, msg: ClientMessage) -> None:
        task = asyncio.create_task(self._send_message_async(msg))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        base_url = self._opts.base_url
        if base_url.startswith(("http://", "https://")):
            base_url = base_url.replace("http", "ws", 1)

        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
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
        case = msg.WhichOneof("message")
        match case:
            case "eot_prediction":
                prediction: EotPrediction = msg.eot_prediction
                request_id = msg.request_id
                if request_id != self._active_request_id:
                    logger.debug(
                        "stale request id received",
                        extra={
                            "incoming_request_id": request_id,
                            "active_request_id": self._active_request_id,
                            "status": self._status.value,
                        },
                    )
                    return

                probability = prediction.probability
                inference_stats = prediction.inference_stats
                window_started_at_ms = self._active_window_min_client_created_at_ms
                request_sent_at_ms = inference_stats.latest_client_created_at.ToMilliseconds()

                if window_started_at_ms is not None and request_sent_at_ms < window_started_at_ms:
                    logger.debug(
                        "ignoring stale eot prediction",
                        extra={
                            "request_id": request_id,
                            "request_sent_at_ms": request_sent_at_ms,
                            "window_started_at_ms": window_started_at_ms,
                            "probability": probability,
                            "stats": {
                                "client_e2e_latency_ms": inference_stats.client_e2e_latency.ToMilliseconds(),
                                "server_e2e_latency_ms": inference_stats.server_e2e_latency.ToMilliseconds(),
                                "preprocessing_duration_ms": inference_stats.preprocessing_duration.ToMilliseconds(),
                                "inference_duration_ms": inference_stats.inference_duration.ToMilliseconds(),
                            },
                        },
                    )
                    return

                fut = self._active_request_fut
                if fut is not None:
                    with contextlib.suppress(asyncio.InvalidStateError):
                        fut.set_result(probability)

                current_time = Timestamp()
                current_time.GetCurrentTime()
                detection_delay_ms = current_time.ToMilliseconds() - request_sent_at_ms
                inference_duration_ms = inference_stats.server_e2e_latency.ToMilliseconds()

                logger.trace(
                    "turn detection result received",
                    extra={
                        "probability": probability,
                        "detection_delay_ms": detection_delay_ms or 0.0,
                        "inference_duration_ms": inference_duration_ms or 0.0,
                        "request_id": request_id,
                        "queued": not self.is_active,
                    },
                )

                if self.is_active:
                    self._emit_prediction(probability)
                else:
                    # FSM: in WARMING_UP, predictions are delayed until activation.
                    # _on_activate replays the latest held prediction.
                    self._held_probability = probability
                self._active_request_fut = asyncio.Future[float]()

                client_e2e_ms = inference_stats.client_e2e_latency.ToMilliseconds()
                self._detector.emit(
                    "metrics_collected",
                    AudioEOTMetrics(
                        timestamp=time.time(),
                        total_duration=client_e2e_ms / 1000.0,
                        prediction_duration=inference_duration_ms / 1000.0,
                        detection_delay=detection_delay_ms / 1000.0,
                        num_requests=1,
                        metadata=Metadata(
                            model_name=self._detector.model,
                            model_provider=self._detector.provider,
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

    # endregion

    # region: main task

    async def _main_task(self) -> None:
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
        closing_ws = False

        @utils.log_exceptions(logger=logger)
        async def send_audio_task() -> None:
            nonlocal closing_ws
            await self._drain_audio_channel()
            closing_ws = True
            await self._send_message_async(ClientMessage(session_close=SessionClose()))

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
                    if closing_ws or self._session.closed:
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
            tasks = [
                asyncio.create_task(send_audio_task()),
                asyncio.create_task(recv_task(ws)),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                await aio.gracefully_cancel(*tasks)
        finally:
            self._ws = None
            if ws is not None:
                await ws.close()

    # endregion
