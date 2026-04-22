from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

import aiohttp
from google.protobuf.timestamp_pb2 import Timestamp

from livekit import rtc
from livekit.protocol.agent_pb.agent_inference import (
    AUDIO_ENCODING_OPUS,
    ClientMessage,
    EotInputChatContext,
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
from livekit.protocol.agent_pb.agent_session import (
    ASSISTANT as CHAT_ROLE_ASSISTANT,
    USER as CHAT_ROLE_USER,
    ChatMessage as PbChatMessage,
)

from ... import utils
from ..._exceptions import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    create_api_error_from_http,
)
from ...language import LanguageCode
from ...llm.chat_context import ChatContext
from ...log import logger
from ...types import APIConnectOptions
from ...utils import aio
from ...utils.codecs import AudioStreamEncoder
from .._utils import create_access_token, get_inference_headers
from .detector import TurnDetectionEvent, TurnDetectorOptions

MAX_HISTORY_TURNS = 6
MIN_SILENCE_DURATION_MS = 200
_ENCODER_BIT_RATE = 24000

if TYPE_CHECKING:
    from .detector import MultimodalTurnDetector


def _extract_messages(chat_ctx: ChatContext) -> list[PbChatMessage]:
    """Extract the last few messages from the chat context."""
    messages: list[PbChatMessage] = []
    for msg in reversed(chat_ctx.messages()):
        text = " ".join([part for part in msg.content if isinstance(part, str)]).strip()
        if text and msg.role in ("assistant", "user"):
            messages.append(
                PbChatMessage(
                    role=CHAT_ROLE_ASSISTANT if msg.role == "assistant" else CHAT_ROLE_USER,
                    content=[PbChatMessage.ChatContent(text=text)],
                )
            )
        if len(messages) == MAX_HISTORY_TURNS:
            break
    return messages[::-1]


class _InferenceStatus(str, Enum):
    DEACTIVATED = "deactivated"
    WARMING_UP = "warming_up"
    ACTIVE = "active"
    FLUSHED = "flushed"


class TurnDetectionStream:
    @dataclass
    class _FlushSentinel:
        reason: str | None = None

    def __init__(
        self,
        *,
        detector: MultimodalTurnDetector,
        opts: TurnDetectorOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        self._detector = detector
        self._opts = opts
        self._conn_options = conn_options
        self._session = detector._ensure_session()

        self._audio_input_sample_rate: int | None = None
        self._audio_input_num_channels: int | None = None
        self._audio_resampler: rtc.AudioResampler | None = None
        self._audio_ch = aio.Chan[rtc.AudioFrame | TurnDetectionStream._FlushSentinel]()

        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._event_ch = aio.Chan[TurnDetectionEvent]()
        self._num_retries = 0

        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())
        self._tasks: set[asyncio.Task[None]] = set()

        self._messages: list[PbChatMessage] = []

        # Turn detection states:
        #
        # | state        | inference state | results admitted | audio streaming |
        # |--------------|-----------------|------------------|-------------------|
        # | warming up   | running         | delayed          | yes               | <- VAD silence detected (>200ms)
        # | active       | running         | yes              | yes               | <- user not speaking (VAD EOS)
        # | not active   | stopped         | no               | yes               | <- user speaking (VAD SOS)
        # | flushed      | stopped         | no               | cleared           | <- agent speaking started/ended

        # stateDiagram-v2
        #     state "warming up" as warming_up
        #     state "not active" as not_active
        #     warming_up --> not_active
        #     warming_up --> active
        #     active --> not_active
        #     not_active --> active
        #     active --> flushed
        #     flushed --> warming_up

        self._status: _InferenceStatus = _InferenceStatus.DEACTIVATED
        self._active_request_id: str | None = None
        self._active_request_fut: asyncio.Future[float] | None = None
        self._active_window_min_client_created_at_ms: int | None = None

    @property
    def model(self) -> str:
        return self._detector.model

    @property
    def provider(self) -> str:
        return self._detector.provider

    @property
    def is_active(self) -> bool:
        return self._status == _InferenceStatus.ACTIVE

    @property
    def is_inference_running(self) -> bool:
        return self._status in (_InferenceStatus.WARMING_UP, _InferenceStatus.ACTIVE)

    async def unlikely_threshold(
        self, language: LanguageCode | None, modality: Literal["multimodal", "text"] = "multimodal"
    ) -> float:
        return await self._detector.unlikely_threshold(language, modality)

    async def supports_language(
        self, language: LanguageCode | None, modality: Literal["multimodal", "text"] = "multimodal"
    ) -> bool:
        return await self._detector.supports_language(language, modality)

    async def predict_end_of_turn(
        self,
        chat_ctx: ChatContext,
        *,
        timeout: float | None = None,
    ) -> float:
        """
        This is purely for timeout mechanism for each active inference window:
        - time to first prediction
        - time to next prediction since last prediction
        """
        timeout = timeout if timeout is not None else 0.5
        fut: asyncio.Future[float] | None = None
        try:
            self.update_chat_ctx(chat_ctx)
            fut = self.warmup()
            self._activate()
            done, _ = await asyncio.wait([fut], timeout=timeout)
            if not done:
                raise asyncio.TimeoutError()
            return done.pop().result()
        except asyncio.TimeoutError:
            logger.warning(
                "eot prediction timed out, returning a default value",
                extra={"timeout": timeout, "request_id": self._active_request_id, "default": 1.0},
            )
            if fut is not None:
                with contextlib.suppress(asyncio.InvalidStateError):
                    fut.set_result(1.0)
            self._active_request_fut = None
            self._active_request_id = None
            self._active_window_min_client_created_at_ms = None
            # default to a positive prediction so min_endpointing_delay is used
            return 1.0

    async def __anext__(self) -> TurnDetectionEvent:
        try:
            return await self._event_ch.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled():
                exc = self._task.exception()
                if exc is not None:
                    raise exc  # noqa: B904
            raise StopAsyncIteration from None

    def __aiter__(self) -> AsyncIterator[TurnDetectionEvent]:
        return self

    async def _main_task(self) -> None:
        max_retries = self._conn_options.max_retry
        while self._num_retries <= max_retries:
            try:
                return await self._run()
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

    # region: utility functions
    def _build_auth_headers(self) -> dict[str, str]:
        return {
            **get_inference_headers(),
            "Authorization": f"Bearer {create_access_token(self._opts.api_key, self._opts.api_secret)}",
        }

    def _resample_audio_frame(self, frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
        if self._audio_input_sample_rate is None or self._audio_input_num_channels is None:
            self._audio_input_sample_rate = frame.sample_rate
            self._audio_input_num_channels = frame.num_channels
            if self._audio_input_sample_rate != self._opts.sample_rate:
                self._audio_resampler = rtc.AudioResampler(
                    input_rate=self._audio_input_sample_rate,
                    output_rate=self._opts.sample_rate,
                    num_channels=self._audio_input_num_channels,
                    quality=rtc.AudioResamplerQuality.QUICK,
                )
        elif (
            frame.sample_rate != self._audio_input_sample_rate
            or frame.num_channels != self._audio_input_num_channels
        ):
            logger.error(
                "a frame with different audio format was already pushed",
                extra={
                    "sample_rate": frame.sample_rate,
                    "expected_sample_rate": self._audio_input_sample_rate,
                    "num_channels": frame.num_channels,
                    "expected_num_channels": self._audio_input_num_channels,
                },
            )
            return []

        if self._audio_resampler is None:
            return [frame]
        return self._audio_resampler.push(frame)

    def _flush_audio_resampler(self) -> list[rtc.AudioFrame]:
        frames = self._audio_resampler.flush() if self._audio_resampler is not None else []
        self._reset_audio_resampler()
        return frames

    def _reset_audio_resampler(self) -> None:
        self._audio_resampler = None
        self._audio_input_sample_rate = None
        self._audio_input_num_channels = None

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
                        encoding=AUDIO_ENCODING_OPUS,
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
        match msg.WhichOneof("message"):
            case "eot_prediction":
                prediction: EotPrediction = msg.eot_prediction
                backend: Literal["multimodal", "text"] = {
                    EotPrediction.EotBackend.EOT_BACKEND_MULTIMODAL: "multimodal",
                    EotPrediction.EotBackend.EOT_BACKEND_TEXT: "text",
                }.get(prediction.backend, "multimodal")
                request_id = msg.request_id
                if request_id != self._active_request_id:
                    logger.trace("stale request id received: %s", request_id)
                    return

                probability = prediction.probability
                stats = prediction.processing_stats
                inference_stats = stats.inference_stats
                window_started_at_ms = self._active_window_min_client_created_at_ms
                request_sent_at_ms = stats.latest_client_created_at.ToMilliseconds()

                if window_started_at_ms is not None and request_sent_at_ms < window_started_at_ms:
                    logger.trace(
                        "ignoring stale eot prediction",
                        extra={
                            "request_id": request_id,
                            "request_sent_at_ms": request_sent_at_ms,
                            "window_started_at_ms": window_started_at_ms,
                            "probability": probability,
                            "backend": backend,
                            "stats": {
                                "e2e_latency_ms": stats.e2e_latency.ToMilliseconds(),
                                "inference_e2e_latency_ms": inference_stats.e2e_latency.ToMilliseconds(),
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
                inference_duration_ms = inference_stats.e2e_latency.ToMilliseconds()

                logger.debug(
                    "turn detection result received",
                    extra={
                        "backend": backend,
                        "probability": probability,
                        "detection_delay_ms": detection_delay_ms or 0.0,
                        "inference_duration_ms": inference_duration_ms or 0.0,
                        "request_id": request_id,
                        "queued": not self.is_active,
                    },
                )

                self._event_ch.send_nowait(
                    TurnDetectionEvent(
                        type="eot_prediction",
                        last_speaking_time=time.time(),
                        end_of_turn_probability=probability,
                        backend=backend,
                    )
                )
                self._active_request_fut = asyncio.Future[float]()
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

    # region: state management
    def warmup(self) -> asyncio.Future[float]:
        """Start running warmup inference."""
        if not self.is_inference_running:
            self._warmup()

        if self._active_request_id is not None:
            self._send_message_sync(
                ClientMessage(eot_input_chat_context=EotInputChatContext(messages=self._messages))
            )
        if self._active_request_fut is None:
            raise RuntimeError("eot detection warmup failed, no request future")
        return self._active_request_fut

    def stop_warmup(self) -> None:
        """Stop warmup inference."""
        if not self.is_inference_running:
            return
        self._status = _InferenceStatus.DEACTIVATED
        self._active_request_id = None
        self._active_window_min_client_created_at_ms = None
        if self._active_request_fut is not None:
            with contextlib.suppress(asyncio.InvalidStateError):
                self._active_request_fut.set_result(0.0)
            self._active_request_fut = None

    def _warmup(self) -> None:
        if self._status == _InferenceStatus.WARMING_UP:
            return
        self._status = _InferenceStatus.WARMING_UP
        request_id = utils.shortuuid("turn_request_")
        self._active_request_id = request_id
        self._active_request_fut = asyncio.Future[float]()

        request_id = self._active_request_id
        created_at = Timestamp()
        created_at.GetCurrentTime()
        msg = ClientMessage(
            inference_start=InferenceStart(request_id=request_id), created_at=created_at
        )
        self._send_message_sync(msg)
        self._active_window_min_client_created_at_ms = msg.created_at.ToMilliseconds()

    def _activate(self) -> None:
        if self._status == _InferenceStatus.ACTIVE:
            return

        self._status = _InferenceStatus.ACTIVE

    def _deactivate(self) -> None:
        if self._status == _InferenceStatus.DEACTIVATED:
            return
        self._active_request_id = None
        self._active_window_min_client_created_at_ms = None
        if self._active_request_fut is not None:
            with contextlib.suppress(asyncio.InvalidStateError):
                self._active_request_fut.set_result(0.0)
            self._active_request_fut = None
        self._status = _InferenceStatus.DEACTIVATED

    def flush(self, reason: str | None = None) -> None:
        if self._audio_ch.closed:
            return
        if self._status == _InferenceStatus.FLUSHED:
            return

        for resampled_frame in self._flush_audio_resampler():
            self._audio_ch.send_nowait(resampled_frame)
        self._audio_ch.send_nowait(TurnDetectionStream._FlushSentinel())
        logger.trace("turn detection audio flushed", extra={"reason": reason})
        self._deactivate()
        self._status = _InferenceStatus.FLUSHED
        created_at = Timestamp()
        created_at.GetCurrentTime()
        msg = ClientMessage(inference_stop=InferenceStop(), created_at=created_at)
        self._send_message_sync(msg)

    def set_active(self, active: bool, trigger: str | None = None) -> None:
        """Start inference when the user stops talking."""
        ws = self._ws
        if ws is None or ws.closed:
            return

        if active:
            if not self.is_active:
                if self._status != _InferenceStatus.WARMING_UP:
                    logger.warning("eot detector not warmed up before activation")
                    self.warmup()

                logger.trace(
                    "turn detection activated",
                    extra={"trigger": trigger},
                )
                self._activate()
            return

        if not self.is_active:
            return

        logger.trace("turn detection deactivated", extra={"trigger": trigger})
        self._deactivate()
        created_at = Timestamp()
        created_at.GetCurrentTime()
        msg = ClientMessage(inference_stop=InferenceStop(), created_at=created_at)
        self._send_message_sync(msg)

    # endregion

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._audio_ch.closed:
            return
        for resampled_frame in self._resample_audio_frame(frame):
            self._audio_ch.send_nowait(resampled_frame)

    def update_chat_ctx(self, chat_ctx: ChatContext) -> None:
        """Update the assistant messages for the active request."""
        self._messages = _extract_messages(chat_ctx)
        self._send_message_sync(
            ClientMessage(eot_input_chat_context=EotInputChatContext(messages=self._messages))
        )

    def end_input(self) -> None:
        self.flush()
        self._audio_ch.close()

    async def _run(self) -> None:
        closing_ws = False

        @utils.log_exceptions(logger=logger)
        async def send_audio_task() -> None:
            nonlocal closing_ws

            encoder = AudioStreamEncoder(
                codec="opus",
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                bit_rate=_ENCODER_BIT_RATE,
                codec_options={
                    "application": "lowdelay",
                    "frame_duration": "60",
                    "compression_level": "0",
                    "vbr": "on",
                },
            )

            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=self._opts.sample_rate // 50 * 3,  # 60ms chunks
            )

            async def _send_encoded(ogg_bytes: bytes, num_samples: int) -> None:
                audio_created_at = Timestamp()
                audio_created_at.GetCurrentTime()
                await self._send_message_async(
                    ClientMessage(
                        input_audio=InputAudio(
                            audio=ogg_bytes,
                            num_samples=num_samples,
                            created_at=audio_created_at,
                        )
                    )
                )

            async for frame in self._audio_ch:
                if isinstance(frame, TurnDetectionStream._FlushSentinel):
                    for chunk in audio_bstream.flush():
                        ogg_bytes, num_samples = encoder.push(chunk).as_tuple()
                        if ogg_bytes:
                            await _send_encoded(ogg_bytes, num_samples)
                    await self._send_message_async(ClientMessage(session_flush=SessionFlush()))
                    continue

                for chunk in audio_bstream.push(frame.data):
                    ogg_bytes, num_samples = encoder.push(chunk).as_tuple()
                    if ogg_bytes:
                        await _send_encoded(ogg_bytes, num_samples)

            for chunk in audio_bstream.flush():
                ogg_bytes, num_samples = encoder.push(chunk).as_tuple()
                if ogg_bytes:
                    await _send_encoded(ogg_bytes, num_samples)

            final_bytes, num_samples = encoder.close().as_tuple()
            if final_bytes:
                await _send_encoded(final_bytes, num_samples)

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

    async def aclose(self) -> None:
        self.end_input()
        await aio.cancel_and_wait(self._task)
        await aio.cancel_and_wait(*self._tasks)
        if self._active_request_fut is not None:
            with contextlib.suppress(asyncio.InvalidStateError):
                self._active_request_fut.set_result(0.0)
        self._active_request_fut = None
        self._active_request_id = None
        self._active_window_min_client_created_at_ms = None
        self._status = _InferenceStatus.DEACTIVATED
        self._messages = []
