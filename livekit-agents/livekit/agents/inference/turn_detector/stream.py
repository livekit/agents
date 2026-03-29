from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import aiohttp
from google.protobuf.timestamp_pb2 import Timestamp

from livekit import rtc

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
from .._utils import create_access_token
from .detector import INFERENCE_TIMEOUT, TurnDetectionEvent, TurnDetectorOptions
from .proto.livekit_agent_turn_detector_pb2 import (
    TD_CHAT_ROLE_ASSISTANT,
    TD_CHAT_ROLE_USER,
    EouPrediction,
    InferenceStart,
    InferenceStop,
    InputAudio,
    InputChatContext,
    SessionCreate,
    SessionFinalize,
    SessionFlush,
    SessionSettings,
    TdChatMessage as PbChatMessage,
    TurnDetectorClientMessage,
    TurnDetectorServerMessage,
)

MAX_HISTORY_TURNS = 6

if TYPE_CHECKING:
    from .detector import MultiModalTurnDetector


def _extract_messages(chat_ctx: ChatContext) -> list[PbChatMessage]:
    """Extract only the last assistant message from the chat context."""
    messages: list[PbChatMessage] = []
    for msg in reversed(chat_ctx.messages()):
        text = " ".join([part for part in msg.content if isinstance(part, str)]).strip()
        if text:
            messages.append(
                PbChatMessage(
                    role=TD_CHAT_ROLE_ASSISTANT if msg.role == "assistant" else TD_CHAT_ROLE_USER,
                    content=text,
                )
            )
        if len(messages) == MAX_HISTORY_TURNS:
            break
    return messages


class TurnDetectionStream:
    """WebSocket-based stream that sends audio frame-by-frame."""

    class _FlushSentinel:
        pass

    def __init__(
        self,
        *,
        detector: MultiModalTurnDetector,
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

        self._messages: tuple[list[PbChatMessage], str | None] = ([], None)

        # Turn detection states:
        #
        # | state        | inference state | results admitted | audio streaming |
        # |--------------|-----------------|------------------|-------------------|
        # | warming up   | running         | no               | yes               | <- VAD silence detected
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

        self._warming_up = False
        self._active = False
        self._flushed = False
        self._active_request_id: str | None = None
        self._active_request_fut: asyncio.Future[float] | None = None
        # make sure the VAD EOS is always observed before emitting any turn detection results
        self._active_evt = asyncio.Event()

    @property
    def model(self) -> str:
        return self._detector.model

    @property
    def provider(self) -> str:
        return self._detector.provider

    @property
    def is_active(self) -> bool:
        return self._active
    
    @property
    def active_evt(self) -> asyncio.Event:
        return self._active_evt

    @property
    def is_inference_running(self) -> bool:
        return self._warming_up or self._active

    @property
    def is_admitting_results(self) -> bool:
        return self.is_active

    async def unlikely_threshold(self, language: LanguageCode | None) -> float:
        return await self._detector.unlikely_threshold(language)

    async def supports_language(self, language: LanguageCode | None) -> bool:
        return await self._detector.supports_language(language)

    async def predict_end_of_turn(
        self,
        chat_ctx: ChatContext,
        *,
        timeout: float | None = INFERENCE_TIMEOUT,
    ) -> float:
        """
        This is purely for timeout mechanism for each active inference window:
        - time to first prediction
        - time to next prediction since last prediction
        """
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
                "EOU prediction timed out",
                extra={"timeout": timeout, "request_id": self._active_request_id},
            )
            return 0.0

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

    def _build_auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": (
                f"Bearer {create_access_token(self._opts.api_key, self._opts.api_secret)}"
            )
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

    def warmup(self) -> asyncio.Future[float]:
        if not self.is_inference_running:
            self._warmup()

        if self._active_request_id is not None:
            self._send_message_sync(
                TurnDetectorClientMessage(
                    input_chat_context=InputChatContext(messages=self._messages)
                )
            )
        assert self._active_request_fut is not None
        return self._active_request_fut

    def stop_warmup(self) -> None:
        if not self.is_inference_running:
            return
        self._warming_up = False
        self._active = False
        self._flushed = False
        self._active_request_id = None
        if self._active_request_fut is not None:
            with contextlib.suppress(asyncio.InvalidStateError):
                self._active_request_fut.set_result(0.0)
            self._active_request_fut = None

    def _warmup(self) -> None:
        if self._warming_up:
            return
        self._warming_up = True
        self._active = False
        self._flushed = False
        self._active_evt.clear()
        request_id = utils.shortuuid("turn_request_")
        self._active_request_id = request_id
        self._active_request_fut = asyncio.Future[float]()
        request_id = self._active_request_id
        created_at = Timestamp()
        created_at.GetCurrentTime()
        msg = TurnDetectorClientMessage(
            inference_start=InferenceStart(request_id=request_id), created_at=created_at
        )
        self._send_message_sync(msg)

    def _activate(self) -> None:
        if self._active:
            return
        self._warming_up = False
        self._active = True
        self._flushed = False
        self._active_evt.set()

    def _deactivate(self) -> None:
        if not self._active:
            return
        self._active_request_id = None
        if self._active_request_fut is not None:
            with contextlib.suppress(asyncio.InvalidStateError):
                self._active_request_fut.set_result(0.0)
            self._active_request_fut = None
        self._active = False
        self._flushed = False
        self._warming_up = False
        self._active_evt.clear()

    def _send_message_sync(self, msg: TurnDetectorClientMessage) -> None:
        if self._ws is None or self._ws.closed:
            return
        created_at = Timestamp()
        created_at.GetCurrentTime()
        msg.created_at.CopyFrom(created_at)
        task = asyncio.create_task(self._ws.send_bytes(msg.SerializeToString()))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _send_message_async(self, msg: TurnDetectorClientMessage) -> None:
        if self._ws is None or self._ws.closed:
            return
        created_at = Timestamp()
        created_at.GetCurrentTime()
        msg.created_at.CopyFrom(created_at)
        return await self._ws.send_bytes(msg.SerializeToString())

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._audio_ch.closed:
            return
        for resampled_frame in self._resample_audio_frame(frame):
            self._audio_ch.send_nowait(resampled_frame)

    def update_chat_ctx(self, chat_ctx: ChatContext) -> None:
        """
        Update the assistant messages if needed for the active request.
        It should start when either the assistant finishes speaking or when the user starts speaking.

        Returns:
            A future that will be set to the end-of-turn probability.
        """
        self._messages = _extract_messages(chat_ctx)
        self._send_message_sync(
            TurnDetectorClientMessage(input_chat_context=InputChatContext(messages=self._messages))
        )

    def flush(self, reason: str | None = None) -> None:
        if self._audio_ch.closed:
            return
        if self._flushed:
            return

        for resampled_frame in self._flush_audio_resampler():
            self._audio_ch.send_nowait(resampled_frame)
        self._audio_ch.send_nowait(TurnDetectionStream._FlushSentinel())
        logger.trace("turn detection audio flushed", extra={"reason": reason})
        self._flushed = True
        self._deactivate()
        created_at = Timestamp()
        created_at.GetCurrentTime()
        msg = TurnDetectorClientMessage(inference_stop=InferenceStop(), created_at=created_at)
        self._send_message_sync(msg)

    def set_active(self, active: bool, trigger: str | None = None) -> None:
        """Start inference when the user stops talking."""
        ws = self._ws
        if ws is None or ws.closed:
            return

        if active:
            if not self.is_active:
                logger.trace("turn detection activated", extra={"trigger": trigger})
                self._activate()
            return

        logger.trace("turn detection deactivated", extra={"trigger": trigger})
        self._deactivate()
        created_at = Timestamp()
        created_at.GetCurrentTime()
        msg = TurnDetectorClientMessage(inference_stop=InferenceStop(), created_at=created_at)
        self._send_message_sync(msg)

    def end_input(self) -> None:
        self.flush()
        self._audio_ch.close()

    async def _run(self) -> None:
        closing_ws = False

        @utils.log_exceptions(logger=logger)
        async def send_audio_task() -> None:
            nonlocal closing_ws

            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=self._opts.sample_rate // 20,  # 50ms chunks
            )

            async for frame in self._audio_ch:
                if isinstance(frame, TurnDetectionStream._FlushSentinel):
                    for chunk in audio_bstream.flush():
                        await self._send_message_async(
                            TurnDetectorClientMessage(
                                input_audio=InputAudio(audio=chunk.data.tobytes())
                            )
                        )
                    await self._send_message_async(
                        TurnDetectorClientMessage(session_flush=SessionFlush())
                    )
                    continue

                for chunk in audio_bstream.push(frame.data):
                    await self._send_message_async(
                        TurnDetectorClientMessage(
                            input_audio=InputAudio(audio=chunk.data.tobytes())
                        )
                    )

            for chunk in audio_bstream.flush():
                await self._send_message_async(
                    TurnDetectorClientMessage(input_audio=InputAudio(audio=chunk.data.tobytes()))
                )

            closing_ws = True
            await self._send_message_async(
                TurnDetectorClientMessage(session_finalize=SessionFinalize())
            )

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

                server_msg = TurnDetectorServerMessage()
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

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        base_url = self._opts.base_url
        if base_url.startswith(("http://", "https://")):
            base_url = base_url.replace("http", "ws", 1)

        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    f"{base_url}/turn-detector",
                    headers=self._build_auth_headers(),
                ),
                self._conn_options.timeout,
            )
            await self._send_message_async(
                TurnDetectorClientMessage(
                    session_create=SessionCreate(
                        settings=SessionSettings(
                            sample_rate=self._opts.sample_rate,
                        ),
                        model=self._detector.model,
                    )
                )
            )
        except aiohttp.ClientResponseError as e:
            raise create_api_error_from_http(e.message, status=e.status) from e
        except asyncio.TimeoutError as e:
            raise APITimeoutError("turn detector connection timed out") from e
        except aiohttp.ClientConnectorError as e:
            raise APIConnectionError("failed to connect to turn detector") from e
        return ws

    def _process_message(self, msg: TurnDetectorServerMessage) -> None:
        match msg.WhichOneof("message"):
            case "eou_prediction":
                request_id = msg.request_id
                if request_id != self._active_request_id:
                    return

                prediction: EouPrediction = msg.eou_prediction
                probability = prediction.probability
                stats = msg.eou_prediction.processing_stats
                fut = self._active_request_fut
                if fut is not None:
                    with contextlib.suppress(asyncio.InvalidStateError):
                        fut.set_result(probability)

                current_time = Timestamp()
                current_time.GetCurrentTime()
                detection_delay_ms = (
                    current_time.ToMilliseconds() - stats.latest_client_created_at.ToMilliseconds()
                )
                inference_duration_ms = (
                    stats.batching_wait_duration.ToMilliseconds()
                    + stats.preprocessing_duration.ToMilliseconds()
                    + stats.inference_duration.ToMilliseconds()
                )

                logger.trace(
                    "turn prediction result received",
                    extra={
                        "probability": probability,
                        "detection_delay_ms": detection_delay_ms or 0.0,
                        "inference_duration_ms": inference_duration_ms or 0.0,
                        "request_id": request_id,
                        "queued": not self.is_admitting_results,
                    },
                )

                self._event_ch.send_nowait(
                    TurnDetectionEvent(
                        type="eou_prediction",
                        last_speaking_time=time.time(),
                        end_of_turn_probability=probability,
                    )
                )
                # Reset the future for the next prediction
                if fut is not None and probability < 0.3:
                    self._active_request_fut = asyncio.Future[float]()
            case (
                "session_created"
                | "session_finalized"
                | "session_closed"
                | "inference_started"
                | "inference_stopped"
            ):
                current_time = Timestamp()
                current_time.GetCurrentTime()
                if (
                    transport_latency := current_time.ToMilliseconds()
                    - msg.client_created_at.ToMilliseconds()
                ) > 200 and msg.client_created_at.ToMilliseconds() > 0:
                    logger.warning(
                        "turn detection transport latency is too high: %sms",
                        transport_latency,
                    )
            case "error":
                raise APIStatusError(
                    f"turn detector returned error: {msg.error.message}",
                    status_code=msg.error.code,
                    request_id=msg.request_id,
                )
            case _:
                logger.warning("unexpected turn detector message: %s", msg.WhichOneof("message"))

    async def aclose(self) -> None:
        self.end_input()
        await aio.cancel_and_wait(self._task)
        if self._active_request_fut is not None:
            with contextlib.suppress(asyncio.InvalidStateError):
                self._active_request_fut.set_result(0.0)
        self._active_request_fut = None
        self._active_request_id = None
        self._active = False
        self._flushed = False
        self._warming_up = False
        self._messages = []
