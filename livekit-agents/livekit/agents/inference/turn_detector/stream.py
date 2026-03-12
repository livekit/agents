from __future__ import annotations

import abc
import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import aiohttp

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
from ...llm.chat_context import ChatContext, ChatMessage
from ...log import logger
from ...types import APIConnectOptions
from ...utils import aio
from ...utils.audio import AudioFrameBuffer
from .._utils import create_access_token
from .detector import INFERENCE_INTERVAL, INFERENCE_TIMEOUT, TurnDetectionEvent, TurnDetectorOptions
from .proto.livekit_agent_turn_detector_pb2 import (
    TD_CHAT_ROLE_ASSISTANT,
    TD_CHAT_ROLE_USER,
    EouPrediction,
    InferenceStart,
    InferenceStop,
    InputAudio,
    InputChatContext,
    PredictRequest,
    SessionCreate,
    SessionFinalize,
    SessionFlush,
    SessionSettings,
    TdChatContext as PbChatContext,
    TdChatMessage as PbChatMessage,
    TurnDetectorClientMessage,
    TurnDetectorServerMessage,
)

if TYPE_CHECKING:
    from .detector import MultiModalTurnDetector


# TODO: @chenghao-mou add tests for this function
def _chat_ctx_to_proto(chat_ctx: ChatContext) -> PbChatContext:
    def _merge_trailing_user_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
        while len(messages) > 1 and messages[-2].role == messages[-1].role == "user":
            messages[-2].content.extend(messages.pop().content)
        return messages

    _ROLE_MAP = {
        "user": TD_CHAT_ROLE_USER,
        "assistant": TD_CHAT_ROLE_ASSISTANT,
    }
    # merge the trailing user messages into one user message to save on tokens
    # truncate to the last agent + user messages
    ctx_data: list[ChatMessage] = _merge_trailing_user_messages(chat_ctx.messages())[-2:]
    messages: list[PbChatMessage] = []
    for item in ctx_data:
        if (pb_role := _ROLE_MAP.get(item.role)) is None:
            continue
        if not (text := " ".join([part for part in item.content if isinstance(part, str)]).strip()):
            continue
        messages.append(PbChatMessage(role=pb_role, content=text))
    return PbChatContext(messages=messages)


class MultiModalTurnDetectionStream(abc.ABC):
    def __init__(
        self,
        *,
        detector: MultiModalTurnDetector,
        opts: TurnDetectorOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        """This class follows the _TurnDetector protocol so that it can be used as a turn detector."""
        self._detector = detector
        self._opts = opts
        self._conn_options = conn_options
        self._session = detector._ensure_session()
        self._audio_input_sample_rate: int | None = None
        self._audio_input_num_channels: int | None = None
        self._audio_resampler: rtc.AudioResampler | None = None

        self._event_ch = aio.Chan[TurnDetectionEvent]()
        self._num_retries = 0
        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())

    @property
    def model(self) -> str:
        return self._detector.model

    @property
    def provider(self) -> str:
        return self._detector.provider

    async def unlikely_threshold(self, language: LanguageCode | None) -> float | None:
        return await self._detector.unlikely_threshold(language)

    async def supports_language(self, language: LanguageCode | None) -> bool:
        return await self._detector.supports_language(language)

    @abc.abstractmethod
    def push_audio(self, frame: rtc.AudioFrame) -> None: ...

    @abc.abstractmethod
    def push_chat_ctx(self, chat_ctx: ChatContext) -> asyncio.Future[float]: ...

    @abc.abstractmethod
    def flush(self) -> None: ...

    @abc.abstractmethod
    def set_active(self, active: bool) -> None:
        """Signal the stream to start or stop producing inferences.

        active=True when the user stops speaking (VAD EOS).
        active=False when the user starts speaking (VAD SOS).
        """
        ...

    @abc.abstractmethod
    def end_input(self) -> None: ...

    async def predict_end_of_turn(
        self,
        chat_ctx: ChatContext,
        *,
        timeout: float | None = INFERENCE_TIMEOUT,
    ) -> float:
        try:
            fut = self.push_chat_ctx(chat_ctx)
            done, _ = await asyncio.wait([fut], timeout=timeout)
            if not done:
                raise asyncio.TimeoutError()
            return done.pop().result()
        except asyncio.TimeoutError:
            logger.warning("EOU prediction timed out", extra={"timeout": timeout})
            return 0.0

    async def aclose(self) -> None:
        self.end_input()
        await aio.cancel_and_wait(self._task)

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
                if max_retries == 0:
                    raise
                elif self._num_retries == max_retries:
                    raise APIConnectionError(
                        f"failed to connect livekit turn detector after {self._num_retries} attempts",
                    ) from e
                else:
                    retry_interval = self._conn_options._interval_for_retry(self._num_retries)
                    logger.warning(
                        "livekit turn detector connection failed: %s, retrying in %ss",
                        e,
                        retry_interval,
                        extra={"attempt": self._num_retries},
                    )
                    await asyncio.sleep(retry_interval)

                self._num_retries += 1

    @abc.abstractmethod
    async def _run(self) -> None: ...

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


class WSStream(MultiModalTurnDetectionStream):
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
        self._audio_ch = aio.Chan[rtc.AudioFrame | WSStream._FlushSentinel]()
        self._chat_ctx_ch = aio.Chan[tuple[ChatContext, asyncio.Future[float], str]]()
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._futures: dict[str, asyncio.Future[float]] = {}
        super().__init__(detector=detector, opts=opts, conn_options=conn_options)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._audio_ch.closed:
            return
        for resampled_frame in self._resample_audio_frame(frame):
            self._audio_ch.send_nowait(resampled_frame)

    def push_chat_ctx(self, chat_ctx: ChatContext) -> asyncio.Future[float]:
        fut = asyncio.Future[float]()
        if self._chat_ctx_ch.closed:
            fut.set_result(0.0)
            return fut
        request_id = utils.shortuuid("turn_request_")
        self._chat_ctx_ch.send_nowait((chat_ctx, fut, request_id))
        return fut

    def flush(self) -> None:
        if self._audio_ch.closed:
            return
        for resampled_frame in self._flush_audio_resampler():
            self._audio_ch.send_nowait(resampled_frame)
        self._audio_ch.send_nowait(WSStream._FlushSentinel())

    def set_active(self, active: bool) -> None:
        ws = self._ws
        if ws is None or ws.closed:
            return

        if active:
            msg = TurnDetectorClientMessage(inference_start=InferenceStart())
        else:
            msg = TurnDetectorClientMessage(inference_stop=InferenceStop())
        task = asyncio.create_task(ws.send_bytes(msg.SerializeToString()))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.remove)

    def end_input(self) -> None:
        self.flush()
        self._audio_ch.close()
        self._chat_ctx_ch.close()

    async def _run(self) -> None:
        closing_ws = False
        self._futures = {}

        @utils.log_exceptions(logger=logger)
        async def send_audio_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=self._opts.sample_rate // 20,  # 50ms chunks
            )

            async def _send_audio_frame(frame: rtc.AudioFrame) -> None:
                msg = TurnDetectorClientMessage(input_audio=InputAudio(audio=frame.data.tobytes()))
                await ws.send_bytes(msg.SerializeToString())

            async for frame in self._audio_ch:
                if isinstance(frame, WSStream._FlushSentinel):
                    for chunk in audio_bstream.flush():
                        await _send_audio_frame(chunk)
                    flush_msg = TurnDetectorClientMessage(session_flush=SessionFlush())
                    await ws.send_bytes(flush_msg.SerializeToString())
                    continue

                for chunk in audio_bstream.push(frame.data):
                    await _send_audio_frame(chunk)

            for chunk in audio_bstream.flush():
                await _send_audio_frame(chunk)

            closing_ws = True
            finalize_msg = TurnDetectorClientMessage(session_finalize=SessionFinalize())
            await ws.send_bytes(finalize_msg.SerializeToString())

        @utils.log_exceptions(logger=logger)
        async def send_chat_ctx_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            async for chat_ctx, fut, request_id in self._chat_ctx_ch:
                self._futures[request_id] = fut
                pb_ctx = _chat_ctx_to_proto(chat_ctx)
                msg = TurnDetectorClientMessage(
                    input_chat_context=InputChatContext(chat_context=pb_ctx, request_id=request_id)
                )
                await ws.send_bytes(msg.SerializeToString())

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
                asyncio.create_task(send_audio_task(ws)),
                asyncio.create_task(send_chat_ctx_task(ws)),
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
                    f"{base_url}/turn-detector?model={self._detector.model}",
                    headers=self._build_auth_headers(),
                ),
                self._conn_options.timeout,
            )
            create_msg = TurnDetectorClientMessage(
                session_create=SessionCreate(
                    settings=SessionSettings(
                        sample_rate=self._opts.sample_rate,
                    ),
                    model=self._detector.model,
                )
            )
            await ws.send_bytes(create_msg.SerializeToString())
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
                prediction: EouPrediction = msg.eou_prediction
                probability = prediction.probability
                fut = self._futures.pop(prediction.request_id, None)
                if fut is not None:
                    with contextlib.suppress(asyncio.InvalidStateError):
                        fut.set_result(probability)
                self._event_ch.send_nowait(
                    TurnDetectionEvent(
                        type="eou_prediction",
                        last_speaking_time=time.time(),
                        end_of_turn_probability=probability,
                    )
                )
            case "session_created" | "session_finalized" | "session_closed":
                pass
            case "error":
                raise APIError(f"turn detector returned error: {msg.error.message}")
            case _:
                logger.warning("unexpected turn detector message: %s", msg.WhichOneof("message"))

    async def aclose(self) -> None:
        await super().aclose()
        for fut in self._futures.values():
            with contextlib.suppress(asyncio.InvalidStateError):
                fut.set_result(0.0)
        self._futures.clear()


class HTTPStream(MultiModalTurnDetectionStream):
    """HTTP-based stream that buffers audio and sends periodic POST requests."""

    def __init__(
        self,
        *,
        detector: MultiModalTurnDetector,
        opts: TurnDetectorOptions,
        conn_options: APIConnectOptions,
        inference_interval: float = INFERENCE_INTERVAL,
        max_buffer_duration: float = 5.0,
    ) -> None:
        self._audio_buffer = AudioFrameBuffer(max_duration=max_buffer_duration)
        self._latest_chat_ctx: tuple[ChatContext, str] | None = None
        self._active = asyncio.Event()
        self._closed = False
        self._inference_interval = inference_interval
        self._futures: dict[str, asyncio.Future[float]] = {}
        super().__init__(detector=detector, opts=opts, conn_options=conn_options)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            return
        for resampled_frame in self._resample_audio_frame(frame):
            self._audio_buffer.push(resampled_frame)

    def push_chat_ctx(self, chat_ctx: ChatContext) -> asyncio.Future[float]:
        fut = asyncio.Future[float]()
        if self._closed:
            fut.set_result(0.0)
            return fut
        request_id = utils.shortuuid("turn_request_")
        self._latest_chat_ctx = (chat_ctx, request_id)
        self._futures[request_id] = fut
        return fut

    def flush(self) -> None:
        if self._closed:
            return
        self._reset_audio_resampler()
        self._audio_buffer.reset()

    def set_active(self, active: bool) -> None:
        if active:
            self._active.set()
        else:
            self._active.clear()

    def end_input(self) -> None:
        self.flush()
        self._closed = True
        self._active.set()  # unblock any waiter so the task can exit

    async def _run(self) -> None:
        base_url = self._opts.base_url
        url = f"{base_url}/turn-detector?model={self._detector.model}"
        headers = {
            **self._build_auth_headers(),
            "Content-Type": "application/x-protobuf",
        }

        while not self._closed:
            await self._active.wait()
            if self._closed:
                return
            last_speaking_time = time.time()
            await asyncio.sleep(self._inference_interval)
            if self._closed or not self._active.is_set():
                continue

            snapshot = self._audio_buffer.snapshot()
            if snapshot is None:
                continue

            if self._latest_chat_ctx is None:
                continue

            chat_ctx, request_id = self._latest_chat_ctx

            req = PredictRequest(
                model=self._detector.model,
                audio=snapshot.data.tobytes(),
                settings=SessionSettings(sample_rate=self._opts.sample_rate),
                request_id=request_id,
            )
            req.chat_context.CopyFrom(_chat_ctx_to_proto(chat_ctx))

            try:
                async with self._session.post(
                    url,
                    data=req.SerializeToString(),
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self._conn_options.timeout),
                ) as resp:
                    resp.raise_for_status()
                    body = await resp.read()
                    pb_resp = EouPrediction()
                    pb_resp.ParseFromString(body)
                    probability = pb_resp.probability
                    with contextlib.suppress(asyncio.InvalidStateError):
                        self._futures[request_id].set_result(probability)
                        self._event_ch.send_nowait(
                            TurnDetectionEvent(
                                type="eou_prediction",
                                end_of_turn_probability=probability,
                                last_speaking_time=last_speaking_time,
                            )
                        )
            except aiohttp.ClientResponseError as e:
                raise create_api_error_from_http(
                    e.message, status=e.status, request_id=request_id
                ) from e
            except asyncio.TimeoutError as e:
                raise APITimeoutError("turn detector HTTP request timed out") from e
            except aiohttp.ClientConnectorError as e:
                raise APIConnectionError("failed to connect to turn detector") from e

    async def aclose(self) -> None:
        await super().aclose()
        for fut in self._futures.values():
            with contextlib.suppress(asyncio.InvalidStateError):
                fut.set_result(0.0)
        self._futures.clear()
