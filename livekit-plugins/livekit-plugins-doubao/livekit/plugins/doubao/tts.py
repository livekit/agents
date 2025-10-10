from __future__ import annotations

import asyncio
import json
import os
import uuid
import weakref
from dataclasses import dataclass, replace
from typing import Any, Union

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .protocol import EventType, Message, MsgType, MsgTypeFlagBits


# Use unidirectional stream endpoint to match official demo behavior
_DEFAULT_ENDPOINT = "wss://openspeech.bytedance.com/api/v3/tts/unidirectional/stream"
_DEFAULT_ENCODING = "wav"  # 'wav' | 'mp3' | 'pcm'
_DEFAULT_SAMPLE_RATE = 24000
WS_INACTIVITY_TIMEOUT = 10


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        app_id: NotGivenOr[str] = NOT_GIVEN,
        access_token: NotGivenOr[str] = NOT_GIVEN,
        voice_type: NotGivenOr[str] = NOT_GIVEN,
        resource_id: NotGivenOr[str] = NOT_GIVEN,
        endpoint: NotGivenOr[str] = NOT_GIVEN,
        encoding: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        inactivity_timeout: int = WS_INACTIVITY_TIMEOUT,
        auto_mode: NotGivenOr[bool] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
        mix_speakers: NotGivenOr[list[dict[str, object]]] = NOT_GIVEN,
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer | tokenize.SentenceTokenizer] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        speech_rate: float = 0.0, # [-50, 100]
        enable_timestamp: bool = False,
    ) -> None:
        if not is_given(encoding):
            encoding = _DEFAULT_ENCODING
        if not is_given(sample_rate):
            sample_rate = _DEFAULT_SAMPLE_RATE

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=False),
            sample_rate=int(sample_rate),
            num_channels=1,
        )

        app_id_val = app_id if is_given(app_id) else os.environ.get("DOUBAO_APP_ID")
        access_token_val = (
            access_token
            if is_given(access_token)
            else (
                os.environ.get("DOUBAO_ACCESS_TOKEN")
                or os.environ.get("DOUBAO_APP_ACCESS_TOKEN")
            )
        )
        if not app_id_val or not access_token_val:
            raise ValueError(
                "Doubao TTS requires app_id and access_token (DOUBAO_APP_ID/DOUBAO_ACCESS_TOKEN)"
            )

        if not is_given(auto_mode):
            auto_mode = True

        if not is_given(word_tokenizer):
            word_tokenizer = (
                tokenize.basic.WordTokenizer(ignore_punctuation=False)
                if not auto_mode
                else tokenize.blingfire.SentenceTokenizer()
            )

        if not is_given(endpoint):
            endpoint = _DEFAULT_ENDPOINT

        self._opts = _TTSOptions(
            app_id=app_id_val,
            access_token=access_token_val,
            voice_type=(
                voice_type if is_given(voice_type) else os.environ.get("DOUBAO_VOICE_TYPE", "zh_female_cancan_mars_bigtts")
            ),
            resource_id=resource_id if is_given(resource_id) else None,
            endpoint=str(endpoint),
            encoding=str(encoding),
            sample_rate=int(sample_rate),
            inactivity_timeout=inactivity_timeout,
            word_tokenizer=word_tokenizer,
            auto_mode=auto_mode,
            model=(model if is_given(model) else None),
            mix_speakers=(mix_speakers if is_given(mix_speakers) else None),
            speech_rate=speech_rate,
            enable_timestamp=enable_timestamp,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

        self._current_connection: _Connection | None = None
        self._connection_lock = asyncio.Lock()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session or getattr(self._session, "closed", False):
            self._session = utils.http_context.http_session()
        return self._session

    async def list_voices(self) -> list[str]:  # placeholder for compatibility
        return []

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        voice_settings: NotGivenOr[Any] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
        mix_speakers: NotGivenOr[list[dict[str, object]]] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        changed = False

        if is_given(voice_id):
            self._opts.voice_type = voice_id
            changed = True

        if is_given(model):
            self._opts.model = model  # type: ignore[assignment]
            changed = True

        if is_given(mix_speakers):
            self._opts.mix_speakers = mix_speakers  # type: ignore[assignment]
            changed = True

        if changed and self._current_connection:
            self._current_connection.mark_non_current()
            self._current_connection = None

    async def current_connection(self) -> _Connection:
        async with self._connection_lock:
            if (
                self._current_connection
                and self._current_connection.is_current
                and not self._current_connection._closed
            ):
                return self._current_connection

            session = self._ensure_session()
            conn = _Connection(self._opts, session)
            await conn.connect()
            self._current_connection = conn
            return conn

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()

        if self._current_connection:
            await self._current_connection.aclose()
            self._current_connection = None


class ChunkedStream(tts.ChunkedStream):
    """Synthesize by delegating to streaming API (compatibility shim)"""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        # Single-shot request using unidirectional endpoint (official demo style)
        context_id = utils.shortuuid()

        output_emitter.initialize(
            request_id=context_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            stream=False,
            mime_type=(
                "audio/pcm"
                if self._opts.encoding == "pcm"
                else ("audio/mp3" if self._opts.encoding == "mp3" else "audio/wav")
            ),
        )

        session = self._tts._ensure_session()

        headers = {
            "X-Api-App-Key": self._opts.app_id,
            "X-Api-Access-Key": self._opts.access_token,
            "X-Api-Resource-Id": (
                self._opts.resource_id if self._opts.resource_id else _get_resource_id(self._opts.voice_type)
            ),
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }

        req = {
            "user": {"uid": str(uuid.uuid4())},
            "req_params": {
                "speaker": self._opts.voice_type,
                "audio_params": {
                    "format": self._opts.encoding,
                    "sample_rate": self._opts.sample_rate,
                    "enable_timestamp": self._opts.enable_timestamp,
                    "speech_rate": self._opts.speech_rate,
                },
                "text": self._input_text,
                "additions": json.dumps({"disable_markdown_filter": False}),
            },
        }

        try:
            ws = await session.ws_connect(
                self._opts.endpoint,
                headers=headers,
                timeout=self._conn_options.timeout,
                max_msg_size=10 * 1024 * 1024,
            )
            try:
                logid = getattr(getattr(ws, "_response", None), "headers", {}).get("x-tt-logid")
                if logid:
                    logger.info(f"doubao ws connected, x-tt-logid={logid}")
            except Exception:
                pass
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except Exception as e:
            raise APIConnectionError("could not connect to Doubao") from e

        try:
            # Send single full request
            msg = Message(type=MsgType.FullClientRequest, flag=MsgTypeFlagBits.NoSeq)
            msg.payload = json.dumps(req).encode()
            await ws.send_bytes(msg.marshal())

            # Receive audio until SessionFinished
            while True:
                incoming = await ws.receive()
                if incoming.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    code = getattr(ws, "close_code", None)
                    reason = getattr(ws, "close_message", None)
                    raise APIStatusError(f"connection closed (code={code}, reason={reason})")
                if incoming.type != aiohttp.WSMsgType.BINARY:
                    continue
                packet = Message.from_bytes(incoming.data)
                if packet.type == MsgType.AudioOnlyServer and packet.payload:
                    output_emitter.push(packet.payload)
                elif packet.type == MsgType.FullServerResponse:
                    if packet.event == EventType.SessionFinished:
                        break
                    if packet.event == EventType.SessionFailed:
                        raise APIError("doubao session failed")
        finally:
            await ws.close()


class SynthesizeStream(tts.SynthesizeStream):
    """Doubao bidirectional TTS using custom protocol over WebSocket"""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        self._context_id = utils.shortuuid()
        # sentence/word tokenizer stream to chunk input
        self._sent_tokenizer_stream = self._opts.word_tokenizer.stream()
        self._text_buffer = ""
        self._connection: _Connection | None = None

    async def aclose(self) -> None:
        try:
            await self._sent_tokenizer_stream.aclose()
        except Exception:
            pass
        await super().aclose()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        # initialize stream output
        output_emitter.initialize(
            request_id=self._context_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            stream=True,
            mime_type=(
                "audio/pcm"
                if self._opts.encoding == "pcm"
                else ("audio/mp3" if self._opts.encoding == "mp3" else "audio/wav")
            ),
        )
        output_emitter.start_segment(segment_id=self._context_id)

        # Read all input text until end_input, then send a single request
        segment_text = ""
        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                # ignore flush in unidirectional mode; we send one request
                continue
            if isinstance(data, str):
                segment_text += data

        segment_text = segment_text.strip()

        if not segment_text:
            output_emitter.end_segment()
            return

        # Perform a single unidirectional streaming request, forwarding audio chunks
        session = self._tts._ensure_session()

        headers = {
            "X-Api-App-Key": self._opts.app_id,
            "X-Api-Access-Key": self._opts.access_token,
            "X-Api-Resource-Id": (
                self._opts.resource_id if self._opts.resource_id else _get_resource_id(self._opts.voice_type)
            ),
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }

        req = {
            "user": {"uid": str(uuid.uuid4())},
            "req_params": {
                "speaker": self._opts.voice_type,
                "audio_params": {
                    "format": self._opts.encoding,
                    "sample_rate": self._opts.sample_rate,
                    "enable_timestamp": self._opts.enable_timestamp,
                    "speech_rate": self._opts.speech_rate,
                },
                "text": segment_text,
                "additions": json.dumps({"disable_markdown_filter": False}),
            },
        }

        try:
            ws = await session.ws_connect(
                self._opts.endpoint,
                headers=headers,
                timeout=self._conn_options.timeout,
                max_msg_size=10 * 1024 * 1024,
            )
            try:
                logid = getattr(getattr(ws, "_response", None), "headers", {}).get("x-tt-logid")
                if logid:
                    logger.info(f"doubao ws connected, x-tt-logid={logid}")
            except Exception:
                pass
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except Exception as e:
            raise APIConnectionError("could not connect to Doubao") from e

        # Send full request and stream audio frames to emitter
        try:
            msg = Message(type=MsgType.FullClientRequest, flag=MsgTypeFlagBits.NoSeq)
            msg.payload = json.dumps(req).encode()
            await ws.send_bytes(msg.marshal())

            # notify start
            self._mark_started()

            while True:
                incoming = await ws.receive()
                if incoming.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    code = getattr(ws, "close_code", None)
                    reason = getattr(ws, "close_message", None)
                    raise APIStatusError(f"connection closed (code={code}, reason={reason})")
                if incoming.type != aiohttp.WSMsgType.BINARY:
                    continue
                packet = Message.from_bytes(incoming.data)
                if packet.type == MsgType.AudioOnlyServer and packet.payload:
                    output_emitter.push(packet.payload)
                elif packet.type == MsgType.FullServerResponse:
                    if packet.event == EventType.SessionFinished:
                        break
                    if packet.event == EventType.SessionFailed:
                        raise APIError("doubao session failed")
        finally:
            await ws.close()

        output_emitter.end_segment()


@dataclass
class _TTSOptions:
    app_id: str
    access_token: str
    voice_type: str
    resource_id: str | None
    endpoint: str
    encoding: str
    sample_rate: int
    word_tokenizer: tokenize.WordTokenizer | tokenize.SentenceTokenizer
    inactivity_timeout: int
    auto_mode: NotGivenOr[bool]
    model: str | None
    mix_speakers: list[dict[str, object]] | None
    speech_rate: float
    enable_timestamp: bool
@dataclass
class _SynthesizeContent:
    context_id: str
    text: str


@dataclass
class _CloseContext:
    context_id: str


@dataclass
class _StreamData:
    emitter: tts.AudioEmitter
    stream: SynthesizeStream
    waiter: asyncio.Future[None]
    timeout_timer: asyncio.TimerHandle | None = None


class _Connection:
    """Manages a single WebSocket connection with send/recv loops for multi-context TTS"""

    def __init__(self, opts: _TTSOptions, session: aiohttp.ClientSession):
        self._opts = opts
        self._session = session
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._is_current = True
        self._active_contexts: set[str] = set()
        self._input_queue = utils.aio.Chan[Union[_SynthesizeContent, _CloseContext]]()

        self._context_data: dict[str, _StreamData] = {}

        self._send_task: asyncio.Task | None = None
        self._recv_task: asyncio.Task | None = None
        self._closed = False
        self._connected_fut: asyncio.Future[None] | None = None

    @property
    def is_current(self) -> bool:
        return self._is_current

    def mark_non_current(self) -> None:
        self._is_current = False

    async def connect(self) -> None:
        if self._ws or self._closed:
            return

        headers = {
            "X-Api-App-Key": self._opts.app_id,
            "X-Api-Access-Key": self._opts.access_token,
            "X-Api-Resource-Id": (
                self._opts.resource_id
                if self._opts.resource_id
                else _get_resource_id(self._opts.voice_type)
            ),
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }
        self._ws = await self._session.ws_connect(
            self._opts.endpoint, headers=headers, max_msg_size=10 * 1024 * 1024
        )

        self._send_task = asyncio.create_task(self._send_loop())
        self._recv_task = asyncio.create_task(self._recv_loop())

        # StartConnection handshake
        self._connected_fut = asyncio.get_running_loop().create_future()
        start = Message(type=MsgType.FullClientRequest, flag=MsgTypeFlagBits.WithEvent)
        start.event = EventType.StartConnection
        start.payload = b"{}"
        await self._ws.send_bytes(start.marshal())
        await asyncio.wait_for(self._connected_fut, timeout=10)

    def register_stream(
        self, stream: SynthesizeStream, emitter: tts.AudioEmitter, done_fut: asyncio.Future[None]
    ) -> None:
        context_id = stream._context_id
        self._context_data[context_id] = _StreamData(
            emitter=emitter, stream=stream, waiter=done_fut
        )

    def send_content(self, content: _SynthesizeContent) -> None:
        if self._closed or not self._ws or self._ws.closed:
            raise APIConnectionError("WebSocket connection is closed")
        self._input_queue.send_nowait(content)

    def close_context(self, context_id: str) -> None:
        if self._closed or not self._ws or self._ws.closed:
            raise APIConnectionError("WebSocket connection is closed")
        self._input_queue.send_nowait(_CloseContext(context_id))

    async def _send_loop(self) -> None:
        try:
            while not self._closed:
                try:
                    msg = await self._input_queue.recv()
                except utils.aio.ChanClosed:
                    break

                if not self._ws or self._ws.closed:
                    break

                if isinstance(msg, _SynthesizeContent):
                    is_new_context = msg.context_id not in self._active_contexts

                    if not self._is_current and is_new_context:
                        continue

                    if is_new_context:
                        # Start session
                        start_req = {
                            "user": {"uid": str(uuid.uuid4())},
                            "req_params": {
                                "speaker": self._opts.voice_type,
                                "audio_params": {
                                    "format": self._opts.encoding,
                                    "sample_rate": self._opts.sample_rate,
                                    "enable_timestamp": self._opts.enable_timestamp,
                                    "speech_rate": self._opts.speech_rate,
                                },
                                "additions": json.dumps({"disable_markdown_filter": False}),
                            },
                        }
                        start = Message(type=MsgType.FullClientRequest, flag=MsgTypeFlagBits.WithEvent)
                        start.event = EventType.StartSession
                        start.session_id = msg.context_id
                        start.payload = json.dumps(start_req).encode()
                        await self._ws.send_bytes(start.marshal())
                        self._active_contexts.add(msg.context_id)

                    # Task request with text
                    task_req = {
                        "user": {"uid": str(uuid.uuid4())},
                        "req_params": {
                            "speaker": self._opts.voice_type,
                            "audio_params": {
                                "format": self._opts.encoding,
                                "sample_rate": self._opts.sample_rate,
                                "enable_timestamp": self._opts.enable_timestamp,
                                "speech_rate": self._opts.speech_rate,
                            },
                            "text": msg.text,
                            "additions": json.dumps({"disable_markdown_filter": False}),
                        },
                    }
                    task = Message(type=MsgType.FullClientRequest, flag=MsgTypeFlagBits.WithEvent)
                    task.event = EventType.TaskRequest
                    task.session_id = msg.context_id
                    task.payload = json.dumps(task_req).encode()

                    self._start_timeout_timer(msg.context_id)
                    await self._ws.send_bytes(task.marshal())

                elif isinstance(msg, _CloseContext):
                    if msg.context_id in self._active_contexts:
                        finish = Message(type=MsgType.FullClientRequest, flag=MsgTypeFlagBits.WithEvent)
                        finish.event = EventType.FinishSession
                        finish.session_id = msg.context_id
                        finish.payload = b"{}"
                        await self._ws.send_bytes(finish.marshal())

        except asyncio.CancelledError:
            logger.debug("send loop cancelled")
            raise  # Re-raise to allow proper cleanup
        except Exception as e:
            logger.warning(f"send loop error: {e}", exc_info=e)
        finally:
            if not self._closed:
                await self.aclose()

    async def _recv_loop(self) -> None:
        try:
            # Add receive timeout to prevent infinite blocking
            recv_timeout = self._opts.inactivity_timeout

            while not self._closed and self._ws and not self._ws.closed:
                try:
                    # Set timeout for WebSocket receive to avoid infinite waiting
                    msg = await asyncio.wait_for(self._ws.receive(), timeout=recv_timeout)
                except asyncio.TimeoutError:
                    logger.warning(f"doubao tts receive timeout after {recv_timeout}s, closing connection")
                    # Cleanup all pending contexts
                    for ctx in self._context_data.values():
                        if not ctx.waiter.done():
                            ctx.waiter.set_exception(
                                APITimeoutError(f"no data received for {recv_timeout} seconds")
                            )
                    break
                except asyncio.CancelledError:
                    logger.debug("recv loop cancelled during cleanup")
                    break

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if not self._closed:
                        logger.warning("websocket closed unexpectedly")
                    break

                if msg.type != aiohttp.WSMsgType.BINARY:
                    continue

                try:
                    packet = Message.from_bytes(msg.data)
                except Exception:
                    continue

                # Connection handshake
                if (
                    packet.type == MsgType.FullServerResponse
                    and packet.event == EventType.ConnectionStarted
                    and self._connected_fut
                    and not self._connected_fut.done()
                ):
                    self._connected_fut.set_result(None)
                    continue

                context_id = packet.session_id
                if not context_id or context_id not in self._context_data:
                    continue

                ctx = self._context_data[context_id]
                emitter = ctx.emitter

                if packet.type == MsgType.AudioOnlyServer:
                    if packet.payload:
                        emitter.push(packet.payload)
                        if ctx.timeout_timer:
                            ctx.timeout_timer.cancel()
                    continue

                if packet.type == MsgType.FullServerResponse:
                    if packet.event in (EventType.SessionFinished, EventType.SessionFailed):
                        if not ctx.waiter.done():
                            if packet.event == EventType.SessionFailed:
                                ctx.waiter.set_exception(APIError("doubao session failed"))
                            else:
                                ctx.waiter.set_result(None)
                        self._cleanup_context(context_id)
                        if not self._is_current and not self._active_contexts:
                            logger.debug("no active contexts, shutting down connection")
                            break
        except asyncio.CancelledError:
            logger.debug("recv loop cancelled")
            raise  # Re-raise to allow proper cleanup
        except Exception as e:
            logger.warning(f"recv loop error: {e}", exc_info=e)
            # Notify all pending contexts about the error
            for ctx in self._context_data.values():
                if not ctx.waiter.done():
                    # Wrap network errors as retryable APIConnectionError
                    if isinstance(e, (aiohttp.ClientError, ConnectionError, OSError)):
                        ctx.waiter.set_exception(
                            APIConnectionError(f"network error in tts recv loop: {e}")
                        )
                    else:
                        ctx.waiter.set_exception(e)
                if ctx.timeout_timer:
                    ctx.timeout_timer.cancel()
            self._context_data.clear()
        finally:
            if not self._closed:
                await self.aclose()

    def _cleanup_context(self, context_id: str) -> None:
        ctx = self._context_data.pop(context_id, None)
        if ctx and ctx.timeout_timer:
            ctx.timeout_timer.cancel()
        self._active_contexts.discard(context_id)

    def _start_timeout_timer(self, context_id: str) -> None:
        if not (ctx := self._context_data.get(context_id)) or ctx.timeout_timer:
            return
        timeout = ctx.stream._conn_options.timeout

        def _on_timeout() -> None:
            if not ctx.waiter.done():
                ctx.waiter.set_exception(
                    APITimeoutError(f"doubao tts timed out after {timeout} seconds")
                )
            self._cleanup_context(context_id)

        ctx.timeout_timer = asyncio.get_event_loop().call_later(timeout, _on_timeout)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._input_queue.close()

        for ctx in self._context_data.values():
            if not ctx.waiter.done():
                ctx.waiter.set_exception(APIStatusError("connection closed"))
            if ctx.timeout_timer:
                ctx.timeout_timer.cancel()
        self._context_data.clear()

        # try to gracefully end connection
        try:
            if self._ws and not self._ws.closed:
                finish = Message(type=MsgType.FullClientRequest, flag=MsgTypeFlagBits.WithEvent)
                finish.event = EventType.FinishConnection
                finish.payload = b"{}"
                await self._ws.send_bytes(finish.marshal())
        except Exception:
            pass
        finally:
            if self._ws:
                await self._ws.close()

        if self._send_task:
            await utils.aio.gracefully_cancel(self._send_task)
        if self._recv_task:
            await utils.aio.gracefully_cancel(self._recv_task)

        self._ws = None


def _get_resource_id(voice: str) -> str:
    try:
        if voice and voice.startswith("S_"):
            return "volc.megatts.default"
    except Exception:
        pass
    return "volc.service_type.10029"
