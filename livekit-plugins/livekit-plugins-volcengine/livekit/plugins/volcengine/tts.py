from __future__ import annotations

import asyncio
import os
import weakref
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .protocol import (
    Event,
    MessageType,
    ProtocolError,
    ServerMessage,
    build_client_message,
    parse_server_message as parse_message,
)

DEFAULT_ENDPOINT = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
DEFAULT_RESOURCE_ID = "seed-tts-2.0"
SUPPORTED_SAMPLE_RATES = {8000, 16000, 22050, 24000, 32000, 44100, 48000}


@dataclass
class _TTSOptions:
    api_key: str
    voice: str
    resource_id: str
    endpoint: str
    sample_rate: int
    speech_rate: int
    loudness_rate: int
    emotion: str | None
    emotion_scale: int


class TTS(tts.TTS[Any]):
    """Volcengine bidirectional streaming text-to-speech client."""

    def __init__(
        self,
        *,
        voice: str | None = None,
        resource_id: str | None = None,
        sample_rate: int = 24000,
        speech_rate: int = 0,
        loudness_rate: int = 0,
        emotion: str | None = None,
        emotion_scale: int = 4,
        api_key: str | None = None,
        endpoint: str = DEFAULT_ENDPOINT,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a Volcengine TTS client.

        Args:
            voice: Volcengine speaker ID. Defaults to ``VOLCENGINE_TTS_VOICE``.
            resource_id: Product and model resource ID. Defaults to
                ``VOLCENGINE_TTS_RESOURCE_ID`` or ``seed-tts-2.0``.
            sample_rate: PCM sample rate in Hz.
            speech_rate: Speech rate adjustment in the range [-50, 100].
            loudness_rate: Loudness adjustment in the range [-50, 100].
            emotion: Optional emotion supported by the selected voice.
            emotion_scale: Emotion intensity in the range [1, 5].
            api_key: Volcengine API key. Defaults to ``VOLCENGINE_API_KEY``.
            endpoint: Bidirectional TTS WebSocket endpoint.
            http_session: Existing aiohttp client session.
        """
        _validate_options(sample_rate, speech_rate, loudness_rate, emotion_scale)
        resolved_api_key = api_key or os.environ.get("VOLCENGINE_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Volcengine API key is required. Set VOLCENGINE_API_KEY or provide api_key."
            )
        resolved_voice = voice or os.environ.get("VOLCENGINE_TTS_VOICE")
        if not resolved_voice:
            raise ValueError(
                "Volcengine voice is required. Set VOLCENGINE_TTS_VOICE or provide voice."
            )
        resolved_resource_id = (
            resource_id or os.environ.get("VOLCENGINE_TTS_RESOURCE_ID") or DEFAULT_RESOURCE_ID
        )

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._opts = _TTSOptions(
            api_key=resolved_api_key,
            voice=resolved_voice,
            resource_id=resolved_resource_id,
            endpoint=endpoint,
            sample_rate=sample_rate,
            speech_rate=speech_rate,
            loudness_rate=loudness_rate,
            emotion=emotion,
            emotion_scale=emotion_scale,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )

    @property
    def model(self) -> str:
        """Return the configured Volcengine resource ID."""
        return self._opts.resource_id

    @property
    def provider(self) -> str:
        """Return the provider name used in metrics."""
        return "Volcengine"

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        speech_rate: NotGivenOr[int] = NOT_GIVEN,
        loudness_rate: NotGivenOr[int] = NOT_GIVEN,
        emotion: NotGivenOr[str | None] = NOT_GIVEN,
        emotion_scale: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """Update synthesis options used by subsequently created streams.

        Args:
            voice: Volcengine speaker ID.
            speech_rate: Speech rate adjustment in the range [-50, 100].
            loudness_rate: Loudness adjustment in the range [-50, 100].
            emotion: Optional emotion supported by the selected voice.
            emotion_scale: Emotion intensity in the range [1, 5].
        """
        next_speech_rate = speech_rate if is_given(speech_rate) else self._opts.speech_rate
        next_loudness_rate = loudness_rate if is_given(loudness_rate) else self._opts.loudness_rate
        next_emotion_scale = emotion_scale if is_given(emotion_scale) else self._opts.emotion_scale
        _validate_options(
            self._opts.sample_rate,
            next_speech_rate,
            next_loudness_rate,
            next_emotion_scale,
        )
        if is_given(voice):
            self._opts.voice = voice
        if is_given(speech_rate):
            self._opts.speech_rate = speech_rate
        if is_given(loudness_rate):
            self._opts.loudness_rate = loudness_rate
        if is_given(emotion):
            self._opts.emotion = emotion
        if is_given(emotion_scale):
            self._opts.emotion_scale = emotion_scale

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        """Synthesize complete text through the streaming implementation."""
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        """Create a bidirectional text input and PCM audio output stream."""
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        """Open and initialize a reusable WebSocket connection."""
        self._pool.prewarm()

    async def aclose(self) -> None:
        """Close active streams and pooled connections."""
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._pool.aclose()

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        ws = await asyncio.wait_for(
            self._ensure_session().ws_connect(
                self._opts.endpoint,
                headers={
                    "X-Api-Key": self._opts.api_key,
                    "X-Api-Resource-Id": self._opts.resource_id,
                },
            ),
            timeout,
        )
        try:
            await ws.send_bytes(build_client_message(Event.START_CONNECTION))
            response = await _receive_binary_message(ws, timeout)
            _require_event(response, Event.CONNECTION_STARTED)
            return ws
        except BaseException:
            await ws.close()
            raise

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        if not ws.closed:
            try:
                await ws.send_bytes(build_client_message(Event.FINISH_CONNECTION))
            except Exception:
                logger.debug("Failed to finish Volcengine connection", exc_info=True)
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session


class SynthesizeStream(tts.SynthesizeStream):
    """LiveKit streaming TTS adapter for a Volcengine connection."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        segments = utils.aio.Chan[utils.aio.Chan[str]]()

        async def collect_input() -> None:
            segment: utils.aio.Chan[str] | None = None
            async for item in self._input_ch:
                if isinstance(item, str):
                    if segment is None:
                        segment = utils.aio.Chan[str]()
                        segments.send_nowait(segment)
                    segment.send_nowait(item)
                elif isinstance(item, self._FlushSentinel) and segment is not None:
                    segment.close()
                    segment = None
            if segment is not None:
                segment.close()
            segments.close()

        async def synthesize_segments() -> None:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                self._acquire_time = self._tts._pool.last_acquire_time
                self._connection_reused = self._tts._pool.last_connection_reused
                async for segment in segments:
                    await self._run_session(ws, segment, output_emitter)

        tasks = [asyncio.create_task(collect_input()), asyncio.create_task(synthesize_segments())]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except aiohttp.ClientResponseError as error:
            raise APIStatusError(
                message=error.message,
                status_code=error.status,
                request_id=request_id,
                body=None,
            ) from error
        except Exception as error:
            raise APIConnectionError(f"Volcengine TTS connection failed: {error}") from error
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_session(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        segment: utils.aio.Chan[str],
        output_emitter: tts.AudioEmitter,
    ) -> None:
        session_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=session_id)
        await ws.send_bytes(
            build_client_message(
                Event.START_SESSION,
                _session_payload(self._opts),
                session_id=session_id,
            )
        )
        _require_event(
            await _receive_binary_message(ws, self._conn_options.timeout), Event.SESSION_STARTED
        )

        async def send_text() -> None:
            async for text in segment:
                if not text:
                    continue
                self._mark_started()
                await ws.send_bytes(
                    build_client_message(
                        Event.TASK_REQUEST,
                        {"event": Event.TASK_REQUEST, "req_params": {"text": text}},
                        session_id=session_id,
                    )
                )
            await ws.send_bytes(build_client_message(Event.FINISH_SESSION, session_id=session_id))

        async def receive_audio() -> None:
            while True:
                response = await _receive_binary_message(ws, self._conn_options.timeout)
                if response.message_type == MessageType.ERROR_RESPONSE:
                    raise _response_error(response)
                if response.event == Event.TTS_RESPONSE:
                    output_emitter.push(response.payload)
                elif response.event == Event.SESSION_FINISHED:
                    _raise_for_status(response)
                    output_emitter.end_segment()
                    return
                elif response.event in {
                    Event.CONNECTION_FAILED,
                    Event.SESSION_CANCELED,
                    Event.SESSION_FAILED,
                }:
                    raise _response_error(response)

        tasks = [asyncio.create_task(send_text()), asyncio.create_task(receive_audio())]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)


def _session_payload(options: _TTSOptions) -> dict[str, Any]:
    audio_params: dict[str, Any] = {
        "format": "pcm",
        "sample_rate": options.sample_rate,
        "speech_rate": options.speech_rate,
        "loudness_rate": options.loudness_rate,
    }
    if options.emotion is not None:
        audio_params["emotion"] = options.emotion
        audio_params["emotion_scale"] = options.emotion_scale
    return {
        "event": Event.START_SESSION,
        "namespace": "BidirectionalTTS",
        "req_params": {
            "speaker": options.voice,
            "audio_params": audio_params,
        },
    }


async def _receive_binary_message(
    ws: aiohttp.ClientWebSocketResponse, timeout: float
) -> ServerMessage:
    message = await ws.receive(timeout=timeout)
    if message.type == aiohttp.WSMsgType.BINARY:
        return parse_message(message.data)
    if message.type in {
        aiohttp.WSMsgType.CLOSE,
        aiohttp.WSMsgType.CLOSED,
        aiohttp.WSMsgType.CLOSING,
    }:
        raise APIStatusError(
            "Volcengine WebSocket closed unexpectedly",
            status_code=ws.close_code or -1,
            body=f"{message.data=} {message.extra=}",
        )
    raise ProtocolError(f"Unexpected Volcengine WebSocket message type: {message.type}")


def _require_event(message: ServerMessage, expected_event: Event) -> None:
    if message.message_type == MessageType.ERROR_RESPONSE:
        raise _response_error(message)
    if message.event in {Event.CONNECTION_FAILED, Event.SESSION_FAILED}:
        raise _response_error(message)
    if message.event != expected_event:
        raise ProtocolError(f"Expected Volcengine event {expected_event.name}, got {message.event}")
    _raise_for_status(message)


def _raise_for_status(message: ServerMessage) -> None:
    payload = message.json_payload()
    status_code = payload.get("status_code")
    if status_code is not None and status_code != 20000000:
        raise _response_error(message)


def _response_error(message: ServerMessage) -> APIStatusError:
    payload: dict[str, Any] = {}
    if message.payload:
        try:
            payload = message.json_payload()
        except ProtocolError:
            pass
    status_code = message.error_code or payload.get("status_code") or -1
    error_message = payload.get("message") or message.payload.decode("utf-8", errors="replace")
    return APIStatusError(
        message=f"Volcengine TTS error: {error_message}",
        status_code=status_code,
        request_id=None,
        body=payload or error_message,
    )


def _validate_options(
    sample_rate: int, speech_rate: int, loudness_rate: int, emotion_scale: int
) -> None:
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise ValueError(f"Unsupported Volcengine sample_rate: {sample_rate}")
    if not -50 <= speech_rate <= 100:
        raise ValueError("speech_rate must be between -50 and 100")
    if not -50 <= loudness_rate <= 100:
        raise ValueError("loudness_rate must be between -50 and 100")
    if not 1 <= emotion_scale <= 5:
        raise ValueError("emotion_scale must be between 1 and 5")
