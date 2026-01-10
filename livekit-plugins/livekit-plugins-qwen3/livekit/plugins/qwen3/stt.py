from __future__ import annotations

import asyncio
import base64
import json
import os
import weakref
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from .log import logger
from .models import DEFAULT_BASE_URL

# Qwen3 ASR Model
STTModel = str  # "qwen3-asr-flash-realtime"

# Supported languages for ASR
STTLanguage = str  # "zh", "en", "yue" (Cantonese), etc.

# Default values for ASR
DEFAULT_STT_MODEL = "qwen3-asr-flash-realtime"
DEFAULT_STT_LANGUAGE = "zh"
DEFAULT_STT_SAMPLE_RATE = 16000


@dataclass
class _STTOptions:
    api_key: str
    base_url: str
    model: STTModel
    language: STTLanguage
    sample_rate: int


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: STTModel = DEFAULT_STT_MODEL,
        language: STTLanguage = DEFAULT_STT_LANGUAGE,
        sample_rate: int = DEFAULT_STT_SAMPLE_RATE,
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ):
        """Qwen3 ASR (STT) plugin for LiveKit Agents.

        Args:
            model: The Qwen3 ASR model to use. Defaults to "qwen3-asr-flash-realtime".
            language: The language for recognition. Options: "zh", "en", "yue" (Cantonese), etc.
            sample_rate: Audio sample rate in Hz. Defaults to 16000.
            api_key: DashScope API key. Defaults to DASHSCOPE_API_KEY env var.
            base_url: WebSocket base URL. Defaults to China region.
            http_session: Optional aiohttp ClientSession to reuse.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
            )
        )

        base_url_val = (
            base_url
            if utils.is_given(base_url)
            else os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)
        )

        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY must be set either as an argument or environment variable"
            )

        self._opts = _STTOptions(
            api_key=api_key,
            base_url=base_url_val,
            model=model,
            language=language,
            sample_rate=sample_rate,
        )

        self._session = http_session
        self._streams: weakref.WeakSet[RecognizeStream] = weakref.WeakSet()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Qwen3"

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModel] = NOT_GIVEN,
        language: NotGivenOr[STTLanguage] = NOT_GIVEN,
    ) -> None:
        """Update STT options dynamically."""
        if utils.is_given(model):
            self._opts.model = model
        if utils.is_given(language):
            self._opts.language = language

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        url = f"{self._opts.base_url}?model={self._opts.model}"
        headers = {"Authorization": f"Bearer {self._opts.api_key}"}
        session = self._ensure_session()
        ws = await asyncio.wait_for(session.ws_connect(url, headers=headers, heartbeat=30), timeout)
        logger.debug(f"Qwen3 ASR WebSocket connected to {url}")
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Non-streaming recognition - sends entire audio buffer at once."""
        raise NotImplementedError(
            "Non-streaming recognition not implemented. Use stream() instead."
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        stream = RecognizeStream(
            stt=self,
            conn_options=conn_options,
            language=language if utils.is_given(language) else self._opts.language,
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()


class RecognizeStream(stt.RecognizeStream):
    """WebSocket-based streaming STT for Qwen3 ASR."""

    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        language: str,
    ):
        super().__init__(
            stt=stt,
            conn_options=conn_options,
            sample_rate=stt._opts.sample_rate,
        )
        self._stt: STT = stt
        self._opts = replace(stt._opts)
        self._opts.language = language

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        session_ready = asyncio.Future[str]()

        async def _send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            """Send audio frames to the WebSocket."""
            # Wait for session to be ready
            try:
                session_id = await asyncio.wait_for(session_ready, self._conn_options.timeout)
            except asyncio.TimeoutError as e:
                raise APITimeoutError("session.updated timed out") from e

            logger.debug(f"Qwen3 ASR session ready: {session_id}")

            # Send audio frames
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    # Send silence to trigger final recognition
                    silence_data = bytes(1024)
                    for _ in range(10):
                        audio_b64 = base64.b64encode(silence_data).decode("ascii")
                        event = {
                            "type": "input_audio_buffer.append",
                            "event_id": f"event_{utils.shortuuid()}",
                            "audio": audio_b64,
                        }
                        await ws.send_str(json.dumps(event))
                        await asyncio.sleep(0.01)
                    continue

                # Convert AudioFrame to base64 PCM
                if isinstance(data, rtc.AudioFrame):
                    # Get raw PCM bytes from frame
                    pcm_bytes = data.data.tobytes()
                    audio_b64 = base64.b64encode(pcm_bytes).decode("ascii")

                    event = {
                        "type": "input_audio_buffer.append",
                        "event_id": f"event_{utils.shortuuid()}",
                        "audio": audio_b64,
                    }
                    await ws.send_str(json.dumps(event))

            # End input
            await asyncio.sleep(0.1)

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            """Receive transcription events from the WebSocket."""
            session_id = None

            while True:
                msg = await ws.receive()

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if not session_ready.done():
                        session_ready.set_exception(
                            APIConnectionError("WebSocket closed before session ready")
                        )
                    logger.debug("Qwen3 ASR WebSocket connection closed")
                    break

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning(f"Unexpected Qwen3 ASR message type: {msg.type}")
                    continue

                data: dict[str, Any] = json.loads(msg.data)
                event_type = data.get("type")

                if event_type == "error":
                    error = data.get("error", {})
                    error_msg = f"Qwen3 ASR error: {error}"
                    logger.error(error_msg)
                    if not session_ready.done():
                        session_ready.set_exception(APIError(error_msg))
                    raise APIError(error_msg)

                elif event_type == "session.created":
                    session_id = data.get("session", {}).get("id")
                    logger.debug(f"Qwen3 ASR session created: {session_id}")

                elif event_type == "session.updated":
                    session_id = data.get("session", {}).get("id")
                    if not session_ready.done():
                        session_ready.set_result(session_id)
                    logger.debug(f"Qwen3 ASR session updated: {session_id}")

                elif event_type == "input_audio_buffer.speech_started":
                    # Emit start of speech event
                    logger.debug("Qwen3 ASR speech started")
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.START_OF_SPEECH,
                            request_id=request_id,
                        )
                    )

                elif event_type == "input_audio_buffer.speech_stopped":
                    # Emit end of speech event
                    logger.debug("Qwen3 ASR speech stopped")
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.END_OF_SPEECH,
                            request_id=request_id,
                        )
                    )

                elif event_type == "conversation.item.input_audio_transcription.text":
                    # Interim transcript (stash result)
                    stash_text = data.get("stash", "")
                    if stash_text:
                        logger.debug(f"Qwen3 ASR interim: {stash_text[:50]}...")
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                request_id=request_id,
                                alternatives=[
                                    stt.SpeechData(
                                        language=self._opts.language,
                                        text=stash_text,
                                    )
                                ],
                            )
                        )

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    # Final transcript
                    transcript = data.get("transcript", "")
                    if transcript:
                        logger.debug(f"Qwen3 ASR final: {transcript[:50]}...")
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                request_id=request_id,
                                alternatives=[
                                    stt.SpeechData(
                                        language=self._opts.language,
                                        text=transcript,
                                    )
                                ],
                            )
                        )

                elif event_type == "session.finished":
                    logger.debug("Qwen3 ASR session finished")
                    break

                elif event_type not in ("input_audio_buffer.committed",):
                    logger.debug(f"Qwen3 ASR event: {event_type}")

        try:
            ws = await self._stt._connect_ws(self._conn_options.timeout)

            # Send session configuration for ASR
            config = {
                "type": "session.update",
                "event_id": f"event_{utils.shortuuid()}",
                "session": {
                    "output_modalities": ["text"],
                    "enable_input_audio_transcription": True,
                    "transcription_params": {
                        "language": self._opts.language,
                        "sample_rate": self._opts.sample_rate,
                        "input_audio_format": "pcm",
                    },
                },
            }
            await ws.send_str(json.dumps(config))
            logger.debug(
                f"Sent Qwen3 ASR session config: language={self._opts.language}, "
                f"sample_rate={self._opts.sample_rate}"
            )

            tasks = [
                asyncio.create_task(_send_task(ws)),
                asyncio.create_task(_recv_task(ws)),
            ]

            try:
                await asyncio.gather(*tasks)
            finally:
                await self._stt._close_ws(ws)
                await utils.aio.gracefully_cancel(*tasks)

        except asyncio.TimeoutError:
            logger.error(f"Qwen3 ASR WebSocket timeout after {self._conn_options.timeout}s")
            raise APITimeoutError(
                f"WebSocket ASR recognition timed out after {self._conn_options.timeout}s"
            ) from None
        except aiohttp.ClientResponseError as e:
            logger.error(f"Qwen3 ASR WebSocket HTTP error: {e.status} {e.message}")
            raise APIError(
                message=f"WebSocket HTTP {e.status}: {e.message}",
            ) from e
        except Exception as e:
            if not isinstance(e, (APITimeoutError, APIConnectionError, APIError)):
                logger.error(f"Qwen3 ASR WebSocket error: {type(e).__name__}: {e}")
            raise APIConnectionError(f"WebSocket connection failed: {e}") from e
