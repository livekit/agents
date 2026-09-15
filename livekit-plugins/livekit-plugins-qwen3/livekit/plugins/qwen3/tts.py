from __future__ import annotations

import asyncio
import base64
import json
import os
import weakref
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from .log import logger
from .models import (
    DEFAULT_BASE_URL,
    DEFAULT_LANGUAGE,
    DEFAULT_MODE,
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_VOICE,
    TTSLanguage,
    TTSMode,
    TTSModel,
    TTSVoice,
)


@dataclass
class _TTSOptions:
    api_key: str
    base_url: str
    model: TTSModel | str
    voice: TTSVoice | str
    language: TTSLanguage | str
    mode: TTSMode
    sample_rate: int


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModel | str = DEFAULT_MODEL,
        voice: TTSVoice | str = DEFAULT_VOICE,
        language: TTSLanguage | str = DEFAULT_LANGUAGE,
        mode: TTSMode = DEFAULT_MODE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ):
        """Qwen3 TTS plugin for LiveKit Agents.

        Args:
            model: The Qwen3 TTS model to use. Defaults to "qwen3-tts-flash-realtime".
            voice: The voice to use. Options include "Kiki", "Rocky", "Cherry", etc.
            language: The language for synthesis. Options: "Auto", "Chinese", "English", "Cantonese", etc.
            mode: Session mode. "server_commit" auto-triggers synthesis, "commit" requires manual trigger.
            sample_rate: Audio sample rate in Hz. Defaults to 24000.
            tokenizer: Sentence tokenizer for text segmentation.
            api_key: DashScope API key. Defaults to DASHSCOPE_API_KEY env var.
            base_url: WebSocket base URL. Defaults to China region.
            http_session: Optional aiohttp ClientSession to reuse.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=False),
            sample_rate=sample_rate,
            num_channels=1,
        )

        base_url = (
            base_url
            if utils.is_given(base_url)
            else os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)
        )

        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY must be set either as an argument or environment variable"
            )

        self._sentence_tokenizer = (
            tokenizer if utils.is_given(tokenizer) else tokenize.basic.SentenceTokenizer()
        )

        self._opts = _TTSOptions(
            api_key=api_key,
            base_url=base_url,
            model=model,
            voice=voice,
            language=language,
            mode=mode,
            sample_rate=sample_rate,
        )

        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Qwen3"

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModel | str] = NOT_GIVEN,
        voice: NotGivenOr[TTSVoice | str] = NOT_GIVEN,
        language: NotGivenOr[TTSLanguage | str] = NOT_GIVEN,
    ) -> None:
        """Update TTS options dynamically."""
        if utils.is_given(model):
            self._opts.model = model
        if utils.is_given(voice):
            self._opts.voice = voice
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
        logger.debug(f"Qwen3 WebSocket connected to {url}")
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

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


class SynthesizeStream(tts.SynthesizeStream):
    """WebSocket-based streaming TTS."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
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

        sentence_stream = self._tts._sentence_tokenizer.stream()
        session_ready = asyncio.Future[str]()

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sentence_stream.flush()
                    continue
                sentence_stream.push_text(data)
            sentence_stream.end_input()

        async def _send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            # Wait for session to be ready
            try:
                session_id = await asyncio.wait_for(session_ready, self._conn_options.timeout)
            except asyncio.TimeoutError as e:
                raise APITimeoutError("session.updated timed out") from e

            logger.debug(f"Qwen3 session ready: {session_id}")

            # Send text fragments
            async for sentence in sentence_stream:
                self._mark_started()
                event = {
                    "type": "input_text_buffer.append",
                    "event_id": f"event_{utils.shortuuid()}",
                    "text": sentence.token,
                }
                await ws.send_str(json.dumps(event))
                logger.debug(f"Sent text: {sentence.token[:50]}...")

            # In server_commit mode, wait for auto-commit
            # Give some time for the server to process
            await asyncio.sleep(0.5)

            # Finish session
            await ws.send_str(
                json.dumps(
                    {
                        "type": "session.finish",
                        "event_id": f"event_{utils.shortuuid()}",
                    }
                )
            )

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
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
                    logger.debug("Qwen3 WebSocket connection closed")
                    break

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning(f"Unexpected Qwen3 message type: {msg.type}")
                    continue

                data: dict[str, Any] = json.loads(msg.data)
                event_type = data.get("type")

                if event_type == "error":
                    error = data.get("error", {})
                    error_msg = f"Qwen3 error: {error}"
                    logger.error(error_msg)
                    if not session_ready.done():
                        session_ready.set_exception(APIError(error_msg))
                    raise APIError(error_msg)

                elif event_type == "session.created":
                    session_id = data.get("session", {}).get("id")
                    logger.debug(f"Qwen3 session created: {session_id}")

                elif event_type == "session.updated":
                    session_id = data.get("session", {}).get("id")
                    if not session_ready.done():
                        session_ready.set_result(session_id)
                    logger.debug(f"Qwen3 session updated: {session_id}")

                elif event_type == "response.created":
                    response_id = data.get("response", {}).get("id")
                    output_emitter.start_segment(segment_id=response_id or request_id)
                    logger.debug(f"Qwen3 response created: {response_id}")

                elif event_type == "response.audio.delta":
                    delta = data.get("delta", "")
                    if delta:
                        audio_bytes = base64.b64decode(delta)
                        output_emitter.push(audio_bytes)

                elif event_type == "response.audio.done":
                    logger.debug("Qwen3 audio generation complete")
                    output_emitter.flush()

                elif event_type == "response.done":
                    logger.debug("Qwen3 response complete")
                    output_emitter.end_input()

                elif event_type == "session.finished":
                    logger.debug("Qwen3 session finished")
                    break

                elif event_type not in (
                    "input_text_buffer.committed",
                    "response.output_item.added",
                    "response.output_item.done",
                    "response.content_part.added",
                    "response.content_part.done",
                ):
                    logger.debug(f"Qwen3 event: {event_type}")

        try:
            ws = await self._tts._connect_ws(self._conn_options.timeout)

            # Send session configuration
            config = {
                "type": "session.update",
                "event_id": f"event_{utils.shortuuid()}",
                "session": {
                    "mode": self._opts.mode,
                    "voice": self._opts.voice,
                    "language_type": self._opts.language,
                    "response_format": "pcm",
                    "sample_rate": self._opts.sample_rate,
                },
            }
            await ws.send_str(json.dumps(config))
            logger.debug(
                f"Sent Qwen3 session config: voice={self._opts.voice}, language={self._opts.language}"
            )

            tasks = [
                asyncio.create_task(_input_task()),
                asyncio.create_task(_send_task(ws)),
                asyncio.create_task(_recv_task(ws)),
            ]

            try:
                await asyncio.gather(*tasks)
            finally:
                await self._tts._close_ws(ws)
                await sentence_stream.aclose()
                await utils.aio.gracefully_cancel(*tasks)

        except asyncio.TimeoutError:
            logger.error(f"Qwen3 WebSocket timeout after {self._conn_options.timeout}s")
            raise APITimeoutError(
                f"WebSocket TTS synthesis timed out after {self._conn_options.timeout}s"
            ) from None
        except aiohttp.ClientResponseError as e:
            logger.error(f"Qwen3 WebSocket HTTP error: {e.status} {e.message}")
            raise APIStatusError(
                message=f"WebSocket HTTP {e.status}: {e.message}",
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from e
        except Exception as e:
            if not isinstance(e, (APIStatusError, APITimeoutError, APIConnectionError, APIError)):
                logger.error(f"Qwen3 WebSocket error: {type(e).__name__}: {e}")
            raise APIConnectionError(f"WebSocket connection failed: {e}") from e


class ChunkedStream(tts.ChunkedStream):
    """Non-streaming TTS using WebSocket (synthesizes full text at once)."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        if not self._input_text.strip():
            return

        request_id = utils.shortuuid()

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )

        session_ready = asyncio.Future[str]()
        audio_done = asyncio.Future[None]()

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            while True:
                msg = await ws.receive()

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    break

                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                data: dict[str, Any] = json.loads(msg.data)
                event_type = data.get("type")

                if event_type == "error":
                    error = data.get("error", {})
                    raise APIError(f"Qwen3 error: {error}")

                elif event_type == "session.updated":
                    if not session_ready.done():
                        session_ready.set_result(data.get("session", {}).get("id", ""))

                elif event_type == "response.audio.delta":
                    delta = data.get("delta", "")
                    if delta:
                        audio_bytes = base64.b64decode(delta)
                        output_emitter.push(audio_bytes)

                elif event_type == "response.done":
                    output_emitter.flush()
                    if not audio_done.done():
                        audio_done.set_result(None)

                elif event_type == "session.finished":
                    break

        try:
            ws = await self._tts._connect_ws(self._conn_options.timeout)

            # Start receiver task
            recv_task = asyncio.create_task(_recv_task(ws))

            # Send session config
            config = {
                "type": "session.update",
                "event_id": f"event_{utils.shortuuid()}",
                "session": {
                    "mode": self._opts.mode,
                    "voice": self._opts.voice,
                    "language_type": self._opts.language,
                    "response_format": "pcm",
                    "sample_rate": self._opts.sample_rate,
                },
            }
            await ws.send_str(json.dumps(config))

            # Wait for session ready
            await asyncio.wait_for(session_ready, self._conn_options.timeout)

            # Send text
            await ws.send_str(
                json.dumps(
                    {
                        "type": "input_text_buffer.append",
                        "event_id": f"event_{utils.shortuuid()}",
                        "text": self._input_text,
                    }
                )
            )

            # Wait for response
            await asyncio.sleep(0.5)

            # Finish session
            await ws.send_str(
                json.dumps(
                    {
                        "type": "session.finish",
                        "event_id": f"event_{utils.shortuuid()}",
                    }
                )
            )

            # Wait for audio completion
            await asyncio.wait_for(audio_done, timeout=30)

            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

            await self._tts._close_ws(ws)

        except asyncio.TimeoutError:
            raise APITimeoutError("TTS synthesis timed out") from None
        except Exception as e:
            if not isinstance(e, (APIStatusError, APITimeoutError, APIConnectionError, APIError)):
                logger.error(f"Qwen3 TTS error: {e}")
            raise APIConnectionError(f"Connection failed: {e}") from e
