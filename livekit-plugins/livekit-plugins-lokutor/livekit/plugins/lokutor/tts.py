from __future__ import annotations

import asyncio
import json
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
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from .log import logger
from .models import DEFAULT_VOICE_ID, VoiceID


@dataclass
class _TTSOptions:
    api_key: str
    voice: str
    language: str | None
    speed: float
    steps: int
    visemes: bool
    base_url: str
    sample_rate: int

    def get_ws_url(self) -> str:
        return f"{self.base_url}/ws/tts"


class TTS(tts.TTS):
    """Lokutor Text-to-Speech integration.

    Connects to the Lokutor WebSocket API to synthesize speech. Supports
    streaming and non-streaming synthesis with multiple voices and languages.

    Args:
        api_key: Lokutor API key. If not provided, reads from ``LOKUTOR_API_KEY``
            environment variable.
        voice: Voice ID to use. See ``VoiceID`` for available options.
            Defaults to ``"F1"``.
        language: Language code (e.g. ``"en"``, ``"es"``, ``"fr"``).
            Defaults to ``"en"``.
        speed: Speed multiplier between 0.5 and 2.0. Defaults to ``1.05``.
        steps: Number of diffusion steps between 3 and 10. Higher values
            improve quality but increase latency. Defaults to ``5``.
        visemes: Whether to enable viseme (lip-sync) data. Defaults to
            ``False``.
        sample_rate: Audio sample rate in Hz. Defaults to ``44100``.
        base_url: Base URL for the Lokutor API. Defaults to
            ``"wss://api.lokutor.com"``.
        http_session: Optional shared ``aiohttp.ClientSession``. If not
            provided, a new session will be created.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        voice: VoiceID | str = DEFAULT_VOICE_ID,
        language: str | None = "en",
        speed: float = 1.05,
        steps: int = 5,
        visemes: bool = False,
        sample_rate: int = 44100,
        base_url: str = "wss://api.lokutor.com",
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        lokutor_api_key = api_key or os.environ.get("LOKUTOR_API_KEY")
        if not lokutor_api_key:
            raise ValueError(
                "Lokutor API key is required, either as argument or set"
                " LOKUTOR_API_KEY environment variable"
            )

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=False,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )

        self._opts = _TTSOptions(
            api_key=lokutor_api_key,
            voice=voice,
            language=language,
            speed=speed,
            steps=steps,
            visemes=visemes,
            base_url=base_url,
            sample_rate=sample_rate,
        )

        self._session = http_session
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def model(self) -> str:
        """Returns the model name ``"versa-1.0"``."""
        return "versa-1.0"

    @property
    def provider(self) -> str:
        """Returns the provider name ``"Lokutor"``."""
        return "Lokutor"

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        url = f"{self._opts.get_ws_url()}?api_key={self._opts.api_key}"
        ws = await asyncio.wait_for(
            session.ws_connect(url, max_msg_size=0),
            timeout,
        )
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def prewarm(self) -> None:
        """Pre-warm the WebSocket connection pool."""
        self._pool.prewarm()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        """Synthesize text to speech in a single request.

        Args:
            text: The text to synthesize.
            conn_options: Connection options for the request.

        Returns:
            A ``ChunkedStream`` that yields ``SynthesizedAudio`` events.
        """
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        """Create a streaming TTS session.

        Args:
            conn_options: Connection options for the stream.

        Returns:
            A ``SynthesizeStream`` for streaming synthesis.
        """
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """Clean up all stream and connection resources."""
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    """Non-streaming TTS synthesis.

    Sends the full text in a single WebSocket request and collects all
    audio before yielding it.
    """

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )

        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                request = _build_request(self._opts, self._input_text)
                await ws.send_str(json.dumps(request))

                while True:
                    msg = await ws.receive(timeout=self._conn_options.timeout)
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        raise APIStatusError(
                            "Lokutor connection closed unexpectedly",
                            request_id=request_id,
                            status_code=ws.close_code or -1,
                            body=f"{msg.data=} {msg.extra=}",
                        )

                    if msg.type == aiohttp.WSMsgType.BINARY:
                        output_emitter.push(msg.data)
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        if msg.data == "EOS":
                            break
                        try:
                            data = json.loads(msg.data)
                            if isinstance(data, dict) and data.get("type") == "error":
                                raise APIError(f"Lokutor error: {data.get('message', 'unknown')}")
                        except json.JSONDecodeError:
                            logger.warning("unexpected text message: %s", msg.data[:200])

                output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Streaming TTS synthesis.

    Reads text from ``push_text()`` calls and sends each chunk as a separate
    WebSocket request, streaming audio back in real-time.
    """

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

        output_emitter.start_segment(segment_id=request_id)

        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                self._acquire_time = self._tts._pool.last_acquire_time
                self._connection_reused = self._tts._pool.last_connection_reused

                async for data in self._input_ch:
                    if isinstance(data, self._FlushSentinel):
                        output_emitter.end_segment()
                        continue

                    request = _build_request(self._opts, data)
                    self._mark_started()
                    await ws.send_str(json.dumps(request))

                    while True:
                        msg = await ws.receive(timeout=self._conn_options.timeout)
                        if msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.CLOSING,
                        ):
                            raise APIStatusError(
                                "Lokutor connection closed unexpectedly",
                                request_id=request_id,
                                status_code=ws.close_code or -1,
                                body=f"{msg.data=} {msg.extra=}",
                            )

                        if msg.type == aiohttp.WSMsgType.BINARY:
                            output_emitter.push(msg.data)
                        elif msg.type == aiohttp.WSMsgType.TEXT:
                            if msg.data == "EOS":
                                break
                            try:
                                data = json.loads(msg.data)
                                if isinstance(data, dict) and data.get("type") == "error":
                                    raise APIError(
                                        f"Lokutor error: {data.get('message', 'unknown')}"
                                    )
                            except json.JSONDecodeError:
                                logger.warning("unexpected text message: %s", msg.data[:200])

                    output_emitter.flush()

                output_emitter.end_segment()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


def _build_request(opts: _TTSOptions, text: str) -> dict[str, Any]:
    request: dict[str, Any] = {
        "text": text,
        "voice": opts.voice,
        "speed": opts.speed,
        "steps": opts.steps,
        "visemes": opts.visemes,
    }
    if opts.language:
        request["lang"] = opts.language
    return request
