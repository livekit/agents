# Copyright 2026 Maya Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from __future__ import annotations

import asyncio
import os
import sys
import weakref
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit, urlunsplit

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

from ._protocol import (
    NUM_CHANNELS,
    SAMPLE_RATE,
    Turn,
    cancel_unfinished,
    receive_audio,
    receive_json,
    send_text,
    validate_metadata,
)
from .models import TTSLanguages, TTSModels

DEFAULT_BASE_URL = "https://tts.mayaresearch.ai"


@dataclass(frozen=True)
class _Settings:
    model: str
    voice: str
    language: NotGivenOr[str]

    def start(self) -> dict[str, object]:
        result: dict[str, object] = {
            "type": "start",
            "v2": True,
            "model": self.model,
            "voice": self.voice,
        }
        if is_given(self.language):
            result["language"] = self.language
        return result


@dataclass(eq=False)
class _Connection:
    ws: aiohttp.ClientWebSocketResponse
    settings: _Settings


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(c in value for c in "\r\n"):
        raise ValueError(f"{name} must be a nonempty single-line string")
    return value


class TTS(tts.TTS):
    """Maya Research TTS over the public persistent WebSocket v2 protocol."""

    def __init__(
        self,
        *,
        model: TTSModels | str = "Maya Calyx",
        voice: str = "Aarav",
        language: NotGivenOr[TTSLanguages | str] = NOT_GIVEN,
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
    ) -> None:
        """Create a Maya Research speech provider.

        Args:
            model: Exact server-supported model ID. Currently defaults to Maya Calyx.
            voice: Exact voice name for the chosen model. Defaults to Aarav.
            language: Documented language code; omit for mixed-language text.
            api_key: API key, or the MAYA_API_KEY environment variable.
            base_url: API root, or MAYA_BASE_URL. Defaults to the public Maya endpoint.
            http_session: Optional caller-owned aiohttp session.
            tokenizer: Sentence tokenizer. The default BlingFire tokenizer primarily
                recognizes western punctuation; supply an Indic-aware tokenizer when
                early danda-delimited sentence delivery is required.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._api_key = _nonempty(api_key or os.getenv("MAYA_API_KEY") or "", "MAYA_API_KEY")
        self._settings = _Settings(
            model=_nonempty(model, "model"),
            voice=_nonempty(voice, "voice"),
            language=_nonempty(language, "language") if is_given(language) else NOT_GIVEN,
        )
        root = base_url if is_given(base_url) else os.getenv("MAYA_BASE_URL", DEFAULT_BASE_URL)
        parsed = urlsplit(root)
        if (
            parsed.scheme not in ("http", "https", "ws", "wss")
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base_url must be an HTTP/WebSocket root without credentials or query")
        scheme = {"https": "wss", "http": "ws"}.get(parsed.scheme, parsed.scheme)
        self._url = urlunsplit(
            (scheme, parsed.netloc, parsed.path.rstrip("/") + "/v1/tts/stream", "", "")
        )
        self._session = http_session
        self._closed = False
        self._pool = utils.ConnectionPool[_Connection](
            connect_cb=self._connect,
            close_cb=self._disconnect,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )
        self._streams: weakref.WeakSet[ChunkedStream | SynthesizeStream] = weakref.WeakSet()
        self._tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.blingfire.SentenceTokenizer()
        )

    @property
    def model(self) -> str:
        """Model ID selected for subsequent turns."""
        return self._settings.model

    @property
    def provider(self) -> str:
        """Stable provider name, independent of the selected model."""
        return "Maya Research"

    async def _connect(self, timeout: float) -> _Connection:
        settings = self._settings
        session = self._session or utils.http_context.http_session()
        try:
            ws = await asyncio.wait_for(
                session.ws_connect(self._url, headers={"Authorization": f"Bearer {self._api_key}"}),
                timeout,
            )
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as exc:
            raise APIStatusError(
                "Maya websocket handshake failed", status_code=exc.status
            ) from None
        except Exception:
            raise APIConnectionError("Could not connect to Maya") from None
        if TYPE_CHECKING and sys.version_info < (3, 11):
            # aiohttp's 3.10 stubs omit the default decode_text overload.
            ws = cast(aiohttp.ClientWebSocketResponse, ws)
        try:
            await asyncio.wait_for(ws.send_json(settings.start()), timeout)
            validate_metadata(await receive_json(ws, timeout))
        except BaseException as exc:
            await ws.close()
            if isinstance(exc, (APIError, asyncio.CancelledError)):
                raise
            if isinstance(exc, asyncio.TimeoutError):
                raise APITimeoutError() from None
            if isinstance(exc, Exception):
                raise APIConnectionError("Maya startup handshake failed") from None
            raise
        return _Connection(ws, settings)

    async def _disconnect(self, connection: _Connection) -> None:
        await connection.ws.close()

    @asynccontextmanager
    async def _connection(self, timeout: float) -> AsyncIterator[_Connection]:
        if self._closed:
            raise APIError("Maya TTS is closed", retryable=False)
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise APITimeoutError()
            connection = await self._pool.get(timeout=remaining)
            if not connection.ws.closed and connection.settings == self._settings:
                break
            # Settings belong to the socket. Retire only a socket acquired for a
            # NEW turn; changing options must never cut off an active speaker.
            self._pool.remove(connection)
            await connection.ws.close()
        try:
            yield connection
        except BaseException as exc:
            self._pool.remove(connection)
            await connection.ws.close()
            if isinstance(exc, (APIError, asyncio.CancelledError)):
                raise
            if isinstance(exc, asyncio.TimeoutError):
                raise APITimeoutError() from None
            if isinstance(exc, Exception):
                raise APIConnectionError("Maya websocket transport failed") from None
            raise
        else:
            self._pool.put(connection)

    def prewarm(self) -> None:
        """Prepare a connection without sending text or generating speech."""
        if not self._closed:
            self._pool.prewarm()

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[TTSLanguages | str] = NOT_GIVEN,
    ) -> None:
        """Change settings for the next acquired turn; active turns keep their voice."""
        values: dict[str, str] = {}
        for name, value in (("model", model), ("voice", voice), ("language", language)):
            if is_given(value):
                values[name] = _nonempty(value, name)
        self._settings = replace(self._settings, **values)

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        """Synthesize a complete text input as an asynchronous audio stream."""
        stream = ChunkedStream(tts=self, input_text=text, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        """Create a new incremental-text stream for one LiveKit segment."""
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """Close owned streams and sockets, never a caller-provided HTTP session."""
        self._closed = True
        for stream in list(self._streams):
            await stream.aclose()
        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    """One-shot text synthesis through a pooled Maya WebSocket."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._maya = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        turn = Turn(utils.shortuuid())
        output_emitter.initialize(
            request_id=turn.context_id,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )
        if not self._input_text.strip():
            return
        try:
            async with self._maya._connection(self._conn_options.timeout) as connection:
                self._acquire_time = self._maya._pool.last_acquire_time
                self._connection_reused = self._maya._pool.last_connection_reused
                try:
                    await send_text(connection.ws, turn, self._input_text, more=False)
                    await receive_audio(
                        connection.ws, turn, output_emitter, self._conn_options.timeout
                    )
                finally:
                    await cancel_unfinished(connection.ws, turn)
        except APIError as exc:
            if turn.audio_bytes:
                exc.retryable = False  # Never automatically repeat already-delivered speech.
            raise


class SynthesizeStream(tts.SynthesizeStream):
    """Incremental sentences sharing one Maya context and terminal closer."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._maya = tts

    async def _sentences(self) -> AsyncGenerator[str, None]:
        tokenizer = self._maya._tokenizer.stream()

        async def feed() -> None:
            try:
                async for value in self._input_ch:
                    if isinstance(value, str):
                        tokenizer.push_text(value)
                    else:
                        tokenizer.flush()
            finally:
                tokenizer.end_input()

        task = asyncio.create_task(feed())
        try:
            async for token in tokenizer:
                if token.token.strip():
                    yield token.token.rstrip() + " "
            await task
        finally:
            await utils.aio.gracefully_cancel(task)
            await tokenizer.aclose()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        turn = Turn(utils.shortuuid())
        output_emitter.initialize(
            request_id=turn.context_id,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        sentences = self._sentences()
        try:
            # Waiting for an LLM is not a Maya response timeout. An empty turn
            # must not send a closer for a context that was never opened.
            first = await anext(sentences, None)
            if first is None:
                return
            output_emitter.start_segment(segment_id=turn.context_id)
            async with self._maya._connection(self._conn_options.timeout) as connection:
                self._acquire_time = self._maya._pool.last_acquire_time
                self._connection_reused = self._maya._pool.last_connection_reused
                tasks: list[asyncio.Task[None]] = []
                try:
                    await send_text(connection.ws, turn, first, more=True)
                    self._mark_started()

                    async def send_remaining() -> None:
                        async for text in sentences:
                            await send_text(connection.ws, turn, text, more=True)
                        await send_text(connection.ws, turn, "", more=False)

                    sender = asyncio.create_task(send_remaining())
                    receiver = asyncio.create_task(
                        receive_audio(
                            connection.ws, turn, output_emitter, self._conn_options.timeout
                        )
                    )
                    tasks = [sender, receiver]
                    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    if receiver.done():
                        await receiver
                    else:
                        await sender
                        await receiver
                    output_emitter.end_segment()
                finally:
                    await utils.aio.gracefully_cancel(*tasks)
                    await cancel_unfinished(connection.ws, turn)
        except APIError as exc:
            if turn.audio_bytes:
                exc.retryable = False
            raise
        finally:
            await sentences.aclose()
