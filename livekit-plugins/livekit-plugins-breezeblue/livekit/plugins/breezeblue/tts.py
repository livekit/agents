import asyncio
import contextlib
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Literal
from urllib.parse import quote
from weakref import WeakSet

import aiohttp
from async_timeout import timeout as timeout_scope

from livekit.agents import APIConnectionError, APIStatusError, tts, utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
)

from ._http import HttpChunkedStream, HttpSynthesizeStream
from ._realtime import RealtimeConnection, RealtimeStream
from .version import __version__


class TTS(tts.TTS):
    """Stream Breeze Developer API speech into LiveKit."""

    def __init__(
        self,
        *,
        voice: str,
        instructions: str | None = None,
        model: str | None = None,
        language: str | None = None,
        api_key: str | None = None,
        base_url: str = "https://api.breeze.blue",
        http_session: aiohttp.ClientSession | None = None,
        transport: Literal["http", "websocket"] = "http",
        word_timestamps: bool = False,
        timestamp_mode: Literal["chunk", "lookahead"] = "chunk",
        voice_settings: Mapping[str, float | bool] | None = None,
        max_characters: int = 1000,
        sentence_pause_ms: int = 160,
        input_mode: Literal["sentence", "whole"] = "whole",
        input_chunk_length_schedule: Sequence[int] | None = None,
    ) -> None:
        """Configure a Breeze synthesis provider.

        Args:
            voice: Saved Breeze voice ID.
            instructions: Natural-language delivery style.
            model: Breeze model ID, or use the service default.
            language: Two-letter speech language code.
            api_key: Key for the public API. Defaults to BREEZE_API_KEY.
            base_url: Breeze Developer API base URL.
            http_session: Existing aiohttp session managed by the caller.
            transport: HTTP or Realtime WebSocket for streaming LiveKit input.
            word_timestamps: Request aligned words over HTTP.
            timestamp_mode: Breeze HTTP timing strategy.
            voice_settings: Synthesis controls passed to Breeze.
            max_characters: Client-side length limit per synthesis request.
            sentence_pause_ms: Minimum pause when joining independent audio segments.
            input_mode: Wait for the full reply or synthesize sentence by sentence.
            input_chunk_length_schedule: Optional character counts for successive streaming text
                chunks. Overrides input_mode and repeats the final count until the reply ends.
        """
        if input_mode not in ("sentence", "whole"):
            raise ValueError("input_mode must be sentence or whole")
        if transport not in ("http", "websocket"):
            raise ValueError("transport must be http or websocket")
        if word_timestamps and transport == "websocket":
            raise ValueError("Breeze word timestamps require the HTTP transport")
        if not isinstance(sentence_pause_ms, int) or not 0 <= sentence_pause_ms <= 500:
            raise ValueError("sentence_pause_ms must be an integer between 0 and 500")
        if not 1 <= max_characters <= 2000:
            raise ValueError("max_characters must be between 1 and 2000")
        schedule = (
            tuple(input_chunk_length_schedule) if input_chunk_length_schedule is not None else None
        )
        if schedule is not None and (
            not schedule
            or any(
                type(length) is not int or not 1 <= length <= max_characters for length in schedule
            )
        ):
            raise ValueError(
                "input_chunk_length_schedule must contain integers between 1 and max_characters"
            )
        if timestamp_mode not in ("chunk", "lookahead"):
            raise ValueError("timestamp_mode must be chunk or lookahead")
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=word_timestamps,
            ),
            sample_rate=24000,
            num_channels=1,
        )
        self._api_key = api_key or os.getenv("BREEZE_API_KEY") or os.getenv("BREEZEBLUE_API_KEY")
        if not self._api_key:
            raise ValueError("Set BREEZE_API_KEY or pass api_key")
        if not voice:
            raise ValueError("A Breeze voice ID is required")
        self._transport = transport
        self._input_mode = input_mode
        self._input_chunk_length_schedule = schedule
        self._word_timestamps = word_timestamps
        self._timestamp_mode = timestamp_mode
        self._voice_settings = None if voice_settings is None else dict(voice_settings)
        self._max_characters = max_characters
        self._sentence_pause_ms = sentence_pause_ms
        self._headers = {
            "xi-api-key": self._api_key,
            "User-Agent": f"livekit-plugins-breezeblue/{__version__}",
        }
        self._realtime: tuple[str, RealtimeConnection] | None = None
        self._realtime_lock = asyncio.Lock()
        self._active_streams: WeakSet[tts.ChunkedStream | tts.SynthesizeStream] = WeakSet()
        self._prewarm_task: asyncio.Task[None] | None = None
        self._voice = voice
        self._instructions = instructions
        self._model = model
        self._language = language
        self._base_url = base_url.rstrip("/")
        self._session = http_session
        self._owned_session: aiohttp.ClientSession | None = None

    @property
    def model(self) -> str:
        return self._model or "auto"

    @property
    def provider(self) -> str:
        return "BreezeBlue"

    def update_options(self, *, voice: str | None = None, instructions: str | None = None) -> None:
        """Update the voice or speaking instructions for subsequent synthesis."""
        if voice is not None:
            if not voice:
                raise ValueError("A Breeze voice ID is required")
            self._voice = voice
        if instructions is not None:
            self._instructions = instructions or None

    def _options(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "model_id": self._model,
                "language_code": self._language,
                "instructions": self._instructions,
                "voice_settings": self._voice_settings,
            }.items()
            if value is not None
        }

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        stream = HttpChunkedStream(
            self, text, conn_options, quote(self._voice, safe=""), self._options()
        )
        self._active_streams.add(stream)
        return stream

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.SynthesizeStream:
        stream: tts.SynthesizeStream
        if self._transport == "websocket":
            stream = RealtimeStream(self, self._voice, self._options(), conn_options)
        else:
            stream = HttpSynthesizeStream(self, conn_options)
        self._active_streams.add(stream)
        return stream

    async def _get_realtime(
        self, voice: str, options: dict[str, Any], timeout: float
    ) -> tuple[RealtimeConnection, bool]:
        if self._realtime:
            previous_voice, connection = self._realtime
            if previous_voice == voice and connection.reusable():
                if connection.options == options:
                    return connection, True
                instructions = options.get("instructions")
                if instructions and {**connection.options, "instructions": instructions} == options:
                    try:
                        await connection.update(instructions, timeout)
                    except APIStatusError as error:
                        if error.retryable:
                            raise
                    else:
                        return connection, True
            await self._close_realtime()
        async with timeout_scope(timeout):
            connection = await RealtimeConnection.connect(self, voice, options, timeout)
        self._realtime = voice, connection
        return connection, False

    async def _close_realtime(self) -> None:
        if self._realtime:
            _, connection = self._realtime
            self._realtime = None
            await connection.close()

    def prewarm(self) -> None:
        if self._transport == "websocket" and (
            self._prewarm_task is None or self._prewarm_task.done()
        ):
            self._prewarm_task = asyncio.create_task(self._prewarm())

    async def _prewarm(self) -> None:
        async with self._realtime_lock:
            with contextlib.suppress(
                APIStatusError, APIConnectionError, aiohttp.ClientError, asyncio.TimeoutError
            ):
                await self._get_realtime(
                    self._voice, self._options(), DEFAULT_API_CONNECT_OPTIONS.timeout
                )

    @staticmethod
    async def _raise_for_status(response: aiohttp.ClientResponse, timeout: float) -> None:
        retryable = response.status in (408, 429) or response.status >= 500
        retry_after = response.headers.get("Retry-After")
        if retryable and retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                try:
                    delay = (
                        parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)
                    ).total_seconds()
                except (ValueError, TypeError, OverflowError):
                    delay = 0
            if delay > timeout:
                retryable = False
            elif delay > 0:
                response.release()
                await asyncio.sleep(delay)
        raise APIStatusError(
            f"Breeze TTS returned HTTP {response.status}",
            status_code=response.status,
            request_id=response.headers.get("history-item-id"),
            retryable=retryable,
        )

    def _http_session(self) -> aiohttp.ClientSession:
        if self._session is not None:
            return self._session
        try:
            return utils.http_context.http_session()
        except RuntimeError:
            if self._owned_session is None or self._owned_session.closed:
                self._owned_session = aiohttp.ClientSession()
            return self._owned_session

    async def aclose(self) -> None:
        if self._prewarm_task:
            self._prewarm_task.cancel()
            await asyncio.gather(self._prewarm_task, return_exceptions=True)
        await asyncio.gather(*(stream.aclose() for stream in list(self._active_streams)))
        async with self._realtime_lock:
            await self._close_realtime()
        if self._owned_session is not None:
            await self._owned_session.close()
