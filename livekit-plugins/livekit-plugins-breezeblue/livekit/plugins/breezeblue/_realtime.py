from __future__ import annotations

import asyncio
import sys
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import aiohttp
from async_timeout import timeout as timeout_scope

from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError, tts, utils
from livekit.agents.types import APIConnectOptions

from ._pcm import PcmPause
from ._text import take_prefix, text_segments, utf8_chunks

if TYPE_CHECKING:
    from .tts import TTS


_PROTOCOL = "breeze-realtime-v1"


class RealtimeConnection:
    """Reuse a public Breeze session for a serial sequence of turns."""

    def __init__(
        self,
        socket: aiohttp.ClientWebSocketResponse,
        ready: dict[str, Any],
        options: dict[str, Any],
    ) -> None:
        self.socket = socket
        self.options = options
        self.max_append_bytes = int(ready.get("max_append_bytes", 2048))
        self.max_characters = int(ready.get("max_turn_characters", 1000))
        self.idle_timeout = float(ready.get("inactivity_timeout_seconds", 30))
        self.expires = time.monotonic() + float(ready.get("max_session_seconds", 1800))
        self.last_used = time.monotonic()

    @classmethod
    async def connect(
        cls, provider: TTS, voice: str, options: dict[str, Any], timeout: float
    ) -> RealtimeConnection:
        url = f"{provider._base_url}/v1/text-to-speech/{quote(voice, safe='')}/realtime-sessions"
        http = provider._http_session()
        async with http.post(
            url,
            headers=provider._headers,
            json={**options, "output_format": "pcm"},
            timeout=aiohttp.ClientTimeout(total=timeout),
            allow_redirects=False,
        ) as response:
            if response.status != 200 and response.status != 201:
                await provider._raise_for_status(response, timeout)
            credentials = await response.json()

        direct = credentials.get("direct_websocket_url")
        socket = await http.ws_connect(
            direct or credentials["websocket_url"],
            protocols=(_PROTOCOL, f"breeze-realtime-token.{credentials['client_secret']}")
            if direct
            else (_PROTOCOL,),
            timeout=aiohttp.ClientWSTimeout(ws_close=2),
            headers={"User-Agent": provider._headers["User-Agent"]},
        )
        try:
            if socket.protocol != _PROTOCOL:
                raise APIConnectionError("Breeze did not negotiate the realtime subprotocol")
            async with timeout_scope(timeout):
                ready = await cls.receive(socket)
            if not isinstance(ready, dict) or ready.get("type") != "session.ready":
                raise APIConnectionError("Breeze realtime session did not send session.ready")
            return cls(socket, ready, options)
        except BaseException:
            await socket.close()
            raise

    def reusable(self) -> bool:
        now = time.monotonic()
        return (
            not self.socket.closed
            and now < self.expires - 2
            and now < self.last_used + self.idle_timeout - 2
        )

    @staticmethod
    async def receive(socket: aiohttp.ClientWebSocketResponse) -> bytes | dict[str, Any]:
        message = await socket.receive()
        if message.type == aiohttp.WSMsgType.BINARY:
            return bytes(message.data)
        if message.type != aiohttp.WSMsgType.TEXT:
            raise APIConnectionError("Breeze realtime connection closed")
        event = message.json()
        if not isinstance(event, dict):
            raise APIConnectionError("Invalid Breeze realtime event")
        if event.get("type") == "error":
            code = event.get("code")
            error_types = {
                "AUTH_REQUIRED": (401, False),
                # The next attempt creates a new session with fresh credentials.
                "AUTH_SESSION_EXPIRED": (502, True),
                "VALIDATION_ERROR": (400, False),
                "BAD_REQUEST": (400, False),
                "RESOURCE_NOT_FOUND": (404, False),
                "BILLING_INSUFFICIENT_CREDITS": (402, False),
                "API_KEY_CREDIT_LIMIT_EXCEEDED": (402, False),
                "RATE_LIMITED": (429, True),
                "GENERATION_CONCURRENCY_EXCEEDED": (429, True),
                "GENERATION_CAPACITY_EXCEEDED": (429, True),
                "UPSTREAM_TIMEOUT": (504, True),
            }
            known_code = code if isinstance(code, str) and code in error_types else None
            status, retryable = error_types.get(known_code or "", (502, True))
            retry_after = 0.0
            meta = event.get("meta")
            if isinstance(meta, dict):
                seconds = meta.get("retry_after_seconds")
                milliseconds = meta.get("retry_after_ms")
                if (
                    isinstance(seconds, (int, float))
                    and not isinstance(seconds, bool)
                    and 0 < seconds <= sys.float_info.max
                ):
                    retry_after = float(seconds)
                elif (
                    isinstance(milliseconds, (int, float))
                    and not isinstance(milliseconds, bool)
                    and 0 < milliseconds <= sys.float_info.max
                ):
                    retry_after = milliseconds / 1000
            raise APIStatusError(
                f"Breeze realtime error: {known_code or 'UNKNOWN'}",
                status_code=status,
                body={"retry_after": retry_after} if retry_after else None,
                retryable=retryable,
            )
        return event

    async def send(self, kind: str, **fields: str) -> None:
        await self.socket.send_json({"type": kind, **fields})

    async def update(self, instructions: str, timeout: float) -> None:
        await self.send("session.update", instructions=instructions)
        async with timeout_scope(timeout):
            while True:
                event = await self.receive(self.socket)
                if isinstance(event, dict) and event["type"] == "session.updated":
                    self.options = {**self.options, "instructions": instructions}
                    self.last_used = time.monotonic()
                    return

    async def close(self) -> None:
        await self.socket.close()


class RealtimeStream(tts.SynthesizeStream):
    def __init__(
        self, provider: TTS, voice: str, options: dict[str, Any], conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=provider, conn_options=conn_options)
        self.provider = provider
        self.voice = voice
        self.options = options

    async def _run(self, emitter: tts.AudioEmitter) -> None:
        self._wire: RealtimeConnection | None = None
        self._turn: str | None = None
        self._reader: asyncio.Task[str] | None = None
        self._emitted = False
        self._discard_audio = False
        self._chars = 0
        self._pause = PcmPause(self.provider._sentence_pause_ms)
        timeout = self._conn_options.timeout
        try:
            async with self.provider._realtime_lock:
                try:
                    await self._synthesize(emitter, timeout)
                except asyncio.CancelledError:
                    if self._turn:
                        await self._cancel_turn(timeout)
                    raise
                except BaseException:
                    await self.provider._close_realtime()
                    raise
                finally:
                    if self._reader:
                        self._reader.cancel()
                        await asyncio.gather(self._reader, return_exceptions=True)
        except asyncio.TimeoutError as error:
            raise APITimeoutError(
                "Breeze realtime timed out", retryable=not self._emitted
            ) from error
        except aiohttp.WSServerHandshakeError as error:
            raise APIStatusError(
                f"Breeze realtime handshake returned HTTP {error.status}",
                status_code=error.status,
                retryable=not self._emitted and error.status >= 500,
            ) from None
        except aiohttp.ClientError:
            raise APIConnectionError(
                "Breeze realtime connection failed", retryable=not self._emitted
            ) from None
        except (APIConnectionError, APIStatusError) as error:
            if self._emitted:
                error.retryable = False
            if (
                isinstance(error, APIStatusError)
                and error.retryable
                and isinstance(error.body, dict)
            ):
                delay = error.body.get("retry_after", 0)
                if delay > timeout:
                    error.retryable = False
                elif delay > 0:
                    await asyncio.sleep(delay)
            raise

    async def _synthesize(self, emitter: tts.AudioEmitter, timeout: float) -> None:
        request_id = utils.shortuuid()
        emitter.initialize(
            request_id=request_id,
            sample_rate=24000,
            num_channels=1,
            mime_type="audio/pcm",
            frame_size_ms=20,
            stream=True,
        )
        segment = False
        async for text in text_segments(
            self._input_ch,
            self.provider._max_characters,
            whole_response=self.provider._input_mode == "whole",
            input_chunk_length_schedule=self.provider._input_chunk_length_schedule,
        ):
            if text is None:
                await self._finish_turn(timeout)
                if segment:
                    emitter.end_segment()
                    segment = False
            else:
                if not segment:
                    emitter.start_segment(segment_id=utils.shortuuid())
                    segment = True
                await self._append(text, emitter, timeout)
        await self._finish_turn(timeout)
        if segment:
            emitter.end_segment()

    async def _append(self, text: str, emitter: tts.AudioEmitter, timeout: float) -> None:
        if not self._wire:
            begin = time.monotonic()
            self._wire, self._connection_reused = await self.provider._get_realtime(
                self.voice, self.options, timeout
            )
            self._acquire_time = time.monotonic() - begin
        wire = self._wire
        limit = min(wire.max_characters, self.provider._max_characters)
        while text:
            if self._reader and self._reader.done():
                self._reader.result()
            if self._turn and self._chars + len(text) > limit:
                await self._finish_turn(timeout)
            part, text = take_prefix(text, limit)
            if not part.strip() and not self._turn:
                continue
            if not self._turn:
                self._pause.begin()
                self._mark_started()
                self._turn = utils.shortuuid()
                self._chars = 0
                await wire.send("turn.start", turn_id=self._turn)
                self._last_frame = time.monotonic()
                self._reader = asyncio.create_task(self._receive_turn(wire, self._turn, emitter))
            for chunk in utf8_chunks(part, wire.max_append_bytes):
                await wire.send("text.append", text=chunk)
            self._chars += len(part)
            await wire.send("text.flush")
            if self.provider._sentence_pause_ms:
                # Breeze exposes audio boundaries by turn. Finish each sentence
                # on the same connection so the next join is sample-accurate.
                await self._finish_turn(timeout)

    async def _receive_turn(
        self, wire: RealtimeConnection, turn_id: str, emitter: tts.AudioEmitter
    ) -> str:
        while True:
            event = await wire.receive(wire.socket)
            self._last_frame = time.monotonic()
            if isinstance(event, bytes):
                if not self._discard_audio:
                    gap, pcm = self._pause.push(event)
                    if gap:
                        emitter.push(gap)
                    if pcm:
                        emitter.push(pcm)
                        self._emitted = True
            elif (
                event["type"] in {"turn.done", "turn.cancelled"} and event.get("turn_id") == turn_id
            ):
                if not self._discard_audio and (remaining := self._pause.finish()):
                    emitter.push(remaining)
                    self._emitted = True
                return str(event["type"])

    async def _finish_turn(self, timeout: float) -> None:
        if self._turn is None:
            return
        wire, reader = self._wire, self._reader
        if wire is None or reader is None:
            raise APIConnectionError("Breeze realtime turn was not started")
        await wire.send("turn.end")
        self._last_frame = time.monotonic()
        while not reader.done():
            remaining = timeout - (time.monotonic() - self._last_frame)
            if remaining <= 0:
                raise asyncio.TimeoutError
            await asyncio.wait((reader,), timeout=remaining)
        result = reader.result()
        wire.last_used = time.monotonic()
        self._turn = None
        self._reader = None
        if result != "turn.done":
            raise APIConnectionError("Breeze cancelled the turn before synthesis completed")

    async def _cancel_turn(self, timeout: float) -> None:
        self._discard_audio = True
        wire, reader = self._wire, self._reader
        if wire is None or reader is None:
            await self.provider._close_realtime()
            return
        try:
            if not reader.done():
                await wire.send("turn.cancel")
                async with timeout_scope(min(2, timeout)):
                    await asyncio.shield(reader)
            reader.result()
            wire.last_used = time.monotonic()
        except (
            APIConnectionError,
            APIStatusError,
            aiohttp.ClientError,
            asyncio.TimeoutError,
            ValueError,
        ):
            await self.provider._close_realtime()
