# Copyright 2025 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vakyam realtime WebSocket protocol for sentence-wise TTS.

Copied from the Vakyam Python SDK protocol (config → text → binary* →
end_of_utterance, with cancel + drain on barge-in). Implemented here directly
so this plugin does not wrap ``vakyamai``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field, JsonValue, TypeAdapter, ValidationError

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    __version__ as livekit_version,
    http,
)

from ._utils import (
    ErrorResponse,
    normalize_base_url,
    normalize_voice,
    raise_ws_error,
    validate_text,
    validate_tts_options,
    websocket_url,
)
from .models import DEFAULT_BASE_URL, DEFAULT_LANGUAGE, DEFAULT_MODEL, DEFAULT_SAMPLE_RATE
from .version import __version__

USER_AGENT = f"LiveKit-Agents-Vakyam/{__version__} livekit-agents/{livekit_version}"


@dataclass(frozen=True)
class TTSSessionConfig:
    """Immutable synthesis settings applied once per WebSocket session."""

    model: str = DEFAULT_MODEL
    voice: str = "Archana"
    language: str = DEFAULT_LANGUAGE
    sample_rate: int = DEFAULT_SAMPLE_RATE
    speed: float = 1.0
    output_format: str = "pcm"

    def to_wire_message(self) -> dict[str, JsonValue]:
        return {
            "type": "config",
            "model_id": self.model,
            "voice": normalize_voice(self.voice),
            "language": self.language,
            "output_format": self.output_format,
            "sample_rate": self.sample_rate,
            "speed": self.speed,
        }


@dataclass(frozen=True)
class WebSocketSpeechResult:
    """Terminal result for one utterance (``end_of_utterance`` or ``cancellation``)."""

    characters_used: int
    duration_seconds: float
    truncated: bool = False
    truncation_reason: str | None = None
    cancelled: bool = False


def text_message(text: str) -> dict[str, str]:
    validate_text(text)
    return {"type": "text", "text": text}


def cancel_message() -> dict[str, str]:
    return {"type": "cancel"}


def ping_message() -> dict[str, str]:
    return {"type": "ping"}


def disconnect_message() -> dict[str, str]:
    return {"type": "disconnect"}


class _ControlMessage(ErrorResponse):
    type: Literal["connected", "configured", "error", "pong"]


class _Cancellation(ErrorResponse):
    type: Literal["cancellation"]
    characters_used: int = 0
    duration_seconds: float = 0.0


class _EndOfUtterance(ErrorResponse):
    type: Literal["end_of_utterance"]
    characters_used: int
    duration_seconds: float
    truncated: bool = False
    reason: str | None = None


_Message = Annotated[_ControlMessage | _Cancellation | _EndOfUtterance, Field(discriminator="type")]
_MESSAGE: TypeAdapter[_ControlMessage | _Cancellation | _EndOfUtterance] = TypeAdapter(_Message)


def _decode_message(message: str | bytes) -> _ControlMessage | _Cancellation | _EndOfUtterance:
    try:
        return _MESSAGE.validate_json(message)
    except ValidationError as exc:
        raise APIConnectionError(
            "Vakyam TTS returned an invalid WebSocket message", retryable=False
        ) from exc


def speech_result_from_terminal(
    data: _Cancellation | _EndOfUtterance,
) -> WebSocketSpeechResult:
    if isinstance(data, _Cancellation):
        return WebSocketSpeechResult(
            characters_used=data.characters_used,
            duration_seconds=data.duration_seconds,
            cancelled=True,
        )
    return WebSocketSpeechResult(
        characters_used=data.characters_used,
        duration_seconds=data.duration_seconds,
        truncated=data.truncated,
        truncation_reason=data.reason,
    )


class AsyncStreamingTTSSession:
    """Low-level async WebSocket session that streams PCM audio per utterance.

    Protocol:
      connected → config → configured →
      (text → binary* → end_of_utterance | cancel → binary* → cancellation)* →
      disconnect

    After ``cancel``, keep receiving until ``cancellation`` (drain trailing
    audio). Do not send the next ``text`` until then. The socket stays open.
    """

    def __init__(
        self,
        *,
        api_key: str,
        config: TTSSessionConfig,
        session: http.HttpSession,
        base_url: str | None = None,
        allow_insecure_base_url: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        validate_tts_options(
            model=config.model,
            language=config.language,
            sample_rate=config.sample_rate,
            speed=config.speed,
            voice=config.voice,
        )
        resolved_base = normalize_base_url(
            base_url or DEFAULT_BASE_URL, allow_insecure_base_url=allow_insecure_base_url
        )
        self._api_key = api_key
        self._config = config
        self._url = websocket_url(resolved_base)
        self._session = session
        self._connection: http.WebSocket | None = None
        self._utterance_active = False
        self.last_result: WebSocketSpeechResult | None = None

    @property
    def sample_rate(self) -> int:
        return int(self._config.sample_rate)

    @property
    def connected(self) -> bool:
        return self._connection is not None

    @property
    def utterance_active(self) -> bool:
        """True while waiting for ``end_of_utterance`` or ``cancellation``."""
        return self._utterance_active

    async def connect(self, conn_options: APIConnectOptions) -> None:
        if self._connection is not None:
            return
        self._connection = await self._session.ws_connect(
            self._url,
            headers={"Authorization": f"Bearer {self._api_key}", "User-Agent": USER_AGENT},
            conn_options=conn_options,
        )
        try:
            connected = await self._recv_control(conn_options.read_timeout)
            if connected.type != "connected":
                raise APIConnectionError("Unexpected WebSocket handshake response")
            await self._connection.send_json(self._config.to_wire_message())
            configured = await self._recv_control(conn_options.read_timeout)
            if configured.type == "error":
                raise_ws_error(configured)
            if configured.type != "configured":
                raise APIConnectionError("Unexpected WebSocket config response")
        except BaseException:
            await self.close()
            raise

    async def synthesize_stream(
        self, text: str, *, timeout: float | None = 10.0
    ) -> AsyncIterator[bytes]:
        """Send one utterance and yield raw audio until EOU or cancellation.

        If this coroutine is cancelled mid-stream (LiveKit barge-in), it sends
        ``cancel``, drains until ``cancellation``, then re-raises so the
        WebSocket can be reused.
        """
        ws = self._socket()
        payload = text_message(text)
        self.last_result = None
        self._utterance_active = True
        try:
            await ws.send_json(payload)
            while True:
                message = await self._recv(timeout)
                if isinstance(message, bytes):
                    if message:
                        yield message
                    continue
                if message.type == "error":
                    raise_ws_error(message)
                if message.type == "pong":
                    continue
                if isinstance(message, (_EndOfUtterance, _Cancellation)):
                    self.last_result = speech_result_from_terminal(message)
                    return
                raise APIConnectionError(f"Unexpected WebSocket message type: {message.type}")
        except asyncio.CancelledError:
            await asyncio.shield(self._abort_and_drain(timeout=timeout))
            raise
        finally:
            self._utterance_active = False

    async def cancel(
        self, *, drain: bool = True, timeout: float | None = 10.0
    ) -> WebSocketSpeechResult | None:
        """Send barge-in ``cancel`` and optionally drain until ``cancellation``."""
        await self._socket().send_json(cancel_message())
        if not drain:
            return None
        return await self._drain_until_terminal(timeout=timeout)

    async def ping(self, *, timeout: float | None = 10.0) -> bool:
        await self._socket().send_json(ping_message())
        response = await self._recv_control(timeout)
        if response.type == "error":
            raise_ws_error(response)
        if response.type != "pong":
            raise APIConnectionError(
                f"Unexpected Vakyam TTS WebSocket ping response: {response.type}"
            )
        return True

    async def close(self) -> None:
        if self._connection is None:
            return
        connection, self._connection = self._connection, None
        self._utterance_active = False
        with suppress(APIError):
            await connection.send_json(disconnect_message())
        await connection.close()

    def _socket(self) -> http.WebSocket:
        if self._connection is None:
            raise APIConnectionError("WebSocket session is not connected")
        return self._connection

    async def _recv(
        self, timeout: float | None
    ) -> bytes | _ControlMessage | _Cancellation | _EndOfUtterance:
        ws = self._socket()
        message = await ws.receive(timeout=timeout)
        if message.type is http.WebSocketMessageType.BINARY:
            return message.binary
        if message.type is http.WebSocketMessageType.TEXT:
            return _decode_message(message.text)
        if ws.close_code == _AUTH_FAILED:
            raise APIStatusError("Vakyam TTS authentication failed", status_code=401)
        raise ws.close_error("Vakyam TTS WebSocket closed")

    async def _recv_control(
        self, timeout: float | None
    ) -> _ControlMessage | _Cancellation | _EndOfUtterance:
        message = await self._recv(timeout)
        if isinstance(message, bytes):
            raise APIConnectionError("Unexpected binary WebSocket message", retryable=False)
        return message

    async def _abort_and_drain(self, *, timeout: float | None) -> None:
        """Best-effort cancel + drain after the consumer task was cancelled."""
        if self._connection is None:
            return
        with suppress(APIError):
            await self._connection.send_json(cancel_message())
            self.last_result = await self._drain_until_terminal(timeout=timeout)

    async def _drain_until_terminal(self, *, timeout: float | None) -> WebSocketSpeechResult:
        while True:
            message = await self._recv(timeout)
            if isinstance(message, bytes) or message.type == "pong":
                continue
            if message.type == "error":
                raise_ws_error(message)
            if isinstance(message, (_EndOfUtterance, _Cancellation)):
                result = speech_result_from_terminal(message)
                self.last_result = result
                return result
            raise APIConnectionError(
                f"Unexpected WebSocket message type while draining: {message.type}"
            )


# Vakyam's documented close code for a rejected API key: the shared close-code rule would retry it.
_AUTH_FAILED = 4001
