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
import json
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from livekit.agents import APIConnectionError, APIStatusError, __version__ as livekit_version

from ._utils import (
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

    def to_wire_message(self) -> dict[str, Any]:
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


def speech_result_from_terminal(data: dict[str, Any]) -> WebSocketSpeechResult:
    msg_type = data.get("type")
    if msg_type == "cancellation":
        return WebSocketSpeechResult(
            characters_used=int(data.get("characters_used", 0)),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            cancelled=True,
        )
    if msg_type == "end_of_utterance":
        return WebSocketSpeechResult(
            characters_used=int(data["characters_used"]),
            duration_seconds=float(data["duration_seconds"]),
            truncated=bool(data.get("truncated", False)),
            truncation_reason=(str(data["reason"]) if data.get("reason") is not None else None),
        )
    raise APIConnectionError(f"Not a terminal WebSocket message type: {msg_type}")


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
        self._connection: Any = None
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

    async def __aenter__(self) -> AsyncStreamingTTSSession:
        await self.connect()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: object
    ) -> None:
        await self.close()

    async def connect(self) -> None:
        if self._connection is not None:
            return
        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:
            raise APIConnectionError("websockets is required for Vakyam TTS streaming") from exc

        try:
            self._connection = await connect(
                self._url,
                additional_headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "User-Agent": USER_AGENT,
                },
            )
            connected = json.loads(await self._connection.recv())
            if connected.get("type") != "connected":
                raise APIConnectionError("Unexpected WebSocket handshake response")

            await self._connection.send(json.dumps(self._config.to_wire_message()))
            configured = json.loads(await self._connection.recv())
            if configured.get("type") == "error":
                raise_ws_error(configured)
            if configured.get("type") != "configured":
                raise APIConnectionError("Unexpected WebSocket config response")
        except (APIStatusError, APIConnectionError):
            await self.close()
            raise
        except Exception as exc:
            await self.close()
            raise APIConnectionError(f"Vakyam TTS WebSocket connection failed: {exc}") from exc

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Send one utterance and yield raw audio until EOU or cancellation.

        If this coroutine is cancelled mid-stream (LiveKit barge-in), it sends
        ``cancel``, drains until ``cancellation``, then re-raises so the
        WebSocket can be reused.
        """
        if self._connection is None:
            raise APIConnectionError("WebSocket session is not connected")

        self.last_result = None
        self._utterance_active = True
        await self._connection.send(json.dumps(text_message(text)))

        try:
            while True:
                message = await self._connection.recv()
                if isinstance(message, bytes):
                    if message:
                        yield message
                    continue

                data = json.loads(message)
                msg_type = data.get("type")
                if msg_type == "error":
                    raise_ws_error(data)
                if msg_type == "pong":
                    continue
                if msg_type in {"end_of_utterance", "cancellation"}:
                    self.last_result = speech_result_from_terminal(data)
                    return
                raise APIConnectionError(f"Unexpected WebSocket message type: {msg_type}")
        except asyncio.CancelledError:
            await asyncio.shield(self._abort_and_drain())
            raise
        finally:
            self._utterance_active = False

    async def cancel(self, *, drain: bool = True) -> WebSocketSpeechResult | None:
        """Send barge-in ``cancel`` and optionally drain until ``cancellation``."""
        if self._connection is None:
            raise APIConnectionError("WebSocket session is not connected")

        await self._connection.send(json.dumps(cancel_message()))
        if not drain:
            return None
        return await self._drain_until_terminal()

    async def ping(self) -> bool:
        if self._connection is None:
            raise APIConnectionError("WebSocket session is not connected")
        await self._connection.send(json.dumps(ping_message()))
        response = json.loads(await self._connection.recv())
        return bool(response.get("type") == "pong")

    async def close(self) -> None:
        if self._connection is None:
            return
        try:
            with suppress(Exception):
                await self._connection.send(json.dumps(disconnect_message()))
            with suppress(Exception):
                await self._connection.close()
        finally:
            self._connection = None
            self._utterance_active = False

    async def _abort_and_drain(self) -> None:
        """Best-effort cancel + drain after the consumer task was cancelled."""
        if self._connection is None:
            return
        with suppress(Exception):
            await self._connection.send(json.dumps(cancel_message()))
        with suppress(Exception):
            self.last_result = await self._drain_until_terminal()

    async def _drain_until_terminal(self) -> WebSocketSpeechResult:
        if self._connection is None:
            raise APIConnectionError("WebSocket session is not connected")

        while True:
            message = await self._connection.recv()
            if isinstance(message, bytes):
                continue
            data = json.loads(message)
            msg_type = data.get("type")
            if msg_type == "error":
                raise_ws_error(data)
            if msg_type == "pong":
                continue
            if msg_type in {"end_of_utterance", "cancellation"}:
                result = speech_result_from_terminal(data)
                self.last_result = result
                return result
            raise APIConnectionError(
                f"Unexpected WebSocket message type while draining: {msg_type}"
            )
