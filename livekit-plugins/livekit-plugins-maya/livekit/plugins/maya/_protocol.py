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
import base64
import binascii
import contextlib
import json
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from livekit.agents import APIConnectionError, APIError, APIStatusError, APITimeoutError, tts

SAMPLE_RATE = 24000
NUM_CHANNELS = 1


@dataclass
class Turn:
    context_id: str
    submitted: bool = False
    input_closed: bool = False
    terminal: bool = False
    audio_bytes: int = 0
    response_started_at: float | None = None
    activity: asyncio.Event = field(default_factory=asyncio.Event, repr=False)


async def receive_json(
    ws: aiohttp.ClientWebSocketResponse, timeout: float | None
) -> dict[str, Any]:
    if timeout is not None and timeout <= 0:
        raise APITimeoutError()
    try:
        message = await ws.receive(timeout=timeout)
    except asyncio.TimeoutError:
        raise APITimeoutError() from None
    if message.type != aiohttp.WSMsgType.TEXT:
        raise APIConnectionError("Maya websocket closed or returned a non-JSON message")
    try:
        data = json.loads(message.data)
    except (ValueError, TypeError):
        raise APIError("Maya returned malformed JSON", retryable=False) from None
    if not isinstance(data, dict):
        raise APIError("Maya returned a non-object JSON message", retryable=False)
    return data


def validate_metadata(data: dict[str, Any]) -> None:
    if data.get("type") != "metadata":
        raise APIError("Maya rejected the connection settings", retryable=False)
    if (
        type(data.get("sample_rate")) is not int
        or data["sample_rate"] != SAMPLE_RATE
        or type(data.get("channels")) is not int
        or data["channels"] != NUM_CHANNELS
        or data.get("encoding") != "pcm_s16le"
    ):
        raise APIError(
            "Unsupported Maya audio format: expected 24000 Hz mono pcm_s16le", retryable=False
        )


async def send_text(
    ws: aiohttp.ClientWebSocketResponse, turn: Turn, text: str, *, more: bool, timeout: float
) -> None:
    # Mark before yielding: a partially completed send may already reach the server.
    turn.submitted = True
    if not more:
        turn.input_closed = True
    # New text needs response progress, and the final closer needs a terminal.
    # More input must not keep extending a wait that has received no audio.
    if turn.response_started_at is None or not more:
        turn.response_started_at = asyncio.get_running_loop().time()
    turn.activity.set()
    await asyncio.wait_for(
        ws.send_json(
            {"type": "text", "context_id": turn.context_id, "text": text, "continue": more}
        ),
        timeout,
    )


async def _receive_turn_json(
    ws: aiohttp.ClientWebSocketResponse, turn: Turn, timeout: float
) -> dict[str, Any]:
    # Keep one receive alive while the sender changes the deadline. Cancelling
    # and recreating a receive on each input event could lose an incoming frame.
    receive = asyncio.create_task(receive_json(ws, None))
    try:
        while True:
            turn.activity.clear()
            remaining = (
                None
                if turn.response_started_at is None
                else turn.response_started_at + timeout - asyncio.get_running_loop().time()
            )
            if remaining is not None and remaining <= 0:
                raise APITimeoutError()
            changed = asyncio.create_task(turn.activity.wait())
            try:
                done, _ = await asyncio.wait(
                    (receive, changed), timeout=remaining, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                changed.cancel()
                await asyncio.gather(changed, return_exceptions=True)
            if receive in done:
                return receive.result()
            if not done:
                raise APITimeoutError()
    finally:
        receive.cancel()
        await asyncio.gather(receive, return_exceptions=True)


async def receive_audio(
    ws: aiohttp.ClientWebSocketResponse, turn: Turn, emitter: tts.AudioEmitter, timeout: float
) -> None:
    carry = b""
    loop = asyncio.get_running_loop()
    while True:
        data = await _receive_turn_json(ws, turn, timeout)
        kind, context = data.get("type"), data.get("context_id")
        if kind == "error" and context in (None, turn.context_id):
            # Do not echo service bodies: they can contain submitted text or secrets.
            raise APIError("Maya reported a synthesis error", retryable=False)
        if context != turn.context_id:
            continue  # Includes unscoped audio and stale audio/terminators.
        if kind == "audio":
            try:
                audio = base64.b64decode(data["audio"], validate=True)
            except (KeyError, ValueError, TypeError, binascii.Error):
                raise APIError("Maya returned invalid base64 audio", retryable=False) from None
            if not audio:
                continue
            # v2 has no per-sentence completion ACK. Once audio has progressed,
            # an open input may simply be waiting for the LLM. Re-arm on new
            # text; after the closer, every audio gap/end wait stays bounded.
            turn.response_started_at = loop.time() if turn.input_closed else None
            turn.audio_bytes += len(audio)
            carry += audio
            complete = len(carry) - len(carry) % 2
            if complete:
                emitter.push(carry[:complete])
                carry = carry[complete:]
        elif kind == "end":
            turn.terminal = True
            if not turn.input_closed:
                raise APIError("Maya ended the turn before text input closed", retryable=False)
            if carry or not turn.audio_bytes:
                raise APIError("Maya ended with truncated or empty PCM audio", retryable=False)
            return
        elif kind == "cancelled":
            turn.terminal = True
            # LiveKit treats 499 as a graceful cancellation, even with no audio.
            raise APIStatusError("Maya turn cancelled", status_code=499, retryable=False)


async def cancel_unfinished(ws: aiohttp.ClientWebSocketResponse, turn: Turn) -> None:
    if turn.submitted and not turn.terminal and not ws.closed:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                ws.send_json({"type": "cancel", "context_id": turn.context_id}), timeout=1.0
            )
