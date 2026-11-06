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

"""Smallest AI Lightning v4 text-to-speech over the ``/lightning-v4/live`` session.

Lightning v4 is a dedicated endpoint, held separately from the Lightning v3.1
family's ``/tts/live`` session implemented in :mod:`.tts`. Its live session is a
different wire protocol: connect-time query-string parameters instead of an
``init`` message, a turn-based ``speak``/``turn_start``/``turn_end`` exchange
with binary audio frames instead of v3.1's ``context_id``/``continue``
continuation protocol, and a session held for the whole call rather than a
socket serializing one request at a time. Because of that, it is implemented as
its own :class:`TTS`, named ``LightningV4TTS``, rather than a model option on
the existing :class:`~.tts.TTS`.

Lightning v4 is beta and gated per account; an account without access gets a
``model_access_denied`` error and the connection is closed. It is English-only
(``en``/``auto``) and does not support word timestamps.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlencode

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger

NUM_CHANNELS = 1
DEFAULT_BASE_URL = "wss://api.smallest.ai/waves/v1/lightning-v4/live"
DEFAULT_VOICE_ID = "brannock"
DEFAULT_SAMPLE_RATE = 48000

SAMPLE_RATES = (8000, 16000, 24000, 44100, 48000)
SPEED_MIN = 0.5
SPEED_MAX = 2.0

_READY_TIMEOUT = 10.0

# How long a barge-in waits for the interrupted turn's terminal event before
# giving up on the socket. `interrupt` is a state flip on the server, so this
# only ever covers a round trip; anything slower is a socket worth replacing.
_INTERRUPT_DRAIN_TIMEOUT = 0.5

# Session-level error codes the server reports with the connection still open
# (`{"event": "error", ...}`), vs. connect-time rejections that close it
# (`{"status": "error", ...}`) — both are surfaced through the same `_api_error`.
_FATAL_ERROR_CODES = frozenset({"model_access_denied", "invalid_voice", "invalid_query"})


@dataclass
class _TTSOptions:
    voice_id: str
    language: LanguageCode
    sample_rate: int
    speed: float
    content_filter: NotGivenOr[bool]
    content_filter_action: NotGivenOr[str]
    base_url: str


def _resolved_language(language: LanguageCode) -> str:
    """Lightning v4 is English-only; anything else falls back to `auto`
    (server-side detection) rather than being rejected client-side."""
    return "en" if language.language == "en" else "auto"


class LightningV4TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        voice_id: str = DEFAULT_VOICE_ID,
        language: str = "en",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        speed: float = 1.0,
        content_filter: NotGivenOr[bool] = NOT_GIVEN,
        content_filter_action: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = DEFAULT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a new instance of Smallest AI Lightning v4 TTS.

        Args:
            api_key: Your Smallest AI API key. Falls back to the
                ``SMALLEST_API_KEY`` environment variable.
            voice_id: A Lightning v4 voice id. Fetch the current catalogue from
                the Lightning v4 ``get_voices`` endpoint if you want a voice
                other than the default.
            language: ``"en"`` or anything else, which resolves to ``"auto"``
                (server-side detection) — Lightning v4 is English-only.
            sample_rate: Output sample rate in Hz, one of 8000, 16000, 24000,
                44100, 48000. Defaults to 48000.
            speed: Playback speed multiplier, 0.5-2.0.
            content_filter: Screen each turn's text before synthesis before
                sending it. If not given, the API default (off) applies.
            content_filter_action: ``"reject"`` drops a turn that matches the
                filter; ``"flag"`` synthesizes it and records the match. If
                not given, the API default (``"reject"``) applies.
            base_url: Override the Lightning v4 live-session WebSocket URL.
            http_session: Optional ``aiohttp.ClientSession`` to reuse.
        """
        if sample_rate not in SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {SAMPLE_RATES}, got {sample_rate}")
        if not SPEED_MIN <= speed <= SPEED_MAX:
            raise ValueError(f"speed must be within {SPEED_MIN}-{SPEED_MAX}, got {speed}")

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        smallest_api_key = api_key if is_given(api_key) else os.environ.get("SMALLEST_API_KEY")
        if not smallest_api_key:
            raise ValueError(
                "Smallest.ai API key is required, either as `api_key` argument or "
                "`SMALLEST_API_KEY` environment variable"
            )

        self._api_key = smallest_api_key
        self._opts = _TTSOptions(
            voice_id=voice_id,
            language=LanguageCode(language),
            sample_rate=sample_rate,
            speed=speed,
            content_filter=content_filter,
            content_filter_action=content_filter_action,
            base_url=base_url,
        )
        self._session = http_session
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=3600,
            mark_refreshed_on_get=False,
        )

    @property
    def provider(self) -> str:
        return "SmallestAI"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def _ws_url(self) -> str:
        params: dict[str, Any] = {
            "voice_id": self._opts.voice_id,
            "language": _resolved_language(self._opts.language),
            "sample_rate": self._opts.sample_rate,
            "output_format": "pcm",
            "speed": self._opts.speed,
        }
        if is_given(self._opts.content_filter):
            params["content_filter"] = str(self._opts.content_filter).lower()
        if is_given(self._opts.content_filter_action):
            params["content_filter_action"] = self._opts.content_filter_action
        return f"{self._opts.base_url}?{urlencode(params)}"

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        """Open the socket and wait for the session to be fully attached.

        The server sends `ready` twice: first to confirm admission (carrying
        `session_id`), then once a voice server is attached (carrying the
        negotiated `sample_rate`). `speak` is only valid after the second.
        """
        ws = await asyncio.wait_for(
            self._ensure_session().ws_connect(
                self._ws_url(),
                headers={"Authorization": f"Bearer {self._api_key}"},
            ),
            timeout,
        )
        try:
            dispatched = await self._recv_ready(ws, timeout)
            attached = await self._recv_ready(ws, timeout)

            negotiated_rate = attached.get("sample_rate")
            if negotiated_rate and negotiated_rate != self._opts.sample_rate:
                logger.warning(
                    "Lightning v4 negotiated %s Hz, expected %s Hz; using the negotiated rate",
                    negotiated_rate,
                    self._opts.sample_rate,
                )
                self._opts.sample_rate = negotiated_rate

            logger.debug(
                "Lightning v4 session ready", extra={"session_id": dispatched.get("session_id")}
            )
        except BaseException:
            try:
                await ws.close()
            except Exception:
                pass
            raise
        return ws

    async def _recv_ready(
        self, ws: aiohttp.ClientWebSocketResponse, timeout: float
    ) -> dict[str, Any]:
        msg = await asyncio.wait_for(ws.receive(), timeout)
        if msg.type is not aiohttp.WSMsgType.TEXT:
            raise APIError(f"Lightning v4 did not acknowledge the session: {msg.type}")
        data: dict[str, Any] = json.loads(msg.data)
        if data.get("event") != "ready":
            raise _api_error(data)
        return data

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        if ws.closed:
            return
        try:
            await ws.send_str(json.dumps({"event": "end"}))
        except Exception:
            pass
        finally:
            await ws.close()

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        content_filter: NotGivenOr[bool] = NOT_GIVEN,
        content_filter_action: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Update TTS options.

        ``voice_id``, ``language``, ``speed``, ``content_filter`` and
        ``content_filter_action`` are all negotiated in the connection's query
        string, so changing any of them invalidates the pool: a pooled socket
        would otherwise keep serving the old ones for the rest of its life.
        """
        if is_given(speed) and not SPEED_MIN <= speed <= SPEED_MAX:
            raise ValueError(f"speed must be within {SPEED_MIN}-{SPEED_MAX}, got {speed}")

        changed = False
        if is_given(voice_id):
            self._opts.voice_id = voice_id
            changed = True
        if is_given(language):
            self._opts.language = LanguageCode(language)
            changed = True
        if is_given(speed):
            self._opts.speed = speed
            changed = True
        if is_given(content_filter):
            self._opts.content_filter = content_filter
            changed = True
        if is_given(content_filter_action):
            self._opts.content_filter_action = content_filter_action
            changed = True

        if changed:
            self._pool.invalidate()

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        raise NotImplementedError(
            "Lightning v4 is only available over its live session; use `stream()`, or "
            "`SmallestAI.TTS` (Lightning v3.1) for one-shot HTTP synthesis."
        )

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        return SynthesizeStream(tts=self, conn_options=conn_options)

    def prewarm(self) -> None:
        self._pool.prewarm()

    async def aclose(self) -> None:
        await self._pool.aclose()


class SynthesizeStream(tts.SynthesizeStream):
    """Stream a whole segment as one Lightning v4 turn.

    Lightning v4's `speak` takes one complete utterance, not incremental
    fragments — there is no continuation protocol on this endpoint, unlike
    Lightning v3.1. So unlike a token-streaming provider, this accumulates the
    whole segment before sending anything, then sends exactly one `speak` and
    waits for `turn_start`, binary audio, and `turn_end`.
    """

    def __init__(self, *, tts: LightningV4TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: LightningV4TTS = tts
        self._pool = tts._pool
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        turn_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=turn_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=turn_id)

        parts: list[str] = []
        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                continue
            parts.append(data)
        text = "".join(parts)

        if not text.strip():
            output_emitter.end_segment()
            return

        speak_sent = False

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal speak_sent
            self._mark_started()
            await ws.send_str(json.dumps({"event": "speak", "turn_id": turn_id, "text": text}))
            speak_sent = True

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            while True:
                msg = await ws.receive(timeout=self._conn_options.timeout)
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Lightning v4 connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )
                if msg.type is aiohttp.WSMsgType.BINARY:
                    output_emitter.push(msg.data)
                    continue
                if msg.type is not aiohttp.WSMsgType.TEXT:
                    continue

                data = json.loads(msg.data)
                event = data.get("event")
                if event == "turn_start":
                    continue
                elif event in ("turn_end", "interrupted"):
                    if data.get("turn_id") != turn_id:
                        # Belongs to a turn this stream already abandoned.
                        continue
                    output_emitter.end_segment()
                    return
                elif event == "error" or data.get("status") == "error":
                    raise _api_error(data)
                else:
                    logger.warning("unexpected Lightning v4 message %s", data)

        async def interrupt_and_drain(ws: aiohttp.ClientWebSocketResponse) -> None:
            """Barge in and consume the turn's terminal event before reuse."""
            await ws.send_str(json.dumps({"event": "interrupt"}))
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Lightning v4 connection closed while interrupting a turn",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )
                if msg.type is not aiohttp.WSMsgType.TEXT:
                    continue
                data = json.loads(msg.data)
                if data.get("turn_id") != turn_id:
                    continue
                if data.get("event") in ("interrupted", "turn_end"):
                    return
                if data.get("event") == "error" or data.get("status") == "error":
                    raise _api_error(data)

        cancelled: asyncio.CancelledError | None = None
        try:
            async with self._pool.connection(timeout=self._conn_options.timeout) as ws:
                self._acquire_time = self._pool.last_acquire_time
                self._connection_reused = self._pool.last_connection_reused
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                ]
                try:
                    await asyncio.gather(*tasks)
                except asyncio.CancelledError as e:
                    await utils.aio.gracefully_cancel(*tasks)
                    if not speak_sent:
                        # Cancelled before the turn ever started; nothing to interrupt.
                        raise
                    try:
                        # Bounded: a socket that does not answer must not hold up the
                        # barge-in that is waiting on this teardown.
                        await asyncio.wait_for(
                            interrupt_and_drain(ws), timeout=_INTERRUPT_DRAIN_TIMEOUT
                        )
                    except asyncio.CancelledError:
                        try:
                            await ws.close()
                        except Exception:
                            pass
                        raise
                    except BaseException as drain_error:
                        # A session that failed to settle cleanly is not safe to hand
                        # back to the pool — a still-active turn would leak audio into
                        # whatever the next stream sends.
                        logger.debug(
                            "Lightning v4 interrupt handshake failed",
                            extra={"error": str(drain_error)},
                        )
                        try:
                            await ws.close()
                        except Exception:
                            pass
                        raise e from drain_error
                    # Exit the pool context normally so this clean, drained session
                    # (and its accumulated server-side context) is kept for the next
                    # turn, then preserve caller cancellation.
                    cancelled = e
                except BaseException:
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    raise
                finally:
                    await utils.aio.gracefully_cancel(*tasks)
            if cancelled is not None:
                raise cancelled
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


def _api_error(data: dict[str, Any]) -> APIError:
    """Unwrap either of Lightning v4's two error shapes: a session-level
    `{"event": "error", "code", "message"}` (connection stays open), or a
    connect-time rejection `{"status": "error", "error": {"code", "message"}}`."""
    nested = data.get("error")
    error: dict[str, Any] = nested if isinstance(nested, dict) else data
    code = error.get("code")
    message = error.get("message") or "Lightning v4 returned an error"
    return APIError(
        f"{code}: {message}" if code else message,
        body=data,
        retryable=code not in _FATAL_ERROR_CODES,
    )
