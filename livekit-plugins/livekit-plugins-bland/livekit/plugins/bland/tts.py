# Copyright 2023 LiveKit, Inc.
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
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger

DEFAULT_BASE_URL = "https://api.bland.ai/v2"
DEFAULT_VOICE_ID = "2f29fdbb-c55e-4add-9c7c-93437ebf379d"
DEFAULT_SAMPLE_RATE = 48000
NUM_CHANNELS = 1

SAMPLE_RATES = (8000, 16000, 24000, 44100, 48000)

# Bland ends a session that goes 60s without a client message. The pool clock is
# refreshed on every acquire, so this expires a socket that has sat unused across a
# long conversational gap rather than handing back one the server already dropped.
_MAX_SESSION_DURATION = 50

# How long a barge-in waits for the cancelled turn's terminal before giving up on
# the socket. Answering `cancel` is a state flip on the server, so this only ever
# covers a round trip; anything slower is a socket worth replacing. Deliberately
# not derived from `conn_options.timeout` — that budget is for establishing a
# connection, and spending it here stalls the next turn behind a dead one.
_CANCEL_DRAIN_TIMEOUT = 0.5

# Errors that a retry cannot fix: bad credentials, a bad request, or an account that
# needs attention. Everything else (synthesis failures, a busy concurrency pool) is
# worth another attempt.
_FATAL_ERROR_CODES = frozenset(
    {
        "AUTH_FAILED",
        "INSUFFICIENT_CREDITS",
        "ORG_DELETED",
        "ORG_STRIPE_OVERDUE",
        "ORG_SUSPENDED",
        "USER_BANNED",
        "already_initialized",
        "context_overflow",
        "init_required",
        "insufficient_credits",
        "invalid_message",
        "invalid_request",
        "unsupported_encoding",
        "unsupported_sample_rate",
        "unsupported_voice",
        "voice_not_found",
        "voice_not_live",
    }
)


@dataclass
class _TTSOptions:
    voice_id: str
    sample_rate: int
    expressiveness: NotGivenOr[float]
    stability: NotGivenOr[float]
    base_url: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice_id: str = DEFAULT_VOICE_ID,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        expressiveness: NotGivenOr[float] = NOT_GIVEN,
        stability: NotGivenOr[float] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = DEFAULT_BASE_URL,
        streaming: bool = True,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a new instance of the Bland TTS.

        Args:
            voice_id: Bland voice UUID; names are not accepted. Defaults to a ``BTTS_V3``
                voice. ``BTTS_V2`` voices work, but the controls below are calibrated for
                ``BTTS_V3``.
            sample_rate: Output sample rate in Hz, one of 8000, 16000, 24000, 44100, 48000.
                Defaults to 48000, the rate ``BTTS_V3`` renders natively.
            expressiveness: 0.0-1.0. Higher is more varied intonation.
            stability: 0.0-1.0. Higher is more consistent between renders.
            api_key: Bland API key. Falls back to the ``BLAND_API_KEY`` environment variable.
            base_url: Override the Bland API base URL.
            streaming: Stream text into one realtime WebSocket session, which is what a
                voice agent wants: audio starts before the sentence is finished, and a
                barge-in cancels the turn in place. Set False to synthesize each
                utterance with a single HTTP request instead — no session is held open,
                and no concurrency slot with it, which suits a pipeline that speaks
                rarely. ``synthesize()`` uses HTTP either way.
            http_session: Optional ``aiohttp.ClientSession`` to reuse.
        """
        if sample_rate not in SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {SAMPLE_RATES}, got {sample_rate}")

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=streaming),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        bland_api_key = api_key if is_given(api_key) else os.environ.get("BLAND_API_KEY")
        if not bland_api_key:
            raise ValueError(
                "Bland API key is required, either as `api_key` argument or "
                "`BLAND_API_KEY` environment variable"
            )

        self._api_key = bland_api_key
        self._opts = _TTSOptions(
            voice_id=voice_id,
            sample_rate=sample_rate,
            expressiveness=expressiveness,
            stability=stability,
            base_url=base_url.rstrip("/"),
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        # No pool when streaming is off: nothing would ever check a socket out of it,
        # and an open session holds a concurrency slot for as long as it lives.
        self._pool = (
            utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
                connect_cb=self._connect_ws,
                close_cb=self._close_ws,
                max_session_duration=_MAX_SESSION_DURATION,
                mark_refreshed_on_get=True,
            )
            if streaming
            else None
        )

    @property
    def provider(self) -> str:
        return "Bland"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        """Open a session and hold it at `ready`, so a turn starts on the first `speak`."""
        try:
            ws = await asyncio.wait_for(
                self._ensure_session().ws_connect(
                    _ws_url(self._opts.base_url),
                    headers={"Authorization": f"Bearer {self._api_key}"},
                ),
                timeout,
            )
        except aiohttp.WSServerHandshakeError as e:
            # Bland answers an upgrade carrying no credential with a real 401 rather
            # than accepting and closing, so this is the path a bad key takes.
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=e.headers.get("x-request-id") if e.headers else None,
            ) from e
        try:
            init: dict[str, Any] = {
                "type": "init",
                "voice": self._opts.voice_id,
                "audio": {"encoding": "pcm_s16le", "sample_rate": self._opts.sample_rate},
            }
            if controls := _controls(self._opts):
                init["controls"] = controls
            await ws.send_str(json.dumps(init))

            msg = await asyncio.wait_for(ws.receive(), timeout)
            if msg.type is not aiohttp.WSMsgType.TEXT:
                raise APIError(f"Bland did not acknowledge init: {msg.type}")

            data = json.loads(msg.data)
            if data.get("type") != "ready":
                raise _api_error(data)
            if (
                data.get("encoding") != "pcm_s16le"
                or data.get("sample_rate") != self._opts.sample_rate
            ):
                raise APIError(
                    "Bland acknowledged an unexpected audio format",
                    body=data,
                    retryable=False,
                )
        except BaseException:
            try:
                await ws.close()
            except Exception:
                pass
            raise

        logger.debug("Bland TTS session ready", extra={"session_id": data.get("session_id")})
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Settle the session rather than dropping the socket, so usage is reconciled once."""
        if ws.closed:
            return
        try:
            await ws.send_str(json.dumps({"type": "close"}))
            await asyncio.wait_for(ws.receive(), timeout=1.0)
        except Exception as e:
            logger.debug("Bland TTS close handshake skipped", extra={"error": str(e)})
        finally:
            await ws.close()

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        expressiveness: NotGivenOr[float] = NOT_GIVEN,
        stability: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        changed = False
        if is_given(voice_id):
            self._opts.voice_id = voice_id
            changed = True
        if is_given(expressiveness):
            self._opts.expressiveness = expressiveness
            changed = True
        if is_given(stability):
            self._opts.stability = stability
            changed = True

        if changed:
            # `init` fixes the voice and controls for the life of a session, so a pooled
            # socket would keep serving the old ones.
            if self._pool is not None:
                self._pool.invalidate()

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        if self._pool is None:
            raise RuntimeError(
                "streaming is disabled on this Bland TTS instance; construct it with "
                "`streaming=True`, or wrap it in a `tts.StreamAdapter`"
            )
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        if self._pool is not None:
            self._pool.prewarm()

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        if self._pool is not None:
            await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize a complete string over HTTP."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        body: dict[str, Any] = {
            "text": self._input_text,
            "voice": self._opts.voice_id,
            "audio": {"encoding": "pcm_s16le", "sample_rate": self._opts.sample_rate},
        }
        if controls := _controls(self._opts):
            body["controls"] = controls

        try:
            async with self._tts._ensure_session().post(
                f"{self._opts.base_url}/tts",
                headers={
                    "authorization": self._tts._api_key,
                    "content-type": "application/json",
                },
                json=body,
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
            ) as resp:
                if resp.status != 200:
                    raise APIStatusError(
                        message=await _error_message(resp),
                        status_code=resp.status,
                        request_id=resp.headers.get("x-request-id"),
                        body=None,
                    )

                output_emitter.initialize(
                    request_id=resp.headers.get("x-request-id") or utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm",
                )

                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

                output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Stream text into a session and receive audio as it is rendered.

    Bland accumulates the text deltas server-side and picks its own synthesis
    boundaries, so tokens go out as they arrive: no sentence tokenizer, no character
    threshold, and no flush after every fragment.
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        # `TTS.stream()` refuses to build one of these without a pool, so this is
        # always present — bound once here rather than re-narrowed at each use.
        assert tts._pool is not None
        self._pool = tts._pool
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        # One stream is one segment, which is one Bland turn: the socket carries a
        # single turn at a time and the framework opens a stream per segment.
        context_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=context_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=context_id)
        input_sent = asyncio.Event()
        text_sent = False

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal text_sent
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    continue

                text_sent = True
                self._mark_started()
                await ws.send_str(
                    json.dumps({"type": "speak", "context_id": context_id, "text": data})
                )
                input_sent.set()

            if not text_sent:
                output_emitter.end_segment()
                input_sent.set()
                return

            # Bland holds a short tail back waiting for more context; this releases it.
            await ws.send_str(json.dumps({"type": "end_of_turn", "context_id": context_id}))
            input_sent.set()

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            await input_sent.wait()
            if not text_sent:
                return
            while True:
                msg = await ws.receive(timeout=self._conn_options.timeout)
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Bland connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                if msg.type is aiohttp.WSMsgType.BINARY:
                    output_emitter.push(msg.data)
                    continue

                if msg.type is not aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Bland message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                event = data.get("type")
                if event == "utterance_end":
                    # A stale terminator can only belong to a turn this stream already
                    # abandoned, so it is not this turn's boundary.
                    if data.get("context_id") != context_id:
                        continue
                    if (reason := data.get("reason")) != "complete":
                        raise APIError(f"Bland turn ended as {reason}", body=data)
                    output_emitter.end_segment()
                    return
                elif event == "utterance_start":
                    continue
                elif event == "error":
                    raise _api_error(data)
                else:
                    logger.warning("unexpected Bland message %s", data)

        async def cancel_and_drain(ws: aiohttp.ClientWebSocketResponse) -> None:
            """Cancel the active turn and consume its terminal event before reuse."""
            await ws.send_str(json.dumps({"type": "cancel", "context_id": context_id}))
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Bland connection closed while cancelling a turn",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )
                if msg.type is aiohttp.WSMsgType.BINARY:
                    continue
                if msg.type is not aiohttp.WSMsgType.TEXT:
                    continue

                data = json.loads(msg.data)
                if data.get("context_id") != context_id:
                    continue
                if data.get("type") == "utterance_end":
                    if data.get("reason") in ("cancelled", "complete"):
                        return
                    raise APIError("Bland did not cancel the turn cleanly", body=data)
                if data.get("type") == "error":
                    # An admission failure creates no turn and emits no terminal, so
                    # there is nothing left to drain to. Returning here would hand the
                    # socket back reusable — but from this side an admission refusal
                    # is indistinguishable from a mid-turn error whose terminal is
                    # still in flight, and that terminal would surface against the
                    # next, unrelated turn. Raising closes the socket, which costs one
                    # reconnect on a path that has already failed and keeps turns from
                    # contaminating each other.
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
                    turn_was_sent = input_sent.is_set()
                    input_sent.set()
                    await utils.aio.gracefully_cancel(*tasks)
                    if text_sent and not turn_was_sent:
                        # Cancellation interrupted the first write, so whether the
                        # server owns this context is unknowable. Do not reuse it.
                        try:
                            await ws.close()
                        except Exception:
                            pass
                        raise
                    if text_sent:
                        try:
                            # Bounded: a socket that does not answer must not hold up
                            # the barge-in that is waiting on this teardown.
                            await asyncio.wait_for(
                                cancel_and_drain(ws), timeout=_CANCEL_DRAIN_TIMEOUT
                            )
                        except asyncio.CancelledError:
                            try:
                                await ws.close()
                            except Exception:
                                pass
                            raise
                        except BaseException as drain_error:
                            # Tidying up failed, but this is still a cancellation, and
                            # it has to leave as one. Letting the drain's own error
                            # escape would turn a barge-in into a retryable API error:
                            # the framework replays the buffered text, so the caller
                            # hears the interrupted sentence a second time — and when
                            # the cancel came from `aclose()` before `end_input()`, the
                            # replay waits forever on an input channel nothing will
                            # close, with the one cancellation already spent.
                            logger.debug(
                                "Bland cancel handshake failed",
                                extra={"error": str(drain_error)},
                            )
                            try:
                                await ws.close()
                            except Exception:
                                pass
                            raise e from drain_error
                    # Exit the pool context normally so this clean, drained session is
                    # returned for the next turn, then preserve caller cancellation.
                    cancelled = e
                except BaseException:
                    # A failed stream cannot safely return a socket with unread turn
                    # state to the pool.
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    raise
                finally:
                    input_sent.set()
                    await utils.aio.gracefully_cancel(*tasks)
            if cancelled is not None:
                raise cancelled
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


def _ws_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    return f"{base_url.replace('https://', 'wss://', 1).replace('http://', 'ws://', 1)}/tts/ws"


def _controls(opts: _TTSOptions) -> dict[str, float]:
    controls: dict[str, float] = {}
    if is_given(opts.expressiveness):
        controls["expressiveness"] = opts.expressiveness
    if is_given(opts.stability):
        controls["stability"] = opts.stability
    return controls


def _api_error(data: dict[str, Any]) -> APIError:
    code = data.get("code")
    message = data.get("message") or "Bland returned an error"
    return APIError(
        f"{code}: {message}" if code else message,
        body=data,
        retryable=code not in _FATAL_ERROR_CODES,
    )


async def _error_message(resp: aiohttp.ClientResponse) -> str:
    """Unwrap the v2 ``{"error": {"code", "message"}}`` envelope, falling back to the raw body."""
    try:
        payload = await resp.json()
    except Exception:
        return resp.reason or "request failed"

    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        code, message = error.get("code"), error.get("message")
        if code and message:
            return f"{code}: {message}"
        if message:
            return str(message)
    return str(payload)
