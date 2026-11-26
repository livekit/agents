# Copyright 2026 LiveKit, Inc.
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
import functools
import json
import os
import re
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import Any

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
from .version import __version__

_WS_URL = "wss://fi.production.flamapis.com/flam-voice-svc/api/v1/public/stream"
_API_BASE = "https://fi.production.flamapis.com/flam-voice-svc/api/v1/public"
_SUBPROTOCOL = "finesse.v1"

# A direct session is minted with a generous character budget so a single call rarely exhausts
# it; the server clamps this to its own maximum. When a session does run out (or its ~1 hour TTL
# expires) the next synthesis mints a fresh one.
_SESSION_BUDGET_CHARS = 200_000

# The server ends a direct session roughly an hour after it is minted, regardless of use. A
# direct pool caps its connection's lifetime below that (and does not refresh the clock on reuse),
# so a long-lived agent recycles the socket — minting a fresh session — before the server would
# end it mid-reply. Kept comfortably under the ~3600 s server TTL to absorb an in-flight turn.
_DIRECT_MAX_SESSION_DURATION = 3300

# Finesse synthesizes at 24 kHz and upsamples above it. Below it, it produces only 8 kHz
# telephony audio, so the rates it returns are 8000 and 24000 to 48000.
NATIVE_SAMPLE_RATE = 24000
_TELEPHONY_SAMPLE_RATE = 8000
_MAX_SAMPLE_RATE = 48000

# Finesse uses base ISO 639 codes, except where it names a macrolanguage member.
_LANGUAGE_OVERRIDES = {"ar": "arb", "ne": "npi", "no": "nb", "or": "ory"}
# Arabic regions with their own Finesse dialect: Egyptian, and Gulf (Khaleeji).
_ARABIC_REGIONS = {"EG": "arz", "AE": "afb", "KW": "afb", "QA": "afb", "BH": "afb", "OM": "afb"}

# Text frames over 1 MiB close the socket. Frames are sent as UTF-8 (ensure_ascii=False), so a
# character costs at most 6 bytes (a JSON-escaped control character) and this stays far below it.
_MAX_TEXT_CHARS = 100_000

# Finesse error codes are snake_case identifiers; anything else is not put in an error message.
_ERROR_CODE_RE = re.compile(r"^[a-z0-9_]{1,64}$")

# Error codes that fail the same way on retry.
_NON_RETRYABLE_CODES = frozenset(
    {
        "insufficient_credits",
        "quota_exhausted",
        "payment_required",
        "voice_not_found",
        "voice_not_allowed",
        "invalid_language",
        "invalid_request",
        "invalid_message",
        "text_required",
        "token_invalid",
        "key_expired",
        "insufficient_scope",
        "feature_disabled",
        "plan_limit_reached",
    }
)

# On a direct session these codes signal the session's own budget is spent (not the account's
# credits), so a freshly minted session continues; see _event_error.
_SESSION_RENEWABLE_CODES = frozenset({"insufficient_credits"})

# Close codes that fail the same way on a new connection: out of credits, frame too large.
# Any other close (normal, going away, service restart, engine failure) is retried.
_NON_RETRYABLE_CLOSE_CODES = frozenset({1008, 1009})


def _finesse_language(language: LanguageCode) -> str:
    base = language.language
    if base == "ar" and language.region in _ARABIC_REGIONS:
        return _ARABIC_REGIONS[language.region]
    return _LANGUAGE_OVERRIDES.get(base, base)


@dataclass
class _TTSOptions:
    api_key: str
    voice: str
    language: LanguageCode
    sample_rate: int
    num_step: int | None
    guidance_scale: float | None
    length_scale: float | None
    idle_flush_ms: int | None
    ws_url: str
    api_base: str
    use_direct_session: bool


def _validate(
    *,
    voice: NotGivenOr[str] = NOT_GIVEN,
    sample_rate: NotGivenOr[int] = NOT_GIVEN,
    num_step: NotGivenOr[int | None] = NOT_GIVEN,
    guidance_scale: NotGivenOr[float | None] = NOT_GIVEN,
    length_scale: NotGivenOr[float | None] = NOT_GIVEN,
    idle_flush_ms: NotGivenOr[int | None] = NOT_GIVEN,
) -> None:
    if is_given(voice) and not voice.strip():
        raise ValueError("Finesse voice is required: pass the id of a ready voice")
    if (
        is_given(sample_rate)
        and sample_rate != _TELEPHONY_SAMPLE_RATE
        and not NATIVE_SAMPLE_RATE <= sample_rate <= _MAX_SAMPLE_RATE
    ):
        raise ValueError(
            f"sample_rate must be {_TELEPHONY_SAMPLE_RATE}, or between {NATIVE_SAMPLE_RATE} "
            f"and {_MAX_SAMPLE_RATE} Hz"
        )
    if is_given(num_step) and num_step is not None and not 1 <= num_step <= 32:
        raise ValueError("num_step must be between 1 and 32")
    if is_given(guidance_scale) and guidance_scale is not None and not 0 <= guidance_scale <= 10:
        raise ValueError("guidance_scale must be between 0 and 10")
    if is_given(length_scale) and length_scale is not None and not 0 < length_scale <= 5:
        raise ValueError("length_scale must be greater than 0 and at most 5")
    if is_given(idle_flush_ms) and idle_flush_ms is not None and not 200 <= idle_flush_ms <= 5000:
        raise ValueError("idle_flush_ms must be between 200 and 5000")


class _DirectConnectionPool(utils.ConnectionPool[aiohttp.ClientWebSocketResponse]):
    """A pool that closes a retired connection before opening its replacement.

    Every direct connection holds a Finesse realtime session, and each open session occupies one
    of the account's concurrent-stream slots. When a session ages out, the base pool queues its
    socket for closing but only drains that queue on the next checkout, so the obsolete socket
    keeps its slot until then — and if the agent falls silent right after a rollover, until the
    server expires the session. Draining first releases the aged-out session's slot before the
    replacement is minted, so a rollover never transiently needs two slots for one voice.

    Only the direct pool needs this; the relayed pool's sockets carry no per-session slot.
    """

    async def _connect(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        # get() queues an expired socket onto _to_close just before calling _connect; close it
        # now, before the replacement handshake, rather than deferring to the next checkout.
        await self._drain_to_close()
        return await super()._connect(timeout)


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str,
        api_key: str | None = None,
        language: str = "en",
        sample_rate: int = NATIVE_SAMPLE_RATE,
        num_step: int | None = None,
        guidance_scale: float | None = None,
        length_scale: float | None = None,
        idle_flush_ms: int | None = None,
        use_direct_session: bool = False,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a new instance of Finesse TTS.

        Speech streams over the Finesse realtime WebSocket. Each synthesis is one Finesse
        context: text is sent as it arrives, and the audio comes back as raw 16-bit PCM.

        See https://docs.finesse.flamapp.ai for more details on the Finesse API.

        Args:
            voice (str): The id of a ready Finesse voice, one of your own (including cloned
                voices) or from the public catalog.
            api_key (str, optional): The Finesse API key. If not provided, it is read from the
                FINESSE_API_KEY environment variable.
            language (str, optional): The language to speak, as a language code or name.
                Defaults to "en".
            sample_rate (int, optional): The audio sample rate in Hz: 8000 for telephony, or
                from 24000 (native) up to 48000. Defaults to 24000.
            num_step (int, optional): Quality vs latency, 1-32. Defaults to the server's 32.
            guidance_scale (float, optional): 0-10. Defaults to the server's 2.0.
            length_scale (float, optional): Speaking rate, greater than 0 up to 5; higher is
                slower. Defaults to the server's 1.0.
            idle_flush_ms (int, optional): Speak a pending partial sentence after this many
                milliseconds without new text, 200-5000. Defaults to the server's 800.
            use_direct_session (bool, optional): Connect straight to the synthesis GPU in your
                account's region over a short-lived, voice-scoped session, instead of the
                relayed public stream. Lowers first-audio latency for live agents. The session
                is minted for the current voice; changing the voice mints a new one. Defaults
                to False.
            http_session (aiohttp.ClientSession, optional): An existing aiohttp ClientSession
                to use. If not provided, a new session is created.
        """
        _validate(
            voice=voice,
            sample_rate=sample_rate,
            num_step=num_step,
            guidance_scale=guidance_scale,
            length_scale=length_scale,
            idle_flush_ms=idle_flush_ms,
        )
        finesse_api_key = api_key or os.environ.get("FINESSE_API_KEY")
        if not finesse_api_key:
            raise ValueError(
                "Finesse API key is required, either as argument or set"
                " FINESSE_API_KEY environment variable"
            )

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )

        self._opts = _TTSOptions(
            api_key=finesse_api_key,
            voice=voice,
            language=LanguageCode(language),
            sample_rate=sample_rate,
            num_step=num_step,
            guidance_scale=guidance_scale,
            length_scale=length_scale,
            idle_flush_ms=idle_flush_ms,
            ws_url=_WS_URL,
            api_base=_API_BASE,
            use_direct_session=use_direct_session,
        )
        self._session = http_session
        # Every setting travels in each context's task_start, so a settings change applies to
        # the next synthesis on a pooled connection without reconnecting.
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )
        # A direct session is minted for one voice, so each voice gets its own pool: a stream
        # always draws from the pool for the voice it snapshotted, which guarantees the session
        # it uses was minted for that exact voice. Classic connections are voice-agnostic (the
        # voice travels in each task_start), so they share the single pool above.
        self._direct_pools: dict[str, utils.ConnectionPool[aiohttp.ClientWebSocketResponse]] = {}
        # Count of streams currently synthesizing on each direct-session voice, so an idle pool
        # for a voice no longer in use can be closed without severing an active stream.
        self._direct_inuse: dict[str, int] = {}
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def model(self) -> str:
        return "finesse"

    @property
    def provider(self) -> str:
        return "Finesse"

    def _pool_for(self, voice: str) -> utils.ConnectionPool[aiohttp.ClientWebSocketResponse]:
        """Return the connection pool a stream for ``voice`` should use.

        Classic streams all share one voice-agnostic pool. Direct sessions are minted for a
        single voice, so each voice gets its own pool; the pool's connections are therefore
        always minted for the voice the caller asked for.
        """
        if not self._opts.use_direct_session:
            return self._pool
        pool = self._direct_pools.get(voice)
        if pool is None:
            pool = _DirectConnectionPool(
                connect_cb=functools.partial(self._connect_direct, voice=voice),
                close_cb=self._close_ws,
                # A direct connection carries a server session that expires regardless of use, so
                # its lifetime is capped below that TTL and the clock is not refreshed on reuse:
                # the pool recycles it (minting a fresh session) before the server ends it. The
                # retired socket is closed before its replacement opens (see _DirectConnectionPool)
                # so a rollover never holds two of the account's concurrency slots for one voice.
                max_session_duration=_DIRECT_MAX_SESSION_DURATION,
                mark_refreshed_on_get=False,
            )
            self._direct_pools[voice] = pool
        return pool

    @asynccontextmanager
    async def _pooled_connection(
        self, voice: str, timeout: float
    ) -> AsyncIterator[
        tuple[
            utils.ConnectionPool[aiohttp.ClientWebSocketResponse], aiohttp.ClientWebSocketResponse
        ]
    ]:
        """Hold a connection from ``voice``'s pool, evicting now-idle direct pools on release."""
        pool = self._pool_for(voice)
        direct = self._opts.use_direct_session
        if direct:
            self._direct_inuse[voice] = self._direct_inuse.get(voice, 0) + 1
        try:
            async with pool.connection(timeout=timeout) as ws:
                yield pool, ws
        finally:
            if direct:
                await self._release_direct(voice)

    async def _release_direct(self, voice: str) -> None:
        """Drop a stream's hold on ``voice`` and close the pools no voice is using any more.

        A direct session holds a GPU session open, so once the voice is no longer current and no
        stream is still synthesizing on it, its pool is closed rather than left idle. Pools with
        an active stream are kept (their checked-out connection is never severed), and a later
        stream for an evicted voice simply mints a fresh session.
        """
        remaining = self._direct_inuse.get(voice, 0) - 1
        if remaining > 0:
            self._direct_inuse[voice] = remaining
        else:
            self._direct_inuse.pop(voice, None)
        for v in list(self._direct_pools):
            if v != self._opts.voice and self._direct_inuse.get(v, 0) == 0:
                # pop with a default: a concurrent release may have already removed this pool
                # while we awaited an earlier aclose().
                pool = self._direct_pools.pop(v, None)
                if pool is not None:
                    await pool.aclose()

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        try:
            return await asyncio.wait_for(
                session.ws_connect(
                    self._opts.ws_url,
                    protocols=(_SUBPROTOCOL,),
                    headers={
                        "X-API-Key": self._opts.api_key,
                        "X-Finesse-Client": f"livekit-agents/{__version__}",
                    },
                ),
                timeout,
            )
        except aiohttp.WSServerHandshakeError as e:
            # The handshake error keeps the request headers, including the API key.
            raise APIStatusError(
                "Finesse refused the connection",
                status_code=e.status,
            ) from None

    async def _connect_direct(
        self, timeout: float, *, voice: str
    ) -> aiohttp.ClientWebSocketResponse:
        """Mint a direct realtime session and connect straight to the GPU region.

        The backend issues a short-lived, voice-scoped session token; the client then opens the
        socket against the synthesis GPU itself, one hop closer than the relayed stream. The
        per-turn protocol is identical.
        """
        session = self._ensure_session()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        minted = await self._mint_session(session, timeout, voice)
        remaining = max(0.0, deadline - loop.time())
        try:
            return await asyncio.wait_for(
                session.ws_connect(
                    minted["url"],
                    # The session token is the auth; it travels as the second subprotocol.
                    protocols=(_SUBPROTOCOL, minted["token"]),
                    headers={"X-Finesse-Client": f"livekit-agents/{__version__}"},
                ),
                remaining,
            )
        except aiohttp.WSServerHandshakeError as e:
            raise APIStatusError(
                "Finesse refused the session connection",
                status_code=e.status,
            ) from None

    async def _mint_session(
        self, session: aiohttp.ClientSession, timeout: float, voice: str
    ) -> dict[str, Any]:
        body = {"voice_ids": [voice], "budget_chars": _SESSION_BUDGET_CHARS}
        try:
            async with session.post(
                f"{self._opts.api_base}/realtime/sessions",
                json=body,
                headers={
                    "X-API-Key": self._opts.api_key,
                    "X-Finesse-Client": f"livekit-agents/{__version__}",
                },
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                try:
                    payload: Any = await resp.json(content_type=None)
                except (ValueError, aiohttp.ClientError):
                    payload = None
                if resp.status not in (200, 201):
                    code = payload.get("code") if isinstance(payload, dict) else None
                    raise APIStatusError(
                        f"Finesse could not create a realtime session: {code or resp.reason}",
                        status_code=resp.status,
                        body=payload,
                    )
                data = payload.get("data") if isinstance(payload, dict) else None
                if not isinstance(data, dict) or not data.get("url") or not data.get("token"):
                    # The body is deliberately omitted: a malformed-but-successful response can
                    # carry the session token, which must never reach the retry/error logs.
                    raise APIStatusError(
                        "Finesse returned an unexpected session response",
                        status_code=resp.status,
                    )
                return data
        except aiohttp.ClientError as e:
            raise APIConnectionError() from e

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def prewarm(self) -> None:
        self._pool_for(self._opts.voice).prewarm()

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        num_step: NotGivenOr[int | None] = NOT_GIVEN,
        guidance_scale: NotGivenOr[float | None] = NOT_GIVEN,
        length_scale: NotGivenOr[float | None] = NOT_GIVEN,
    ) -> None:
        """Update the TTS options. They apply from the next synthesis.

        Args:
            voice (str, optional): The id of a ready Finesse voice.
            language (str, optional): The language to speak, as a language code or name.
            num_step (int, optional): Quality vs latency, 1-32; None for the server default.
            guidance_scale (float, optional): 0-10; None for the server default.
            length_scale (float, optional): Speaking rate, greater than 0 up to 5; None for
                the server default.
        """
        _validate(
            voice=voice,
            num_step=num_step,
            guidance_scale=guidance_scale,
            length_scale=length_scale,
        )
        if is_given(voice) and voice != self._opts.voice:
            self._opts.voice = voice
            # In direct mode the next synthesis snapshots the new voice and draws from that
            # voice's own pool (see _pool_for), so it never reuses a session minted for the
            # previous voice. The previous voice's pool is closed by _release_direct once no
            # stream is using it any more.
        if is_given(language):
            self._opts.language = LanguageCode(language)
        if is_given(num_step):
            self._opts.num_step = num_step
        if is_given(guidance_scale):
            self._opts.guidance_scale = guidance_scale
        if is_given(length_scale):
            self._opts.length_scale = length_scale

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        await self._pool.aclose()
        # A stream finishing during an awaited aclose() can evict a pool via _release_direct,
        # so iterate over a snapshot of the values.
        for pool in list(self._direct_pools.values()):
            await pool.aclose()
        self._direct_pools.clear()
        self._direct_inuse.clear()


def _task_start(opts: _TTSOptions, context_id: str, text: str, *, finalize: bool) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "type": "task_start",
        "context_id": context_id,
        "voice_id": opts.voice,
        "text": text,
        "finalize": finalize,
        "language": _finesse_language(opts.language),
        "output_format": "s16le",
        "sample_rate": opts.sample_rate,
    }
    for name in ("num_step", "guidance_scale", "length_scale", "idle_flush_ms"):
        value = getattr(opts, name)
        if value is not None:
            msg[name] = value
    return msg


async def _send_text(
    ws: aiohttp.ClientWebSocketResponse,
    opts: _TTSOptions,
    context_id: str,
    text: str,
    *,
    opened: bool,
    finalize: bool = False,
) -> None:
    """Send text to a context, opening it with task_start when it isn't open yet.

    Long text is split so that every frame stays under the server's 1 MiB limit.
    """
    chunks = [text[i : i + _MAX_TEXT_CHARS] for i in range(0, len(text), _MAX_TEXT_CHARS)]
    if not chunks:
        chunks = [""]
    for i, chunk in enumerate(chunks):
        last = finalize and i == len(chunks) - 1
        if not opened and i == 0:
            msg = _task_start(opts, context_id, chunk, finalize=last)
        else:
            msg = {
                "type": "task_continue",
                "context_id": context_id,
                "text": chunk,
                "finalize": last,
            }
        await ws.send_str(json.dumps(msg, ensure_ascii=False))


async def _cancel_context(ws: aiohttp.ClientWebSocketResponse, context_id: str) -> None:
    """Tell Finesse to stop an interrupted context.

    The pool only closes a removed connection when the next synthesis starts, so without this
    the server would keep synthesizing the interrupted turn until then.
    """
    try:
        msg = json.dumps({"type": "task_cancel", "context_id": context_id})
        await asyncio.shield(ws.send_str(msg))
    except (Exception, asyncio.CancelledError):
        # Best effort: the connection may already be gone, which stops the context too.
        pass


def _event_error(data: dict[str, Any], request_id: str, *, direct: bool = False) -> APIStatusError:
    code = data.get("code")
    if not isinstance(code, str) or not _ERROR_CODE_RE.match(code):
        code = "unknown_error"
    retryable = code not in _NON_RETRYABLE_CODES
    if direct and code in _SESSION_RENEWABLE_CODES:
        # On a direct session this means the session's character budget is spent, not that the
        # account is out of credits: the next attempt mints a fresh session and continues. A
        # truly empty account instead fails when that mint is refused.
        retryable = True
    return APIStatusError(
        f"Finesse {data.get('type')}: {code}",
        request_id=request_id,
        body={"code": code},
        retryable=retryable,
    )


async def _receive(
    ws: aiohttp.ClientWebSocketResponse, timeout: float, input_done: asyncio.Event | None
) -> aiohttp.WSMessage:
    """Wait for the next frame.

    The timeout only runs once the input has ended: while the LLM is still writing (or
    calling a tool), the server can rightly have nothing to send.
    """
    if input_done is None or input_done.is_set():
        return await ws.receive(timeout=timeout)

    recv = asyncio.ensure_future(ws.receive())
    done = asyncio.ensure_future(input_done.wait())
    try:
        await asyncio.wait({recv, done}, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:
        recv.cancel()
        raise
    finally:
        done.cancel()
    if recv.done():
        return recv.result()
    # The input ended while waiting: from now on the server has to answer within the timeout.
    return await asyncio.wait_for(recv, timeout)


async def _receive_context(
    ws: aiohttp.ClientWebSocketResponse,
    *,
    context_id: str,
    sample_rate: int,
    output_emitter: tts.AudioEmitter,
    timeout: float,
    direct: bool = False,
    input_done: asyncio.Event | None = None,
) -> None:
    """Push the context's audio until it finishes.

    Frames for other contexts are dropped: a pooled connection can still carry the tail of
    an earlier, cancelled context.
    """
    # Each binary frame belongs to the context named by the audio_chunk just before it.
    audio_context: str | None = None
    while True:
        msg = await _receive(ws, timeout, input_done)
        if msg.type in (
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSING,
        ):
            close_code = ws.close_code
            raise APIStatusError(
                "Finesse connection closed unexpectedly",
                status_code=close_code or -1,
                request_id=context_id,
                retryable=close_code not in _NON_RETRYABLE_CLOSE_CODES,
            )

        if msg.type == aiohttp.WSMsgType.BINARY:
            if audio_context == context_id:
                output_emitter.push(msg.data)
            audio_context = None
            continue

        if msg.type != aiohttp.WSMsgType.TEXT:
            logger.warning("unexpected Finesse message type %s", msg.type)
            continue

        data = json.loads(msg.data)
        kind = data.get("type")
        event_context = data.get("context_id")

        if kind == "audio_chunk":
            audio_context = event_context
        elif kind == "error":
            # A fatal error ends the session; a non-fatal one only rejected one message.
            if data.get("fatal") or event_context in (None, context_id):
                raise _event_error(data, context_id, direct=direct)
        elif kind == "session_ended":
            raise APIStatusError(
                "Finesse session ended", request_id=context_id, body={"reason": data.get("reason")}
            )
        elif event_context != context_id:
            continue
        elif kind == "task_started":
            if data.get("sample_rate") not in (None, sample_rate):
                logger.warning(
                    "Finesse returned a different sample rate",
                    extra={"requested": sample_rate, "returned": data["sample_rate"]},
                )
        elif kind == "task_done":
            return
        elif kind == "task_cancelled":
            # The plugin only cancels a context it has stopped reading, so this came from the
            # server: the speech is incomplete.
            raise APIStatusError("Finesse cancelled the context", request_id=context_id)
        elif kind == "task_failed":
            raise _event_error(data, context_id, direct=direct)
        elif kind != "sentence_done":
            logger.debug("ignoring Finesse message type %s", kind)


class ChunkedStream(tts.ChunkedStream):
    """Synthesize a whole text as one Finesse context."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        context_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=context_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )
        if not self._input_text.strip():
            output_emitter.flush()
            return

        try:
            async with self._tts._pooled_connection(
                self._opts.voice, self._conn_options.timeout
            ) as (_pool, ws):
                await _send_text(
                    ws, self._opts, context_id, self._input_text, opened=False, finalize=True
                )
                try:
                    await _receive_context(
                        ws,
                        context_id=context_id,
                        sample_rate=self._opts.sample_rate,
                        output_emitter=output_emitter,
                        timeout=self._conn_options.timeout,
                        direct=self._opts.use_direct_session,
                    )
                except asyncio.CancelledError:
                    await _cancel_context(ws, context_id)
                    raise
            output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Stream text into one Finesse context as it arrives."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        context_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=context_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        # Set once the context is open, or once the input ended without any text.
        opened_event = asyncio.Event()
        # Set once the context is finalized: only then must the server answer within the timeout.
        input_done = asyncio.Event()
        opened = False
        finished = False

        async def _send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal opened
            # Leading whitespace waits for real text: the server rejects a context with none.
            pending = ""
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    # The server speaks a pending partial sentence on its own after
                    # idle_flush_ms, so a flush needs no message.
                    continue
                if not opened:
                    pending += data
                    if not pending.strip():
                        continue
                    output_emitter.start_segment(segment_id=context_id)
                    self._mark_started()
                    await _send_text(ws, self._opts, context_id, pending, opened=False)
                    opened = True
                    opened_event.set()
                else:
                    await _send_text(ws, self._opts, context_id, data, opened=True)

            if opened:
                await _send_text(ws, self._opts, context_id, "", opened=True, finalize=True)
            input_done.set()
            opened_event.set()

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal finished
            await opened_event.wait()
            if not opened:
                return
            await _receive_context(
                ws,
                context_id=context_id,
                sample_rate=self._opts.sample_rate,
                output_emitter=output_emitter,
                timeout=self._conn_options.timeout,
                direct=self._opts.use_direct_session,
                input_done=input_done,
            )
            finished = True

        try:
            async with self._tts._pooled_connection(
                self._opts.voice, self._conn_options.timeout
            ) as (pool, ws):
                self._acquire_time = pool.last_acquire_time
                self._connection_reused = pool.last_connection_reused
                tasks = [
                    asyncio.create_task(_send_task(ws)),
                    asyncio.create_task(_recv_task(ws)),
                ]
                try:
                    await asyncio.gather(*tasks)
                except asyncio.CancelledError:
                    # The user interrupted the agent mid-turn.
                    if opened and not finished:
                        await _cancel_context(ws, context_id)
                    raise
                finally:
                    opened_event.set()
                    await utils.aio.gracefully_cancel(*tasks)
            output_emitter.end_input()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
