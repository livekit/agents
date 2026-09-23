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

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import time
import weakref
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field, replace
from types import TracebackType
from typing import Any, Literal, NamedTuple

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .connection import (
    CandidateState,
    PluginEvent,
    TTSConnectionConfig,
    bridge_endpoint,
    bridge_model,
    resolve_base_url,
)
from .gateway_adapter import (
    build_external_tracking_headers,
    build_tts_init_payload,
    extract_error_status,
    is_non_retryable_client_error,
    is_payload_too_large,
    normalize_region_override,
    normalize_world_part_override,
)
from .log import logger
from .sentence_tokenizer import SentenceTokenizer, is_partial_head, separator_after

NUM_CHANNELS = 1
WS_CLOSE_TIMEOUT_S = 1.0
# Client-side WebSocket ping interval. A socket that dies without a close frame
# during a long silence is otherwise invisible to the idle reader, because a
# read on it simply never returns. Transport level, not provider specific.
_WS_HEARTBEAT_S = 20.0
# After an interrupt the plugin sends `clear` and waits up to this long for the
# gateway to acknowledge it, then this much longer, counted from the
# acknowledgement, for the socket to stay quiet. Anything else closes the
# socket, so no late audio from the cancelled reply can reach the next one. The
# wait runs in the background, so it never delays the next reply; it only
# decides whether this socket can be reused. A second is generous on purpose:
# the acknowledgement arrives once the provider's in-flight audio has drained,
# measured at 300-500 ms, and giving up early throws away a socket the gateway
# was still happy with. Frames arrive in order, so nothing the gateway sent
# before its acknowledgement can arrive after it; the quiet window is only a
# guard against a gateway that keeps sending.
_CANCEL_SETTLE_S = 1.0
_CANCEL_QUIET_S = 0.1
# How long a reply will wait for an interrupt's drain to hand the held socket
# back instead of opening a private one. It waits only when the drain is due to
# finish within this long: once the gateway has acknowledged, that is always
# the case, and before then only near the deadline. This is about what the
# connect and init it avoids cost, so waiting is never more expensive than the
# thing it avoids.
_SETTLE_HANDBACK_WAIT_S = 0.15
_PREWARM_CONNECT_TIMEOUT_S = 10.0
# How long a reply waits for a background connect already in flight rather than
# opening a socket of its own, counted from when that connect started. Several
# times what a healthy connect takes, so only a stalled one is given up on, and
# a reply never waits out the whole prewarm timeout before connecting itself.
_PENDING_CONNECT_WAIT_S = 2.0
# Two limits on how long the held socket lives. SLNG closes a connection that
# reaches its maximum duration, which on some plans is 30 minutes, even in the
# middle of a reply, and a reply cut off after its audio has started cannot be
# retried. So a socket this old is replaced between replies, in the background.
# And a socket idle this long is closed and not reopened: an instance nobody
# uses any more (a TTS replaced at an agent handoff is never closed) would
# otherwise hold a concurrent session for the rest of the call, because the
# keepalive below stops SLNG reaping it. The next reply, if one comes, connects
# as the first one did.
_MAX_CONNECTION_AGE_S = 20 * 60.0
_MAX_IDLE_S = 5 * 60.0
# Nothing closes a TTS that an agent handoff replaces, but its AgentSession
# stops listening to it. An instance with no one listening for this long closes
# its connection and stops reopening one, rather than waiting out the idle limit
# above. A handoff that keeps the same TTS stops listening and starts again
# with the new agent's MCP servers connecting in between, so this only has to
# outlast that.
_DETACH_GRACE_S = 10.0
# A gateway that accepts a connection and immediately closes it must not cause a
# connect storm. Applies only to sockets that closed without serving a reply:
# each one waits twice as long as the last, and after this many in a row the
# instance stops reopening in the background altogether. A settings error the
# gateway refuses at init would otherwise be retried for the length of the
# call. Replies still connect on demand, so giving up costs a cold start, not
# the call, and the first socket that carries text starts the count over.
_BACKGROUND_CONNECT_MIN_INTERVAL_S = 2.0
_BACKGROUND_CONNECT_MAX_INTERVAL_S = 30.0
_MAX_BACKGROUND_REOPENS = 3
# Reasons the plugin closes the held socket itself. Nothing failed, so these
# reopen at once and are not counted against the limit above.
_PLANNED_REOPENS = frozenset({"options_changed", "max_age"})

# Where "phrase" mode cuts a frame, alongside `phrase_max_chars`. The other
# modes never consult it: a sentence is a frame, and so is a word.
_PHRASE_FLUSH_SUFFIXES = (".", "!", "?", ",", ";", ":")

# Every WebSocket message type that ends a read loop. ERROR is one of them,
# but it is a transport failure carrying an exception, not a close, so the read
# loops log it separately instead of treating it as a tidy end of stream.
_WS_CLOSED_TYPES = (
    aiohttp.WSMsgType.CLOSE,
    aiohttp.WSMsgType.CLOSED,
    aiohttp.WSMsgType.CLOSING,
)
_WS_END_TYPES = (*_WS_CLOSED_TYPES, aiohttp.WSMsgType.ERROR)
# How often the idle reader sends a keepalive. The gateway's idle timer counts
# text and binary frames only, so the transport-level ping above does not hold
# a silent socket open.
_IDLE_KEEPALIVE_S = 20.0

# How a cancelled reply ended. Only "acknowledged" lets the socket be reused,
# so this is a closed set rather than free-form strings.
_CancelOutcome = Literal[
    "acknowledged",
    "timeout",
    "closed",
    "error",
    "audio_after_terminal",
    "send_failed",
    "drain_failed",
    # A socket closed without a drain, because it could not be kept anyway.
    "not_reusable",
]


def _describe_ws_end(
    ws: aiohttp.ClientWebSocketResponse, msg: aiohttp.WSMessage
) -> dict[str, object]:
    """Why a read loop ended: the close code and reason, or the transport error.

    Worth carrying into every log line, because a gateway that reaps an idle or
    over-long connection sends no error frame first; the close code (1008, 1011,
    1012) and its reason are then the only account of what happened.
    """
    code: object = None
    reason: object = None
    if msg.type is aiohttp.WSMsgType.CLOSE:
        code, reason = msg.data, msg.extra
    if code is None:
        code = getattr(ws, "close_code", None)
    error = msg.data if msg.type is aiohttp.WSMsgType.ERROR else ws.exception()
    return {
        "ws_close_code": code,
        "ws_close_reason": reason,
        "ws_error": str(error) if error is not None else None,
    }


def _transport_failed(ws: aiohttp.ClientWebSocketResponse, msg: aiohttp.WSMessage) -> bool:
    """Whether a read loop ended because the connection failed, not because it closed.

    ERROR carries the failure itself. aiohttp reports a lost connection (a
    reset, a heartbeat pong that never came) as CLOSED with 1006, because no
    close frame arrived. A peer that ends a turn by closing sends one, which
    arrives as CLOSE, and a clean end of stream is CLOSED with 1000.
    """
    return msg.type is aiohttp.WSMsgType.ERROR or (
        msg.type is aiohttp.WSMsgType.CLOSED
        and getattr(ws, "close_code", None) == aiohttp.WSCloseCode.ABNORMAL_CLOSURE
    )


def _log_ws_end(
    ws: aiohttp.ClientWebSocketResponse,
    msg: aiohttp.WSMessage,
    *,
    context: str,
    **extra: object,
) -> dict[str, object]:
    """Log the end of a read loop, at warning when the transport itself failed."""
    info = _describe_ws_end(ws, msg)
    fields = {"context": context, **extra, **info}
    if msg.type is aiohttp.WSMsgType.ERROR:
        logger.warning("[TTS] websocket transport failed", extra=fields)
    else:
        logger.debug("[TTS] websocket closed", extra=fields)
    return info


async def _receive_by(ws: aiohttp.ClientWebSocketResponse, deadline: float) -> aiohttp.WSMessage:
    """Read one frame, or raise TimeoutError at ``deadline`` (a ``perf_counter`` time).

    Not ``receive(timeout=...)``: aiohttp answers pings and swallows pongs
    inside ``receive()`` and restarts that timeout on each one, so a server that
    pings, or this plugin's own heartbeat, can keep it from ever firing.
    """
    return await asyncio.wait_for(ws.receive(), timeout=max(deadline - time.perf_counter(), 0.001))


async def _close_ws(ws: aiohttp.ClientWebSocketResponse, *, context: str) -> None:
    try:
        await asyncio.wait_for(ws.close(), timeout=WS_CLOSE_TIMEOUT_S)
    except (TimeoutError, asyncio.TimeoutError):
        logger.warning(
            "[TTS] websocket close timed out",
            extra={"context": context, "timeout_s": WS_CLOSE_TIMEOUT_S},
        )
    except Exception:
        logger.warning("[TTS] websocket close failed", extra={"context": context}, exc_info=True)


def _extract_audio_b64(resp: dict[str, object]) -> str | None:
    data = resp.get("data")
    if isinstance(data, str) and data:
        return data
    if isinstance(data, dict):
        nested_audio = data.get("audio")
        if isinstance(nested_audio, str) and nested_audio:
            return nested_audio

    audio = resp.get("audio")
    if isinstance(audio, str) and audio:
        return audio

    return None


def _extract_error_message(resp: dict[str, object]) -> str:
    data = resp.get("data")
    if isinstance(data, dict):
        nested = data.get("message") or data.get("description") or data.get("error")
        if isinstance(nested, str) and nested:
            return nested

    top_level = resp.get("message") or resp.get("description") or resp.get("error")
    if isinstance(top_level, str) and top_level:
        return top_level

    return "Unknown error"


def _extract_error_status(resp: dict[str, object]) -> int | None:
    """Extract an HTTP status from an error frame's int/numeric/string code."""
    return extract_error_status(resp)


def _contains_letter(text: str) -> bool:
    """True if the text has at least one alphabetic character in any script.

    ``str.isalpha`` covers every Unicode letter — Latin, Devanagari, Bengali,
    Tamil, etc. — so this is script-agnostic. Tokens with only punctuation,
    digits, whitespace, or symbols (e.g. "—", "4.5", "5,000") return False.
    Some providers (notably Sarvam Bulbul) reject a per-token WebSocket frame
    that carries no allowed-language character, so such tokens must be merged
    into a neighbouring word rather than sent on their own.
    """
    return any(ch.isalpha() for ch in text)


class _TextFrame(NamedTuple):
    """One text frame for the gateway. ``flush`` marks the reply's last one.

    The flush itself is sent as its own message straight after that frame.
    ``partial`` marks the opening of a sentence whose remainder follows, which
    tells the gateway to start on it without treating it as a whole utterance.
    """

    text: str
    flush: bool
    partial: bool = False


@dataclass
class _SegmentInput:
    """One reply's tokenizer stream plus the flag that lets the sender spot the last frame."""

    stream: tokenize.WordStream | tokenize.SentenceStream
    # Set by _tokenize_input immediately before it calls stream.end_input().
    # The tokenizer retains its last token until end_input(), so a token pulled
    # while this is False is never the reply's last; once True, every remaining
    # token is already queued in the closed channel and can be drained at once.
    input_ended: bool = False


class _FrameBatcher:
    """Groups tokenizer tokens into the text frames sent to the gateway.

    In ``"sentence"`` mode every token is already a whole sentence, so each one
    becomes its own frame. Punctuation is not consulted: a sentence that ends
    in a closing quote, a bracket, an ellipsis or a CJK full stop is still one
    sentence, and cutting on ASCII punctuation instead would merge it with the
    next. In ``"phrase"`` mode words are re-batched at ``_PHRASE_FLUSH_SUFFIXES``
    or once the buffer reaches ``max_chars``. In ``"word"`` mode every spoken
    word is a frame.

    A frame is never cut before it holds a letter (see ``_contains_letter``),
    so a letterless token stays attached to a neighbouring word: some providers
    reject a frame with no allowed-language character. ``finish`` is the
    exception. At the end of a reply there is no neighbour left, so it returns
    whatever is buffered. Sending a bare "4200." gets it voiced by providers
    that can, and refused loudly by providers that cannot, where dropping it
    would leave the caller listening to silence.
    """

    def __init__(self, *, mode: Literal["sentence", "word", "phrase"], max_chars: int) -> None:
        self._mode = mode
        self._max_chars = max_chars
        self._buf = ""
        self._has_letter = False

    def push(self, token: str, *, separator: str = " ") -> str | None:
        """Add a token; return a completed frame when this token closes one.

        ``separator`` is what follows the token in the frame. A space, except
        after the opening of a sentence in a script written without spaces,
        where one would end up inside the sentence once it is joined back up.
        """
        piece = f"{token}{separator}"
        if self._mode == "sentence":
            self._buf += piece
            self._has_letter = self._has_letter or _contains_letter(token)
            return self._take() if self._has_letter else None
        if self._mode == "phrase":
            self._buf += piece
            self._has_letter = self._has_letter or _contains_letter(token)
            stripped = self._buf.rstrip()
            at_boundary = bool(stripped) and (
                stripped.endswith(_PHRASE_FLUSH_SUFFIXES) or len(stripped) >= self._max_chars
            )
            return self._take() if at_boundary and self._has_letter else None
        if not self._buf:
            self._buf = piece
            self._has_letter = _contains_letter(token)
            return None
        if _contains_letter(token):
            # A new spoken word arrived. Emit the buffered word (with any
            # letterless tokens that trailed it); if the buffer so far is only
            # leading punctuation, prepend it to this word instead.
            if self._has_letter:
                frame = self._take()
                self._buf = piece
                self._has_letter = True
                return frame
            self._buf += piece
            self._has_letter = True
            return None
        self._buf += piece
        return None

    def finish(self) -> str | None:
        """Return the trailing buffer as a frame, letters or not, or None if empty."""
        if self._buf.strip():
            return self._take()
        self._buf, self._has_letter = "", False
        return None

    def _take(self) -> str:
        frame, self._buf, self._has_letter = self._buf, "", False
        return frame


@dataclass(frozen=True)
class _ReceivedWsEvent:
    kind: Literal["audio_chunk", "audio_end", "error", "ignore", "unknown"]
    audio: bytes | None = None
    error: str | None = None
    error_status: int | None = None


def _decode_b64_audio(audio_b64: str, *, context: str) -> bytes | None:
    """Decode one base64 audio payload, or warn and drop it.

    ``binascii.Error`` subclasses ``ValueError``, so this catches a malformed
    payload without swallowing anything the caller does with the result.
    """
    try:
        return base64.b64decode(audio_b64)
    except ValueError:
        logger.warning("[TTS] invalid base64 audio (%s)", context, exc_info=True)
        return None


def _decode_audio_payload(resp: dict[str, object], *, context: str) -> bytes | None:
    audio_b64 = _extract_audio_b64(resp)
    if not audio_b64:
        return None
    return _decode_b64_audio(audio_b64, context=context)


def _is_final_frame(resp: dict[str, object]) -> bool:
    """Whether a raw provider frame (no ``type``) says it is the last one."""
    value = resp.get("isFinal")
    return (
        value is True
        or value == 1
        or (isinstance(value, str) and value.strip().lower() in ("true", "1"))
    )


def _normalize_ws_message_type(resp: dict[str, object]) -> str | None:
    mtype = resp.get("type")
    if not isinstance(mtype, str):
        return None

    if mtype in ("Metadata", "Open", "control_ack", "ready"):
        return None
    if mtype == "Flushed":
        return "audio_end"
    if mtype in ("audio_chunk", "Audio", "audio", "chunk"):
        return "audio_chunk"
    if mtype == "done":
        return "audio_end"
    if mtype in ("Error", "error"):
        return "error"
    if mtype == "event":
        data = resp.get("data")
        if isinstance(data, dict):
            event_name = data.get("event") or data.get("event_type")
            if isinstance(event_name, str):
                normalized = event_name.strip().lower()
                if normalized in {"complete", "completed", "done", "end", "final"}:
                    return "audio_end"
                if normalized in {"error", "failed"}:
                    return "error"
    return mtype


def _parse_ws_event(resp: dict[str, object]) -> _ReceivedWsEvent:
    mtype = _normalize_ws_message_type(resp)
    if mtype is None:
        return _ReceivedWsEvent(kind="ignore")
    if mtype == "audio_chunk":
        return _ReceivedWsEvent(
            kind="audio_chunk",
            audio=_decode_audio_payload(resp, context="audio_chunk"),
        )
    if mtype in ("audio_end", "end", "flushed"):
        return _ReceivedWsEvent(
            kind="audio_end",
            audio=_decode_audio_payload(resp, context="audio_end"),
        )
    if mtype == "error":
        return _ReceivedWsEvent(
            kind="error",
            error=_extract_error_message(resp),
            error_status=_extract_error_status(resp),
        )
    return _ReceivedWsEvent(kind="unknown")


@dataclass
class _TTSOptions:
    model_endpoint: str
    model: str
    voice: str
    language: NotGivenOr[str]
    sample_rate: int
    encoding: Literal["linear16"]
    speed: NotGivenOr[float]
    word_tokenizer: tokenize.WordTokenizer | tokenize.SentenceTokenizer
    api_key: str
    model_options: dict[str, object]
    extra_headers: dict[str, str]
    runtime_init: dict[str, Any] | None
    warm_standby_enabled: bool
    text_chunking: Literal["sentence", "word", "phrase"]
    phrase_max_chars: int


class _StaleConnection(Exception):
    """A reused socket turned out to be dead before this reply produced audio."""


@dataclass
class _HeldConnection:
    """A WebSocket a TTS instance runs replies on.

    Usually the one it holds for the whole call. Also covers the two sockets
    that are never installed: the private one a reply opens when the held
    socket is still busy, and the one a background connect opened just as
    another was installed, which is closed instead.
    """

    ws: aiohttp.ClientWebSocketResponse
    # Settings epoch the init was sent under; stale once the TTS epoch moves on.
    epoch: int
    opened_by: str
    timing: _WsConnectionTiming
    in_use: bool = False
    reused: bool = False
    served_segments: int = 0
    # When the socket was opened, for _MAX_CONNECTION_AGE_S.
    opened_at: float = field(default_factory=time.perf_counter)
    # True once a reply wrote text on this socket. A socket the gateway closed
    # before that has nothing to show for itself, so reopening it is throttled.
    text_sent: bool = False
    # Whether this gateway said it can take the opening of a sentence early.
    accepts_partial_text: bool = False
    idle_reader: asyncio.Task[None] | None = None
    gateway_request_id: str | None = None
    gateway_session_id: str | None = None

    def note_ready(self, resp: dict[str, object]) -> None:
        """Record what a `ready` frame says, keeping any field it omits."""
        request_id = resp.get("slng_request_id")
        if isinstance(request_id, str):
            self.gateway_request_id = request_id
        session_id = resp.get("session_id")
        if isinstance(session_id, str):
            self.gateway_session_id = session_id
        self.accepts_partial_text = resp.get("accepts_partial_text") is True


@dataclass
class _PendingSettle:
    """An interrupt's drain: the socket it owns, and when it is due to release it."""

    task: asyncio.Task[None]
    conn: _HeldConnection
    ends_at: float


@dataclass
class _WsConnectionTiming:
    ws_connect_ms: float
    init_send_ms: float
    connect_total_ms: float


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000


def _has_running_loop() -> bool:
    """Whether a task can be started here. ``prewarm()`` is callable without one."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class TTS(tts.TTS):
    def __bool__(self) -> bool:
        # LiveKit Agents code may use truthiness checks like `tts or None`.
        # Some EventEmitter-style bases can be falsy when they have no listeners,
        # which would unintentionally disable TTS.
        return True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        connections: list[str | TTSConnectionConfig] | None = None,
        model_endpoint: str | None = None,
        provider_api_key: str | None = None,
        voice: str,
        slng_base_url: str | None = None,
        region_override: str | list[str] | None = None,
        world_part_override: str | None = None,
        external_agent_id: str | None = None,
        external_session_id: str | None = None,
        language: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: int = 24000,
        speed: NotGivenOr[float] = NOT_GIVEN,
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer | tokenize.SentenceTokenizer] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        extra_headers: dict[str, str] | None = None,
        # Advanced / optional. Used by integrations that drive the session
        # themselves; a typical client can ignore these.
        runtime_init: dict[str, Any] | None = None,
        warm_standby_enabled: bool = True,
        text_chunking: Literal["auto", "sentence", "word", "phrase"] = "auto",
        phrase_max_chars: int = 60,
        first_audio_timeout_s: float | None = None,
        fallback_recovery_cooldown_s: float = 60.0,
        _candidate: bool = False,
        **model_options: Any,
    ) -> None:
        """
        Create a new instance of SLNG TTS.

        Args:
            api_key (str): SLNG API key. Falls back to the ``SLNG_API_KEY`` env var.
            model: Unmute Bridge model identifier, e.g. "deepgram/aura:2".
            connections: Ordered model, endpoint, typed config, or SLNG TTS candidates.
            provider_api_key: Optional BYOK provider credential, sent as the
                ``X-Slng-Provider-Key`` header (external providers only).
            slng_base_url: Host of your SLNG region, for example
                "us-east.api.slng.ai". Falls back to the ``SLNG_BASE_URL`` env
                var. Required unless every connection is a full endpoint URL.
            region_override: Optional gateway region override, sent as the
                ``X-Region-Override`` header. Accepts a single region or a list
                of preferred regions in priority order.
            world_part_override: Optional gateway world-part override, sent as
                the ``X-World-Part-Override`` header. Constrains routing to a
                broad geographic zone (for example "eu", "na", "ap") when an
                exact region is not required. ``region_override`` takes
                precedence when both are set.
            external_agent_id: Optional tracking ID attached to usage events as
                the ``X-SLNG-Agent-Id`` header (max 128 chars).
            external_session_id: Optional tracking ID attached to usage events as
                the ``X-SLNG-Session-Id`` header (max 128 chars).
            voice: Required voice identifier.
            language: Optional language code. When omitted, the model's catalog
                default applies.
            speed: Optional speed multiplier. When omitted, the model's catalog
                default applies.
            sample_rate (int): Sample rate of audio. Defaults to 24000.
            word_tokenizer: Optional tokenizer for processing text. Defaults to
                ``slng.SentenceTokenizer()`` in sentence mode, which splits on
                the sentence terminators of every script with no language
                setting, and ``tokenize.basic.WordTokenizer(ignore_punctuation=False)``
                otherwise.
            warm_standby_enabled: Hold one connection for the whole call: open
                it at session start (``prewarm()``), reuse it for every reply,
                and reopen it in the background if the gateway closes it. It
                counts as one concurrent session for the whole call, until it
                has been idle for five minutes or no session has used the
                instance for 10 seconds. Defaults to True. False opens a
                connection for each reply and closes it afterwards.
            text_chunking: How LLM text is cut into gateway frames. ``"sentence"``
                sends one frame per sentence, in any script, and requires a
                ``SentenceTokenizer``. ``"phrase"`` re-batches words at clause
                punctuation or every ``phrase_max_chars``. ``"word"`` sends one
                frame per word. ``"auto"``, the default, is ``"phrase"`` when
                ``word_tokenizer`` is a ``WordTokenizer`` and ``"sentence"``
                otherwise.
            phrase_max_chars: In ``"phrase"`` mode, cut a frame once the buffer
                reaches this many characters. Ignored in the other modes.
            http_session (aiohttp.ClientSession): Optional aiohttp session to use for requests.
        """
        if model_endpoint is not None:
            raise ValueError(
                "model_endpoint was removed in 2.0: the plugin always connects "
                "through the Unmute Bridge. Pass model='provider/model:variant' "
                "or connections=[...] instead."
            )
        resolved_key = api_key or os.environ.get("SLNG_API_KEY")
        if not resolved_key:
            raise ValueError("api_key is required, or set the SLNG_API_KEY environment variable")
        slng_base_url = resolve_base_url(slng_base_url)

        if not voice.strip():
            raise ValueError("voice is required")
        if text_chunking not in {"auto", "sentence", "word", "phrase"}:
            raise ValueError("text_chunking must be 'auto', 'sentence', 'word', or 'phrase'")
        if phrase_max_chars <= 0:
            raise ValueError("phrase_max_chars must be positive")
        if "encoding" in model_options:
            # Everything else is forwarded to the gateway untouched, but this
            # one decides how the plugin reads the audio coming back. An
            # override here leaves the init and the decoder disagreeing, and
            # the caller hears noise rather than an error. (`sample_rate`
            # cannot arrive this way: it is a named argument.)
            raise ValueError(
                "encoding cannot be passed as a model option: the plugin decodes the "
                "audio itself and always requests linear16"
            )
        if model is not None and connections:
            raise ValueError("use model or connections, not both")

        raw_connections: list[str | TTSConnectionConfig]
        if connections:
            raw_connections = connections
        elif model is not None:
            raw_connections = [model]
        else:
            raise ValueError("model or connections is required")

        primary = raw_connections[0]
        if isinstance(primary, TTSConnectionConfig):
            resolved_model_endpoint = primary.endpoint
            endpoint_model = bridge_model(primary.endpoint, "tts")
            if primary.model is not None and primary.model != endpoint_model:
                raise ValueError("TTS connection model must match its endpoint")
            model = endpoint_model
            voice = primary.voice or voice
        else:
            resolved_model_endpoint = (
                primary if "://" in primary else bridge_endpoint(slng_base_url, "tts", primary)
            )
            model = bridge_model(resolved_model_endpoint, "tts")
            primary = TTSConnectionConfig(
                endpoint=resolved_model_endpoint,
                model=model,
                voice=voice,
            )

        headers = dict(extra_headers or {})
        region_override_header = normalize_region_override(region_override)
        if region_override_header:
            headers.setdefault("X-Region-Override", region_override_header)
        world_part_header = normalize_world_part_override(world_part_override)
        if world_part_header:
            headers.setdefault("X-World-Part-Override", world_part_header)
        headers.update(primary.headers)
        headers.update(
            build_external_tracking_headers(
                external_agent_id=external_agent_id,
                external_session_id=external_session_id,
            )
        )
        if provider_api_key is not None:
            byok_key = provider_api_key.strip()
            if not byok_key:
                raise ValueError("provider_api_key must not be empty")
            headers["X-Slng-Provider-Key"] = byok_key

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        resolved_chunking: Literal["sentence", "word", "phrase"]
        if text_chunking != "auto":
            resolved_chunking = text_chunking
        elif isinstance(word_tokenizer, tokenize.WordTokenizer):
            # "auto" meant phrase batching before sentence mode existed, and a
            # WordTokenizer is what a caller from then passes. Keep that
            # working rather than refusing it below.
            resolved_chunking = "phrase"
        else:
            resolved_chunking = "sentence"
        if not is_given(word_tokenizer):
            word_tokenizer = (
                SentenceTokenizer()
                if resolved_chunking == "sentence"
                else tokenize.basic.WordTokenizer(ignore_punctuation=False)
            )
        elif resolved_chunking == "sentence" and not isinstance(
            word_tokenizer, tokenize.SentenceTokenizer
        ):
            # Sentence mode sends one frame per token, so a tokenizer that
            # emits anything smaller sends a frame per word while the caller
            # believes it asked for sentences. Say so instead of degrading
            # quietly.
            raise ValueError(
                "text_chunking='sentence' needs a SentenceTokenizer; pass "
                "text_chunking='phrase' or 'word' to use a WordTokenizer"
            )

        self._opts = _TTSOptions(
            model_endpoint=resolved_model_endpoint,
            model=model,
            voice=voice,
            language=language,
            sample_rate=sample_rate,
            # LiveKit expects raw PCM. Some SLNG models default to MP3 unless explicitly requested.
            encoding="linear16",
            speed=speed,
            word_tokenizer=word_tokenizer,
            api_key=resolved_key,
            model_options=dict(model_options),
            extra_headers=headers,
            runtime_init=(
                dict(primary.init)
                if primary.init is not None
                else dict(runtime_init)
                if runtime_init is not None
                else None
            ),
            warm_standby_enabled=warm_standby_enabled,
            text_chunking=resolved_chunking,
            phrase_max_chars=phrase_max_chars,
        )
        self._session = http_session
        self._first_audio_timeout_s = first_audio_timeout_s
        self._candidate_state = CandidateState(len(raw_connections), fallback_recovery_cooldown_s)
        self._active_candidate_index = 0
        self._candidate_tts: list[TTS] = [self]
        self._is_candidate = _candidate
        # Set by the parent chain on the candidates it constructs: a
        # candidate's voice is only updated at runtime when it was inherited
        # from the chain-level default rather than set explicitly per
        # candidate.
        self._inherits_voice = False
        self._streams = weakref.WeakSet[SynthesizeStream]()
        # Bumped on every option change; a held connection opened under an older
        # epoch carries a stale init and is retired rather than reused.
        self._ws_epoch = 0
        # What the last `ready` said about early sentence openings. None until
        # one has been read, which with warm standby happens at session start;
        # a reply that starts before that sends whole sentences.
        self._partial_text_supported: bool | None = None
        # One connection per TTS instance, held across the replies of a call.
        # Deliberately not utils.ConnectionPool: that pool has no maximum size,
        # so returning the private socket a busy reply opened would leave two
        # connections in the pool and bill the call for two concurrent
        # sessions. It also has no equivalent of the idle reader or of the
        # cancel-and-settle handshake below.
        self._ws_lock = asyncio.Lock()
        self._held: _HeldConnection | None = None
        # Retained refs to release / cancel-drain / idle-reader tasks so the
        # event loop cannot drop them mid-run.
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._connect_task: asyncio.Task[None] | None = None
        # The interrupt drain, retained (like _connect_task) so that a reply
        # starting while it runs can wait for the socket it is about to hand
        # back, and can tell whether that is the socket it wants.
        self._settle: _PendingSettle | None = None
        self._last_background_connect_at: float | None = None
        # Pending retry of a connect that was throttled; see _defer_connect.
        self._reconnect_handle: asyncio.TimerHandle | None = None
        # Consecutive reopens whose socket died before carrying any text, which
        # is what the backoff and the give-up rule above are counted in.
        self._failed_reopens = 0
        # False once another candidate in the chain takes over: this instance
        # then neither keeps a socket nor reopens one, so a failover does not
        # leave the previous candidate holding a connection for the whole call.
        self._hold_allowed = True
        self._closing = False
        # What listens for this instance's metrics, which is how an
        # AgentSession's activity attaches to a TTS; see _DETACH_GRACE_S.
        self._metrics_listeners: set[Callable[..., Any]] = set()
        self._detach_handle: asyncio.TimerHandle | None = None
        self._detached = False

        if not _candidate:
            for fallback in raw_connections[1:]:
                config = (
                    fallback
                    if isinstance(fallback, TTSConnectionConfig)
                    else TTSConnectionConfig(
                        endpoint=(
                            fallback
                            if "://" in fallback
                            else bridge_endpoint(slng_base_url, "tts", fallback)
                        )
                    )
                )
                candidate = TTS(
                    api_key=resolved_key,
                    connections=[config],
                    provider_api_key=provider_api_key,
                    voice=config.voice or voice,
                    slng_base_url=slng_base_url,
                    region_override=region_override,
                    world_part_override=world_part_override,
                    external_agent_id=external_agent_id,
                    external_session_id=external_session_id,
                    language=language,
                    sample_rate=sample_rate,
                    speed=speed,
                    word_tokenizer=word_tokenizer,
                    http_session=http_session,
                    extra_headers=extra_headers,
                    runtime_init=runtime_init,
                    warm_standby_enabled=warm_standby_enabled,
                    text_chunking=resolved_chunking,
                    phrase_max_chars=phrase_max_chars,
                    first_audio_timeout_s=first_audio_timeout_s,
                    fallback_recovery_cooldown_s=fallback_recovery_cooldown_s,
                    _candidate=True,
                    **model_options,
                )
                candidate._inherits_voice = config.voice is None
                if (
                    candidate.sample_rate != self.sample_rate
                    or candidate.num_channels != self.num_channels
                ):
                    raise ValueError("all TTS candidates must use the same audio format")
                self._candidate_tts.append(candidate)
                candidate.on("metrics_collected", self._forward_metrics)
                candidate.on("slng_event", self._forward_plugin_event)

    def on(self, event: Any, callback: Callable[..., Any] | None = None) -> Callable[..., Any]:
        registered = super().on(event, callback)
        if event == "metrics_collected" and callback is not None and not self._is_candidate:
            self._metrics_listeners.add(callback)
            self._on_session_attached()
        return registered

    def off(self, event: Any, callback: Callable[..., Any]) -> None:
        super().off(event, callback)
        if event == "metrics_collected" and callback in self._metrics_listeners:
            self._metrics_listeners.discard(callback)
            if not self._metrics_listeners:
                self._on_session_detached()

    def _on_session_attached(self) -> None:
        if self._detach_handle is not None:
            self._detach_handle.cancel()
            self._detach_handle = None
        if self._detached:
            self._detached = False
            # The activity attaching now called prewarm() before it attached,
            # while this instance was still refusing to hold a socket, so open
            # the connection here instead.
            self._candidate_tts[self._candidate_state.start()]._hold_allowed = True
            self.prewarm()

    def _on_session_detached(self) -> None:
        if self._closing or self._detach_handle is not None or not _has_running_loop():
            return
        self._detach_handle = asyncio.get_running_loop().call_later(
            _DETACH_GRACE_S, self._detach_if_unused
        )

    def _detach_if_unused(self) -> None:
        self._detach_handle = None
        if self._metrics_listeners or self._closing:
            return
        self._detached = True
        logger.info(
            "[TTS] no session is using this instance; closing its connection",
            extra={"tts_model": self._opts.model},
        )
        # Switched off here rather than in the task below, so that nothing
        # reopens from this point and a session attaching before that task
        # runs is the last word. A reply still running closes its socket when
        # it ends, and a background connect in flight declines to install
        # its socket, unless a session attaches first, when it installs it.
        for candidate in self._candidate_tts:
            candidate._hold_allowed = False
            candidate._cancel_deferred_connect()
        self._spawn(self._release_detached())

    async def _release_detached(self) -> None:
        """Close the sockets the candidates still hold, unless the instance is in use again."""
        for candidate in self._candidate_tts:
            async with candidate._ws_lock:
                # Checked under the lock, after any wait for it: a session that
                # attached, or a stream that started, keeps what it has.
                if not self._detached:
                    return
                held = candidate._held
                if held is None or held.in_use:
                    continue
                candidate._held = None
                await candidate._stop_idle_reader(held)
            await _close_ws(held.ws, context="detached")

    def _forward_metrics(self, metrics: Any) -> None:
        self.emit("metrics_collected", metrics)

    def _forward_plugin_event(self, event: PluginEvent) -> None:
        self.emit("slng_event", event)

    def _emit_plugin_event(
        self,
        name: str,
        level: Literal["info", "warning", "error"] = "info",
        **data: Any,
    ) -> None:
        self.emit(
            "slng_event",
            PluginEvent(name=name, component="tts", level=level, data=data),
        )

    @property
    def model(self) -> str:
        return self._candidate_tts[self._active_candidate_index]._opts.model

    @property
    def provider(self) -> str:
        return "SLNG"

    @property
    def warm_standby_enabled(self) -> bool:
        return self._opts.warm_standby_enabled

    def _is_ws_usable(self, ws: aiohttp.ClientWebSocketResponse) -> bool:
        return not bool(getattr(ws, "closed", False))

    def _note_ready(self, conn: _HeldConnection, resp: dict[str, object]) -> None:
        """Record a `ready` frame against the connection and the instance.

        The capability is remembered on the instance because a reply's tokenizer
        stream is created before its connection is acquired, and the stream has
        to know then whether it may release an opening early.
        """
        conn.note_ready(resp)
        self._partial_text_supported = conn.accepts_partial_text

    def _may_keep(self, conn: _HeldConnection) -> bool:
        """Whether this connection can stay held for the next reply.

        Call it under ``_ws_lock``, or with no await between the check and what
        is done with it: it reads the held-socket bookkeeping. False for a
        private socket, a socket whose init predates the current options, one
        past its maximum age, a dead socket, an instance another candidate
        replaced, or one closing.
        """
        return (
            self._held is conn
            and self._hold_allowed
            and conn.epoch == self._ws_epoch
            and not self._past_max_age(conn)
            and self._is_ws_usable(conn.ws)
            and not self._closing
        )

    def _past_max_age(self, conn: _HeldConnection) -> bool:
        return time.perf_counter() - conn.opened_at >= _MAX_CONNECTION_AGE_S

    def _reopen_reason(self, conn: _HeldConnection) -> str:
        """Why the held socket is being replaced, which decides how its reopen is paced."""
        if conn.epoch != self._ws_epoch:
            return "options_changed"
        if self._past_max_age(conn):
            return "max_age"
        return "reconnect"

    async def _connect_ws(
        self, timeout: float
    ) -> tuple[aiohttp.ClientWebSocketResponse, _WsConnectionTiming]:
        session = self._ensure_session()

        # Connect to WebSocket
        model_endpoint = self._opts.model_endpoint
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}",
            "X-API-Key": self._opts.api_key,
        }
        headers.update(self._opts.extra_headers)
        connect_started_at = time.perf_counter()
        ws = await asyncio.wait_for(
            session.ws_connect(
                model_endpoint,
                headers=headers,
                heartbeat=_WS_HEARTBEAT_S,
            ),
            timeout,
        )
        ws_connect_ms = _elapsed_ms(connect_started_at)

        # SLNG-specific: send init. The `ready` that answers it is read by
        # whoever owns the socket next, the idle reader or a reply.
        init_payload = self._opts.runtime_init
        if init_payload is None:
            init_payload = build_tts_init_payload(
                model=self._opts.model,
                voice=self._opts.voice,
                sample_rate=self._opts.sample_rate,
                encoding=self._opts.encoding,
                language=self._opts.language if is_given(self._opts.language) else None,
                speed=self._opts.speed if is_given(self._opts.speed) else None,
                model_options=self._opts.model_options,
            )

        try:
            init_started_at = time.perf_counter()
            await ws.send_str(json.dumps(init_payload))
            init_send_ms = _elapsed_ms(init_started_at)
        except BaseException:
            await _close_ws(ws, context="init_failure")
            raise

        timing = _WsConnectionTiming(
            ws_connect_ms=ws_connect_ms,
            init_send_ms=init_send_ms,
            connect_total_ms=_elapsed_ms(connect_started_at),
        )
        logger.info(
            "[TTS] websocket connected",
            extra={
                "tts_model": self._opts.model,
                "tts_provider": self.provider,
                "tts_endpoint": model_endpoint,
                "ws_connect_ms": timing.ws_connect_ms,
                "init_send_ms": timing.init_send_ms,
                "connect_total_ms": timing.connect_total_ms,
            },
        )

        return ws, timing

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        """Run connection bookkeeping in the background, retained and logged on failure."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)

        def _done(done: asyncio.Task[None]) -> None:
            self._background_tasks.discard(done)
            if done.cancelled():
                return
            exc = done.exception()
            if exc is not None:
                logger.warning(
                    "[TTS] connection task failed",
                    extra={"tts_model": self._opts.model},
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

        task.add_done_callback(_done)
        return task

    async def _acquire_connection(self, *, timeout: float) -> _HeldConnection:
        """Return the held connection if it is alive and current, else open one.

        The returned connection is marked in-use; the caller hands it back with
        ``_release_connection`` or ``_release_after_cancel``. If another reply
        still owns the held socket (an interrupted reply draining, or a reply
        generated while the previous one is still speaking), this opens a
        private socket that is never installed and is closed on release. So
        is every socket when warm standby is off.

        ``timeout`` bounds the whole acquisition, waits included: every wait
        below gets what is left of it, and a stale socket is closed in the
        background, so a reply fails on time rather than after a wait and a
        full connect. Stopping the idle reader is the one step it does not
        bound, as that ends at the reader's next read.
        """
        deadline = time.perf_counter() + timeout
        # A background connect already in flight is a latency win: wait for it
        # rather than opening a second socket alongside it. asyncio.wait keeps
        # that task's failure, and its cancellation, out of this reply. The
        # wait is bounded by how long that connect has already been running,
        # so a stalled one costs this reply at most _PENDING_CONNECT_WAIT_S
        # before it connects on its own.
        pending_connect = self._connect_task
        if pending_connect is not None and not pending_connect.done():
            started_at = self._last_background_connect_at or time.perf_counter()
            budget = min(deadline, started_at + _PENDING_CONNECT_WAIT_S) - time.perf_counter()
            if budget > 0:
                await asyncio.wait({pending_connect}, timeout=budget)

        # An interrupt's drain owns the held socket until it finishes, so a
        # reply that starts inside that window opens a private socket and pays
        # a connect, an init and a second concurrent session for one reply.
        # Waiting for the drain is worth it only at the end of it; see
        # _SETTLE_HANDBACK_WAIT_S. Anything longer falls through to the private
        # socket exactly as before. The wait is outside _ws_lock because the
        # drain takes that lock to hand the socket back.
        settle = self._settle
        if (
            settle is not None
            and not settle.task.done()
            # Only a drain of the socket that is still the held one can hand
            # it back. One that aclose has since detached closes it instead,
            # so waiting for it would delay a reply for nothing.
            and self._held is settle.conn
            and settle.conn.in_use
            and settle.ends_at - time.perf_counter() <= _SETTLE_HANDBACK_WAIT_S
            and deadline - time.perf_counter() > 0
        ):
            await asyncio.wait(
                {settle.task},
                timeout=min(_SETTLE_HANDBACK_WAIT_S, deadline - time.perf_counter()),
            )

        stale: _HeldConnection | None = None
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise asyncio.TimeoutError()
        await asyncio.wait_for(self._ws_lock.acquire(), timeout=remaining)
        try:
            held = self._held
            if held is not None and not held.in_use:
                # Not bounded by the deadline: the reader stops at its next
                # read, and the socket must not be handed on until it has, or
                # two readers would share it.
                await self._stop_idle_reader(held)
                if (
                    held.epoch == self._ws_epoch
                    and not self._past_max_age(held)
                    and self._is_ws_usable(held.ws)
                ):
                    held.in_use = True
                    held.reused = True
                    return held
                self._held = None
                stale = held
        finally:
            self._ws_lock.release()
        if stale is not None:
            # In the background: a dead socket can take the whole close
            # timeout to give up, and this reply has nothing to wait for.
            self._spawn(_close_ws(stale.ws, context="acquire_stale"))

        epoch = self._ws_epoch
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise asyncio.TimeoutError()
        ws, timing = await self._connect_ws(timeout=remaining)
        try:
            conn = _HeldConnection(
                ws=ws, epoch=epoch, opened_by="segment", timing=timing, in_use=True
            )
            async with self._ws_lock:
                # Without warm standby nothing is held between replies, so no
                # session stays open while the call is silent.
                if (
                    self._held is None
                    and self.warm_standby_enabled
                    and self._hold_allowed
                    and epoch == self._ws_epoch
                    and not self._closing
                ):
                    self._held = conn
            return conn
        except BaseException:
            # An interrupt can land while this waits for the lock. Without this
            # the socket is open, unreferenced and never closed: a session the
            # gateway keeps billing until it times out on its own.
            await _close_ws(ws, context="acquire_cancelled")
            raise

    async def _release_connection(self, conn: _HeldConnection, *, keep: bool) -> None:
        """Hand a connection back after a reply.

        ``keep=True`` holds it for the next reply, but only when ``_may_keep``
        still allows it; a private socket, a stale init, a socket past its
        maximum age, a dead socket, a replaced candidate or a closing instance
        is closed here instead. ``keep=False`` always closes.
        """
        async with self._ws_lock:
            conn.in_use = False
            current = self._held is conn
            if keep and self._may_keep(conn):
                self._start_idle_reader(conn)
                return
            if current:
                self._held = None
        await _close_ws(conn.ws, context="release_keep" if keep else "release_error")
        if keep and current and not self._closing:
            self._on_connection_lost(conn, reason=self._reopen_reason(conn))

    def _release_after_cancel(
        self, conn: _HeldConnection, *, in_flight: bool, tasks_done: bool
    ) -> None:
        """Interrupt path. Runs inside a CancelledError handler, so it cannot await.

        ``in_flight``: text was sent and ``audio_end`` was not seen, so the
        gateway has to be told to cancel. ``tasks_done``: the segment's send and
        receive tasks have finished, so nobody else is inside ``ws.receive()``.

        Both true, on a socket that could be kept: send ``clear`` and drain,
        keeping the socket only if the gateway ends the reply cleanly. Both
        true on any other socket (a private one, or the held one of an instance
        that is closing, replaced or out of date): it is closed whatever the
        gateway answers, so it is closed at once, which stops the reply as
        surely as ``clear`` would. Neither in flight nor still reading: nothing
        of this reply reached the wire and nobody is mid-read, so the socket is
        clean and is kept as is. Anything else leaves the socket's state
        unknown, so it is closed.
        """
        if not (in_flight and tasks_done):
            self._spawn(self._release_connection(conn, keep=not in_flight and tasks_done))
        elif self._may_keep(conn):
            # Retained so that a reply starting meanwhile can wait for the
            # handback. Only the held socket's drain is recorded: no other
            # socket can be handed back.
            self._settle = _PendingSettle(
                task=self._spawn(self._cancel_and_settle(conn)),
                conn=conn,
                ends_at=time.perf_counter() + _CANCEL_SETTLE_S + _CANCEL_QUIET_S,
            )
        else:
            self._spawn(self._cancel_and_settle(conn, drain=False))

    async def _cancel_and_settle(self, conn: _HeldConnection, *, drain: bool = True) -> None:
        """Send clear, then decide whether the socket is clean enough to keep.

        With ``drain=False`` the socket is closed without either, but goes
        through the same bookkeeping, so the held socket is still reopened.
        """
        started_at = time.perf_counter()
        audio_after_cancel = 0
        cancel_sent = False
        outcome: _CancelOutcome = "send_failed" if drain else "not_reusable"
        try:
            if drain:
                await conn.ws.send_str(SynthesizeStream._CLEAR_MSG)
                cancel_sent = True
                outcome, audio_after_cancel = await self._settle_cancel(
                    conn, deadline=started_at + _CANCEL_SETTLE_S
                )
        except BaseException:
            # A socket that was already dead when the cancel was written is a
            # different problem from one that failed while draining, so keep
            # "send_failed" in that case rather than relabelling every failure.
            if cancel_sent:
                outcome = "drain_failed"
            raise
        finally:
            async with self._ws_lock:
                conn.in_use = False
                was_held = self._held is conn
                keep = outcome == "acknowledged" and self._may_keep(conn)
                if keep:
                    self._start_idle_reader(conn)
                elif was_held:
                    self._held = None
            # Only once the socket is back: a reply arriving before then still
            # finds the drain and can wait for it.
            if self._settle is not None and self._settle.conn is conn:
                self._settle = None
            if not keep:
                await _close_ws(conn.ws, context=f"cancel_{outcome}")
                # This closed the call's held socket, so reopen it (throttled
                # when it never carried text). Gating on a completed reply
                # instead would drop warm standby for the rest of a call whose
                # very first reply was interrupted.
                if was_held and not self._closing:
                    self._on_connection_lost(conn, reason=self._reopen_reason(conn))
            # Only a clean acknowledgement, a socket the gateway had already
            # closed, or one that was never going to be kept is routine; the
            # rest mean the interrupt did not go the way the code expects and
            # the socket was thrown away.
            logger.log(
                logging.INFO
                if outcome in ("acknowledged", "closed", "not_reusable")
                else logging.WARNING,
                "[TTS] cancel settled",
                extra={
                    "tts_model": self._opts.model,
                    "tts_provider": self.provider,
                    "cancel_outcome": outcome,
                    "settle_ms": _elapsed_ms(started_at),
                    "audio_chunks_after_cancel": audio_after_cancel,
                    "socket_kept": keep,
                },
            )
            self._emit_plugin_event(
                "tts.cancel",
                outcome=outcome,
                socket_kept=keep,
                audio_chunks_after_cancel=audio_after_cancel,
            )

    async def _settle_cancel(
        self, conn: _HeldConnection, *, deadline: float
    ) -> tuple[_CancelOutcome, int]:
        """Read until the cancelled reply ends and the socket goes quiet, or give up.

        Returns ``(outcome, audio_chunks_seen)``. Only "acknowledged" (a terminal
        frame before the deadline, then ``_CANCEL_QUIET_S`` with nothing behind
        it) means the socket may be reused; every other outcome means it has to
        be closed. The quiet window runs from the terminal frame, so a gateway
        that acknowledges in 350 ms frees the socket about 100 ms later rather
        than at the deadline.
        """
        ws = conn.ws
        audio = 0
        quiet_until: float | None = None

        def note_terminal() -> None:
            nonlocal quiet_until
            if quiet_until is not None:
                return
            quiet_until = time.perf_counter() + _CANCEL_QUIET_S
            settle = self._settle
            if settle is not None and settle.conn is conn:
                # A reply waiting to reuse this socket can now tell that it
                # is about to be handed back.
                settle.ends_at = quiet_until

        while True:
            limit = deadline if quiet_until is None else quiet_until
            remaining = limit - time.perf_counter()
            if remaining <= 0:
                return ("timeout" if quiet_until is None else "acknowledged"), audio
            try:
                msg = await _receive_by(ws, limit)
            except (TimeoutError, asyncio.TimeoutError):
                return ("timeout" if quiet_until is None else "acknowledged"), audio
            if msg.type in _WS_END_TYPES:
                _log_ws_end(ws, msg, context="cancel_settle", tts_model=self._opts.model)
                return "closed", audio
            if msg.type == aiohttp.WSMsgType.BINARY:
                audio += 1
                if quiet_until is not None:
                    return "audio_after_terminal", audio
                continue
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                resp = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.debug("[TTS] ignoring non-JSON frame while settling a cancel")
                continue
            if not isinstance(resp, dict):
                continue
            if resp.get("type") == "cleared":
                # The acknowledgement of the `clear` sent above. `cancel` aborts
                # the same turn but is answered with nothing at all, which is
                # why the interrupt does not use it.
                note_terminal()
                continue
            event = _parse_ws_event(resp)
            if event.kind == "audio_end":
                note_terminal()
                continue
            if event.kind == "error":
                # The gateway's own account of why the cancelled turn ended
                # badly; the caller only records the outcome word.
                logger.warning(
                    "[TTS] error while settling a cancel",
                    extra={
                        "tts_model": self._opts.model,
                        # A provider's message can quote the text it refused.
                        "lk.pii.error": event.error,
                        "error_status": event.error_status,
                    },
                )
                return "error", audio
            if event.kind == "audio_chunk" or isinstance(resp.get("audio"), str):
                audio += 1
                if quiet_until is not None:
                    return "audio_after_terminal", audio

    def _start_idle_reader(self, conn: _HeldConnection) -> None:
        if conn.idle_reader is not None and not conn.idle_reader.done():
            return
        conn.idle_reader = self._spawn(self._idle_read(conn))

    async def _stop_idle_reader(self, conn: _HeldConnection) -> None:
        task, conn.idle_reader = conn.idle_reader, None
        if task is not None:
            await utils.aio.gracefully_cancel(task)

    async def _idle_read(self, conn: _HeldConnection) -> None:
        """Own the socket between replies: notice closes, read `ready`, drop strays.

        Cancellation (a reply taking the socket over) propagates out of
        ``ws.receive()`` and ends the task before any bookkeeping, which is the
        intended exit. Without this reader a gateway close would go unnoticed
        until the next reply wrote to a dead socket, and any late frame from a
        finished reply would be read by the next one.

        It also retires the socket: once it has been idle for ``_MAX_IDLE_S``,
        with no reopen, and with a reopen in the background once it passes
        ``_MAX_CONNECTION_AGE_S``, when audio arrives, or when an error arrives
        before any reply has used it.
        """
        discarded = 0
        idle_since = time.perf_counter()
        # Why this reader retired the socket itself, if it did.
        retire: Literal["idle", "max_age", "stray_audio", "error_before_text"] | None = None

        def note_stray(frame_type: str) -> None:
            # Between replies the gateway should send nothing. A stray frame
            # here is the one signal that a reply's audio could reach the next
            # one, so say so the first time rather than only counting it.
            nonlocal discarded
            discarded += 1
            if discarded == 1:
                # Report on the first one, not on reader exit: a socket that
                # survives the whole call never exits, so an exit-time summary
                # would never be seen for the case that matters most.
                logger.warning(
                    "[TTS] stray frame on idle connection",
                    extra={"tts_model": self._opts.model, "frame_type": frame_type},
                )
                self._emit_plugin_event("tts.idle_stray_frame", "warning", frame_type=frame_type)

        next_keepalive_at = idle_since + _IDLE_KEEPALIVE_S
        try:
            while True:
                # Checked on every pass, so neither control traffic nor a run
                # of stray frames can put them off.
                now = time.perf_counter()
                if now - idle_since >= _MAX_IDLE_S:
                    retire = "idle"
                    break
                if self._past_max_age(conn):
                    retire = "max_age"
                    break
                if now >= next_keepalive_at:
                    # This reader owns the socket only between replies. The
                    # gateway's idle timer ignores transport pings, so without
                    # this frame a silent call loses its held socket on
                    # schedule and every later reply pays a reconnect.
                    await conn.ws.send_str(SynthesizeStream._KEEPALIVE_MSG)
                    next_keepalive_at = now + _IDLE_KEEPALIVE_S
                    continue
                try:
                    msg = await _receive_by(conn.ws, next_keepalive_at)
                except (TimeoutError, asyncio.TimeoutError):
                    continue
                if msg.type in _WS_END_TYPES:
                    _log_ws_end(conn.ws, msg, context="idle", tts_model=self._opts.model)
                    break
                if msg.type == aiohttp.WSMsgType.BINARY:
                    # Audio between replies means a turn is still producing
                    # it. Dropping this chunk does not make the socket quiet,
                    # and the next reply to read it would play the rest as its
                    # own, so the socket goes.
                    note_stray(str(msg.type))
                    retire = "stray_audio"
                    break
                if msg.type != aiohttp.WSMsgType.TEXT:
                    note_stray(str(msg.type))
                    continue
                try:
                    resp = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.debug("[TTS] ignoring non-JSON frame on idle connection")
                    continue
                if not isinstance(resp, dict):
                    continue
                if resp.get("type") == "ready":
                    self._note_ready(conn, resp)
                    self._emit_plugin_event(
                        "gateway.session",
                        gateway_request_id=conn.gateway_request_id,
                        gateway_session_id=conn.gateway_session_id,
                    )
                    continue
                event = _parse_ws_event(resp)
                if event.kind == "error":
                    logger.warning(
                        "[TTS] error on idle connection",
                        extra={
                            "tts_model": self._opts.model,
                            # A provider's message can quote the text it refused.
                            "lk.pii.error": event.error or _extract_error_message(resp),
                            "error_status": event.error_status,
                        },
                    )
                    self._emit_plugin_event(
                        "tts.idle_error",
                        "warning",
                        error=event.error or _extract_error_message(resp),
                        status=event.error_status,
                    )
                    if not conn.text_sent:
                        # No reply has used this socket, so the session it
                        # opened has never worked, as when the provider refuses
                        # the voice. A reply handed it would wait out its
                        # timeout; retiring it lets the reconnect backoff apply
                        # and the next reply report the error itself.
                        retire = "error_before_text"
                        break
                    # Otherwise not a terminator: the gateway reports a bad
                    # pronunciation reference, a not-ready backend or a failed
                    # translation on a connection it keeps. Tearing it down here
                    # would close a socket the gateway never closed, and a
                    # repeating error would do it in a loop.
                    continue
                if event.kind == "audio_chunk" or isinstance(resp.get("audio"), str):
                    note_stray(event.kind)
                    retire = "stray_audio"
                    break
                note_stray(event.kind)
        except Exception:
            # A read failure must still clear the held socket below; letting the
            # task die here would leave a dead socket installed and handed to
            # the next reply as if it were warm.
            logger.warning(
                "[TTS] idle connection read failed",
                extra={"tts_model": self._opts.model},
                exc_info=True,
            )

        if discarded:
            logger.debug("[TTS] discarded %d frame(s) on idle connection", discarded)
        async with self._ws_lock:
            owned = self._held is conn and not conn.in_use
            if owned:
                self._held = None
        if not owned:
            return
        if retire is not None:
            # Detached above, so no reply can pick it up while this says goodbye.
            logger.info(
                "[TTS] retiring idle connection",
                extra={"tts_model": self._opts.model, "reason": retire},
            )
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    conn.ws.send_str(SynthesizeStream._CLOSE_MSG), WS_CLOSE_TIMEOUT_S
                )
        await _close_ws(conn.ws, context=f"idle_{retire}" if retire else "idle_closed")
        if not self._closing and retire != "idle":
            self._on_connection_lost(conn, reason=retire or "reconnect")

    def _on_connection_lost(self, conn: _HeldConnection, *, reason: str) -> None:
        # A socket the gateway closed after a reply wrote to it (which some
        # models do every time, ending the turn by closing) is reopened at
        # once, so the next reply is still warm: it demonstrably works, so the
        # backoff below starts over. One that never carried text is backed off
        # and eventually abandoned, unless the plugin retired it itself: a
        # settings change is not a gateway failure.
        planned = reason in _PLANNED_REOPENS
        if conn.text_sent:
            self._failed_reopens = 0
        elif not planned:
            self._failed_reopens += 1
        self._schedule_connect(reason=reason, throttle=not (conn.text_sent or planned))

    def _schedule_connect(self, *, reason: str, throttle: bool = False) -> None:
        """Open the held connection in the background so the next reply finds it warm."""
        if not self.warm_standby_enabled or self._closing or not self._hold_allowed:
            return
        if self._session is not None and self._session.closed:
            # Nothing calls aclose on a TTS, so at the end of a job the HTTP
            # session closes under an instance that is still open, and the
            # held socket dies with it. A reconnect could only fail.
            logger.debug(
                "[TTS] background connect not scheduled: http session closed",
                extra={"tts_model": self._opts.model, "reason": reason},
            )
            return
        if not _has_running_loop():
            # prewarm() is synchronous, so it can be called with no running
            # loop. Warm standby is simply off then, and silence would leave
            # every reply paying a cold connect with nothing to explain it.
            logger.warning(
                "[TTS] background connect not scheduled: no running event loop",
                extra={"tts_model": self._opts.model, "reason": reason},
            )
            return
        if self._connect_task is not None and not self._connect_task.done():
            return
        now = time.perf_counter()
        last = self._last_background_connect_at
        if throttle:
            if self._failed_reopens > _MAX_BACKGROUND_REOPENS:
                logger.warning(
                    "[TTS] giving up on reopening the connection in the background; "
                    "replies will connect on demand",
                    extra={
                        "tts_model": self._opts.model,
                        "reason": reason,
                        "attempts": self._failed_reopens,
                    },
                )
                return
            wait = min(
                _BACKGROUND_CONNECT_MIN_INTERVAL_S * 2 ** (self._failed_reopens - 1),
                _BACKGROUND_CONNECT_MAX_INTERVAL_S,
            )
            if last is not None and now - last < wait:
                self._defer_connect(reason, delay=wait - (now - last))
                return
        self._cancel_deferred_connect()
        self._last_background_connect_at = now
        self._connect_task = self._spawn(self._background_connect(reason))

    def _defer_connect(self, reason: str, *, delay: float) -> None:
        """Wait out the throttle window, then connect.

        Dropping the attempt instead would be permanent: nothing else re-arms
        it, so a socket that dies inside the window (a gateway that accepts and
        closes at once, or a prewarmed connection refused straight away) would
        leave the call with no held connection for the rest of its life.
        """
        if self._reconnect_handle is not None:
            return
        logger.debug(
            "[TTS] background connect deferred",
            extra={"tts_model": self._opts.model, "reason": reason, "delay_s": delay},
        )

        def retry() -> None:
            self._reconnect_handle = None
            self._schedule_connect(reason=reason)

        self._reconnect_handle = asyncio.get_running_loop().call_later(delay, retry)

    def _cancel_deferred_connect(self) -> None:
        handle, self._reconnect_handle = self._reconnect_handle, None
        if handle is not None:
            handle.cancel()

    async def _background_connect(self, reason: str) -> None:
        if self._held is not None:
            # A deferred retry that is no longer needed: a reply opened its own
            # socket while this waited, and that socket is now the held one.
            return
        epoch = self._ws_epoch
        try:
            ws, timing = await self._connect_ws(timeout=_PREWARM_CONNECT_TIMEOUT_S)
        except Exception:
            logger.warning(
                "[TTS] background connect failed",
                extra={"tts_model": self._opts.model, "reason": reason},
                exc_info=True,
            )
            return
        conn = _HeldConnection(ws=ws, epoch=epoch, opened_by=reason, timing=timing)
        installed = False
        try:
            async with self._ws_lock:
                if (
                    not self._closing
                    and self._hold_allowed
                    and self._held is None
                    and epoch == self._ws_epoch
                ):
                    self._held = conn
                    self._start_idle_reader(conn)
                    installed = True
        finally:
            if not installed:
                await _close_ws(ws, context="background_connect_discarded")
                # Whatever refused the socket may have changed by now: new
                # options, or a session attaching to a detached instance. A
                # connect asked for meanwhile was turned away, because this
                # task still counted as in flight, so ask again here.
                if not self._closing and (
                    epoch != self._ws_epoch or (self._hold_allowed and self._held is None)
                ):
                    # Clear the slot first: _schedule_connect refuses to start
                    # while a connect is in flight, and that connect is this
                    # very task, so the reschedule would otherwise be dropped
                    # and the call would run cold from here on.
                    if self._connect_task is asyncio.current_task():
                        self._connect_task = None
                    self._schedule_connect(
                        reason="options_changed" if epoch != self._ws_epoch else reason
                    )

    async def _drop_connection(self, *, context: str) -> None:
        """Retire this candidate's socket and stop it holding or reopening one.

        A socket another reply still owns is left to that reply: holding is
        disallowed first, so its release closes it instead of keeping it, and
        nothing reopens it afterwards.
        """
        self._hold_allowed = False
        self._cancel_deferred_connect()
        connect_task, self._connect_task = self._connect_task, None
        if connect_task is not None and not connect_task.done():
            await utils.aio.gracefully_cancel(connect_task)
        async with self._ws_lock:
            held = self._held
            if held is None:
                return
            if held.in_use:
                logger.debug(
                    "[TTS] held socket still in use at candidate switch; closes on release",
                    extra={"tts_model": self._opts.model, "context": context},
                )
                return
            self._held = None
            await self._stop_idle_reader(held)
        await _close_ws(held.ws, context=context)

    async def _drop_inactive_candidate_connections(self, active: TTS) -> None:
        """Close every other candidate's socket, so a chain holds one in steady state."""
        # A stream is using the instance, whatever its listeners say, so a
        # detach still being carried out stops here.
        self._detached = False
        # A candidate the chain comes back to (the primary, after its recovery
        # cooldown) has to be allowed to hold a socket again.
        active._hold_allowed = True
        for candidate in self._candidate_tts:
            if candidate is not active:
                await candidate._drop_connection(context="candidate_switch")

    async def _invalidate_connection(self) -> None:
        """Retire a held socket whose init predates the current options."""
        async with self._ws_lock:
            held = self._held
            if held is None or held.in_use or held.epoch == self._ws_epoch:
                return
            self._held = None
            await self._stop_idle_reader(held)
        await _close_ws(held.ws, context="options_changed")
        if not self._closing:
            self._on_connection_lost(held, reason="options_changed")

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            voice (str): Voice to use.
            language (str): Language code.
            speed (float): Playback speed multiplier.
        """
        invalidate_connection = False
        if is_given(voice):
            if not voice.strip():
                raise ValueError("voice is required")
            invalidate_connection = invalidate_connection or self._opts.voice != voice
            self._opts.voice = voice
        if is_given(language):
            invalidate_connection = invalidate_connection or self._opts.language != language
            self._opts.language = language
        if is_given(speed):
            invalidate_connection = invalidate_connection or self._opts.speed != speed
            self._opts.speed = speed

        # The held socket was initialized with the old voice/language/speed.
        # Bump the epoch synchronously so nothing hands it out again (no await
        # here, so no acquire can interleave), then retire the socket in the
        # background. An in-use socket is left alone: its release sees the stale
        # epoch and closes it.
        if invalidate_connection:
            self._ws_epoch += 1
            held = self._held
            if held is not None and not held.in_use:
                if _has_running_loop():
                    self._spawn(self._invalidate_connection())
                else:
                    logger.warning(
                        "[TTS] held socket not retired: no running event loop",
                        extra={"tts_model": self._opts.model},
                    )
        # Keep fallback candidates consistent with the primary so a later
        # failover does not synthesize with construction-time settings. A
        # candidate with an explicit per-candidate voice keeps it.
        if not self._is_candidate:
            for candidate in self._candidate_tts[1:]:
                candidate.update_options(
                    voice=voice if is_given(voice) and candidate._inherits_voice else NOT_GIVEN,
                    language=language,
                    speed=speed,
                )

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        if not self._is_candidate:
            return _FallbackChunkedStream(
                parent=self,
                text=text,
                conn_options=conn_options,
            )  # type: ignore[return-value]
        return self._synthesize_candidate(text, conn_options=conn_options)

    def _synthesize_candidate(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions,
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        if not self._is_candidate:
            return _FallbackSynthesizeStream(
                parent=self,
                conn_options=conn_options,
            )  # type: ignore[return-value]
        return self._stream_candidate(conn_options=conn_options)

    def _stream_candidate(
        self,
        *,
        conn_options: APIConnectOptions,
    ) -> SynthesizeStream:
        logger.debug("[TTS] TTS.stream() called, creating SynthesizeStream")
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        logger.debug("[TTS] TTS.stream() returning stream")
        return stream

    def prewarm(self) -> None:
        """Open this instance's connection ahead of the first reply.

        ``AgentSession`` calls this at session start. With ``connections=[...]``
        only the candidate the chain starts on connects. No-op when
        ``warm_standby_enabled`` is False, or when there is no running loop.
        """
        if len(self._candidate_tts) > 1 and not self._is_candidate:
            active = self._candidate_tts[self._candidate_state.start()]
            if active is not self:
                active.prewarm()
                return
        # Open the one connection this instance reuses for the whole call, so
        # the first reply does not pay connect and init. Every later reply
        # reuses it; there is never a second socket in steady state. With
        # warm_standby_enabled=False nothing is held: each reply opens its own.
        self._schedule_connect(reason="prewarm")

    async def aclose(self) -> None:
        self._closing = True
        self._cancel_deferred_connect()
        if self._detach_handle is not None:
            self._detach_handle.cancel()
            self._detach_handle = None
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()

        # Settle a background connect first, so it cannot install a socket after
        # the held one is detached below. Its own cleanup closes what it opened.
        connect_task, self._connect_task = self._connect_task, None
        if connect_task is not None and not connect_task.done():
            _done, still_running = await asyncio.wait({connect_task}, timeout=WS_CLOSE_TIMEOUT_S)
            if still_running:
                await utils.aio.gracefully_cancel(connect_task)

        # Detach the held socket and stop its idle reader first: that reader
        # blocks in receive() indefinitely and must not be waited on below.
        async with self._ws_lock:
            held, self._held = self._held, None
            if held is not None:
                await self._stop_idle_reader(held)
        # Let every release and cancel drain finish; each checks _closing before
        # installing or reopening anything. Bounded: a drain is at most the
        # settle window plus one socket close.
        pending = [task for task in self._background_tasks if not task.done()]
        if pending:
            _done, still_running = await asyncio.wait(
                pending, timeout=WS_CLOSE_TIMEOUT_S + _CANCEL_SETTLE_S + _CANCEL_QUIET_S
            )
            for task in still_running:
                await utils.aio.gracefully_cancel(task)
        if held is not None:
            if self._is_ws_usable(held.ws):
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(
                        held.ws.send_str(SynthesizeStream._CLOSE_MSG), WS_CLOSE_TIMEOUT_S
                    )
            await _close_ws(held.ws, context="tts_aclose")

        if not self._is_candidate:
            for candidate in self._candidate_tts[1:]:
                await candidate.aclose()


class ChunkedStream(tts.ChunkedStream):
    """Non-streaming synthesis: send the full text once over the SLNG WebSocket."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )

        if not self._input_text:
            output_emitter.flush()
            return

        ws: aiohttp.ClientWebSocketResponse | None = None
        audio_received = False

        try:
            # Chunked synthesis always uses a dedicated connection: it is a
            # one-shot call, not part of a call's reply stream.
            ws, timing = await self._tts._connect_ws(timeout=self._conn_options.timeout)
            self._acquire_time = timing.connect_total_ms / 1000
            await ws.send_str(json.dumps({"type": "text", "text": self._input_text}))
            await ws.send_str(SynthesizeStream._FLUSH_MSG)

            while True:
                msg = await ws.receive(timeout=self._conn_options.timeout)
                if msg.type in _WS_END_TYPES:
                    close_info = _log_ws_end(ws, msg, context="chunked", request_id=request_id)
                    if audio_received and _transport_failed(ws, msg):
                        # The audio so far is a fragment; returning it as the
                        # whole synthesis would hide that it was cut off.
                        raise APIConnectionError(
                            "SLNG websocket failed partway through synthesis: "
                            f"{close_info['ws_error'] or 'connection lost'}"
                        )
                    # Several bridge providers (Rime, Cartesia) terminate a
                    # segment by closing the socket. Mirror the streaming path:
                    # a close after audio is a normal end-of-segment, not an
                    # error. Only a close with no audio at all is a failure.
                    if audio_received:
                        output_emitter.flush()
                        break
                    raise APIStatusError("SLNG websocket connection closed unexpectedly")

                if msg.type == aiohttp.WSMsgType.BINARY:
                    output_emitter.push(msg.data)
                    audio_received = True
                    continue

                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    resp = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.debug(
                        "[SLNG TTS] ignoring non-JSON text frame",
                        extra={"lk.pii.data": msg.data},
                    )
                    continue

                if not isinstance(resp, dict):
                    continue

                if "type" not in resp:
                    # Raw provider passthrough (e.g. ElevenLabs): {"audio", "isFinal"}.
                    is_final = _is_final_frame(resp)
                    audio_b64 = resp.get("audio")
                    if isinstance(audio_b64, str) and audio_b64:
                        # The decode is guarded on its own: an emitter failure
                        # is not an encoding problem, and swallowing it here
                        # would resurface as "connection closed unexpectedly".
                        decoded = _decode_b64_audio(audio_b64, context="chunked synthesis")
                        if decoded is not None:
                            output_emitter.push(decoded)
                            audio_received = True

                    if is_final:
                        output_emitter.flush()
                        break

                    if resp.get("error") is not None:
                        raise APIStatusError(
                            f"SLNG TTS error: {resp.get('error')}",
                            status_code=_extract_error_status(resp) or -1,
                        )
                    continue

                event = _parse_ws_event(resp)
                if event.kind == "ignore":
                    continue
                if event.kind == "audio_chunk":
                    if event.audio:
                        output_emitter.push(event.audio)
                        audio_received = True
                elif event.kind == "audio_end":
                    if event.audio:
                        output_emitter.push(event.audio)
                        audio_received = True
                    output_emitter.flush()
                    break
                elif event.kind == "error":
                    raise APIStatusError(
                        f"SLNG TTS error: {event.error}",
                        status_code=event.error_status or -1,
                    )
                else:
                    logger.debug("[TTS] ignoring unknown message", extra={"lk.pii.data": resp})
        except (TimeoutError, asyncio.TimeoutError):
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from None
        except (APIStatusError, APIConnectionError):
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            if ws is not None:
                await _close_ws(ws, context="chunked_synthesis")


class SynthesizeStream(tts.SynthesizeStream):
    # SLNG protocol messages (different from Deepgram). `clear` rather than
    # `cancel` for an interrupt: the two abort the turn identically, but only
    # `clear` is acknowledged, and it is the one most models map to a native
    # stop. A cancel that is never acknowledged costs the settle window and the
    # socket on every barge-in.
    _FLUSH_MSG: str = json.dumps({"type": "flush"})
    _CLEAR_MSG: str = json.dumps({"type": "clear"})
    _CLOSE_MSG: str = json.dumps({"type": "close"})
    _KEEPALIVE_MSG: str = json.dumps({"type": "keepalive"})

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        logger.debug("[TTS] SynthesizeStream.__init__ STARTING")
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        logger.debug("[TTS] SynthesizeStream.__init__ DONE")

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        # Create segments_ch per run so base-class retries after an error get a
        # fresh channel (matching the Deepgram plugin pattern).
        self._segments_ch = utils.aio.Chan[_SegmentInput]()
        request_id = utils.shortuuid()
        logger.debug(f"[TTS] _run starting: request_id={request_id}")
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        async def _tokenize_input() -> None:
            # Converts incoming text into token streams and sends them into _segments_ch
            logger.debug("[TTS] _tokenize_input starting, waiting for input...")
            segment: _SegmentInput | None = None
            text_count = 0
            async for the_input in self._input_ch:
                if isinstance(the_input, str):
                    text_count += 1
                    if text_count == 1:
                        logger.debug(f"[TTS] First text received: '{the_input[:50]}...'")
                    if segment is None:
                        segment = _SegmentInput(stream=self._open_tokenizer_stream())
                        self._segments_ch.send_nowait(segment)
                        logger.debug("[TTS] New token stream created")
                    segment.stream.push_text(the_input)
                elif isinstance(the_input, self._FlushSentinel):
                    logger.debug(f"[TTS] Flush sentinel received after {text_count} texts")
                    if segment is not None:
                        # Mark before ending the input: the sender reads this to
                        # know the next tokens it pulls are the reply's last.
                        segment.input_ended = True
                        segment.stream.end_input()
                    segment = None

            logger.debug(f"[TTS] _tokenize_input done: {text_count} total texts")
            self._segments_ch.close()

        async def _run_segments() -> None:
            logger.debug("[TTS] _run_segments starting, waiting for word_streams...")
            segment_count = 0
            async for segment in self._segments_ch:
                segment_count += 1
                logger.debug(f"[TTS] Processing segment {segment_count}")
                await self._run_ws(segment, output_emitter)
            logger.debug(f"[TTS] _run_segments done: {segment_count} segments")

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_run_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        except (TimeoutError, asyncio.TimeoutError):
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from None
        except (APIStatusError, APIConnectionError):
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    def _open_tokenizer_stream(self) -> tokenize.WordStream | tokenize.SentenceStream:
        """Start one reply's tokenizer stream, saying whether it may run ahead.

        The plugin's own tokenizer can release the opening of a long sentence
        before the sentence is finished, which on a model that starts on part of
        a sentence is worth most of half a second on the first audio of a reply.
        It is only allowed to when the gateway has said it understands such a
        frame: one that has not would either drop it or refuse it, and sending
        an opening as an ordinary frame is the fragment that sentence framing
        exists to avoid.
        """
        tokenizer = self._opts.word_tokenizer
        if isinstance(tokenizer, SentenceTokenizer):
            return tokenizer.stream(partial_head=self._tts._partial_text_supported is True)
        return tokenizer.stream()

    async def _run_ws(
        self,
        segment: _SegmentInput,
        output_emitter: tts.AudioEmitter,
    ) -> None:
        segment_id = utils.shortuuid()
        logger.debug(f"[TTS] _run_ws starting: segment_id={segment_id}")
        segment_started_at = time.perf_counter()
        output_emitter.start_segment(segment_id=segment_id)
        input_sent_event = asyncio.Event()
        # The sender's state lives at segment scope, not inside send_task, so a
        # reconnect replays the whole reply: tokens are consumed from the
        # tokenizer once and a new attempt cannot get them back.
        # `sent_frames` is what reached the wire, `pending_frames` is what was
        # framed but not written yet, and `batcher` holds tokens not yet framed.
        sent_frames: list[_TextFrame] = []
        pending_frames: list[_TextFrame] = []
        batcher = _FrameBatcher(
            mode=self._opts.text_chunking, max_chars=self._opts.phrase_max_chars
        )
        # True once every frame of the reply has been produced (flush included).
        frames_complete = False
        outcome = "completed"
        # What went wrong, when something did: a segment logged with
        # outcome="error" and no cause is a support ticket on its own. The
        # message is kept apart from the type because it can carry a
        # provider's message, and that can quote the text it refused.
        error_type: str | None = None
        error_detail: str | None = None
        ws_close_code: object = None
        ws_connect_ms: float | None = None
        init_send_ms: float | None = None
        connect_total_ms: float | None = None
        ready_ms: float | None = None
        first_text_send_ms: float | None = None
        first_audio_ms: float | None = None
        audio_end_ms: float | None = None
        gateway_request_id: str | None = None
        gateway_session_id: str | None = None
        audio_chunks_seen = 0
        standby_enabled = self._tts.warm_standby_enabled
        # Where this segment's socket came from, and how many times a dead
        # reused socket forced a reconnect-and-replay. `ws_source` says whether
        # this segment opened the socket; `ws_opened_by` says who did open it
        # (prewarm, reconnect, options_changed, segment) and `ws_first_use`
        # whether this is the first reply on it, which is what separates a
        # prewarmed connection from a cold one in the log.
        ws_source: str | None = None
        ws_opened_by: str | None = None
        ws_first_use: bool | None = None
        ws_reconnects = 0
        # Text is on the wire and audio_end has not arrived: an interrupt now
        # has to tell the gateway to cancel.
        in_flight = False
        # Whether this attempt has started writing the reply's flush, and
        # whether the gateway ended the reply before that. An end of reply
        # with text still to send leaves that text to start a turn nobody is
        # reading, so the socket cannot be kept for the next reply.
        flush_sent = False
        ended_early = False

        def capture_ws_timing(conn: _HeldConnection) -> None:
            # Only meaningful when this segment opened the socket: a reused one
            # was paid for by an earlier reply or by prewarm.
            nonlocal ws_connect_ms, init_send_ms, connect_total_ms
            ws_connect_ms = conn.timing.ws_connect_ms
            init_send_ms = conn.timing.init_send_ms
            connect_total_ms = conn.timing.connect_total_ms

        def mark_first_text_sent() -> None:
            nonlocal first_text_send_ms
            if first_text_send_ms is None:
                first_text_send_ms = _elapsed_ms(segment_started_at)

        def mark_first_audio_seen() -> None:
            nonlocal first_audio_ms
            if first_audio_ms is None:
                first_audio_ms = _elapsed_ms(segment_started_at)

        def log_segment_timing() -> None:
            logger.info(
                "[TTS] segment timing",
                extra={
                    "tts_model": self._opts.model,
                    "tts_provider": self._tts.provider,
                    "tts_endpoint": self._opts.model_endpoint,
                    "segment_id": segment_id,
                    "outcome": outcome,
                    "error_type": error_type,
                    "lk.pii.error": error_detail,
                    "ws_close_code": ws_close_code,
                    "ws_connect_ms": ws_connect_ms,
                    "init_send_ms": init_send_ms,
                    "connect_total_ms": connect_total_ms,
                    "ready_ms": ready_ms,
                    "first_text_send_ms": first_text_send_ms,
                    "first_audio_ms": first_audio_ms,
                    "audio_end_ms": audio_end_ms,
                    "segment_total_ms": _elapsed_ms(segment_started_at),
                    "audio_chunks": audio_chunks_seen,
                    "gateway_request_id": gateway_request_id,
                    "gateway_session_id": gateway_session_id,
                    "standby_enabled": standby_enabled,
                    "ws_source": ws_source,
                    "ws_opened_by": ws_opened_by,
                    "ws_first_use": ws_first_use,
                    "ws_reconnects": ws_reconnects,
                },
            )

        async def guarded_send(conn: _HeldConnection, data: str) -> None:
            nonlocal in_flight
            # Set before the write, not after: a cancel can land inside send_str
            # once the transport is applying backpressure, and the bytes may
            # already be on the wire. Assuming they were costs one needless
            # cancel; assuming they were not leaves the gateway holding text
            # that it would voice at the head of the next reply.
            in_flight = True
            try:
                await conn.ws.send_str(data)
            except Exception as exc:
                # A reused socket that fails before this reply produced any audio
                # was closed by the gateway (or died) while it sat idle. Reconnect
                # and replay rather than failing the reply. Every send failure is
                # treated this way on purpose: one replay, then a hard failure
                # that still carries the original cause.
                if conn.reused and audio_chunks_seen == 0:
                    raise _StaleConnection() from exc
                raise
            conn.text_sent = True
            input_sent_event.set()

        async def write_pending(conn: _HeldConnection) -> None:
            """Write queued frames, dropping each only once it is on the wire."""
            nonlocal flush_sent
            while pending_frames:
                frame, flush, partial = pending_frames[0]
                if partial and not conn.accepts_partial_text:
                    # This socket has not said it can take an opening, and an
                    # opening sent as an ordinary frame is the bare fragment
                    # that sentence framing exists to avoid. Put it back with
                    # the piece that finishes the sentence, which costs this
                    # reply its head start and nothing else. Not a rare path:
                    # a freshly opened socket has not read its `ready` yet,
                    # because the reader waits for this sender to write first.
                    if len(pending_frames) > 1:
                        nxt = pending_frames[1]
                        pending_frames[1] = _TextFrame(frame + nxt.text, nxt.flush, nxt.partial)
                        pending_frames.pop(0)
                        continue
                    if not flush:
                        return  # nothing to merge into yet; wait for the rest
                    # The reply ends here, so this frame is the whole of it.
                    partial = False
                self._mark_started()
                payload: dict[str, object] = {"type": "text", "text": frame}
                if partial and conn.accepts_partial_text:
                    # "more of this sentence follows", so the gateway can start
                    # on it without hearing it as a whole utterance. Only ever
                    # set for a gateway that said it understands the field: an
                    # older one would drop or refuse it.
                    payload["partial"] = True
                await guarded_send(conn, json.dumps(payload))
                if flush:
                    # The turn's terminator, as its own message rather than a
                    # flag on the text frame. Both forms are in the bridge
                    # contract, but a model only sees the flag if it declares
                    # one of its own, and a turn that is never terminated plays
                    # to the end and then hangs until the connection times out.
                    # Never sent on its own: without preceding text there is no
                    # turn to terminate, and saying so is a protocol violation.
                    # Marked before the write, as in guarded_send: the end of
                    # the reply can be read before send_str returns.
                    flush_sent = True
                    await guarded_send(conn, self._FLUSH_MSG)
                pending_frames.pop(0)
                sent_frames.append(_TextFrame(frame, flush, partial))
                mark_first_text_sent()

        async def send_task(conn: _HeldConnection) -> None:
            nonlocal frames_complete, flush_sent

            # A reconnect starts a fresh turn on a new socket, so everything an
            # earlier attempt wrote has to go out again, ahead of whatever it
            # had framed but never managed to write.
            if sent_frames:
                pending_frames[:0] = sent_frames
                sent_frames.clear()
            await write_pending(conn)
            if frames_complete:
                return

            stream = segment.stream
            token_count = 0
            while True:
                try:
                    data = await stream.__anext__()
                except StopAsyncIteration:
                    break
                token_count += 1
                if token_count == 1:
                    logger.debug(f"[TTS] send_task: first token '{data.token}'")
                if not segment.input_ended:
                    # More text is still coming, so this token cannot close the
                    # reply: send its frame straight away without the flush.
                    ready_frame = batcher.push(data.token, separator=separator_after(data))
                    if ready_frame is not None:
                        pending_frames.append(_TextFrame(ready_frame, False, is_partial_head(data)))
                        await write_pending(conn)
                    continue

                # The reply's text is complete: every remaining token is already
                # queued behind a closed channel, so draining never suspends.
                # Batch the whole tail first so the flush lands on its last frame.
                tail = [data]
                while True:
                    try:
                        tail.append(await stream.__anext__())
                    except StopAsyncIteration:
                        break
                token_count += len(tail) - 1
                frames: list[_TextFrame] = []
                for item in tail:
                    framed = batcher.push(item.token, separator=separator_after(item))
                    if framed is not None:
                        frames.append(_TextFrame(framed, False, is_partial_head(item)))
                last = batcher.finish()
                if last is not None:
                    frames.append(_TextFrame(last, False))
                if frames:
                    # The reply ends here, so its last frame carries the flush
                    # and cannot be an opening: there is no remainder to follow.
                    frames[-1] = frames[-1]._replace(flush=True, partial=False)
                    pending_frames.extend(frames)
                    frames_complete = True
                await write_pending(conn)
                break

            if not frames_complete:
                last = batcher.finish()
                if last is not None:
                    pending_frames.append(_TextFrame(last, False))
                if pending_frames:
                    # Whatever is still queued is the end of this reply: a tail
                    # the batcher was holding, or an opening held back for a
                    # remainder the reply turned out not to have. It carries the
                    # terminator, and nothing follows it, so it is not an
                    # opening any more.
                    pending_frames[-1] = pending_frames[-1]._replace(flush=True, partial=False)
                    frames_complete = True
                    await write_pending(conn)
                elif sent_frames:
                    # Every frame is already on the wire but none of them
                    # carried the terminator, so it goes now. A tokenizer that
                    # emitted its last token before input_ended was observed
                    # lands here, and so does a reply that ends straight after
                    # an opening this plugin's tokenizer released (the rest was
                    # only whitespace): the flush also completes that opening.
                    flush_sent = True
                    await guarded_send(conn, self._FLUSH_MSG)
                    sent_frames[-1] = sent_frames[-1]._replace(flush=True)
                    frames_complete = True

            logger.debug(f"[TTS] send_task: sent {len(sent_frames)} frames ({token_count} tokens)")
            input_sent_event.set()

        async def recv_task(conn: _HeldConnection) -> None:
            nonlocal ready_ms, audio_end_ms, gateway_request_id, gateway_session_id
            nonlocal audio_chunks_seen, in_flight, outcome, ws_close_code

            def push_audio(data: bytes) -> None:
                nonlocal audio_chunks_seen
                audio_chunks_seen += 1
                mark_first_audio_seen()
                output_emitter.push(data)

            def mark_segment_end() -> None:
                # in_flight = False is what stops the interrupt path sending a
                # cancel for a reply the gateway already finished, so every exit
                # from the loop below goes through here.
                nonlocal audio_end_ms, in_flight
                audio_end_ms = audio_end_ms or _elapsed_ms(segment_started_at)
                in_flight = False
                output_emitter.end_segment()

            def note_reply_end() -> None:
                # The gateway ends a turn on the flush, and when the provider
                # behind it closes. The second can come before this reply
                # finished writing; the reply then ends with the audio it has,
                # and the socket is closed rather than kept.
                nonlocal ended_early, outcome
                if flush_sent:
                    return
                ended_early = True
                outcome = "ended_before_flush"
                logger.warning(
                    "[TTS] gateway ended the reply before its flush; closing the connection",
                    extra={"tts_model": self._opts.model, "segment_id": segment_id},
                )

            ws = conn.ws
            await input_sent_event.wait()
            if not sent_frames and not pending_frames:
                # The tokenizer produced nothing at all (whitespace-only text),
                # so nothing was sent and no audio is coming. Text that carries
                # no letters is sent now, so it no longer lands here.
                logger.debug("[TTS] recv_task: no frames were sent, ending segment")
                output_emitter.end_segment()
                return
            while True:
                try:
                    msg = await ws.receive(timeout=self._conn_options.timeout)
                except (TimeoutError, asyncio.TimeoutError):
                    # A reused socket that never answers died silently while idle.
                    if conn.reused and audio_chunks_seen == 0:
                        raise _StaleConnection() from None
                    raise
                if msg.type in _WS_END_TYPES:
                    close_info = _log_ws_end(ws, msg, context="segment", segment_id=segment_id)
                    ws_close_code = close_info["ws_close_code"]
                    if audio_chunks_seen > 0 and _transport_failed(ws, msg):
                        # The connection broke partway through the reply, so
                        # the audio so far is a fragment. Ending the segment
                        # here would pass it off as the whole reply.
                        raise APIConnectionError(
                            f"SLNG websocket failed after {audio_chunks_seen} audio chunk(s): "
                            f"{close_info['ws_error'] or 'connection lost'}"
                        )
                    if audio_chunks_seen > 0:
                        # No audio_end arrived, so whether the reply finished
                        # is the gateway's word against a closed socket. Some
                        # models do end a turn by closing, so this is not an
                        # error, but it is not the clean path either: say which
                        # one happened rather than calling it terminal output.
                        outcome = "closed_no_audio_end"
                        logger.info(
                            "[TTS] recv_task: websocket closed after %s chunk(s) with no "
                            "audio_end; treating the reply as complete",
                            audio_chunks_seen,
                            extra={"segment_id": segment_id, **close_info},
                        )
                        mark_segment_end()
                        break
                    if conn.reused:
                        raise _StaleConnection()
                    raise APIStatusError("SLNG websocket connection closed unexpectedly")

                # Binary is what the bridge sends; the JSON audio frames below
                # are the compatibility path for providers it passes through
                # untranslated.
                if msg.type == aiohttp.WSMsgType.BINARY:
                    push_audio(msg.data)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        resp = json.loads(msg.data)
                    except json.JSONDecodeError:
                        logger.debug(
                            "[SLNG TTS] ignoring non-JSON text frame",
                            extra={"lk.pii.data": msg.data},
                        )
                        continue

                    if not isinstance(resp, dict):
                        continue

                    if resp.get("type") == "ready":
                        if ready_ms is None:
                            ready_ms = _elapsed_ms(segment_started_at)
                        self._tts._note_ready(conn, resp)
                        gateway_request_id = conn.gateway_request_id
                        gateway_session_id = conn.gateway_session_id
                        self._tts._emit_plugin_event(
                            "gateway.session",
                            gateway_request_id=gateway_request_id,
                            gateway_session_id=gateway_session_id,
                        )
                        continue

                    if "type" not in resp:
                        is_final = _is_final_frame(resp)
                        audio_b64 = resp.get("audio")
                        if isinstance(audio_b64, str) and audio_b64:
                            # Only the decode is guarded. An emitter failure is
                            # not an encoding problem, and swallowing it here
                            # would drop a whole reply's audio while the log
                            # pointed at the gateway.
                            decoded = _decode_b64_audio(
                                audio_b64,
                                context="isFinal frame" if is_final else "audio frame",
                            )
                            if decoded is not None:
                                push_audio(decoded)

                        if is_final:
                            note_reply_end()
                            mark_segment_end()
                            break

                        if resp.get("error") is not None:
                            raise APIStatusError(
                                f"SLNG TTS error: {resp.get('error')}",
                                status_code=_extract_error_status(resp) or -1,
                            )
                        continue

                    event = _parse_ws_event(resp)
                    if event.kind == "ignore":
                        continue

                    if event.kind == "audio_chunk":
                        if event.audio:
                            push_audio(event.audio)

                    # The reply's terminal frame. The bridge sends a bare
                    # audio_end; the provider spellings ("Flushed", "done")
                    # and a payload on the terminal frame are defensive, for
                    # a route that passes a provider through untranslated.
                    elif event.kind == "audio_end":
                        if event.audio:
                            push_audio(event.audio)
                        logger.debug(f"[TTS] recv_task: audio_end after {audio_chunks_seen} chunks")
                        note_reply_end()
                        mark_segment_end()
                        break

                    elif event.kind == "error":
                        raise APIStatusError(
                            f"SLNG TTS error: {event.error}",
                            status_code=event.error_status or -1,
                        )

                    else:
                        logger.debug("[TTS] ignoring unknown message", extra={"lk.pii.data": resp})

        conn: _HeldConnection | None = None
        tasks: list[asyncio.Task[None]] = []
        try:
            while True:
                # Each attempt re-runs both tasks, so the handshake between
                # them starts clear; the replay re-queues the frames first.
                input_sent_event.clear()
                flush_sent = False
                acquire_started_at = time.perf_counter()
                conn = await self._tts._acquire_connection(timeout=self._conn_options.timeout)
                # Reported in this reply's TTSMetrics.
                self._acquire_time = time.perf_counter() - acquire_started_at
                self._connection_reused = conn.reused
                ws_source = "reused" if conn.reused else conn.opened_by
                ws_opened_by = conn.opened_by
                ws_first_use = conn.served_segments == 0
                if not conn.reused:
                    capture_ws_timing(conn)
                gateway_request_id = gateway_request_id or conn.gateway_request_id
                gateway_session_id = gateway_session_id or conn.gateway_session_id
                tasks = [
                    asyncio.create_task(send_task(conn)),
                    asyncio.create_task(recv_task(conn)),
                ]
                try:
                    await asyncio.gather(*tasks)
                except _StaleConnection as exc:
                    await utils.aio.gracefully_cancel(*tasks)
                    ws_reconnects += 1
                    await self._tts._release_connection(conn, keep=False)
                    conn = None
                    in_flight = False
                    if ws_reconnects > 1:
                        raise APIConnectionError(
                            "SLNG websocket closed before producing audio"
                        ) from exc
                    logger.info(
                        "[TTS] reused websocket was dead, reconnecting and replaying segment",
                        extra={"tts_model": self._opts.model, "segment_id": segment_id},
                    )
                    continue
                finally:
                    input_sent_event.set()
                    await utils.aio.gracefully_cancel(*tasks)
                break

            conn.served_segments += 1
            # The socket stays open for the next reply; only the segment ends here.
            await self._tts._release_connection(conn, keep=not ended_early)
            conn = None
        except asyncio.CancelledError:
            outcome = "cancelled"
            if conn is not None:
                # Only drain when nobody can still be inside ws.receive() on this
                # socket: a second cancel can land while the inner finally awaits.
                self._tts._release_after_cancel(
                    conn,
                    in_flight=in_flight,
                    tasks_done=all(task.done() for task in tasks),
                )
                conn = None
            raise
        except Exception as exc:
            outcome = "error"
            error_type = type(exc).__name__
            error_detail = str(exc)
            if conn is not None:
                await self._tts._release_connection(conn, keep=False)
                conn = None
            raise
        finally:
            if conn is not None:
                # Neither handler above ran, so this left on something other
                # than an exception or a cancel. The socket is still marked
                # in use, and a socket that is never handed back is one the
                # instance can never reuse or close.
                logger.warning(
                    "[TTS] segment left its connection in use; closing it",
                    extra={"tts_model": self._opts.model, "segment_id": segment_id},
                )
                self._tts._spawn(self._tts._release_connection(conn, keep=False))
            log_segment_timing()


class _FallbackStreamBase:
    def __init__(self, *, parent: TTS, conn_options: APIConnectOptions) -> None:
        self._parent = parent
        self._conn_options = conn_options
        self._index = parent._candidate_state.start()
        self._parent._active_candidate_index = self._index
        self._stream: Any = None
        self._deadline: float | None = None
        self._started = False
        self._closed = False
        self._attempts = 0

    async def _start_stream(self) -> None:
        raise NotImplementedError

    def __aiter__(self) -> _FallbackStreamBase:
        return self

    async def __aenter__(self) -> _FallbackStreamBase:
        if self._stream is None:
            await self._start_stream()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def __anext__(self) -> Any:
        if self._closed:
            raise StopAsyncIteration
        if self._stream is None:
            # Lazily start so plain `async for` (without `async with` or an
            # explicit __aenter__) works exactly like the non-fallback streams.
            await self._start_stream()
        while True:
            try:
                if not self._started and self._deadline is not None:
                    timeout = max(0.0, self._deadline - time.monotonic())
                    item = await asyncio.wait_for(self._stream.__anext__(), timeout)
                else:
                    item = await self._stream.__anext__()
            except StopAsyncIteration:
                if self._started:
                    raise
                await self._handle_failure(APIConnectionError("TTS produced no audio"))
                continue
            except (TimeoutError, asyncio.TimeoutError) as exc:
                if self._started or self._deadline is None:
                    raise
                await self._handle_failure(exc, allow_retry=False)
                continue
            except Exception as exc:
                if self._started:
                    raise
                await self._handle_failure(exc)
                continue
            self._started = True
            self._deadline = None
            return item

    async def aclose(self) -> None:
        self._closed = True
        stream, self._stream = self._stream, None
        if stream is not None:
            with contextlib.suppress(Exception):
                await stream.aclose()

    async def _handle_failure(self, exc: BaseException, *, allow_retry: bool = True) -> None:
        candidate = self._parent._candidate_tts[self._index]
        if is_payload_too_large(exc):
            # Every candidate receives the same oversized request body, so
            # walking the chain cannot help; surface immediately.
            self._parent._emit_plugin_event(
                "fallback.exhausted",
                "error",
                from_model=candidate._opts.model,
                terminal_status=413,
                error=str(exc),
            )
            raise exc
        if (
            allow_retry
            and not is_non_retryable_client_error(exc)
            and self._attempts < self._conn_options.max_retry
        ):
            self._attempts += 1
            logger.warning(
                "[TTS] attempt failed, retrying the same model",
                extra={"tts_model": candidate._opts.model, "attempt": self._attempts},
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            self._parent._emit_plugin_event(
                "fallback.attempt_failed",
                "warning",
                from_model=candidate._opts.model,
                to_model=candidate._opts.model,
                same_model_retry=True,
                error=str(exc),
            )
            await self._restart()
            return

        next_index = self._parent._candidate_state.advance(self._index)
        if next_index is None:
            logger.error(
                "[TTS] every candidate failed; raising to the caller",
                extra={"tts_model": candidate._opts.model},
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            self._parent._emit_plugin_event(
                "fallback.exhausted",
                "error",
                from_model=candidate._opts.model,
                error=str(exc),
            )
            raise exc

        next_candidate = self._parent._candidate_tts[next_index]
        logger.warning(
            "[TTS] falling over to the next model",
            extra={
                "tts_model": candidate._opts.model,
                "next_tts_model": next_candidate._opts.model,
            },
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        self._parent._emit_plugin_event(
            "fallback.attempt_failed",
            "warning",
            from_model=candidate._opts.model,
            to_model=next_candidate._opts.model,
            error=str(exc),
        )
        self._index = next_index
        self._parent._active_candidate_index = next_index
        self._attempts = 0
        await self._restart()
        self._parent._emit_plugin_event(
            "fallback.switch_succeeded",
            from_model=candidate._opts.model,
            to_model=next_candidate._opts.model,
        )

    async def _restart(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            with contextlib.suppress(Exception):
                await stream.aclose()
        self._closed = False
        self._started = False
        await self._start_stream()


class _FallbackSynthesizeStream(_FallbackStreamBase):
    # Marker stored in the replay buffer so a segment flush is replayed at the
    # right position when a fallback candidate restarts the stream.
    _FLUSH_MARK = object()

    def __init__(self, *, parent: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(parent=parent, conn_options=conn_options)
        self._texts: list[Any] = []
        self._input_ended = False

    def push_text(self, text: str) -> None:
        self._texts.append(text)
        if self._stream is not None:
            self._arm_timeout()
            self._stream.push_text(text)

    def flush(self) -> None:
        self._texts.append(self._FLUSH_MARK)
        if self._stream is not None:
            self._arm_timeout()
            self._stream.flush()

    def end_input(self) -> None:
        self._input_ended = True
        if self._stream is not None:
            self._arm_timeout()
            self._stream.end_input()

    async def _start_stream(self) -> None:
        candidate = self._parent._candidate_tts[self._index]
        await self._parent._drop_inactive_candidate_connections(candidate)
        options = replace(self._conn_options, max_retry=0)
        self._stream = await candidate._stream_candidate(conn_options=options).__aenter__()
        for item in self._texts:
            if item is self._FLUSH_MARK:
                self._stream.flush()
            else:
                self._stream.push_text(item)
        if self._input_ended:
            self._stream.end_input()
        if self._texts:
            timeout = self._parent._first_audio_timeout_s
            self._deadline = time.monotonic() + timeout if timeout is not None else None

    def _arm_timeout(self) -> None:
        timeout = self._parent._first_audio_timeout_s
        if self._deadline is None and timeout is not None:
            self._deadline = time.monotonic() + timeout


class _FallbackChunkedStream(_FallbackStreamBase):
    def __init__(
        self,
        *,
        parent: TTS,
        text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(parent=parent, conn_options=conn_options)
        self._text = text
        timeout = parent._first_audio_timeout_s
        self._deadline = time.monotonic() + timeout if timeout is not None else None

    async def _start_stream(self) -> None:
        candidate = self._parent._candidate_tts[self._index]
        await self._parent._drop_inactive_candidate_connections(candidate)
        options = replace(self._conn_options, max_retry=0)
        self._stream = await candidate._synthesize_candidate(
            self._text,
            conn_options=options,
        ).__aenter__()
        timeout = self._parent._first_audio_timeout_s
        self._deadline = time.monotonic() + timeout if timeout is not None else None

    async def collect(self) -> rtc.AudioFrame:
        """Utility method to collect every frame in a single call"""
        frames = []
        async for ev in self:
            frames.append(ev.frame)

        return rtc.combine_audio_frames(frames)
