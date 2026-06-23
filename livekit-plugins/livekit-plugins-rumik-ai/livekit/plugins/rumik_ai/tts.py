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
import json
import os
import re
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal
from urllib.parse import quote

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    create_api_error_from_http,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger

Model = Literal["muga", "mulberry"]
MugaTone = Literal["happy", "excited", "sad", "angry", "neutral", "whisper"]
MulberrySpeaker = Literal["speaker_1", "speaker_2", "speaker_3", "speaker_4"]

DEFAULT_BASE_URL = "https://silk-api.rumik.ai"
DEFAULT_SAMPLE_RATE = 24000
NUM_CHANNELS = 1
MAX_TEXT_LENGTH = 2000

# Text used only to mint a reusable WebSocket session. The real text is sent per
# synthesis request over the same socket, so this value is never synthesized.
_INIT_TEXT = "init"
# Backstop cap on a pooled session's age. Heartbeat pings (see _WS_HEARTBEAT) keep
# the socket warm so Rumik rarely idle-closes it between turns, and stale sockets are
# still detected on checkout and re-minted -- so this is only a safety cap, not the
# primary refresh mechanism.
_MAX_SESSION_DURATION = 300.0
# aiohttp sends a WebSocket ping every _WS_HEARTBEAT seconds and drops the socket if
# no pong returns. This keeps the reused session alive across conversational pauses so
# the next turn does not pay a fresh mint, mirroring the Rumik Pipecat integration.
_WS_HEARTBEAT = 20.0
# On barge-in or explicit cancel, send {"type": "cancel"} and drain until the server acks
# with {"type": "cancelled"} so the pooled socket is left clean and reusable on the next
# utterance (no re-mint). Bounded so an interruption is never delayed long by a slow ack.
_CANCEL_DRAIN_TIMEOUT = 2.0

MUGA_TONES = {"happy", "excited", "sad", "angry", "neutral", "whisper"}
MUGA_EVENTS = {"laugh", "chuckle", "sigh"}
MUGA_EVENT_COMPATIBILITY = {
    "happy": {"laugh", "chuckle"},
    "excited": {"laugh", "chuckle"},
    "sad": {"sigh"},
    "angry": {"sigh"},
    "neutral": {"laugh", "sigh"},
    "whisper": {"chuckle", "sigh"},
}
MULBERRY_SPEAKERS = {"speaker_1", "speaker_2", "speaker_3", "speaker_4"}

_SQUARE_TAG_RE = re.compile(r"\[([^\]]+)\]")
_TONE_PREFIX_RE = re.compile(r"^\[([^\]]+)\](.*)$", re.DOTALL)
_EVENT_TAG_RE = re.compile(r"<([^>]+)>")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


class _StaleConnectionError(Exception):
    """Raised when a pooled WebSocket is found closed; triggers a reconnect."""


@dataclass
class _TTSOptions:
    model: Model | str
    api_key: str
    tone: MugaTone | str | None
    description: NotGivenOr[str]
    speaker: NotGivenOr[MulberrySpeaker | str]
    f0_up_key: NotGivenOr[float]
    temperature: NotGivenOr[float]
    top_p: NotGivenOr[float]
    top_k: NotGivenOr[int]
    repetition_penalty: NotGivenOr[float]
    max_new_tokens: NotGivenOr[int]
    base_url: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: Model | str = "muga",
        tone: MugaTone | str | None = None,
        description: NotGivenOr[str] = NOT_GIVEN,
        speaker: NotGivenOr[MulberrySpeaker | str] = NOT_GIVEN,
        f0_up_key: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        top_k: NotGivenOr[int] = NOT_GIVEN,
        repetition_penalty: NotGivenOr[float] = NOT_GIVEN,
        max_new_tokens: NotGivenOr[int] = NOT_GIVEN,
        full_response_aggregation: NotGivenOr[bool] = NOT_GIVEN,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a Rumik AI text-to-speech client.

        The client maintains a reusable Rumik WebSocket session: it mints the
        session once and streams every synthesis request over the same socket,
        re-minting only when the socket goes stale. By default ``muga`` buffers the full
        LLM response and synthesizes it in one request so its leading ``[tone]`` tag
        conditions the whole utterance; ``mulberry`` has no tone tag, so it streams
        sentence-by-sentence for lower latency (see ``full_response_aggregation``).

        Args:
            model: Rumik AI Silk model to use. Supports ``"muga"`` and ``"mulberry"``.
                ``muga`` speaks Romanized Hinglish (Roman/Latin script only; Devanagari is
                rejected). ``mulberry`` speaks pure English and Hindi -- write Hindi in
                Devanagari and keep English words in Latin script; it does not use the
                Romanized Hinglish that muga expects.
            tone: Optional Muga fallback tone. When omitted, each Muga input must already
                start with a valid tone tag such as ``[happy]`` or ``[sad]``. When provided,
                untagged input is prefixed with this tone, and existing tags must match it.
            description: Mulberry-only natural language voice description.
            speaker: Optional Mulberry preset speaker, ``speaker_1`` through ``speaker_4``.
            f0_up_key: Mulberry-only pitch shift in semitones, from -12 to 12.
            temperature: Optional sampling temperature. Omitted unless set, so Rumik AI
                applies its own default.
            top_p: Optional nucleus sampling value. Omitted unless set.
            top_k: Optional top-k sampling value. Omitted unless set.
            repetition_penalty: Optional penalty for repeated tokens. Omitted unless set.
            max_new_tokens: Optional output length cap. Omitted unless set.
            full_response_aggregation: When True, buffer the complete LLM response and
                synthesize it in one request to avoid sentence-level TTFB gaps. When
                False, stream sentence-by-sentence via the framework's StreamAdapter for
                lower latency; with muga, set a fallback ``tone`` so each sentence keeps
                a tone tag. Defaults to True for muga (its ``[tone]`` tag must condition
                the whole utterance) and False for mulberry (lower latency).
            api_key: Rumik AI API key. If not provided, reads ``RUMIK_API_KEY``.
            base_url: Rumik AI API base URL.
            http_session: Existing aiohttp session to use.
        """
        # muga buffers the full LLM response and synthesizes it in one request
        # (streaming=True) so its leading [tone] tag conditions the whole utterance and
        # there are no audible gaps from Rumik's per-request TTFB. mulberry has no tone
        # tag, so by default it streams sentence-by-sentence via the framework's
        # StreamAdapter for lower time-to-first-word. An explicit value always wins.
        if is_given(full_response_aggregation):
            full_aggregation = full_response_aggregation
        else:
            full_aggregation = model != "mulberry"

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=full_aggregation, aligned_transcript=False),
            sample_rate=DEFAULT_SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        api_key = api_key or os.environ.get("RUMIK_API_KEY")
        if not api_key:
            raise ValueError(
                "Rumik AI API key is required, either as argument or set RUMIK_API_KEY"
            )

        opts = _TTSOptions(
            model=model,
            api_key=api_key,
            tone=tone,
            description=description,
            speaker=speaker,
            f0_up_key=f0_up_key,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            base_url=base_url.rstrip("/"),
        )
        _validate_options(opts)

        self._opts = opts
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        # Mint the Rumik session once and reuse the socket across requests/turns.
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=_MAX_SESSION_DURATION,
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Rumik AI"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        ws_url = await self._mint_ws_session(timeout)

        async def _open() -> aiohttp.ClientWebSocketResponse:
            # Wrap in a coroutine so wait_for gets a plain awaitable (aiohttp's
            # ws_connect returns a context-manager type that wait_for rejects).
            return await self._ensure_session().ws_connect(ws_url, heartbeat=_WS_HEARTBEAT)

        try:
            return await asyncio.wait_for(_open(), timeout)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except Exception as e:
            raise APIConnectionError() from e

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        # Tell Rumik we're done so it can release the session, then close. Best-effort:
        # the socket may already be gone (idle timeout, server close, interruption).
        try:
            if not ws.closed:
                await ws.send_str(json.dumps({"type": "close"}))
        except Exception:
            pass
        await ws.close()

    async def _mint_ws_session(self, timeout: float) -> str:
        try:
            async with self._ensure_session().post(
                f"{self._opts.base_url}/v1/tts/ws-connect",
                headers={"Authorization": f"Bearer {self._opts.api_key}"},
                json={"model": self._opts.model, "text": _INIT_TEXT},
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                body = await _read_response_body(resp)
                if resp.status >= 400:
                    message = _error_message(body)
                    raise create_api_error_from_http(message, status=resp.status, body=body)

                if not isinstance(body, dict):
                    raise APIStatusError(
                        "Rumik AI ws-connect returned a non-JSON response",
                        status_code=resp.status,
                        body=body,
                        retryable=False,
                    )

                ws_url = body.get("ws_url")
                token = body.get("token")
                if not isinstance(ws_url, str) or not isinstance(token, str):
                    raise APIStatusError(
                        "Rumik AI ws-connect response is missing ws_url or token",
                        status_code=resp.status,
                        body=body,
                        retryable=False,
                    )

                separator = "&" if "?" in ws_url else "?"
                return f"{ws_url}{separator}token={quote(token)}"
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError() from e

    def update_options(
        self,
        *,
        model: NotGivenOr[Model | str] = NOT_GIVEN,
        tone: NotGivenOr[MugaTone | str | None] = NOT_GIVEN,
        description: NotGivenOr[str] = NOT_GIVEN,
        speaker: NotGivenOr[MulberrySpeaker | str] = NOT_GIVEN,
        f0_up_key: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        top_k: NotGivenOr[int] = NOT_GIVEN,
        repetition_penalty: NotGivenOr[float] = NOT_GIVEN,
        max_new_tokens: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """Update TTS options.

        Options are sent on each synthesis request, so changes take effect on the next
        request without reconnecting -- handy for varying mulberry's ``description``,
        ``speaker``, or ``f0_up_key`` between turns. Changing ``model`` is the exception:
        the model is pinned when the WebSocket session is minted, so it invalidates the
        pooled connection and the next request re-mints.
        """
        opts = replace(self._opts)
        model_changed = False
        if is_given(model):
            model_changed = model != opts.model
            opts.model = model
        if is_given(tone):
            opts.tone = tone
        if is_given(description):
            opts.description = description
        if is_given(speaker):
            opts.speaker = speaker
        if is_given(f0_up_key):
            opts.f0_up_key = f0_up_key
        if is_given(temperature):
            opts.temperature = temperature
        if is_given(top_p):
            opts.top_p = top_p
        if is_given(top_k):
            opts.top_k = top_k
        if is_given(repetition_penalty):
            opts.repetition_penalty = repetition_penalty
        if is_given(max_new_tokens):
            opts.max_new_tokens = max_new_tokens

        _validate_options(opts)
        self._opts = opts
        if model_changed:
            # The model is fixed at mint time, so existing pooled sockets are stale.
            self._pool.invalidate()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        """Open the Rumik WebSocket session ahead of the first request."""
        self._pool.prewarm()

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._pool.aclose()

    async def _stream_synthesis(
        self,
        text: str,
        opts: _TTSOptions,
        conn_options: APIConnectOptions,
        output_emitter: tts.AudioEmitter,
        *,
        on_started: Callable[[], None] | None = None,
    ) -> None:
        """Synthesize one request over the pooled WebSocket and push PCM out.

        Shared by SynthesizeStream (full-response mode: one request for the whole
        reply) and ChunkedStream (per-sentence StreamAdapter mode, or a
        FallbackAdapter: one request per fixed piece of text).
        """
        try:
            prepared_text = _prepare_text(text, opts)
            payload = _synthesis_payload(prepared_text, opts)

            # Reconnect-on-stale: a pooled socket may have been idle-closed by Rumik.
            # Detect that before sending and re-mint a fresh session once.
            for attempt in range(2):
                try:
                    async with self._pool.connection(timeout=conn_options.timeout) as ws:
                        if ws.closed:
                            raise _StaleConnectionError
                        cancelled_clean = await self._stream_on_ws(
                            ws, payload, conn_options, output_emitter, on_started
                        )
                    # The pool has returned the socket above. If the generation was
                    # interrupted but stopped cleanly, the socket is already back in the
                    # warm pool; propagate the cancellation now (after it was returned).
                    if cancelled_clean:
                        raise asyncio.CancelledError
                    return
                except _StaleConnectionError:
                    if attempt == 1:
                        raise APIConnectionError("Rumik AI WebSocket is unavailable") from None
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except (APIConnectionError, APIStatusError, APITimeoutError):
            raise
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(message=e.message, status_code=e.status, body=None) from None
        except Exception as e:
            raise APIConnectionError() from e

    async def _stream_on_ws(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        payload: dict[str, Any],
        conn_options: APIConnectOptions,
        output_emitter: tts.AudioEmitter,
        on_started: Callable[[], None] | None,
    ) -> bool:
        """Stream one generation over the pooled socket.

        Returns False on normal completion. Returns True if the generation was interrupted
        (barge-in/cancel) but the socket was stopped cleanly and is safe to reuse -- the
        caller must then propagate the cancellation. Raises on error or an unusable socket.
        """
        if on_started is not None:
            on_started()
        await ws.send_str(json.dumps(payload))

        received_audio = False
        try:
            while True:
                msg = await ws.receive(timeout=conn_options.timeout)
                if msg.type == aiohttp.WSMsgType.BINARY:
                    received_audio = True
                    output_emitter.push(msg.data)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    event = _loads_event(msg.data)
                    event_type = event.get("type")
                    if event_type == "done":
                        # rtf > 1.0 means Rumik generated slower than real time, which can
                        # starve playback and sound robotic; surface it for diagnosis.
                        logger.debug(
                            "Rumik AI TTS synthesis complete",
                            extra={
                                "audio_duration": event.get("audio_duration"),
                                "rtf": event.get("rtf"),
                                "request_id": event.get("request_id"),
                                "credits_used": event.get("credits_used"),
                            },
                        )
                        return False
                    if event_type == "cancelled":
                        # The server stopped this generation (e.g. a barge-in replaced it).
                        # It is a clean terminal frame, so the socket stays reusable.
                        logger.debug(
                            "Rumik AI TTS generation cancelled",
                            extra={
                                "reason": event.get("reason"),
                                "request_id": event.get("request_id"),
                            },
                        )
                        return False
                    if event_type == "error" or "error" in event:
                        raise _provider_error(event)
                    if event_type == "timeout":
                        # Server idle-closed the session; the pool evicts it on the raise.
                        raise APIConnectionError("Rumik AI WebSocket idle timeout")
                    # "queued" and any other informational events are ignored.
                    logger.debug("Ignoring Rumik AI TTS event", extra={"event": event})
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise APIConnectionError(f"Rumik AI WebSocket error: {ws.exception()!r}")
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    close_code = getattr(ws, "close_code", None) or getattr(msg, "data", None)
                    # The server closed the socket, so it cannot be reused.
                    self._pool.remove(ws)
                    if received_audio and close_code in (None, 1000):
                        return False
                    raise APIConnectionError(
                        f"Rumik AI WebSocket closed unexpectedly: code={close_code!r}"
                    )
        except asyncio.CancelledError:
            # Barge-in or explicit cancel: stop the in-flight generation and drain the
            # server's ack so the pooled socket is left clean and reusable on the next
            # utterance (avoids a re-mint). If we can't confirm a clean stop, re-raise so
            # the pool drops the socket.
            if await self._cancel_generation(ws):
                return True
            raise

    async def _cancel_generation(self, ws: aiohttp.ClientWebSocketResponse) -> bool:
        """Stop the in-flight generation and drain the server's ack.

        Sends ``{"type": "cancel"}`` and discards any late audio until the server confirms
        with ``{"type": "cancelled"}`` (or ``done``). Returns True if the socket reached a
        clean idle state and is safe to reuse, False if it should be dropped. Bounded by
        ``_CANCEL_DRAIN_TIMEOUT`` so an interruption is never delayed long by a slow ack.
        """
        try:
            return await asyncio.wait_for(
                self._drain_after_cancel(ws), timeout=_CANCEL_DRAIN_TIMEOUT
            )
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return False
        except Exception:
            return False

    async def _drain_after_cancel(self, ws: aiohttp.ClientWebSocketResponse) -> bool:
        await ws.send_str(json.dumps({"type": "cancel"}))
        while True:
            msg = await ws.receive()
            if msg.type == aiohttp.WSMsgType.BINARY:
                continue  # discard audio that was already in flight before the cancel
            if msg.type == aiohttp.WSMsgType.TEXT:
                event = _loads_event(msg.data)
                event_type = event.get("type")
                if event_type in ("cancelled", "done"):
                    return True
                if event_type == "error" or "error" in event:
                    return False
                continue  # ignore "queued"/informational frames
            # CLOSE / CLOSED / CLOSING / ERROR -> the socket is gone, cannot reuse.
            return False


class ChunkedStream(tts.ChunkedStream):
    """One Rumik request for a fixed piece of text (a StreamAdapter sentence or fallback)."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=DEFAULT_SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )
        await self._tts._stream_synthesis(
            self._input_text, self._opts, self._conn_options, output_emitter
        )


class SynthesizeStream(tts.SynthesizeStream):
    """Synthesizes the full LLM response as a single Rumik request (one segment)."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=DEFAULT_SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )

        # The framework feeds one segment per stream, so we synthesize the whole
        # response in a single request once the input is drained.
        parts: list[str] = []
        async for input_data in self._input_ch:
            if isinstance(input_data, str):
                parts.append(input_data)

        text = "".join(parts).strip()
        if not text:
            return

        output_emitter.start_segment(segment_id=utils.shortuuid())
        try:
            await self._tts._stream_synthesis(
                text,
                self._opts,
                self._conn_options,
                output_emitter,
                on_started=self._mark_started,
            )
        finally:
            output_emitter.end_segment()


def _validate_options(opts: _TTSOptions) -> None:
    if opts.model not in {"muga", "mulberry"}:
        raise ValueError("Rumik AI model must be 'muga' or 'mulberry'")

    if opts.model == "muga":
        if opts.tone is not None and opts.tone not in MUGA_TONES:
            raise ValueError(f"Unsupported Rumik AI Muga tone: {opts.tone}")
        if is_given(opts.description):
            raise ValueError("description is only supported with Rumik AI Mulberry")
        if is_given(opts.speaker):
            raise ValueError("speaker is only supported with Rumik AI Mulberry")
        if is_given(opts.f0_up_key):
            raise ValueError("f0_up_key is only supported with Rumik AI Mulberry")
        return

    if opts.tone is not None:
        raise ValueError("tone is only supported with Rumik AI Muga")
    if is_given(opts.speaker) and opts.speaker not in MULBERRY_SPEAKERS:
        raise ValueError(
            "Rumik AI Mulberry speaker must be speaker_1, speaker_2, speaker_3, or speaker_4"
        )
    if is_given(opts.f0_up_key) and not -12 <= opts.f0_up_key <= 12:
        raise ValueError("Rumik AI Mulberry f0_up_key must be between -12 and 12")


def _prepare_text(text: str, opts: _TTSOptions) -> str:
    # Collapse every run of whitespace (including newlines from buffered LLM output)
    # to a single space: Rumik gets clean text, and muga's "exactly one space after
    # the [tone] tag" rule holds even when the model emits a newline after the tag.
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError("Rumik AI TTS text must not be empty")

    if opts.model == "muga":
        text = _prepare_muga_text(text, opts)
    elif opts.model == "mulberry":
        text = _prepare_mulberry_text(text)

    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError("Rumik AI TTS text must be 2000 characters or fewer")

    return text


def _prepare_muga_text(text: str, opts: _TTSOptions) -> str:
    if _DEVANAGARI_RE.search(text):
        raise ValueError("Rumik AI Muga expects Hinglish in Roman script")

    square_tags = _SQUARE_TAG_RE.findall(text)
    if len(square_tags) > 1:
        raise ValueError("Rumik AI Muga text must contain exactly one global tone tag")

    tone = opts.tone
    match = _TONE_PREFIX_RE.match(text)
    if match:
        text_tone = match.group(1)
        after_tag = match.group(2)
        if text_tone not in MUGA_TONES:
            raise ValueError(f"Unsupported Rumik AI Muga tone tag: [{text_tone}]")
        if tone is not None and text_tone != tone:
            raise ValueError("Rumik AI Muga text tone tag must match the configured tone")
        if not after_tag.startswith(" "):
            raise ValueError("Rumik AI Muga tone tag must be followed by one space")
        tone = text_tone
    elif square_tags:
        raise ValueError("Rumik AI Muga tone tag must be at the start of the text")
    else:
        if tone is None:
            raise ValueError(
                "Rumik AI Muga text must start with one global tone tag "
                "when no fallback tone is configured"
            )
        text = f"[{tone}] {text}"

    assert tone is not None
    events = _EVENT_TAG_RE.findall(text)
    for event in events:
        if event not in MUGA_EVENTS:
            raise ValueError(f"Unsupported Rumik AI Muga event tag: <{event}>")
        if event not in MUGA_EVENT_COMPATIBILITY[tone]:
            raise ValueError(f"Rumik AI Muga event <{event}> is not compatible with [{tone}]")

    if _has_too_many_stacked_events(text):
        raise ValueError("Rumik AI Muga supports at most two stacked event tags")

    return text


def _prepare_mulberry_text(text: str) -> str:
    match = _TONE_PREFIX_RE.match(text)
    if match and match.group(1) in MUGA_TONES:
        raise ValueError("Rumik AI Mulberry does not support Muga tone tags")

    events = _EVENT_TAG_RE.findall(text)
    unsupported_events = sorted(event for event in events if event in MUGA_EVENTS)
    if unsupported_events:
        raise ValueError("Rumik AI Mulberry does not support Muga event tags")

    return text


def _has_too_many_stacked_events(text: str) -> bool:
    consecutive = 0
    last_end = -1
    for match in _EVENT_TAG_RE.finditer(text):
        if match.group(1) not in MUGA_EVENTS:
            consecutive = 0
            last_end = match.end()
            continue

        between = text[last_end : match.start()] if last_end >= 0 else ""
        if last_end >= 0 and between.strip() == "":
            consecutive += 1
        else:
            consecutive = 1
        if consecutive > 2:
            return True
        last_end = match.end()

    return False


def _synthesis_payload(text: str, opts: _TTSOptions) -> dict[str, Any]:
    # The model is fixed when the session is minted, so it is not resent here. Only
    # explicitly-set parameters are included; Rumik AI applies its own defaults for
    # the rest.
    payload: dict[str, Any] = {"text": text}

    if is_given(opts.description):
        payload["description"] = opts.description
    if is_given(opts.speaker):
        payload["speaker"] = opts.speaker
    if is_given(opts.f0_up_key):
        payload["f0_up_key"] = opts.f0_up_key
    if is_given(opts.temperature):
        payload["temperature"] = opts.temperature
    if is_given(opts.top_p):
        payload["top_p"] = opts.top_p
    if is_given(opts.top_k):
        payload["top_k"] = opts.top_k
    if is_given(opts.repetition_penalty):
        payload["repetition_penalty"] = opts.repetition_penalty
    if is_given(opts.max_new_tokens):
        payload["max_new_tokens"] = opts.max_new_tokens

    return payload


async def _read_response_body(resp: aiohttp.ClientResponse) -> object:
    try:
        return await resp.json()
    except Exception:
        return await resp.text()


def _loads_event(data: str) -> dict[str, Any]:
    try:
        event = json.loads(data)
    except json.JSONDecodeError as e:
        raise APIConnectionError("Rumik AI WebSocket returned invalid JSON") from e
    if not isinstance(event, dict):
        raise APIConnectionError("Rumik AI WebSocket returned an invalid event")
    return event


def _provider_error(event: dict[str, Any]) -> APIStatusError:
    message = _error_message(event)
    status_code = _event_status_code(event)
    return APIStatusError(
        message=f"Rumik AI TTS error: {message}",
        status_code=status_code,
        body=event,
        retryable=status_code in (408, 429, 499, 503),
    )


def _error_message(body: object) -> str:
    if isinstance(body, dict):
        error = body.get("error")
        code = body.get("code")
        if error and code:
            return f"{error} ({code})"
        if error:
            return str(error)
    if isinstance(body, str) and body:
        return body
    return "Rumik AI request failed"


def _event_status_code(event: dict[str, Any]) -> int:
    status = event.get("status") or event.get("status_code")
    if isinstance(status, int):
        return status
    if isinstance(status, str) and status.isdecimal():
        return int(status)
    return -1
