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
import base64
import contextlib
import json
import os
import re
import time
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal
from urllib.parse import urlencode

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    tokenize,
    tts,
    utils,
)
from livekit.agents.tokenize.basic import split_words
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TimedString

from .log import logger
from .models import TTSModels

DEFAULT_BASE_URL = "https://api.heygen.com"
DEFAULT_MODEL: TTSModels = "heygen-voice-1-turbo"
# Nadine, a public voice every workspace can use.
DEFAULT_VOICE_ID = "83548de556df48ba8c09c42a57c51d85"
DEFAULT_SAMPLE_RATE = 24000
NUM_CHANNELS = 1

SAMPLE_RATES = (8000, 16000, 22050, 24000, 44100, 48000)

AUTHORIZATION_HEADER = "x-api-key"
_STREAM_PATH = "/v3/models/audio/tts/stream"
_MULTI_STREAM_PATH = "/v3/models/audio/tts/multi-stream-input"

# The longest the server allows, measured from the last client message. It ends a turn's context
# after this long without text, and closes an idle socket.
_INACTIVITY_TIMEOUT = 180

# How long a stream teardown waits for the close handshake. Closing is what stops a context
# already flushed, and a socket that does not answer must not hold up a barge-in.
_CLOSE_TIMEOUT = 1.0

# Without `enable_ssml_parsing` the server closes the socket on a break tag, losing the turn.
_BREAK_TAG = re.compile(r"\s*<break\b[^>]*>", re.IGNORECASE)


@dataclass
class _TTSOptions:
    voice_id: str
    model: TTSModels | str
    language: NotGivenOr[LanguageCode]
    sample_rate: int
    expressiveness_boost: NotGivenOr[float]
    apply_text_normalization: Literal["auto", "on", "off"]
    pronunciation_dictionary: NotGivenOr[dict[str, str]]
    enable_ssml_parsing: bool
    enable_logging: bool
    end_user_id: NotGivenOr[str]
    base_url: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice_id: str = DEFAULT_VOICE_ID,
        model: TTSModels | str = DEFAULT_MODEL,
        language: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        expressiveness_boost: NotGivenOr[float] = NOT_GIVEN,
        apply_text_normalization: Literal["auto", "on", "off"] = "auto",
        pronunciation_dictionary: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        enable_ssml_parsing: bool = False,
        enable_logging: bool = True,
        end_user_id: NotGivenOr[str] = NOT_GIVEN,
        streaming: bool = True,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = DEFAULT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a new instance of HeyGen Voice TTS.

        Args:
            voice_id: HeyGen voice ID: a public voice, or an instant or professional clone in
                your workspace. Defaults to Nadine, a public voice.
            model: ``heygen-voice-1``, ``heygen-voice-1-turbo`` or ``heygen-voice-1-turbo-free``.
                Turbo models serve public voices and instant clones. Professional clones need
                ``heygen-voice-1`` and ``streaming=False``.
            language: Language of the text. Regional tags resolve to their base language, so
                ``en-US`` is sent as ``en``. Left out, HeyGen detects the language.
            sample_rate: Output sample rate in Hz, one of 8000, 16000, 22050, 24000, 44100, 48000.
            expressiveness_boost: 0.0-1.0, for instant voices. Left out, HeyGen uses 1.0, or 0.5
                with pronunciation.
            apply_text_normalization: ``auto``, ``on`` or ``off``. WebSocket only.
            pronunciation_dictionary: Term to IPA map, such as ``{"Acme": "/ˈækmi/"}``. Word timing
                is turned off with a dictionary, since the timings cover the IPA, not the term.
            enable_ssml_parsing: Turn ``<break time="1s"/>`` tags into pauses. WebSocket only. When
                False, break tags are removed from the text.
            enable_logging: Set False to keep no copy of the audio and text at HeyGen. WebSocket
                only: ``synthesize()`` and ``streaming=False`` use HTTP, which has no such option.
            end_user_id: Your identifier for the end user, recorded with HeyGen's usage.
            streaming: Stream text into a WebSocket, with word timing. Set False to synthesize
                each sentence with one HTTP request instead, which every voice supports.
                ``synthesize()`` uses HTTP either way.
            tokenizer: Sentence tokenizer for the text sent over the WebSocket.
            api_key: HeyGen API key. Falls back to the ``HEYGEN_API_KEY`` environment variable.
            base_url: Override the HeyGen API base URL.
            http_session: Optional ``aiohttp.ClientSession`` to reuse.
        """
        if sample_rate not in SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {SAMPLE_RATES}, got {sample_rate}")

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=streaming,
                aligned_transcript=streaming and not is_given(pronunciation_dictionary),
            ),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        heygen_api_key = api_key if is_given(api_key) else os.environ.get("HEYGEN_API_KEY")
        if not heygen_api_key:
            raise ValueError(
                "HeyGen API key is required, either as `api_key` argument or "
                "`HEYGEN_API_KEY` environment variable"
            )

        self._api_key = heygen_api_key
        self._opts = _TTSOptions(
            voice_id=voice_id,
            model=model,
            language=LanguageCode(language) if is_given(language) else NOT_GIVEN,
            sample_rate=sample_rate,
            expressiveness_boost=expressiveness_boost,
            apply_text_normalization=apply_text_normalization,
            pronunciation_dictionary=pronunciation_dictionary,
            enable_ssml_parsing=enable_ssml_parsing,
            enable_logging=enable_logging,
            end_user_id=end_user_id,
            base_url=base_url.rstrip("/"),
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.blingfire.SentenceTokenizer()
        )
        # monotonic time before which no request is sent, see _note_retry_after()
        self._not_before = 0.0
        self._idle_reads: dict[
            aiohttp.ClientWebSocketResponse, asyncio.Task[aiohttp.WSMessage]
        ] = {}
        self._pool = (
            utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
                connect_cb=self._connect_ws, close_cb=self._close_ws
            )
            if streaming
            else None
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "HeyGen"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def _note_retry_after(self, headers: Mapping[str, str] | None) -> None:
        """Hold off new requests for as long as a 429 asked."""
        retry_after = (headers or {}).get("Retry-After", "")
        if retry_after.isdigit():
            self._not_before = max(self._not_before, time.monotonic() + int(retry_after))

    async def _wait_for_retry_after(self, timeout: float) -> None:
        # Refused handshakes count against the per-minute limit, so retrying sooner than asked
        # only pushes the limit further away.
        if (wait := self._not_before - time.monotonic()) > 0:
            if wait > timeout:
                # longer than a reply can wait: stop the retries (a FallbackAdapter moves on)
                raise APIConnectionError(
                    f"HeyGen asked to wait {wait:.1f}s before the next request", retryable=False
                )
            await asyncio.sleep(wait)

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        await self._wait_for_retry_after(timeout)
        try:
            ws = await asyncio.wait_for(
                self._ensure_session().ws_connect(
                    _ws_url(self._opts), headers={AUTHORIZATION_HEADER: self._api_key}
                ),
                timeout,
            )
        except aiohttp.WSServerHandshakeError as e:
            self._note_retry_after(e.headers)
            message = f"HeyGen refused the connection with HTTP {e.status}"
            if e.status == 400:
                message += (
                    ". Check voice_id, model and language. Professional voice clones are not "
                    'served over the WebSocket: use streaming=False and model="heygen-voice-1"'
                )
            # aiohttp discards the response body here, so the error code is out of reach
            raise APIStatusError(
                message,
                status_code=e.status,
                request_id=e.headers.get("x-request-id") if e.headers else None,
            ) from None
        self._read_while_idle(ws)
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        if read := self._idle_reads.pop(ws, None):
            await utils.aio.cancel_and_wait(read)
        await ws.close()

    def _read_while_idle(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        # Pings are only answered during a read, and the server drops a socket that stops
        # answering within 40s. The read also shows when the server closed the socket.
        self._idle_reads[ws] = asyncio.create_task(ws.receive())

    async def _acquire(self, *, timeout: float) -> aiohttp.ClientWebSocketResponse:
        assert self._pool is not None
        while True:
            ws = await self._pool.get(timeout=timeout)
            read = self._idle_reads.pop(ws)
            if not read.done():
                await utils.aio.cancel_and_wait(read)
                return ws
            self._pool.remove(ws)
            if not self._pool.last_connection_reused:
                # closed as soon as it opened: another connection would only spend the rate limit
                raise APIConnectionError("HeyGen closed the connection as soon as it opened")

    def _release(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        assert self._pool is not None
        self._read_while_idle(ws)
        self._pool.put(ws)

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        expressiveness_boost: NotGivenOr[float] = NOT_GIVEN,
        apply_text_normalization: NotGivenOr[Literal["auto", "on", "off"]] = NOT_GIVEN,
        end_user_id: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Update the TTS options. A change takes effect on the next stream.

        Args:
            voice_id: HeyGen voice ID.
            model: HeyGen Voice model.
            language: Language of the text.
            expressiveness_boost: 0.0-1.0, for instant voices.
            apply_text_normalization: ``auto``, ``on`` or ``off``.
            end_user_id: Your identifier for the end user.
        """
        url = _ws_url(self._opts)
        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = LanguageCode(language)
        if is_given(expressiveness_boost):
            self._opts.expressiveness_boost = expressiveness_boost
        if is_given(apply_text_normalization):
            self._opts.apply_text_normalization = apply_text_normalization
        if is_given(end_user_id):
            self._opts.end_user_id = end_user_id

        # The URL fixes these for the life of a socket.
        if self._pool is not None and _ws_url(self._opts) != url:
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
                "streaming is disabled on this HeyGen TTS instance; construct it with "
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
        await self._tts._wait_for_retry_after(self._conn_options.timeout)
        try:
            async with self._tts._ensure_session().post(
                f"{self._opts.base_url}{_STREAM_PATH}",
                headers={AUTHORIZATION_HEADER: self._tts._api_key},
                json=_http_body(self._opts, self._input_text),
                timeout=aiohttp.ClientTimeout(
                    total=None,
                    sock_connect=self._conn_options.timeout,
                    sock_read=self._conn_options.timeout,
                ),
            ) as resp:
                if resp.status != 200:
                    if resp.status == 429:
                        self._tts._note_retry_after(resp.headers)
                    raise await _http_error(resp)

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
    """Stream sentences into a context of the multi-stream WebSocket.

    One stream is one turn and holds its socket alone. The server can end a context on its own, so
    a turn may span more than one, but only ever has one running.
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        # `TTS.stream()` refuses to build one of these without a pool.
        assert tts._pool is not None
        self._pool = tts._pool
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=request_id)

        timeout = self._conn_options.timeout
        sentence_stream = self._tts._sentence_tokenizer.stream()
        contexts = utils.aio.Chan[str]()
        open_context: str | None = None
        input_ended_at: float | None = None

        async def input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sentence_stream.flush()
                    continue
                sentence_stream.push_text(data)
            sentence_stream.end_input()

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal open_context, input_ended_at
            async for ev in sentence_stream:
                text = ev.token
                if not self._opts.enable_ssml_parsing:
                    text = _BREAK_TAG.sub("", text)
                if not text.strip():
                    continue

                # The server joins messages as sent, so sentences need the space between them.
                packet: dict[str, Any] = {"text": f"{text} "}
                if open_context is None:
                    # Naming a context opens it and spends rate limit and billing admission, so
                    # this waits for the first text: a turn with no text sends nothing.
                    open_context = utils.shortuuid()
                    contexts.send_nowait(open_context)
                    if is_given(self._opts.pronunciation_dictionary):
                        packet["pronunciation_dictionary"] = self._opts.pronunciation_dictionary
                packet["context_id"] = open_context
                self._mark_started()
                await ws.send_str(json.dumps(packet))

            input_ended_at = time.monotonic()
            if (context_id := open_context) is not None:
                await ws.send_str(json.dumps({"context_id": context_id, "flush": True}))
                await ws.send_str(json.dumps({"context_id": context_id, "close_context": True}))
            contexts.close()

        async def next_message(ws: aiohttp.ClientWebSocketResponse) -> dict[str, Any]:
            error: dict[str, Any] = {}
            while True:
                try:
                    # Not `receive(timeout=...)`: every server ping would restart that timer.
                    msg = await asyncio.wait_for(ws.receive(), timeout)
                except asyncio.TimeoutError:
                    # The text can pause mid-turn, so silence only counts once it has ended.
                    if input_ended_at is None or time.monotonic() - input_ended_at < timeout:
                        continue
                    raise

                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise _close_error(ws.close_code, error, msg.extra)

                if msg.type is not aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected HeyGen message type %s", msg.type)
                    continue

                data: dict[str, Any] = json.loads(msg.data)
                if isinstance(data.get("error"), dict):
                    # The close that follows says whether this is worth a retry.
                    error = data["error"]
                    continue
                return data

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal open_context
            words = _WordTimer()
            audio_ms = 0.0
            async for _ in contexts:
                while True:
                    data = await next_message(ws)
                    if alignment := data.get("alignment"):
                        output_emitter.push_timed_transcript(words.push(alignment, audio_ms))
                    if audio := data.get("audio"):
                        pcm = base64.b64decode(audio)
                        output_emitter.push(pcm)
                        # alignment restarts at 0 on every message
                        audio_ms += len(pcm) / (2 * self._opts.sample_rate) * 1000
                    if data.get("isFinal"):
                        output_emitter.push_timed_transcript(words.flush())
                        open_context = None
                        break
            output_emitter.end_segment()

        try:
            ws = await self._tts._acquire(timeout=timeout)
            self._acquire_time = self._pool.last_acquire_time
            self._connection_reused = self._pool.last_connection_reused
            tasks = [
                asyncio.create_task(input_task()),
                asyncio.create_task(send_task(ws)),
                asyncio.create_task(recv_task(ws)),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                await sentence_stream.aclose()
                await utils.aio.gracefully_cancel(*tasks)
                if open_context is None and not ws.closed:
                    self._tts._release(ws)
                else:
                    # A running context cannot be recalled: `close_context` lets a flushed one
                    # finish. Only closing stops it, and keeps its audio out of the next turn.
                    self._pool.remove(ws)
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(ws.close(), _CLOSE_TIMEOUT)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


class _WordTimer:
    """Turn per-character timings into timed words, holding back a word that may continue."""

    def __init__(self) -> None:
        self._text = ""
        self._starts_ms: list[float] = []
        self._end_ms = 0.0

    def push(self, alignment: dict[str, Any], offset_ms: float) -> list[TimedString]:
        chars = alignment.get("chars") or []
        starts = alignment.get("charStartTimesMs") or []
        durations = alignment.get("charDurationsMs") or []
        for char, start, duration in zip(chars, starts, durations, strict=False):
            self._text += char
            self._starts_ms += [offset_ms + start] * len(char)
            self._end_ms = offset_ms + start + duration
        return self._take()

    def flush(self) -> list[TimedString]:
        return self._take(final=True)

    def _take(self, *, final: bool = False) -> list[TimedString]:
        words = split_words(self._text, ignore_punctuation=False, split_character=True)
        bounds = [start for _, start, _ in words]
        if final:
            bounds.append(len(self._text))

        timed = [
            TimedString(
                text=self._text[start:end],
                start_time=self._starts_ms[start] / 1000,
                end_time=(self._starts_ms[end] if end < len(self._text) else self._end_ms) / 1000,
            )
            for start, end in zip(bounds, bounds[1:], strict=False)
        ]
        if timed:
            self._text = self._text[bounds[-1] :]
            self._starts_ms = self._starts_ms[bounds[-1] :]
        return timed


def _shared_params(opts: _TTSOptions) -> dict[str, Any]:
    params: dict[str, Any] = {"voice_id": opts.voice_id, "model": opts.model}
    if is_given(opts.language):
        params["language"] = opts.language.language
    if is_given(opts.expressiveness_boost):
        params["expressiveness_boost"] = opts.expressiveness_boost
    if is_given(opts.end_user_id):
        params["end_user_id"] = opts.end_user_id
    return params


def _ws_url(opts: _TTSOptions) -> str:
    params = _shared_params(opts) | {
        "output_format": f"pcm_{opts.sample_rate}",
        # Without it the server waits for 120 characters before it starts on a sentence.
        "auto_mode": "true",
        "inactivity_timeout": _INACTIVITY_TIMEOUT,
        "apply_text_normalization": opts.apply_text_normalization,
        "enable_ssml_parsing": str(opts.enable_ssml_parsing).lower(),
        "enable_logging": str(opts.enable_logging).lower(),
    }
    base_url = opts.base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    return f"{base_url}{_MULTI_STREAM_PATH}?{urlencode(params)}"


def _http_body(opts: _TTSOptions, text: str) -> dict[str, Any]:
    body = _shared_params(opts) | {
        "text": text,
        "output_format": {"encoding": "pcm", "sample_rate_hertz": opts.sample_rate},
    }
    if is_given(opts.pronunciation_dictionary):
        body["pronunciation_dictionary"] = opts.pronunciation_dictionary
    return body


def _error_text(error: dict[str, Any]) -> str:
    return ": ".join(str(part) for part in (error.get("code"), error.get("message")) if part)


def _close_error(
    close_code: int | None, error: dict[str, Any], reason: str | None
) -> APIStatusError:
    return APIStatusError(
        _error_text(error) or reason or "HeyGen connection closed unexpectedly",
        status_code=close_code or -1,
        request_id=error.get("request_id"),
        body=error or None,
        # 1008 refuses the request or rate limits it, so another connection would only spend the
        # limit. Everything else is an interrupted connection, a restart or full capacity.
        retryable=close_code != 1008,
    )


async def _http_error(resp: aiohttp.ClientResponse) -> APIStatusError:
    """Unwrap the ``{"error": {"code", "message", "request_id"}}`` envelope."""
    try:
        payload = await resp.json(content_type=None)
    except Exception:
        payload = None

    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        error = {}
    return APIStatusError(
        _error_text(error) or resp.reason or "request failed",
        status_code=resp.status,
        request_id=error.get("request_id") or resp.headers.get("x-request-id"),
        body=payload,
    )
