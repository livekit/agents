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
import os
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    create_api_error_from_http,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TimedString

from .models import TTSEncoding, TTSModels
from .version import __version__

NUM_CHANNELS = 1
SMALLEST_BASE_URL = "https://api.smallest.ai/waves/v1"
SMALLEST_WS_URL = "wss://api.smallest.ai/waves/v1/tts/live"

# Continuations gives no terminal marker for the last `complete` frame, so completion
# is inferred: this many idle seconds after the closing fragment, with no further
# frames, means done. The connection is never reused after an inferred completion
# (see _run), so a too-short value costs a truncated tail, not a corrupted response.
_CONTINUATIONS_IDLE_TIMEOUT = 0.6


@dataclass
class _TTSOptions:
    model: TTSModels | str
    api_key: str
    voice_id: str
    sample_rate: int
    speed: float
    language: LanguageCode
    output_format: TTSEncoding | str
    word_timestamps: bool
    max_buffer_flush_ms: int
    use_continuations: bool
    max_buffer_delay_ms: int
    base_url: str
    ws_url: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: TTSModels | str = "lightning_v3.1_pro",
        voice_id: str | None = None,
        sample_rate: int = 24000,
        speed: float = 1.0,
        language: str = "en",
        output_format: TTSEncoding | str = "pcm",
        word_timestamps: bool = False,
        max_buffer_flush_ms: int = 0,
        use_continuations: bool = True,
        max_buffer_delay_ms: int = 3000,
        base_url: str = SMALLEST_BASE_URL,
        ws_url: str = SMALLEST_WS_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Smallest AI Lightning TTS.

        Args:
            api_key: Your Smallest AI API key.
            model: The TTS model to use. Use "lightning_v3.1" for the standard model with
                217 voices across 12 languages, or "lightning_v3.1_pro" (default) for the
                premium pool with curated American, British, and Indian voices at 44.1 kHz.
            voice_id: The voice ID to use for synthesis. Defaults to "meher" for
                "lightning_v3.1_pro" and "sophia" for all other models. Pro voices must be
                paired with "lightning_v3.1_pro"; standard voices with "lightning_v3.1".
            sample_rate: Sample rate for the audio output. Both models are natively 44.1 kHz;
                supported rates are 8000, 16000, 24000, and 44100.
            speed: Speed of the speech synthesis (0.5–2.0).
            language: Language of the text to be synthesized. Use "auto" for automatic
                detection and code-switching. Pro supports "en", "hi", and "auto" only.
            output_format: Output format for HTTP synthesize() calls ("pcm", "mp3", "wav",
                "ulaw", "alaw"). WebSocket streaming always returns PCM.
            word_timestamps: Request per-word timing events from the server and emit them
                as timed transcript entries alongside audio. Applies to WebSocket streaming
                only; HTTP synthesize() returns raw audio without word events. Disabled by
                default. Supported on base-queue English + Hindi voices (meher, devansh,
                kartik, maithili, liam, avery); other voices silently emit no word events.
            max_buffer_flush_ms: Server-side buffer bound (milliseconds) for the legacy
                ``continue``/``flush`` WebSocket protocol (used when ``use_continuations``
                is False). As text tokens are streamed in with ``continue: true``, the
                server accumulates them and forces partial audio output once this many
                milliseconds of text have buffered, without waiting for an explicit flush.
                ``0`` (default) disables time-based forced flushing, so audio for a segment
                is produced when the segment's end-of-input flush is sent. Ignored when
                ``use_continuations`` is True.
            use_continuations: Use the ``context_id``-based continuations protocol instead
                of the legacy ``continue``/``flush`` fields. Continuations let the server
                release buffered text as soon as it hits a natural sentence boundary (or
                ``max_buffer_delay_ms`` elapses, whichever comes first) instead of holding
                everything until an explicit flush, and carry prosody across fragments by
                priming each release with the audio of the previous one. Defaults to True;
                set False to fall back to the legacy protocol.
            max_buffer_delay_ms: Upper bound (0-5000ms) on how long the server holds a
                buffered fragment waiting for a clean sentence boundary before speaking it
                anyway. Only applies when ``use_continuations`` is True. Defaults to 3000ms
                (the server default).
            base_url: Base URL for the Smallest AI HTTP API.
            ws_url: WebSocket URL for low-latency streaming synthesis.
            http_session: An existing aiohttp ClientSession to use.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=word_timestamps,
            ),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        api_key = api_key or os.environ.get("SMALLEST_API_KEY")
        if not api_key:
            raise ValueError(
                "Smallest.ai API key is required, either as argument or set"
                " SMALLEST_API_KEY environment variable"
            )

        if voice_id is None:
            voice_id = "meher" if model == "lightning_v3.1_pro" else "sophia"

        self._opts = _TTSOptions(
            model=model,
            api_key=api_key,
            voice_id=voice_id,
            sample_rate=sample_rate,
            speed=speed,
            language=LanguageCode(language),
            output_format=output_format,
            word_timestamps=word_timestamps,
            max_buffer_flush_ms=max_buffer_flush_ms,
            use_continuations=use_continuations,
            max_buffer_delay_ms=max_buffer_delay_ms,
            base_url=base_url,
            ws_url=ws_url,
        )
        self._session = http_session
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=3600,
            mark_refreshed_on_get=False,
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "SmallestAI"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        return await asyncio.wait_for(
            self._ensure_session().ws_connect(
                self._opts.ws_url,
                headers={
                    "Authorization": f"Bearer {self._opts.api_key}",
                    "X-Source": "livekit",
                    "X-LiveKit-Version": __version__,
                },
            ),
            timeout,
        )

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        output_format: NotGivenOr[TTSEncoding | str] = NOT_GIVEN,
        word_timestamps: NotGivenOr[bool] = NOT_GIVEN,
        max_buffer_flush_ms: NotGivenOr[int] = NOT_GIVEN,
        use_continuations: NotGivenOr[bool] = NOT_GIVEN,
        max_buffer_delay_ms: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """Update TTS options."""
        if is_given(model):
            self._opts.model = model
        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(speed):
            self._opts.speed = speed
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if is_given(language):
            self._opts.language = LanguageCode(language)
        if is_given(output_format):
            self._opts.output_format = output_format
        if is_given(word_timestamps):
            self._opts.word_timestamps = word_timestamps
            self._capabilities.aligned_transcript = word_timestamps
        if is_given(max_buffer_flush_ms):
            self._opts.max_buffer_flush_ms = max_buffer_flush_ms
        if is_given(use_continuations):
            self._opts.use_continuations = use_continuations
        if is_given(max_buffer_delay_ms):
            self._opts.max_buffer_delay_ms = max_buffer_delay_ms

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
        return SynthesizeStream(tts=self, conn_options=conn_options)

    def prewarm(self) -> None:
        self._pool.prewarm()

    async def aclose(self) -> None:
        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    """HTTP-based synthesis — used when synthesize() is called directly."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            data = _to_smallest_options(self._opts)
            data["text"] = self._input_text

            headers = {
                "Authorization": f"Bearer {self._opts.api_key}",
                "Content-Type": "application/json",
                "X-Source": "livekit",
                "X-LiveKit-Version": __version__,
            }
            async with self._tts._ensure_session().post(
                f"{self._opts.base_url}/tts",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=self._conn_options.timeout),
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise create_api_error_from_http(body, status=resp.status)

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type=f"audio/{self._opts.output_format}",
                )

                async for chunk, _ in resp.content.iter_chunks():
                    output_emitter.push(chunk)

                output_emitter.flush()

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise create_api_error_from_http(e.message, status=e.status) from None
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """WebSocket-based streaming synthesis — primary path used by the agent pipeline.

    Defaults to the continuations protocol: text fragments are sent under a shared
    ``context_id`` and the server releases audio at natural sentence boundaries (or
    ``max_buffer_delay_ms``, whichever comes first) instead of waiting for the whole
    segment, priming each release with the previous one's audio so prosody stays
    continuous. Falls back to the legacy ``continue``/``flush`` protocol when
    ``use_continuations`` is False.
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
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
        # One stream instance = one segment (the agent creates a new stream per segment).
        output_emitter.start_segment(segment_id=request_id)

        # Reused as the continuations `context_id`; shortuuid()'s output satisfies the
        # server's alphanumeric/length constraints.
        context_id = request_id
        # opened: a fragment was sent under context_id. finalized: the closing
        # (`continue: false`) fragment was sent. Used to detect an interrupted context
        # (e.g. barge-in cancels this stream mid-way) so we can tell the server to drop it.
        ctx_state = {"opened": False, "finalized": False}

        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                self._acquire_time = self._tts._pool.last_acquire_time
                self._connection_reused = self._tts._pool.last_connection_reused

                send_task = asyncio.create_task(self._send_task(ws, context_id, ctx_state))
                recv_task = asyncio.create_task(self._recv_task(ws, output_emitter, ctx_state))
                # Whether recv_task ended on a protocol-confirmed signal (legacy's
                # explicit `complete`) vs. the continuations idle-drain heuristic, which
                # is never fully certain - see _recv_task and the pool.remove() below.
                drained_confidently = True
                try:
                    # send_task reports whether any text was actually sent; if the
                    # segment was empty (no non-whitespace tokens) no context was opened
                    # and the server produces no `complete`, so don't wait on recv_task.
                    sent_any = await send_task
                    if sent_any:
                        drained_confidently = await recv_task
                    else:
                        await utils.aio.gracefully_cancel(recv_task)
                finally:
                    await utils.aio.gracefully_cancel(send_task, recv_task)
                    if self._opts.use_continuations and ctx_state["opened"]:
                        if not ctx_state["finalized"]:
                            # Interrupted (e.g. barge-in) before the context was closed.
                            await self._try_cancel_context(ws, context_id)
                            self._tts._pool.remove(ws)
                        elif not drained_confidently:
                            # Completion was inferred, not confirmed by the server (no
                            # terminal marker exists in this protocol - see _recv_task).
                            # Don't let a possibly-still-active context leak trailing
                            # frames into the next request that reuses this connection.
                            self._tts._pool.remove(ws)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=request_id, body=None
            ) from None
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError() from e

        output_emitter.end_segment()

    async def _try_cancel_context(
        self, ws: aiohttp.ClientWebSocketResponse, context_id: str
    ) -> None:
        # Best-effort: interrupted before the context was closed cleanly (e.g. the
        # agent's speech was cut off by user barge-in). Tell the server to drop
        # whatever's still buffered instead of leaving the context open.
        with contextlib.suppress(Exception):
            pkt = {**self._base_payload(context_id), "cancel_request": True}
            await asyncio.wait_for(ws.send_str(json.dumps(pkt)), timeout=1.0)

    def _base_payload(self, context_id: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._opts.model,
            "voice_id": self._opts.voice_id,
            "sample_rate": self._opts.sample_rate,
            "speed": self._opts.speed,
            "language": self._opts.language.language
            if isinstance(self._opts.language, LanguageCode)
            else self._opts.language,
        }
        if self._opts.word_timestamps:
            payload["word_timestamps"] = True
        if self._opts.use_continuations:
            payload["context_id"] = context_id
            payload["max_buffer_delay_ms"] = self._opts.max_buffer_delay_ms
        else:
            payload["max_buffer_flush_ms"] = self._opts.max_buffer_flush_ms
        return payload

    async def _send_task(
        self, ws: aiohttp.ClientWebSocketResponse, context_id: str, ctx_state: dict[str, bool]
    ) -> bool:
        # Forward each token as it arrives (continuous streaming) instead of buffering
        # the whole segment, so the server can begin synthesis before the text is
        # complete. The flush sentinel (or end of input) closes the segment. Returns
        # whether any non-whitespace text was actually sent.
        sent_any = False
        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                break
            if not data.strip():
                continue
            token_pkt = {**self._base_payload(context_id), "text": data, "continue": True}
            if not self._opts.use_continuations:
                token_pkt["flush"] = False
            self._mark_started()
            await ws.send_str(json.dumps(token_pkt))
            sent_any = True
            ctx_state["opened"] = True

        # Only close when text was sent; an empty segment produces no `complete`, so
        # closing it would leave _recv_task waiting until the connection timeout.
        if sent_any:
            final_pkt = {**self._base_payload(context_id), "text": "", "continue": False}
            if not self._opts.use_continuations:
                final_pkt["flush"] = True
            await ws.send_str(json.dumps(final_pkt))
            ctx_state["finalized"] = True
        return sent_any

    async def _recv_task(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        output_emitter: tts.AudioEmitter,
        ctx_state: dict[str, bool],
    ) -> bool:
        """Returns whether completion was protocol-confirmed (safe to reuse the pooled
        connection) — always True for legacy, never True for continuations, which has
        no way to confirm it (see the loop below and _run's handling of the result).
        """
        if not self._opts.use_continuations:
            while True:
                msg = await self._recv_one(ws, timeout=self._conn_options.timeout)
                status, event = self._parse_status_event(msg, output_emitter)
                if status == "complete":
                    return True

        # Continuations gives no marker for which `complete` is the truly last one (a
        # context can release audio in more than one frame at natural sentence
        # boundaries), so instead of reacting to a specific `complete` event, poll with
        # a short idle timeout and re-check `ctx_state["finalized"]` fresh on every
        # tick. This also sidesteps any race between the closing fragment's send (which
        # flips `finalized`) and an early `complete` reply for it: we never depend on
        # which of the two concurrent tasks the event loop happens to run first, only
        # on the current value of `finalized` at each poll. Because there's still no
        # way to be certain we've drained everything, the caller must never reuse the
        # underlying connection when this heuristic (rather than an explicit signal) is
        # what ended the loop - see _run.
        event_loop = asyncio.get_event_loop()
        hard_deadline = event_loop.time() + self._conn_options.timeout
        while True:
            try:
                msg = await self._recv_one(ws, timeout=_CONTINUATIONS_IDLE_TIMEOUT)
            except asyncio.TimeoutError:
                if ctx_state["finalized"]:
                    return False
                if event_loop.time() >= hard_deadline:
                    raise
                continue
            self._parse_status_event(msg, output_emitter)

    async def _recv_one(
        self, ws: aiohttp.ClientWebSocketResponse, *, timeout: float
    ) -> aiohttp.WSMessage:
        msg = await ws.receive(timeout=timeout)
        if msg.type in (
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSING,
        ):
            raise APIStatusError(
                "SmallestAI WebSocket closed unexpectedly",
                status_code=ws.close_code or -1,
                body=str(msg.data),
            )
        return msg

    def _parse_status_event(
        self, msg: aiohttp.WSMessage, output_emitter: tts.AudioEmitter
    ) -> tuple[str | None, dict[str, Any]]:
        if msg.type != aiohttp.WSMsgType.TEXT:
            return None, {}

        event = json.loads(msg.data)
        status = event.get("status")

        if status == "chunk":
            audio_b64 = event.get("data", {}).get("audio")
            if audio_b64:
                output_emitter.push(base64.b64decode(audio_b64))
        elif status == "word_timestamp":
            data = event.get("data", {})
            word = data.get("word")
            start = data.get("start")
            end = data.get("end")
            if word is not None and start is not None and end is not None:
                output_emitter.push_timed_transcript(
                    TimedString(text=word, start_time=start, end_time=end)
                )
        elif status == "error":
            raise APIConnectionError(
                f"SmallestAI TTS error: {event.get('message', 'unknown error')}"
            )
        return status, event


def _to_smallest_options(opts: _TTSOptions) -> dict[str, Any]:
    return {
        "model": opts.model,
        "voice_id": opts.voice_id,
        "sample_rate": opts.sample_rate,
        "speed": opts.speed,
        "language": opts.language.language
        if isinstance(opts.language, LanguageCode)
        else opts.language,
        "output_format": opts.output_format,
    }
