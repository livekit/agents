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

"""AssemblyAI Streaming TTS, see https://www.assemblyai.com/docs/tts/message-frames.

Sessions are pooled and reused across streams, since opening one costs a handshake and
new sessions per minute are rate limited. Each pooled socket owns a reader that keeps
draining it and drops frames left over from an earlier holder.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import time
import weakref
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
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TimedString

from .log import logger

TTSVoices = Literal[
    # English
    "alba",
    "anna",
    "charles",
    "eve",
    "george",
    "jane",
    "jean",
    "mary",
    "michael",
    "paul",
    "vera",
    # Spanish, German, Italian, Portuguese, French
    "lola",
    "juergen",
    "giovanni",
    "rafael",
    "estelle",
]

TTSLanguages = Literal["english", "spanish", "german", "italian", "portuguese", "french"]

DEFAULT_BASE_URL = "wss://streaming-tts.assemblyai.com"
SUPPORTED_SAMPLE_RATES = (8000, 16000, 22050, 24000, 44100, 48000)

# https://www.assemblyai.com/docs/tts/voices. Each voice speaks one language, and a
# mismatch only fails ~0.5s after `Begin`, so the language is derived from the voice.
VOICE_LANGUAGES: dict[str, str] = {
    "alba": "english",
    "anna": "english",
    "charles": "english",
    "eve": "english",
    "george": "english",
    "jane": "english",
    "jean": "english",
    "mary": "english",
    "michael": "english",
    "paul": "english",
    "vera": "english",
    "lola": "spanish",
    "juergen": "german",
    "giovanni": "italian",
    "rafael": "portuguese",
    "estelle": "french",
}

DEFAULT_VOICES: dict[str, str] = {
    "english": "jane",
    "spanish": "lola",
    "german": "juergen",
    "italian": "giovanni",
    "portuguese": "rafael",
    "french": "estelle",
}

# `language` takes the full English name of the language, not an ISO code.
_ISO_LANGUAGES = {
    "en": "english",
    "es": "spanish",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "fr": "french",
}

# Generate's limit, in code points. A longer sentence becomes several requests rather
# than several Generates under one Flush: together with _MAX_IN_FLIGHT_REQUESTS that
# keeps the unsynthesized backlog under the 8000 chars that close the session (3010).
_MAX_REQUEST_CHARS = 2000
_MAX_IN_FLIGHT_REQUESTS = 3

# don't start a reply on a session about to be closed at its `expires_at` (3008)
_EXPIRY_MARGIN = 60.0

# how long the end of a segment waits for the WordBoundaries trailing its FlushDone
_WORD_BOUNDARIES_GRACE = 0.5

# a rejected credential, or a bad parameter, frame or voice: retrying can't help
_NON_RETRYABLE_CODES = (1008, 3006)
# ...except a 3006 from the inactivity timeout
_INACTIVITY_PREFIX = "No message received"

_FLUSH_FRAME = json.dumps({"type": "Flush"})
_CANCEL_FRAME = json.dumps({"type": "Cancel"})
_KEEPALIVE_FRAME = json.dumps({"type": "KeepAlive"})


@dataclass
class _TTSOptions:
    voice: str
    language: str | None
    sample_rate: int
    word_timestamps: bool
    api_key: str
    base_url: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: NotGivenOr[TTSVoices | str] = NOT_GIVEN,
        language: NotGivenOr[TTSLanguages | str] = NOT_GIVEN,
        sample_rate: int = 24000,
        word_timestamps: bool = True,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = DEFAULT_BASE_URL,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        text_pacing: tts.SentenceStreamPacer | bool = False,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a new instance of AssemblyAI Streaming TTS.

        See https://www.assemblyai.com/docs/tts/getting-started/quickstart for more
        information on the API.

        Args:
            voice: A preset voice, lowercase. Defaults to the default voice of
                ``language``, which is ``"jane"`` for English. See
                https://www.assemblyai.com/docs/tts/voices.
            language: The language to synthesize. Accepts the API's full English name
                (``"spanish"``) or a language code (``"es"``, ``"es-MX"``). Defaults to
                the language of ``voice``; only needed for a voice that is not in
                ``VOICE_LANGUAGES``.
            sample_rate: Output sample rate in Hz, one of 8000, 16000, 22050, 24000,
                44100 or 48000. Defaults to 24000, the service's native rate.
            word_timestamps: Request per-word timings and publish them as an aligned
                transcript. Defaults to True.
            api_key: AssemblyAI API key. Defaults to the ``ASSEMBLYAI_API_KEY``
                environment variable.
            base_url: The Streaming TTS host. Defaults to the global endpoint, which
                routes to the nearest region. Use ``"wss://streaming-tts.us.assemblyai.com"``
                or ``"wss://streaming-tts.eu.assemblyai.com"`` to keep data in one zone.
            tokenizer: Splits the LLM output into the requests sent to the server; each
                sentence is synthesized as its own request. Defaults to
                ``livekit.agents.tokenize.blingfire.SentenceTokenizer``.
            text_pacing: Stream pacer for the TTS. Set to True to use the default
                pacer, False to disable.
            http_session: An existing aiohttp ClientSession to use.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=word_timestamps,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )

        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate must be one of {SUPPORTED_SAMPLE_RATES}, but got {sample_rate}"
            )

        assemblyai_api_key = api_key if is_given(api_key) else os.environ.get("ASSEMBLYAI_API_KEY")
        if not assemblyai_api_key:
            raise ValueError(
                "AssemblyAI API key is required. "
                "Pass one in via the `api_key` parameter, "
                "or set it as the `ASSEMBLYAI_API_KEY` environment variable"
            )

        resolved_voice, resolved_language = _resolve_voice(
            voice if is_given(voice) else None,
            language if is_given(language) else None,
        )

        self._opts = _TTSOptions(
            voice=resolved_voice,
            language=resolved_language,
            sample_rate=sample_rate,
            word_timestamps=word_timestamps,
            api_key=assemblyai_api_key,
            base_url=base_url.rstrip("/"),
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.blingfire.SentenceTokenizer()
        )
        self._stream_pacer: tts.SentenceStreamPacer | None = None
        if text_pacing is True:
            self._stream_pacer = tts.SentenceStreamPacer()
        elif isinstance(text_pacing, tts.SentenceStreamPacer):
            self._stream_pacer = text_pacing

        # Session lifetime is enforced from each socket's own `expires_at`, see _acquire().
        self._pool = utils.ConnectionPool[_Connection](
            connect_cb=self._connect,
            close_cb=self._close,
        )

    @property
    def model(self) -> str:
        return "streaming-tts"

    @property
    def provider(self) -> str:
        return "AssemblyAI"

    @property
    def voice(self) -> str:
        return self._opts.voice

    @property
    def language(self) -> str | None:
        return self._opts.language

    def update_options(
        self,
        *,
        voice: NotGivenOr[TTSVoices | str] = NOT_GIVEN,
        language: NotGivenOr[TTSLanguages | str] = NOT_GIVEN,
    ) -> None:
        """Change the voice or language used by streams created from now on.

        The session configuration is fixed at connect time, so pooled sockets are
        retired and the next stream opens a new session. A stream already speaking
        keeps its voice until it finishes.

        Args:
            voice: A preset voice. When ``language`` is not also given, the language
                follows the new voice.
            language: The language to synthesize. When ``voice`` is not also given and
                the current voice does not speak it, the voice switches to that
                language's default.
        """
        if is_given(voice):
            new_voice, new_language = _resolve_voice(
                voice, language if is_given(language) else None
            )
        elif is_given(language):
            new_language = _normalize_language(language)
            if VOICE_LANGUAGES.get(self._opts.voice, new_language) == new_language:
                new_voice = self._opts.voice
            else:
                new_voice, new_language = _resolve_voice(None, language)
        else:
            return

        if (new_voice, new_language) != (self._opts.voice, self._opts.language):
            self._opts.voice = new_voice
            self._opts.language = new_language
            self._pool.invalidate()

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        # there is no REST endpoint; a one-shot synthesis runs over a pooled socket too.
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        self._pool.prewarm()

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        await self._pool.aclose()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _connect(self, timeout: float) -> _Connection:
        opts = self._opts
        params: dict[str, Any] = {
            "voice": opts.voice,
            "sample_rate": opts.sample_rate,
            "encoding": "pcm_s16le",
            "word_boundaries": "true" if opts.word_timestamps else "false",
        }
        if opts.language:
            params["language"] = opts.language
        url = f"{opts.base_url}/v1/ws?{urlencode(params)}"

        try:
            ws = await asyncio.wait_for(
                # the key goes in raw: "Bearer <key>" is rejected as an unknown key
                self._ensure_session().ws_connect(url, headers={"Authorization": opts.api_key}),
                timeout,
            )
        except asyncio.TimeoutError:
            raise APITimeoutError("timed out connecting to AssemblyAI TTS") from None
        except aiohttp.ClientResponseError as e:
            # RequestInfo carries the request headers, including the API key.
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError("failed to connect to AssemblyAI TTS") from e

        # credentials and parameters are checked after the upgrade: a failure arrives as
        # an Error frame in place of Begin
        try:
            try:
                msg = await ws.receive(timeout=timeout)
            except asyncio.TimeoutError:
                raise APITimeoutError("timed out waiting for AssemblyAI TTS Begin") from None

            begin = _parse_frame(msg)
            if begin is None:
                raise APIStatusError(
                    "AssemblyAI TTS connection closed before Begin",
                    status_code=ws.close_code or -1,
                    retryable=True,
                )
            if begin.get("type") == "Error":
                raise _error_from_frame(begin, session_id=None)
            if begin.get("type") != "Begin":
                raise APIConnectionError(
                    f"expected Begin from AssemblyAI TTS, got {begin.get('type')!r}"
                )
        except BaseException:
            await ws.close()
            raise

        conn = _Connection(ws, begin)
        logger.debug(
            "connected to AssemblyAI TTS",
            extra={"session_id": conn.session_id, "configuration": conn.configuration},
        )

        if conn.sample_rate != opts.sample_rate:
            # the stream would play every frame at the wrong speed and pitch
            await conn.aclose()
            raise APIError(
                f"AssemblyAI TTS session {conn.session_id} was created at "
                f"{conn.sample_rate} Hz instead of the requested {opts.sample_rate} Hz",
                retryable=False,
            )
        if opts.word_timestamps and not conn.word_boundaries:
            logger.warning(
                "AssemblyAI TTS cannot supply word timings for this session, "
                "the transcript will be paced per sentence",
                extra={"session_id": conn.session_id},
            )
        return conn

    async def _close(self, conn: _Connection) -> None:
        await conn.aclose()

    async def _acquire(self, *, timeout: float) -> _Connection:
        while True:
            conn = await self._pool.get(timeout=timeout)
            if conn.usable:
                return conn
            # closed by the server while idle, or too close to its expires_at
            self._pool.remove(conn)

    def _release(self, conn: _Connection, *, discard: bool) -> None:
        if discard or not conn.usable:
            self._pool.remove(conn)
        else:
            self._pool.put(conn)


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        # Acquire before any text arrives, so the handshake overlaps the LLM's first token
        # (text waits in the input channel meanwhile).
        try:
            conn = await self._tts._acquire(timeout=self._conn_options.timeout)
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError("failed to acquire an AssemblyAI TTS connection") from e

        self._acquire_time = self._tts._pool.last_acquire_time
        self._connection_reused = self._tts._pool.last_connection_reused
        if conn.session_id:
            output_emitter._note_provider_request_id(conn.session_id)

        sent_stream = self._tts._sentence_tokenizer.stream()
        if self._tts._stream_pacer:
            sent_stream = self._tts._stream_pacer.wrap(
                sent_stream=sent_stream, audio_emitter=output_emitter
            )

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_stream.flush()
                else:
                    sent_stream.push_text(data)
            sent_stream.end_input()

        run = _StreamRun(conn=conn, listener=conn.attach(), emitter=output_emitter, stream=self)
        tasks = [
            asyncio.create_task(_input_task()),
            asyncio.create_task(run.run(sent_stream)),
        ]
        discard = True
        try:
            await asyncio.gather(*tasks)
            discard = False
        except asyncio.CancelledError:
            # Interrupted (barge-in). Cancel whatever the server still has for us and
            # hand the socket back: frames it already sent for this stream are dropped
            # by the reader until `Cancelled` confirms the server moved on.
            if run.has_outstanding:
                with contextlib.suppress(APIError):
                    conn.send(_CANCEL_FRAME)
                    conn.note_cancel_sent()
            discard = False
            raise
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError(
                f"AssemblyAI TTS stream failed (session {conn.session_id})"
            ) from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            await sent_stream.aclose()
            conn.detach(run.listener)
            self._tts._release(conn, discard=discard)


@dataclass(eq=False)
class _Request:
    text: str
    # where this request's audio starts in the stream, set by its first Audio frame
    segment_offset: float | None = None
    duration: float = 0.0
    done: bool = False
    transcript_pushed: bool = False


class _StreamRun:
    """One attempt of a SynthesizeStream on one connection."""

    def __init__(
        self,
        *,
        conn: _Connection,
        listener: _Listener,
        emitter: tts.AudioEmitter,
        stream: SynthesizeStream,
    ) -> None:
        self.conn = conn
        self.listener = listener
        self._emitter = emitter
        self._stream = stream
        self._timeout = stream._conn_options.timeout
        self._bytes_per_second = stream._opts.sample_rate * 2  # pcm_s16le, mono
        self._push_transcript = stream._opts.word_timestamps
        self._word_timings = stream._opts.word_timestamps and conn.word_boundaries
        # requests in send order: the n-th one gets flush_id `listener.base_flush_id + n`
        self._requests: list[_Request] = []
        self._num_done = 0

    @property
    def has_outstanding(self) -> bool:
        return self._num_done < len(self._requests)

    async def run(self, sent_stream: tokenize.SentenceStream) -> None:
        segment_id = utils.shortuuid()
        segment_bytes = 0
        in_flight = asyncio.Semaphore(_MAX_IN_FLIGHT_REQUESTS)
        input_done = False
        progress = asyncio.Event()
        started = False

        async def _send_task() -> None:
            nonlocal input_done, started
            async for ev in sent_stream:
                for text in _split_request_text(ev.token):
                    await in_flight.acquire()
                    if not started:
                        self._emitter.start_segment(segment_id=segment_id)
                        started = True
                    self.conn.send(json.dumps({"type": "Generate", "text": text}))
                    self.conn.send(_FLUSH_FRAME)
                    self._requests.append(_Request(text=text))
                    self._stream._mark_started()
                    progress.set()
            input_done = True
            progress.set()

        async def _recv_task() -> None:
            nonlocal segment_bytes
            while True:
                if self._num_done == len(self._requests):
                    if input_done:
                        break
                    # nothing outstanding: wait for the next sentence, without a timeout,
                    # since the LLM may still be thinking
                    progress.clear()
                    await progress.wait()
                    continue

                try:
                    item = await asyncio.wait_for(self.listener.recv(), self._timeout)
                except asyncio.TimeoutError:
                    raise APITimeoutError(
                        f"AssemblyAI TTS sent nothing for {self._timeout}s "
                        f"(session {self.conn.session_id})"
                    ) from None
                if isinstance(item, APIError):
                    raise item

                request = self._request_for(item)
                if request is None:
                    continue

                kind = item["type"]
                if kind == "Audio":
                    audio = base64.b64decode(item["audio"])
                    if request.segment_offset is None:
                        request.segment_offset = segment_bytes / self._bytes_per_second
                    self._emitter.push(audio)
                    segment_bytes += len(audio)
                elif kind == "FlushDone":
                    request.done = True
                    request.duration = (item.get("audio_duration_ms") or 0) / 1000
                    self._num_done += 1
                    in_flight.release()
                    if not self._word_timings:
                        self._push_untimed(request)
                elif kind == "WordBoundaries":
                    self._push_word_boundaries(request, item)

            if started:
                # release the buffered audio now; only the end of the segment waits for timings
                self._emitter.flush()
            await self._await_word_boundaries()
            if started:
                self._emitter.end_segment()

        tasks = [asyncio.create_task(_send_task()), asyncio.create_task(_recv_task())]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    def _request_for(self, item: dict[str, Any]) -> _Request | None:
        flush_id = item.get("flush_id")
        base = self.listener.base_flush_id
        if not isinstance(flush_id, int) or base is None:
            return None
        index = flush_id - base
        if not 0 <= index < len(self._requests):
            logger.warning(
                "AssemblyAI TTS sent a frame for an unknown request",
                extra={"session_id": self.conn.session_id, "flush_id": flush_id},
            )
            return None
        return self._requests[index]

    async def _await_word_boundaries(self) -> None:
        """Give the trailing WordBoundaries frames a moment before the segment ends."""
        if not self._word_timings:
            return

        deadline = time.monotonic() + _WORD_BOUNDARIES_GRACE
        while any(self._awaiting_words(r) for r in self._requests):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(self.listener.recv(), remaining)
            except asyncio.TimeoutError:
                break
            if isinstance(item, APIError):
                # the audio is all delivered; only the timings are lost
                logger.warning("AssemblyAI TTS failed while waiting for word boundaries: %s", item)
                break
            if item.get("type") == "WordBoundaries" and (request := self._request_for(item)):
                self._push_word_boundaries(request, item)

        for request in self._requests:
            self._push_untimed(request)

    @staticmethod
    def _awaiting_words(request: _Request) -> bool:
        # a request that produced no audio gets no WordBoundaries
        return request.segment_offset is not None and not request.transcript_pushed

    def _push_word_boundaries(self, request: _Request, item: dict[str, Any]) -> None:
        if request.transcript_pushed or request.segment_offset is None:
            return

        # timings are best-effort: earlier requests still without any get their text
        # untimed now, so the transcript keeps its order
        index = self._requests.index(request)
        for earlier in self._requests[:index]:
            self._push_untimed(earlier)

        # `start`/`end` are on the session timeline, where this request begins at
        # `audio_start_ms`; re-anchor them on the request's position in this segment
        audio_start = (item.get("audio_start_ms") or 0) / 1000
        offset = request.segment_offset - audio_start
        words: list[TimedString] = []
        for word in item.get("words") or []:
            text, start, end = word.get("word"), word.get("start"), word.get("end")
            if not text or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            words.append(
                TimedString(
                    text=f"{text} ",
                    start_time=offset + start / 1000,
                    end_time=offset + end / 1000,
                    confidence=word.get("confidence", NOT_GIVEN),
                )
            )

        request.transcript_pushed = True
        if words:
            self._emitter.push_timed_transcript(words)
        else:
            self._push_text(request)

    def _push_untimed(self, request: _Request) -> None:
        """Push a request's text spanning its whole audio, for when word timings are missing."""
        if request.transcript_pushed or not request.done:
            return
        request.transcript_pushed = True
        self._push_text(request)

    def _push_text(self, request: _Request) -> None:
        if not self._push_transcript or request.segment_offset is None:
            return
        self._emitter.push_timed_transcript(
            TimedString(
                text=f"{request.text} ",
                start_time=request.segment_offset,
                end_time=request.segment_offset + request.duration,
            )
        )


class _Listener:
    """The receiving end of a connection for the stream currently holding it."""

    def __init__(self) -> None:
        self._ch = utils.aio.Chan[dict[str, Any] | APIError]()
        # flush_id of the stream's first request; unknown while a Cancel from the previous
        # holder is unanswered, since `Cancelled` is what settles the next id
        self.base_flush_id: int | None = None

    def push(self, item: dict[str, Any] | APIError) -> None:
        if not self._ch.closed:
            self._ch.send_nowait(item)

    async def recv(self) -> dict[str, Any] | APIError:
        return await self._ch.recv()

    def close(self) -> None:
        self._ch.close()


class _Connection:
    """A Streaming TTS session: one socket, its reader and its writer.

    The reader runs for the socket's whole life, independent of which stream holds it,
    so the socket keeps being drained (the server closes with 3011 when audio piles
    up), a close or Error is noticed while it sits idle in the pool, and frames left
    over from a previous holder never reach the next one.
    """

    def __init__(self, ws: aiohttp.ClientWebSocketResponse, begin: dict[str, Any]) -> None:
        self._ws = ws
        self.session_id: str | None = begin.get("id")
        self.configuration: dict[str, Any] = begin.get("configuration") or {}
        expires_at = begin.get("expires_at")
        self.expires_at = (
            float(expires_at) if isinstance(expires_at, (int, float)) else time.time() + 3600
        )
        self.sample_rate = self.configuration.get("sample_rate")
        self.word_boundaries = self.configuration.get("word_boundaries") is True
        inactivity_timeout = self.configuration.get("inactivity_timeout")

        # FlushDone + Cancelled frames received: the flush_id the next request will get
        self._num_retired = 0
        self._cancels_pending = 0
        self._listener: _Listener | None = None
        self._error: APIError | None = None
        self._closed = False

        self._send_ch = utils.aio.Chan[str]()
        self._recv_atask = asyncio.create_task(self._recv_loop(), name="AssemblyAITTS.recv")
        self._send_atask = asyncio.create_task(
            self._send_loop(
                # the server closes an idle session after `inactivity_timeout` seconds
                # without a client frame; that is in effect only when Begin echoes one
                keepalive_interval=(
                    inactivity_timeout / 2
                    if isinstance(inactivity_timeout, (int, float)) and inactivity_timeout > 0
                    else None
                )
            ),
            name="AssemblyAITTS.send",
        )

    @property
    def usable(self) -> bool:
        return (
            not self._closed
            and self._error is None
            and time.time() < self.expires_at - _EXPIRY_MARGIN
        )

    def attach(self) -> _Listener:
        # While a previous holder's Cancel is unanswered the new holder has no base, so
        # nothing reaches it until `Cancelled` settles the next flush_id. This relies on
        # the Cancel being queued, and note_cancel_sent() called, before the socket is put
        # back in the pool.
        listener = _Listener()
        if self._error is not None:
            listener.push(self._error)
        elif self._cancels_pending == 0:
            listener.base_flush_id = self._num_retired
        self._listener = listener
        return listener

    def detach(self, listener: _Listener) -> None:
        if self._listener is listener:
            self._listener = None
        listener.close()

    def send(self, frame: str) -> None:
        if self._error is not None:
            raise self._error
        if self._closed or self._send_ch.closed:
            raise APIConnectionError(f"AssemblyAI TTS session {self.session_id} is closed")
        self._send_ch.send_nowait(frame)

    def note_cancel_sent(self) -> None:
        self._cancels_pending += 1

    async def aclose(self) -> None:
        self._closed = True
        self._send_ch.close()
        await utils.aio.gracefully_cancel(self._send_atask, self._recv_atask)
        # closing without Terminate makes the server drop whatever is still in flight
        await self._ws.close()

    def _fail(self, error: APIError) -> None:
        if self._error is not None:
            return
        self._error = error
        if self._listener is not None:
            self._listener.push(error)

    async def _send_loop(self, *, keepalive_interval: float | None) -> None:
        try:
            while True:
                try:
                    if keepalive_interval is None:
                        frame = await self._send_ch.recv()
                    else:
                        frame = await asyncio.wait_for(self._send_ch.recv(), keepalive_interval)
                except asyncio.TimeoutError:
                    frame = _KEEPALIVE_FRAME
                except utils.aio.ChanClosed:
                    return
                await self._ws.send_str(frame)
        except Exception as e:
            self._fail(
                APIConnectionError(
                    f"failed to send to AssemblyAI TTS session {self.session_id}: "
                    f"{type(e).__name__}"
                )
            )

    async def _recv_loop(self) -> None:
        try:
            while True:
                msg = await self._ws.receive()
                data = _parse_frame(msg)
                if data is None:
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSING,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        break
                    continue
                self._on_frame(data)
        except Exception as e:
            self._fail(
                APIConnectionError(
                    f"failed to read from AssemblyAI TTS session {self.session_id}: "
                    f"{type(e).__name__}"
                )
            )
            return
        finally:
            self._send_ch.close()

        # Every server-initiated failure close is preceded by an Error frame, which
        # _on_frame has already recorded; anything else was a dropped connection.
        if not self._closed:
            self._fail(
                APIStatusError(
                    f"AssemblyAI TTS session {self.session_id} closed unexpectedly",
                    status_code=self._ws.close_code or 1006,
                    request_id=self.session_id,
                    retryable=True,
                )
            )

    def _on_frame(self, data: dict[str, Any]) -> None:
        kind = data.get("type")
        if kind == "Error":
            error = _error_from_frame(data, session_id=self.session_id)
            logger.warning(
                "AssemblyAI TTS error: %s",
                error.message,
                extra={"session_id": self.session_id, "error_code": data.get("error_code")},
            )
            self._fail(error)
            return

        if kind == "Cancelled":
            self._num_retired += 1
            if self._cancels_pending == 0:
                # would re-base the current holder; every Cancel we send is counted
                logger.warning(
                    "AssemblyAI TTS sent an unexpected Cancelled",
                    extra={"session_id": self.session_id},
                )
                return
            self._cancels_pending -= 1
            if self._cancels_pending == 0 and self._listener is not None:
                self._listener.base_flush_id = self._num_retired
            return

        if kind == "FlushDone":
            self._num_retired += 1

        if kind not in ("Audio", "FlushDone", "WordBoundaries"):
            return  # Begin, Termination, and frame types added later

        listener = self._listener
        flush_id = data.get("flush_id")
        if (
            listener is None
            or listener.base_flush_id is None
            or not isinstance(flush_id, int)
            or flush_id < listener.base_flush_id
        ):
            # no holder, or a WordBoundaries trailing the previous holder's last request
            return
        listener.push(data)


def _parse_frame(msg: aiohttp.WSMessage) -> dict[str, Any] | None:
    if msg.type != aiohttp.WSMsgType.TEXT:
        return None
    try:
        data = json.loads(msg.data)
    except json.JSONDecodeError:
        logger.warning("AssemblyAI TTS sent a frame that is not JSON")
        return None
    return data if isinstance(data, dict) else None


def _error_from_frame(data: dict[str, Any], *, session_id: str | None) -> APIStatusError:
    code = data.get("error_code")
    code = code if isinstance(code, int) else -1
    text = str(data.get("error") or "")
    retryable = code not in _NON_RETRYABLE_CODES or text.startswith(_INACTIVITY_PREFIX)
    return APIStatusError(
        f"AssemblyAI TTS error {code}: {text}",
        status_code=code,
        request_id=session_id,
        body=data,
        retryable=retryable,
    )


def _normalize_language(language: str) -> str:
    name = language.strip().lower()
    if name in DEFAULT_VOICES:
        return name
    try:
        return _ISO_LANGUAGES.get(LanguageCode(name).language, name)
    except Exception:
        # not something LanguageCode understands; let the server decide
        return name


def _resolve_voice(voice: str | None, language: str | None) -> tuple[str, str | None]:
    """Pair a voice with the language it speaks, filling in whichever is missing."""
    normalized = _normalize_language(language) if language else None

    if voice is None:
        lang = normalized or "english"
        if lang not in DEFAULT_VOICES:
            raise ValueError(f"no default voice for language {language!r}; pass `voice` explicitly")
        return DEFAULT_VOICES[lang], lang

    voice_language = VOICE_LANGUAGES.get(voice)
    if voice_language is None:
        if voice.lower() in VOICE_LANGUAGES:
            raise ValueError(f"voice names are case-sensitive, use {voice.lower()!r}")
        # Not in the published catalog, which the server's list can be longer than.
        # Without a language the server assumes English.
        return voice, normalized

    if normalized is not None and normalized != voice_language:
        raise ValueError(f"voice {voice!r} speaks {voice_language}, but language is {normalized!r}")
    return voice, voice_language


def _split_request_text(text: str) -> list[str]:
    """Split text into requests of at most _MAX_REQUEST_CHARS, preferring whitespace."""
    text = text.strip()
    pieces: list[str] = []
    while len(text) > _MAX_REQUEST_CHARS:
        cut = text.rfind(" ", 0, _MAX_REQUEST_CHARS + 1)
        if cut <= 0:
            cut = _MAX_REQUEST_CHARS
        pieces.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        pieces.append(text)
    return pieces
