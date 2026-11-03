# Copyright 2026 Saidwire
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

"""Saidwire real-time speech-to-text for LiveKit Agents.

* ``stream()``: WebSocket streaming with interim and final transcripts, START/END_OF_SPEECH
  events and usage events. ``flush()``, which the agent session calls when its VAD or turn
  detector ends the user's turn, sends the API's ``Finalize`` message; the server answers
  right away with the final transcript.
* ``recognize()``: pre-recorded audio over ``POST /v1/listen``.
* ``early_word`` events (``early_words=True``): each word as soon as it is committed::

      stt = saidwire.STT(early_words=True)
      stt.on("early_word", lambda word: print(word["word"]))
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import time
import urllib.parse
import weakref
from dataclasses import dataclass
from typing import Any

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given
from livekit.agents.voice.io import TimedString

from .log import logger
from .version import __version__

DEFAULT_URL = "wss://api.saidwire.com/v1/listen"
DEFAULT_MODEL = "saidwire-en-2026-10-30"

_KEEPALIVE = json.dumps({"type": "KeepAlive"})
_FINALIZE = json.dumps({"type": "Finalize"})
_CLOSE_STREAM = json.dumps({"type": "CloseStream"})
_USER_AGENT = f"livekit-plugins-saidwire/{__version__}"

_HTTP_HINTS = {
    400: "bad request, or unsupported audio or language",
    401: "invalid API key",
    402: "free credit used up or payment required",
    403: "key revoked",
    429: "too many streams or rate limited",
    503: "service starting up or busy",
}
# Refusals that mean "not now": the service is starting up, busy or rate limited.
_WAIT_HTTP = {429, 503}

# Error codes after which retrying cannot help, with the HTTP status they map to.
_PERMANENT_ERRORS = {
    "INVALID_AUTH": 401,
    "PAYMENT_REQUIRED": 402,
    "QUOTA_EXCEEDED": 402,
    "KEY_REVOKED": 403,
    "FORBIDDEN": 403,
    "BAD_REQUEST": 400,
    "UNSUPPORTED_AUDIO": 400,
    "UNSUPPORTED_LANGUAGE": 400,
}


@dataclass
class STTOptions:
    model: str
    language: str
    sample_rate: int
    interim_results: bool
    punctuate: bool
    endpointing_ms: int | None
    utterance_end_ms: int | None
    vad_events: bool
    early_words: bool
    warmup_timeout: float
    base_url: str


class _ServiceWarming(Exception):
    def __init__(self, status: int, retry_after: float, request_id: str | None) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.retry_after = retry_after
        self.request_id = request_id


def _http_url(ws_url: str) -> str:
    if ws_url.startswith("wss://"):
        return "https://" + ws_url[len("wss://") :]
    if ws_url.startswith("ws://"):
        return "http://" + ws_url[len("ws://") :]
    return ws_url


def _ws_url(url: str) -> str:
    if url.startswith("https://"):
        return "wss://" + url[len("https://") :]
    if url.startswith("http://"):
        return "ws://" + url[len("http://") :]
    return url


def _with_query(url: str, query: dict[str, Any]) -> str:
    def value(x: Any) -> str:
        return ("true" if x else "false") if isinstance(x, bool) else str(x)

    q = {k: value(x) for k, x in query.items() if x is not None}
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{urllib.parse.urlencode(q)}"


def _retry_after(headers: Any, default: float = 5.0) -> float:
    try:
        return max(1.0, min(30.0, float(headers.get("Retry-After"))))
    except (TypeError, ValueError):
        return default


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_URL,
        language: str = "en",
        sample_rate: int = 16000,
        interim_results: bool = True,
        punctuate: bool = True,
        endpointing_ms: int | None = None,
        utterance_end_ms: int | None = None,
        vad_events: bool = False,
        early_words: bool = False,
        warmup_timeout: float = 300.0,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a Saidwire STT.

        Args:
            api_key: Saidwire API key. Defaults to the ``SAIDWIRE_API_KEY`` environment
                variable.
            model: Model id. Defaults to ``saidwire-en-2026-10-30``.
            base_url: Streaming endpoint (``wss://api.saidwire.com/v1/listen``). Pre-recorded
                requests use the same path over HTTPS.
            language: ``en``, ``en-US``, ``en-GB``, ... The API is English-only.
            sample_rate: Rate audio is sent at; frames are resampled to it. 16000 suits most
                agents; use 8000 for telephony audio.
            interim_results: Emit INTERIM_TRANSCRIPT events while the user speaks.
            punctuate: Punctuation and capitals in transcripts.
            endpointing_ms: Pause, in ms, that ends an utterance on the server. ``None`` uses
                the server default. ``0`` turns server endpointing off: finals then come when
                the agent session flushes (VAD or turn detector) and at longer pauses.
            utterance_end_ms: Ask the server for an UtteranceEnd message (mapped to
                END_OF_SPEECH) after this much silence.
            vad_events: Ask the server for SpeechStarted messages (mapped to START_OF_SPEECH).
            early_words: Emit ``early_word`` events on this STT for each word as soon as it is
                committed, usually before the final that contains it.
            warmup_timeout: Seconds to keep retrying while the service answers "starting up"
                or "busy" (HTTP 503 / 429). Audio pushed while waiting is not transcribed.
            http_session: Optional aiohttp session. Defaults to the agent's shared session.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=interim_results,
                aligned_transcript="word",
                offline_recognize=True,
                # The session flushes at end of turn; flush sends Finalize.
                manual_flush=True,
            )
        )
        key = api_key if is_given(api_key) else os.environ.get("SAIDWIRE_API_KEY", "")
        if not key:
            raise ValueError("Saidwire API key is required: pass api_key= or set SAIDWIRE_API_KEY")
        if not language.lower().startswith("en"):
            raise ValueError(f"Saidwire is English-only (got language={language!r})")
        self._api_key = key
        self._opts = STTOptions(
            model=model,
            language=language,
            sample_rate=sample_rate,
            interim_results=interim_results,
            punctuate=punctuate,
            endpointing_ms=endpointing_ms,
            utterance_end_ms=utterance_end_ms,
            vad_events=vad_events,
            early_words=early_words,
            warmup_timeout=warmup_timeout,
            base_url=_ws_url(base_url),
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        """The model id sent to the API."""
        return self._opts.model

    @property
    def provider(self) -> str:
        """The provider name, ``Saidwire``."""
        return "Saidwire"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        lang = language if is_given(language) else self._opts.language
        url = _with_query(
            _http_url(self._opts.base_url),
            {"model": self._opts.model, "punctuate": self._opts.punctuate, "language": lang},
        )
        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "audio/wav",
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
        }
        try:
            async with self._ensure_session().post(
                url,
                data=rtc.combine_audio_frames(buffer).to_wav_bytes(),
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=max(30.0, conn_options.timeout * 3), sock_connect=conn_options.timeout
                ),
            ) as res:
                body = await res.text()
                if res.status != 200:
                    code, msg = _parse_error(body)
                    raise APIStatusError(
                        message=f"{code or 'HTTP ' + str(res.status)}: {msg or ''}".strip(),
                        status_code=res.status,
                        request_id=res.headers.get("x-request-id"),
                        body=body[:500],
                        retryable=res.status in _WAIT_HTTP or res.status >= 500,
                    )
                return _prerecorded_to_event(lang, json.loads(body))
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except APIStatusError:
            raise
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"failed to reach Saidwire ({type(e).__name__})") from None

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        """Open a streaming recognition session."""
        opts = dataclasses.replace(self._opts)
        if is_given(language):
            opts.language = language
        s = SpeechStream(
            stt=self,
            opts=opts,
            conn_options=conn_options,
            api_key=self._api_key,
            http_session=self._ensure_session(),
        )
        self._streams.add(s)
        return s

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        endpointing_ms: NotGivenOr[int | None] = NOT_GIVEN,
        utterance_end_ms: NotGivenOr[int | None] = NOT_GIVEN,
        interim_results: NotGivenOr[bool] = NOT_GIVEN,
        punctuate: NotGivenOr[bool] = NOT_GIVEN,
        early_words: NotGivenOr[bool] = NOT_GIVEN,
        vad_events: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        """Change options. Open streams reconnect with the new settings."""
        changes = {
            k: v
            for k, v in {
                "model": model,
                "endpointing_ms": endpointing_ms,
                "utterance_end_ms": utterance_end_ms,
                "interim_results": interim_results,
                "punctuate": punctuate,
                "early_words": early_words,
                "vad_events": vad_events,
            }.items()
            if is_given(v)
        }
        for k, v in changes.items():
            setattr(self._opts, k, v)
        for s in self._streams:
            s.update_options(**changes)


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._saidwire = stt
        self._opts = opts
        self._api_key = api_key
        self._session = http_session
        self._speaking = False
        self._request_id = ""
        self._reconnect_event = asyncio.Event()
        self._pending_usage = 0.0
        self._last_usage_report = time.monotonic()

    @property
    def request_id(self) -> str:
        """The ``request_id`` of the current connection (quote it when contacting support)."""
        return self._request_id

    def update_options(self, **changes: Any) -> None:
        """Apply option changes and reconnect with them."""
        for k, v in changes.items():
            setattr(self._opts, k, v)
        self._reconnect_event.set()

    def _url(self) -> str:
        o = self._opts
        q: dict[str, Any] = {
            "model": o.model,
            "encoding": "linear16",
            "sample_rate": o.sample_rate,
            "channels": 1,
            "language": o.language,
            "interim_results": o.interim_results,
            "punctuate": o.punctuate,
        }
        if o.endpointing_ms is not None:
            q["endpointing"] = "false" if o.endpointing_ms == 0 else int(o.endpointing_ms)
        if o.utterance_end_ms:
            q["utterance_end_ms"] = int(o.utterance_end_ms)
        if o.vad_events:
            q["vad_events"] = True
        if o.early_words:
            q["early_words"] = True
        return _with_query(o.base_url, q)

    def _add_usage(self, seconds: float, force: bool = False) -> None:
        self._pending_usage += seconds
        now = time.monotonic()
        if self._pending_usage > 0 and (force or now - self._last_usage_report >= 5.0):
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.RECOGNITION_USAGE,
                    request_id=self._request_id,
                    recognition_usage=stt.RecognitionUsage(audio_duration=self._pending_usage),
                )
            )
            self._pending_usage = 0.0
            self._last_usage_report = now

    def _flush_usage(self) -> None:
        """Report audio already sent, when a connection ends before the send task finished."""
        try:
            self._add_usage(0.0, force=True)
        except utils.aio.ChanClosed:
            pass

    async def _run(self) -> None:
        closing = False

        async def keepalive_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            # The server closes a stream after 10 s with no audio and no KeepAlive.
            try:
                while True:
                    await asyncio.sleep(4)
                    await ws.send_str(_KEEPALIVE)
            except (aiohttp.ClientError, ConnectionError):
                return

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing
            samples_50ms = self._opts.sample_rate // 20
            bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_50ms,
            )
            try:
                async for data in self._input_ch:
                    frames: list[rtc.AudioFrame] = []
                    flush = False
                    if isinstance(data, rtc.AudioFrame):
                        frames.extend(bstream.write(data.data.tobytes()))
                    elif isinstance(data, self._FlushSentinel):
                        frames.extend(bstream.flush())
                        flush = True
                    for f in frames:
                        await ws.send_bytes(f.data.tobytes())
                        self._add_usage(f.duration)
                    if flush:
                        await ws.send_str(_FINALIZE)
                closing = True
                self._add_usage(0.0, force=True)
                await ws.send_str(_CLOSE_STREAM)
            except (aiohttp.ClientError, ConnectionError) as e:
                if closing or self._session.closed:
                    return
                raise APIConnectionError("Saidwire connection closed unexpectedly") from e

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            monitor = utils.aio.WebSocketMonitor(
                ws, liveness_timeout=self._conn_options.liveness_timeout
            )
            while True:
                msg = await monitor.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing or self._session.closed:
                        return
                    raise APIStatusError(
                        message="Saidwire connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        request_id=self._request_id,
                        retryable=True,
                    )
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except ValueError:
                    logger.warning("non-JSON message from Saidwire ignored")
                    continue
                self._process_event(data)

        while True:
            ws = await self._connect_ready()
            if ws is None:  # input ended while the service was starting up
                return
            # A new connection restarts the server's audio clock at zero, so anchor the offset
            # on its first frame, and close out a turn the old connection left open.
            self._input_ch.restart_provider_clock()
            self._end_speech()
            tasks = [
                asyncio.create_task(send_task(ws)),
                asyncio.create_task(recv_task(ws)),
                asyncio.create_task(keepalive_task(ws)),
            ]
            group = asyncio.gather(*tasks[:2])
            wait_reconnect = asyncio.create_task(self._reconnect_event.wait())
            try:
                done, _ = await asyncio.wait(
                    (group, wait_reconnect), return_when=asyncio.FIRST_COMPLETED
                )
                for t in done:
                    if t is not wait_reconnect:
                        t.result()
                if wait_reconnect not in done:
                    break
                self._reconnect_event.clear()
            finally:
                await utils.aio.gracefully_cancel(*tasks, wait_reconnect)
                self._flush_usage()
                group.cancel()
                try:
                    group.exception()
                except (asyncio.CancelledError, asyncio.InvalidStateError):
                    pass
                await ws.close()

    async def _connect_ready(self) -> aiohttp.ClientWebSocketResponse | None:
        """Connect, waiting while the service is starting up or busy."""
        deadline = time.monotonic() + self._opts.warmup_timeout
        while True:
            try:
                return await self._connect_ws()
            except _ServiceWarming as w:
                if time.monotonic() + w.retry_after > deadline:
                    raise APIStatusError(
                        message=f"Saidwire was not ready after {self._opts.warmup_timeout:.0f} s",
                        status_code=w.status,
                        request_id=w.request_id,
                        body=None,
                        retryable=False,
                    ) from None
                logger.info(
                    "Saidwire is starting up or busy, retrying",
                    extra={"retry_after": w.retry_after, "status": w.status},
                )
                if not await self._discard_input(w.retry_after):
                    return None

    async def _discard_input(self, seconds: float) -> bool:
        """Drop audio for ``seconds`` so it isn't sent late in a burst.

        Returns False if the input ended meanwhile.
        """
        end = time.monotonic() + seconds
        while (left := end - time.monotonic()) > 0:
            try:
                await asyncio.wait_for(self._input_ch.recv(), left)
            except asyncio.TimeoutError:
                break
            except utils.aio.ChanClosed:
                return False
        return True

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        t0 = time.perf_counter()
        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    self._url(),
                    headers={"Authorization": f"Token {self._api_key}", "User-Agent": _USER_AGENT},
                    # WebSocketMonitor handles pings, so a half-open socket is noticed.
                    autoping=False,
                ),
                self._conn_options.timeout,
            )
        except asyncio.TimeoutError:
            raise APIConnectionError("timed out connecting to Saidwire") from None
        except aiohttp.WSServerHandshakeError as e:
            # The error repr carries the request headers (the API key): never chain it.
            hdrs = e.headers if e.headers is not None else {}
            if e.status in _WAIT_HTTP:
                raise _ServiceWarming(
                    e.status, _retry_after(hdrs), hdrs.get("x-request-id")
                ) from None
            raise APIStatusError(
                message=(
                    f"Saidwire refused the stream (HTTP {e.status}: "
                    f"{_HTTP_HINTS.get(e.status, 'error')})"
                ),
                status_code=e.status,
                request_id=hdrs.get("x-request-id"),
                body=None,
            ) from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(message=e.message, status_code=e.status, body=None) from None
        except Exception as e:
            raise APIConnectionError(
                f"failed to connect to Saidwire ({type(e).__name__})"
            ) from None
        resp = ws._response
        self._request_id = resp.headers.get("x-request-id", "") if resp is not None else ""
        self._report_connection_acquired(time.perf_counter() - t0, False)
        logger.debug("Saidwire stream connected", extra={"request_id": self._request_id})
        return ws

    def _start_speech(self) -> None:
        if not self._speaking:
            self._speaking = True
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))

    def _end_speech(self) -> None:
        if self._speaking:
            self._speaking = False
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

    def _process_event(self, data: dict[str, Any]) -> None:
        typ = data.get("type")
        if typ == "Results":
            self._on_results(data)
            self._mark_provider_progress()
        elif typ == "EarlyWord":
            self._saidwire.emit("early_word", data)
        elif typ == "SpeechStarted":
            self._start_speech()
        elif typ == "UtteranceEnd":
            self._end_speech()
        elif typ == "Error":
            code = data.get("err_code", "")
            if code == "BAD_MESSAGE":
                logger.warning("Saidwire: %s", data.get("err_msg"))
                return
            status = _PERMANENT_ERRORS.get(code)
            # Other errors are followed by a server close, which the receive loop retries.
            if status is not None:
                raise APIStatusError(
                    message=f"Saidwire {code}: {data.get('err_msg', '')}",
                    status_code=status,
                    request_id=data.get("request_id"),
                    retryable=False,
                )
            logger.warning("Saidwire stream error %s: %s", code, data.get("err_msg"))

    def _on_results(self, data: dict[str, Any]) -> None:
        self._request_id = (data.get("metadata") or {}).get("request_id", self._request_id)
        is_final = bool(data.get("is_final"))
        alts = _to_speech_data(self._opts.language, data, self.start_time_offset)
        text = alts[0].text if alts else ""
        if text:
            self._start_speech()
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT
                    if is_final
                    else stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=alts,
                )
            )
        elif is_final and data.get("from_finalize"):
            # The answer to a flush with nothing new: still emit an empty final, so the
            # session stops waiting for one.
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=[
                        stt.SpeechData(language=LanguageCode(self._opts.language), text="")
                    ],
                )
            )
        # A flushed final ends the turn even without speech_final, so the next turn
        # gets its own START_OF_SPEECH.
        if data.get("speech_final") or (is_final and data.get("from_finalize")):
            self._end_speech()


def _timed(word: dict[str, Any], offset: float) -> TimedString:
    return TimedString(
        text=word.get("punctuated_word") or word.get("word", ""),
        start_time=word.get("start", 0) + offset,
        end_time=word.get("end", 0) + offset,
        confidence=word.get("confidence", NOT_GIVEN),
        start_time_offset=offset,
    )


def _to_speech_data(language: str, data: dict[str, Any], offset: float) -> list[stt.SpeechData]:
    out = []
    for alt in (data.get("channel") or {}).get("alternatives") or []:
        words = alt.get("words") or []
        start = words[0].get("start", 0) if words else data.get("start", 0)
        end = words[-1].get("end", 0) if words else data.get("start", 0) + data.get("duration", 0)
        out.append(
            stt.SpeechData(
                language=LanguageCode(language),
                text=(alt.get("transcript") or "").strip(),
                start_time=start + offset,
                end_time=end + offset,
                confidence=alt.get("confidence", 0.0),
                words=[_timed(w, offset) for w in words] or None,
            )
        )
    return out


def _prerecorded_to_event(language: str, data: dict[str, Any]) -> stt.SpeechEvent:
    alts = []
    for alt in data["results"]["channels"][0]["alternatives"]:
        words = alt.get("words") or []
        alts.append(
            stt.SpeechData(
                language=LanguageCode(language),
                text=alt.get("transcript", ""),
                start_time=words[0]["start"] if words else 0.0,
                end_time=words[-1]["end"] if words else 0.0,
                confidence=alt.get("confidence", 0.0),
                words=[_timed(w, 0.0) for w in words] or None,
            )
        )
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        request_id=data.get("metadata", {}).get("request_id", ""),
        alternatives=alts,
    )


def _parse_error(body: str) -> tuple[str | None, str | None]:
    try:
        d = json.loads(body)
        return d.get("err_code"), d.get("err_msg")
    except Exception:
        return None, body[:200]
