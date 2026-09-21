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

"""Smallest AI (Pulse) speech-to-text for LiveKit Agents.

Wire contract of the Pulse streaming API, as this plugin relies on it:

* Audio goes up as raw PCM binary frames; control messages are JSON text frames:
  ``{"type": "finalize"}`` (return everything buffered as a final now, keep the
  socket) and ``{"type": "close_stream"}`` (flush, answer with ``is_last`` and close).
* Transcript messages carry ``transcript`` (the gateway also duplicates it as
  ``transcription``), ``is_final``, ``is_last``, ``from_finalize`` on the answer to a
  finalize, optional ``speech_final``, optional ``joins_previous`` when a final opens
  with a piece that continues the previous final's last word, and ``words`` on finals
  when word timestamps or diarization are on.
* With ``vad_events`` the server sends standalone ``speech_started`` /
  ``speech_ended`` messages from its acoustic VAD. They are the end-of-turn signal:
  a final on its own is not, because the server also cuts finals inside an utterance
  (word-count rollover, blank-token endpoint at a chunk boundary).
* Errors arrive as ``{"error": "..."}``.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
import weakref
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Generic, TypeVar
from urllib.parse import urlencode

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
from livekit.agents.types import NOT_GIVEN, NotGivenOr, TimedString
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger
from .models import STTEncoding, STTModels
from .version import __version__

NUM_CHANNELS = 1
# Base URL for the Smallest AI API.
# Streaming: wss://api.smallest.ai/waves/v1/stt/live?model={model}
# Batch:     https://api.smallest.ai/waves/v1/stt/?model={model}
SMALLEST_STT_BASE_URL = "https://api.smallest.ai/waves/v1"
# Streaming path on the public API. A directly reachable inference server uses "/transcribe".
DEFAULT_STREAM_PATH = "/stt/live"

# Models that support real-time streaming. All others are batch-only and will be
# wrapped with a StreamAdapter by the agent framework automatically.
_STREAMING_MODELS: frozenset[str] = frozenset({"pulse"})

# Any letter or digit in any script; a transcript with none is punctuation only.
_WORD_RE = re.compile(r"\w")

T = TypeVar("T")


class _PeriodicCollector(Generic[T]):
    """Same logic as livekit-plugins-deepgram/_utils.py: sums values, flushes every `duration`."""

    def __init__(self, callback: Callable[[T], None], *, duration: float) -> None:
        self._duration = duration
        self._callback = callback
        self._last_flush_time = time.monotonic()
        self._total: T | None = None

    def push(self, value: T) -> None:
        if self._total is None:
            self._total = value
        else:
            self._total += value  # type: ignore[operator]
        if time.monotonic() - self._last_flush_time >= self._duration:
            self.flush()

    def flush(self) -> None:
        if self._total is not None:
            self._callback(self._total)
            self._total = None
        self._last_flush_time = time.monotonic()


@dataclass
class _STTOptions:
    model: STTModels | str
    api_key: str
    language: str  # BCP-47 code, e.g. "en", "hi"; use "multi" for auto-detection
    sample_rate: int
    encoding: STTEncoding | str
    word_timestamps: bool
    diarize: bool
    format: bool  # punctuation/capitalization; streaming only
    itn_normalize: bool  # spoken -> written numbers ("twenty five" -> "25"); streaming only
    numerals: bool | None  # legacy digits post-pass; None = server default
    keywords: list[tuple[str, float]]  # (keyword, intensifier) pairs; streaming only
    sentence_timestamps: bool  # include sentence-level "utterances"; streaming only
    redact_pii: bool
    redact_pci: bool
    endpointing: bool  # server VAD finalizes the open segment on trailing silence
    endpointing_timeout_ms: int | None  # the VAD silence window; None = server default (600)
    eou_timeout_ms: int | None  # blank-token endpoint; None = server default (800)
    vad_events: bool  # speech_started / speech_ended messages -> START/END_OF_SPEECH
    finalize_on_flush: bool  # {"type":"finalize"} when the framework flushes the stream
    finalize_on_words: bool | None  # word-count rollover of long segments; None = server default
    max_words: int | None  # rollover length; None = server default
    keepalive_interval: float | None  # seconds between {"type":"keepalive"} messages; None = off
    base_url: str
    stream_path: str


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: STTModels | str = "pulse",
        language: str = "en",
        sample_rate: int = 16000,
        encoding: STTEncoding | str = "linear16",
        word_timestamps: bool = True,
        diarize: bool = False,
        format: bool = True,
        itn_normalize: bool = False,
        numerals: NotGivenOr[bool] = NOT_GIVEN,
        keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN,
        sentence_timestamps: bool = False,
        redact_pii: bool = False,
        redact_pci: bool = False,
        endpointing: bool = True,
        endpointing_timeout_ms: int | None = 300,
        eou_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_events: bool = True,
        finalize_on_flush: bool = True,
        finalize_on_words: NotGivenOr[bool] = NOT_GIVEN,
        max_words: NotGivenOr[int] = NOT_GIVEN,
        keepalive_interval: NotGivenOr[float | None] = NOT_GIVEN,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        base_url: str = SMALLEST_STT_BASE_URL,
        stream_path: str = DEFAULT_STREAM_PATH,
    ) -> None:
        """Create a new instance of Smallest AI STT.

        Args:
            model: STT model to use. ``"pulse"`` supports streaming and batch
                transcription. Other models are batch-only; ``stream()`` raises
                ``ValueError`` for them.
            language: BCP-47 language code (e.g. "en", "hi", "fr"). Use "multi" for
                automatic language detection.
            sample_rate: Audio sample rate in Hz. Supported: 8000, 16000, 22050,
                24000, 44100, 48000. Defaults to 16000. Telephony audio should be sent
                at its native 8000 Hz rather than upsampled.
            encoding: PCM encoding of the audio stream; "linear16" is the default.
            word_timestamps: Include per-word start/end timestamps and confidence
                scores on final transcripts. Defaults to True.
            diarize: Enable speaker diarization; each word carries a speaker id.
                Defaults to False.
            format: Apply punctuation and capitalization to streaming transcripts.
                Defaults to True.
            itn_normalize: Run the server's inverse text normalization so spoken
                numbers come back as digits ("twenty five" -> "25"). Independent of
                ``format``. Defaults to False.
            numerals: Legacy digits post-pass; leave unset and use ``itn_normalize``.
            keywords: Boost recognition via ``(keyword, intensifier)`` tuples, e.g.
                ``[("NVIDIA", 2.0)]``. Intensifier ~1.0 (mild) to ~5.0 (strong).
            sentence_timestamps: Include sentence-level timing as an ``"utterances"``
                list in ``SpeechData.metadata``. Defaults to False.
            redact_pii: Mask names, addresses and phone numbers with placeholder
                tokens; matches are listed in ``SpeechData.metadata["redacted_entities"]``.
                Reliable for ``language="en"`` and ``"hi"``. Defaults to False.
            redact_pci: Mask card numbers, CVVs, ZIP codes and account numbers.
                Reliable for ``language="en"`` and ``"hi"``. Defaults to False.
            endpointing: Let the server's VAD finalize the open segment once
                ``endpointing_timeout_ms`` of trailing silence has passed, so the
                last words of a turn arrive as a final without waiting for the
                blank-token endpoint at the next decode boundary. Defaults to True.
            endpointing_timeout_ms: The server VAD silence window in ms, shared by
                ``endpointing`` and ``vad_events``. Defaults to 300 so the server's final
                lands before a VAD-driven agent commits the turn (LiveKit commits about
                0.55 s after speech with its default VAD): measured against 600 on 19 call
                recordings this cut the words leaking into the next turn by 60 % at the
                same accuracy. Shorter windows start cutting inside words. ``None`` uses
                the server default (600).
            eou_timeout_ms: Silence (ms) for the transducer's blank-token endpoint.
                Unset uses the server default (800). Values below the ~1.1 s decode
                chunk make the server cut finals inside utterances; prefer
                ``endpointing`` for low latency.
            vad_events: Ask the server for ``speech_started`` / ``speech_ended``
                messages and map them to START_OF_SPEECH / END_OF_SPEECH. This is the
                only reliable end-of-speech signal: finals alone are also emitted
                mid-utterance. With it off, every final closes the turn. Defaults to True.
            finalize_on_flush: Send ``{"type": "finalize"}`` when the framework flushes
                the stream, so buffered speech is returned as a final immediately
                (Deepgram's ``Finalize``). Defaults to True.
            finalize_on_words: Cut a final every ``max_words`` words during long
                continuous speech. Unset uses the server default (on). The cut is not a
                turn end; END_OF_SPEECH still follows the VAD.
            max_words: Word count for ``finalize_on_words``. Unset uses the server default.
            keepalive_interval: Seconds between ``{"type": "keepalive"}`` messages, which
                reset the public API's inactivity timer during long pauses (it answers with
                ``{"type": "pong"}``). Defaults to 5 s on the public API and off for a
                direct server, which does not accept the message. ``None`` disables it.
            api_key: Smallest AI API key. Falls back to the SMALLEST_API_KEY
                environment variable if not provided.
            http_session: An existing aiohttp ClientSession to reuse.
            base_url: API base URL. For a directly reachable inference server pass its
                ``http://host:port`` and ``stream_path="/transcribe"``.
            stream_path: Path of the streaming endpoint under ``base_url``.
                ``"/stt/live"`` on the public API, ``"/transcribe"`` on a server.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=model in _STREAMING_MODELS,
                interim_results=True,
                diarization=diarize,
                aligned_transcript="word" if word_timestamps else False,
            )
        )

        api_key = api_key or os.environ.get("SMALLEST_API_KEY")
        if not api_key:
            raise ValueError(
                "Smallest AI API key is required, either as argument or set "
                "SMALLEST_API_KEY environment variable"
            )

        self._opts = _STTOptions(
            model=model,
            api_key=api_key,
            language=language,
            sample_rate=sample_rate,
            encoding=encoding,
            word_timestamps=word_timestamps,
            diarize=diarize,
            format=format,
            itn_normalize=itn_normalize,
            numerals=numerals if is_given(numerals) else None,
            keywords=list(keywords) if is_given(keywords) else [],
            sentence_timestamps=sentence_timestamps,
            redact_pii=redact_pii,
            redact_pci=redact_pci,
            endpointing=endpointing,
            endpointing_timeout_ms=endpointing_timeout_ms
            if is_given(endpointing_timeout_ms)
            else None,
            eou_timeout_ms=eou_timeout_ms if is_given(eou_timeout_ms) else None,
            vad_events=vad_events,
            finalize_on_flush=finalize_on_flush,
            finalize_on_words=finalize_on_words if is_given(finalize_on_words) else None,
            max_words=max_words if is_given(max_words) else None,
            keepalive_interval=(
                keepalive_interval
                if is_given(keepalive_interval)
                else (5.0 if stream_path == DEFAULT_STREAM_PATH else None)
            ),
            base_url=base_url,
            stream_path=stream_path,
        )
        self._session = http_session
        self._streams: weakref.WeakSet[SpeechStream] = weakref.WeakSet()

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

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        config = self._sanitize_options(language=language)
        params: dict[str, Any] = {
            "model": config.model,
            "language": config.language,
            "encoding": config.encoding,
            "sample_rate": config.sample_rate,
            "word_timestamps": str(config.word_timestamps).lower(),
            "diarize": str(config.diarize).lower(),
        }

        try:
            async with self._ensure_session().post(
                url=f"{config.base_url}/stt/",
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/octet-stream",
                    "X-Source": "livekit",
                    "X-LiveKit-Version": __version__,
                },
                params=params,
                # to_wav_bytes() produces a valid WAV file; the server auto-detects format.
                data=rtc.combine_audio_frames(buffer).to_wav_bytes(),
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=conn_options.timeout),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return _batch_transcription_to_speech_event(config.language, data)

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from e
        except Exception as e:
            raise APIConnectionError() from e

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        if not self.capabilities.streaming:
            raise ValueError(
                f"{self._opts.model} does not support streaming; use recognize() for batch transcription"
            )
        config = self._sanitize_options(language=language)
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            opts=config,
            http_session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        encoding: NotGivenOr[STTEncoding | str] = NOT_GIVEN,
        format: NotGivenOr[bool] = NOT_GIVEN,
        itn_normalize: NotGivenOr[bool] = NOT_GIVEN,
        keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN,
        sentence_timestamps: NotGivenOr[bool] = NOT_GIVEN,
        redact_pii: NotGivenOr[bool] = NOT_GIVEN,
        redact_pci: NotGivenOr[bool] = NOT_GIVEN,
        endpointing: NotGivenOr[bool] = NOT_GIVEN,
        endpointing_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        eou_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_events: NotGivenOr[bool] = NOT_GIVEN,
        finalize_on_flush: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        """Update STT options; propagates to all active streams (triggers reconnect)."""
        if is_given(model):
            self._opts.model = model
            self._capabilities.streaming = model in _STREAMING_MODELS
        self._opts = _apply_updates(
            self._opts,
            language=language,
            sample_rate=sample_rate,
            encoding=encoding,
            format=format,
            itn_normalize=itn_normalize,
            keywords=keywords,
            sentence_timestamps=sentence_timestamps,
            redact_pii=redact_pii,
            redact_pci=redact_pci,
            endpointing=endpointing,
            endpointing_timeout_ms=endpointing_timeout_ms,
            eou_timeout_ms=eou_timeout_ms,
            vad_events=vad_events,
            finalize_on_flush=finalize_on_flush,
        )
        for stream in self._streams:
            stream.update_options(
                model=model,
                language=language,
                sample_rate=sample_rate,
                encoding=encoding,
                format=format,
                itn_normalize=itn_normalize,
                keywords=keywords,
                sentence_timestamps=sentence_timestamps,
                redact_pii=redact_pii,
                redact_pci=redact_pci,
                endpointing=endpointing,
                endpointing_timeout_ms=endpointing_timeout_ms,
                eou_timeout_ms=eou_timeout_ms,
                vad_events=vad_events,
                finalize_on_flush=finalize_on_flush,
            )

    def _sanitize_options(self, *, language: NotGivenOr[str] = NOT_GIVEN) -> _STTOptions:
        config = replace(self._opts)
        if is_given(language):
            config.language = language
        return config


def _apply_updates(opts: _STTOptions, **updates: Any) -> _STTOptions:
    """Copy `opts` with every given (non-NOT_GIVEN) keyword applied; lists are copied."""
    updated = replace(opts)
    for key, value in updates.items():
        if is_given(value):
            setattr(updated, key, list(value) if isinstance(value, list) else value)
    return updated


class SpeechStream(stt.SpeechStream):
    _FINALIZE_MSG: str = json.dumps({"type": "finalize"})
    _CLOSE_STREAM_MSG: str = json.dumps({"type": "close_stream"})
    _KEEPALIVE_MSG: str = json.dumps({"type": "keepalive"})

    def __init__(
        self,
        *,
        stt: STT,
        opts: _STTOptions,
        conn_options: APIConnectOptions,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._session = http_session
        self._speaking = False
        self._session_id = ""
        self._reconnect_event = asyncio.Event()
        self._audio_duration_collector = _PeriodicCollector(
            callback=self._on_audio_duration_report, duration=5.0
        )

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        encoding: NotGivenOr[STTEncoding | str] = NOT_GIVEN,
        format: NotGivenOr[bool] = NOT_GIVEN,
        itn_normalize: NotGivenOr[bool] = NOT_GIVEN,
        keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN,
        sentence_timestamps: NotGivenOr[bool] = NOT_GIVEN,
        redact_pii: NotGivenOr[bool] = NOT_GIVEN,
        redact_pci: NotGivenOr[bool] = NOT_GIVEN,
        endpointing: NotGivenOr[bool] = NOT_GIVEN,
        endpointing_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        eou_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_events: NotGivenOr[bool] = NOT_GIVEN,
        finalize_on_flush: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        self._opts = _apply_updates(
            self._opts,
            model=model,
            language=language,
            sample_rate=sample_rate,
            encoding=encoding,
            format=format,
            itn_normalize=itn_normalize,
            keywords=keywords,
            sentence_timestamps=sentence_timestamps,
            redact_pii=redact_pii,
            redact_pci=redact_pci,
            endpointing=endpointing,
            endpointing_timeout_ms=endpointing_timeout_ms,
            eou_timeout_ms=eou_timeout_ms,
            vad_events=vad_events,
            finalize_on_flush=finalize_on_flush,
        )
        self._reconnect_event.set()

    async def _run(self) -> None:
        closing_ws = False

        async def keepalive_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            # Resets the public API's inactivity timer while no audio flows; the server
            # answers with a pong that recv_task ignores.
            interval = self._opts.keepalive_interval
            if interval is None or interval <= 0:
                await asyncio.Event().wait()  # disabled: idle for the life of the connection
                return
            try:
                while True:
                    await asyncio.sleep(interval)
                    await ws.send_str(SpeechStream._KEEPALIVE_MSG)
            except (aiohttp.ClientError, ConnectionError) as e:
                # With no audio flowing this write is where a dropped socket shows first.
                if closing_ws or self._session.closed:
                    return
                raise APIConnectionError("Smallest AI STT connection closed unexpectedly") from e

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            # 50 ms packets, the cadence the Pulse docs recommend.
            samples_per_chunk = self._opts.sample_rate // 20
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=NUM_CHANNELS,
                samples_per_channel=samples_per_chunk,
            )

            try:
                async for data in self._input_ch:
                    frames: list[rtc.AudioFrame] = []
                    flushed = False
                    if isinstance(data, rtc.AudioFrame):
                        frames.extend(audio_bstream.write(data.data.tobytes()))
                    elif isinstance(data, self._FlushSentinel):
                        frames.extend(audio_bstream.flush())
                        flushed = True

                    for frame in frames:
                        self._audio_duration_collector.push(frame.duration)
                        await ws.send_bytes(frame.data.tobytes())

                    if flushed:
                        # The framework marked a segment end: the buffered tail is on the
                        # wire, now ask for it back as a final instead of leaving it to
                        # the server's endpointing (Deepgram's Finalize contract).
                        self._audio_duration_collector.flush()
                        if self._opts.finalize_on_flush:
                            await ws.send_str(SpeechStream._FINALIZE_MSG)

                # Input closed: the server flushes, answers with is_last and closes.
                closing_ws = True
                await ws.send_str(SpeechStream._CLOSE_STREAM_MSG)
            except (aiohttp.ClientError, ConnectionError) as e:
                # A dropped socket surfaces on the write side first. Expected closes just
                # return; anything else reconnects through _run, like recv_task.
                if closing_ws or self._session.closed:
                    return
                raise APIConnectionError("Smallest AI STT connection closed unexpectedly") from e

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws or self._session.closed:
                        return
                    raise APIStatusError(
                        message="Smallest AI STT connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                if msg.type == aiohttp.WSMsgType.ERROR:
                    if closing_ws or self._session.closed:
                        return
                    # An unanswered heartbeat ping closes the socket from our side and
                    # arrives here, not as a close frame; ws.exception() holds the reason.
                    raise APIConnectionError("Smallest AI STT connection lost") from ws.exception()

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Smallest AI STT message type: %s", msg.type)
                    continue

                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.warning(
                        "failed to parse Smallest AI STT message", extra={"lk.pii.data": msg.data}
                    )
                    continue

                # Errors: {"type": "error", "status", "message"} from the public API,
                # {"error": "..."} from a server reached directly.
                if data.get("type") == "error" or (
                    "error" in data and "transcript" not in data and "transcription" not in data
                ):
                    raise APIStatusError(
                        message=f"Smallest AI STT error: {data.get('message') or data.get('error')}",
                        status_code=-1,
                        request_id=self._session_id or None,
                        body=data,
                    )
                if data.get("type") == "pong":
                    continue

                self._process_stream_event(data)

                # The session is fully flushed; nothing else will arrive.
                if data.get("is_last"):
                    return

        ws: aiohttp.ClientWebSocketResponse | None = None
        while True:
            try:
                ws = await self._connect_ws()
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                ]
                # Not in the gather: recv_task ends on is_last before the socket closes, and the
                # session must not wait on a keepalive loop. Its failure still ends the session.
                keepalive = asyncio.create_task(keepalive_task(ws))
                tasks_group = asyncio.gather(*tasks)
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())
                try:
                    done, _ = await asyncio.wait(
                        (tasks_group, wait_reconnect_task, keepalive),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task, keepalive)
                    tasks_group.cancel()
                    tasks_group.exception()
            finally:
                if ws is not None:
                    await ws.close()
                self._audio_duration_collector.flush()

    def _stream_params(self) -> dict[str, Any]:
        """Query parameters for the streaming connection; unset knobs are left to the server."""
        o = self._opts
        params: dict[str, Any] = {
            "model": o.model,
            "language": o.language,
            "encoding": o.encoding,
            "sample_rate": o.sample_rate,
            "word_timestamps": str(o.word_timestamps).lower(),
            "diarize": str(o.diarize).lower(),
            "format": str(o.format).lower(),
            "itn_normalize": str(o.itn_normalize).lower(),
            "sentence_timestamps": str(o.sentence_timestamps).lower(),
            "redact_pii": str(o.redact_pii).lower(),
            "redact_pci": str(o.redact_pci).lower(),
            "endpointing": str(o.endpointing).lower(),
            "vad_events": str(o.vad_events).lower(),
        }
        if o.numerals is not None:
            params["numerals"] = str(o.numerals).lower()
        if o.endpointing_timeout_ms is not None:
            params["endpointing_timeout"] = o.endpointing_timeout_ms
        if o.eou_timeout_ms is not None:
            params["eou_timeout_ms"] = o.eou_timeout_ms
        if o.finalize_on_words is not None:
            params["finalize_on_words"] = str(o.finalize_on_words).lower()
        if o.max_words is not None:
            params["max_words"] = o.max_words
        if o.keywords:
            params["keywords"] = ",".join(f"{kw}:{boost:g}" for kw, boost in o.keywords)
        return params

    def _ws_url(self) -> str:
        base = self._opts.base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        return f"{base.rstrip('/')}{self._opts.stream_path}?{urlencode(self._stream_params())}"

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        t0 = time.perf_counter()
        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    self._ws_url(),
                    headers={
                        "Authorization": f"Bearer {self._opts.api_key}",
                        "X-Source": "livekit",
                        "X-LiveKit-Version": __version__,
                    },
                    # Without a heartbeat a half-open socket is never noticed: recv_task
                    # parks on receive() and the exception-driven reconnect never runs.
                    heartbeat=30.0,
                ),
                self._conn_options.timeout,
            )
        except asyncio.TimeoutError:
            raise APIConnectionError("failed to connect to Smallest AI STT") from None
        except aiohttp.ClientResponseError as e:
            # RequestInfo carries the request headers, so chaining this error or
            # formatting it puts the API key in the exception repr.
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError(
                f"failed to connect to Smallest AI STT ({type(e).__name__})"
            ) from None

        self._report_connection_acquired(time.perf_counter() - t0, False)
        self._speaking = False
        logger.debug("established Smallest AI STT WebSocket connection")
        return ws

    def _on_audio_duration_report(self, duration: float) -> None:
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=self._session_id,
                alternatives=[],
                recognition_usage=stt.RecognitionUsage(audio_duration=duration),
            )
        )

    def _start_speech(self) -> None:
        if not self._speaking:
            self._speaking = True
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))

    def _end_speech(self) -> None:
        if self._speaking:
            self._speaking = False
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

    def _process_stream_event(self, data: dict[str, Any]) -> None:
        session_id = data.get("session_id", "")
        if session_id:
            self._session_id = session_id

        msg_type = data.get("type", "")
        # Acoustic VAD messages: the speaker started or stopped, independent of decoding.
        if msg_type == "speech_started":
            self._start_speech()
            return
        if msg_type == "speech_ended":
            self._end_speech()
            return

        # The gateway duplicates the server's `transcript` into `transcription`.
        transcript = data.get("transcript") or data.get("transcription") or ""
        is_final = bool(data.get("is_final", False))
        is_last = bool(data.get("is_last", False))
        from_finalize = bool(data.get("from_finalize", False))

        # Punctuation-only text ("." or "।" landing in its own chunk) is not a user turn.
        if _WORD_RE.search(transcript):
            # Quiet speech the server VAD missed, or vad_events off: the text opens the turn.
            self._start_speech()
            alts = _transcript_to_speech_data(
                language=self._opts.language,
                data=data,
                transcript=transcript,
                start_time_offset=self.start_time_offset,
                diarize=self._opts.diarize,
            )
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=(
                        stt.SpeechEventType.FINAL_TRANSCRIPT
                        if is_final
                        else stt.SpeechEventType.INTERIM_TRANSCRIPT
                    ),
                    request_id=self._session_id,
                    alternatives=alts,
                )
            )

        # A final closes the turn when it answers a finalize, ends the stream, or the
        # server says the speaker paused. Other finals are cuts inside an utterance
        # (word rollover, blank-token endpoint); with vad_events the turn end comes
        # from speech_ended, without them every final has to count as one.
        if is_final and (
            is_last
            or from_finalize
            or data.get("speech_final") is True
            or not self._opts.vad_events
        ):
            self._end_speech()


def _aggregate_confidence(raw_words: list[dict[str, Any]]) -> float:
    """Geometric mean of the per-word confidences (each already a geometric mean over its pieces)."""
    probs = [
        float(w["confidence"]) for w in raw_words if isinstance(w.get("confidence"), (int, float))
    ]
    if not probs:
        return 0.0
    return min(1.0, max(0.0, math.exp(sum(math.log(max(p, 1e-10)) for p in probs) / len(probs))))


def _transcript_to_speech_data(
    language: str,
    data: dict[str, Any],
    *,
    transcript: str,
    start_time_offset: float,
    diarize: bool,
) -> list[stt.SpeechData]:
    raw_words: list[dict[str, Any]] = data.get("words") or []

    words: list[TimedString] | None = (
        [
            TimedString(
                text=w.get("word", ""),
                start_time=w.get("start", 0.0) + start_time_offset,
                end_time=w.get("end", 0.0) + start_time_offset,
                start_time_offset=start_time_offset,
            )
            for w in raw_words
        ]
        if raw_words
        else None
    )

    start_time = raw_words[0].get("start", 0.0) + start_time_offset if raw_words else 0.0
    end_time = raw_words[-1].get("end", 0.0) + start_time_offset if raw_words else 0.0

    # Streaming diarization: per-word speaker ids are integers; the utterance takes the majority.
    speaker_id: str | None = None
    if diarize and raw_words:
        speaker_counts = Counter(w["speaker"] for w in raw_words if "speaker" in w)
        if speaker_counts:
            speaker_id = f"S{speaker_counts.most_common(1)[0][0]}"

    # With language="multi" the server reports the detected language on finals.
    detected_language = data.get("language") or language

    metadata: dict[str, Any] = {}
    raw_utterances: list[dict[str, Any]] = data.get("utterances") or []
    if raw_utterances:
        metadata["utterances"] = [
            {
                **u,
                "start": u.get("start", 0.0) + start_time_offset,
                "end": u.get("end", 0.0) + start_time_offset,
            }
            for u in raw_utterances
        ]
    redacted_entities: list[str] = data.get("redacted_entities") or []
    if redacted_entities:
        metadata["redacted_entities"] = redacted_entities
    if data.get("joins_previous"):
        # This final's first word continues the previous final's last word (a finalize
        # cut the word); a consumer that glues finals should not put a space between them.
        metadata["joins_previous"] = True

    return [
        stt.SpeechData(
            language=LanguageCode(detected_language),
            text=transcript.strip(),
            start_time=start_time,
            end_time=end_time,
            confidence=_aggregate_confidence(raw_words),
            words=words,
            speaker_id=speaker_id,
            metadata=metadata or None,
        )
    ]


def _batch_transcription_to_speech_event(language: str, data: dict[str, Any]) -> stt.SpeechEvent:
    # Batch HTTP response: {"status", "transcription", "audio_length", "words": [...],
    # "utterances": [...] (when word_timestamps=True), "language", "metadata"}.
    transcript = data.get("transcription", "")
    raw_words: list[dict[str, Any]] = data.get("words") or []
    raw_utterances: list[dict[str, Any]] = data.get("utterances") or []
    detected_language = data.get("language") or language

    words: list[TimedString] | None = (
        [
            TimedString(
                text=w.get("word", ""), start_time=w.get("start", 0.0), end_time=w.get("end", 0.0)
            )
            for w in raw_words
        ]
        if raw_words
        else None
    )

    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        request_id=utils.shortuuid(),
        alternatives=[
            stt.SpeechData(
                language=LanguageCode(detected_language),
                text=transcript,
                start_time=raw_words[0].get("start", 0.0) if raw_words else 0.0,
                end_time=raw_words[-1].get("end", 0.0) if raw_words else 0.0,
                confidence=_aggregate_confidence(raw_words),
                words=words,
                metadata={"utterances": raw_utterances} if raw_utterances else None,
            )
        ],
    )
