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

"""Nabrah STT over /api/ext/stt/ws with recognition_model=nabrah_stt_conv_v1.

The model runs speech recognition and turn detection together and decides end of turn per 80 ms frame (smart EOT).
This plugin only maps the WebSocket messages to LiveKit speech events:

    partial with words in the current turn  -> START_OF_SPEECH (once), INTERIM_TRANSCRIPT
    partial with p_end >= preflight          -> PREFLIGHT_TRANSCRIPT (once per text; for preemptive generation)
    final (end of turn)                      -> FINAL_TRANSCRIPT, END_OF_SPEECH, RECOGNITION_USAGE
    {"type": "backchannel"}                  -> "backchannel" event on the STT (the app decides what to do)

Use with `turn_detection="stt"` and no VAD: the server owns the turn.

Timing: each transcript carries `audio_processed` (seconds of our audio it covers). The plugin keeps
the wall time it sent each frame, so START/END_OF_SPEECH times are when the words' audio was
sent, not when the text arrived. LiveKit's transcription_delay / end_of_turn_delay are then
the server's real delay (decode + the end-of-turn wait + network). The decoder emits a word
~1-2 frames after it is spoken, so speech times run up to ~160 ms late.

Usage: audio sent is reported as RECOGNITION_USAGE (STT metrics / usage audio_duration) at each
end of turn and when the stream ends.
"""

from __future__ import annotations

import asyncio
import bisect
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer

from .log import logger

DEFAULT_URL = "wss://api.nabrah.ai/api/ext/stt/ws"  # env NABRAH_STT_URL overrides
MODEL = "nabrah_stt_conv_v1"  # backend MODEL_NAME (wire + billing id)
RATE = 16000  # 16 kHz mono PCM16
# The backend collects model results when audio arrives. Keep packets shorter than the
# model's 80 ms frames so collecting an answer doesn't wait for another whole model frame.
CHUNK = 320
# Server errors that a retry can't fix (anything else, e.g. "Server busy", is retried).
FATAL_ERRORS = {
    "Missing API key": 401,
    "Invalid API key": 401,
    "Invalid config": 400,
    "Insufficient credits": 402,
}
# The only errors before "ready" that a retry can fix; any other is a bad config (e.g. a
# setting that is not a number) and is not retried.
RETRYABLE_HANDSHAKE_ERRORS = {
    "STT is busy. Please try again in a moment.",
    "ASR server unavailable",
    "Timeout waiting for config",
    "Server error in STT streaming",  # backend's generic failure, e.g. credits lookup
}
# Ping every HEARTBEAT s; no pong within HEARTBEAT / 2 s closes the socket as dead (retried).
HEARTBEAT = 10.0


def _server_error(message: str, handshake: bool = False) -> APIStatusError | APIConnectionError:
    if message in FATAL_ERRORS:
        return APIStatusError(
            f"Nabrah STT: {message}", status_code=FATAL_ERRORS[message], retryable=False
        )
    if handshake and message not in RETRYABLE_HANDSHAKE_ERRORS:
        return APIStatusError(f"Nabrah STT: {message}", status_code=400, retryable=False)
    return APIConnectionError(f"Nabrah STT: {message}")


@dataclass
class STTOptions:
    base_url: str
    api_key: str
    preflight_threshold: float | None
    stream_settings: dict[str, Any]  # sent in the config message; None = server default


class STTConv(stt.STT):
    """Nabrah streaming STT with server-side end of turn.

    Turn and backchannel settings keep the server's names; left as None the server's default
    is used (eot_threshold 0.915, eot_debounce 2, min_speech_ms 350, bc_threshold 0.4,
    bc_debounce 2, bc_refractory_ms 1200). The server rejects wrong types and non-finite
    values (not retried) but imposes no range limits.
    Emits custom events: "backchannel" {"text": turn so far} and "end_of_turn" {"text"}.

    preflight_threshold: send PREFLIGHT_TRANSCRIPT (the turn so far) once the server's end-of-turn
    probability reaches this, before the turn really ends at eot_threshold. LiveKit uses it to
    start the reply early when preemptive generation is on. None turns it off.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        eot_threshold: float | None = None,
        eot_debounce: int | None = None,
        min_speech_ms: float | None = None,
        bc_threshold: float | None = None,
        bc_debounce: int | None = None,
        bc_refractory_ms: float | None = None,
        no_text_reset_s: float | None = None,
        preflight_threshold: float | None = 0.5,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True, interim_results=True, offline_recognize=False
            )
        )
        api_key = (
            api_key or os.environ.get("NABRAH_API_KEY") or os.environ.get("NABRAH_STT_V2_API_KEY")
        )
        if not api_key:
            raise ValueError("Nabrah API key required: pass api_key or set NABRAH_API_KEY")
        if preflight_threshold is not None and not 0 <= preflight_threshold <= 1:
            raise ValueError("preflight_threshold must be between 0 and 1, or None")
        self._opts = STTOptions(
            base_url=base_url
            or os.environ.get("NABRAH_STT_URL")
            or os.environ.get("NABRAH_STT_V2_URL")
            or DEFAULT_URL,
            api_key=api_key,
            preflight_threshold=preflight_threshold,
            stream_settings={
                "eot_threshold": eot_threshold,
                "eot_debounce": eot_debounce,
                "min_speech_ms": min_speech_ms,
                "bc_threshold": bc_threshold,
                "bc_debounce": bc_debounce,
                "bc_refractory_ms": bc_refractory_ms,
                "no_text_reset_s": no_text_reset_s,
            },
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return MODEL

    @property
    def provider(self) -> str:
        return "Nabrah"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _connect(
        self, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> aiohttp.ClientWebSocketResponse:
        """Open the backend WebSocket, select the conversational model, wait for ready."""
        opts = self._opts
        config = {"api_key": opts.api_key, "recognition_model": MODEL}
        config.update({k: v for k, v in opts.stream_settings.items() if v is not None})
        try:
            ws = await asyncio.wait_for(
                self._ensure_session().ws_connect(opts.base_url, heartbeat=HEARTBEAT),
                conn_options.timeout,
            )
        except (aiohttp.ClientError, OSError, asyncio.TimeoutError) as e:
            raise APIConnectionError(f"Nabrah STT unreachable at {opts.base_url}: {e}") from e
        try:
            await ws.send_str(json.dumps(config))
            msg = await asyncio.wait_for(ws.receive(), conn_options.timeout)
            reply = json.loads(msg.data) if msg.type == aiohttp.WSMsgType.TEXT else {}
            if not isinstance(reply, dict):
                raise APIConnectionError("Nabrah STT returned an invalid ready response")
        except (asyncio.TimeoutError, aiohttp.ClientError, ValueError) as e:
            await ws.close()
            raise APIConnectionError("Nabrah STT ready handshake failed") from e
        except BaseException:
            await ws.close()
            raise
        if reply.get("type") != "ready":
            await ws.close()
            if reply.get("message"):
                raise _server_error(reply["message"], handshake=True)
            raise _server_error(f"connection closed ({msg.type.name})")
        return ws

    async def check_ready(self) -> None:
        """Connect once and close: fails fast on a bad key, URL or unavailable model."""
        ws = await self._connect()
        await ws.close()

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Nabrah plugin is streaming only; use stream()")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStreamConv:
        return SpeechStreamConv(stt=self, opts=self._opts, conn_options=conn_options)


class SpeechStreamConv(stt.RecognizeStream):
    def __init__(self, *, stt: STTConv, opts: STTOptions, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=RATE)
        self._stt: STTConv = stt
        self._opts = opts
        self._language = LanguageCode("ar-SA")

    async def _run(self) -> None:
        ws = await self._stt._connect(self._conn_options)

        # State is per connection: a retry opens a new server stream that counts audio from 0.
        sent_s: list[float] = []  # audio seconds sent after each frame ...
        sent_wall: list[float] = []  # ... and the wall time it was sent
        reported = 0.0  # audio seconds already reported as usage
        speaking = False
        last_interim = ""
        last_preflight = ""
        # (wall clock, stream audio seconds) of the turn's first and latest words
        first_word = last_word = (0.0, 0.0)
        input_done = False

        def at(audio_s: float | None) -> tuple[float, float]:
            """When the audio up to audio_s was sent (now if the server didn't say)."""
            if audio_s is None or not sent_s:
                return time.time(), self.start_time_offset + (sent_s[-1] if sent_s else 0.0)
            # EOF can produce padded model frames beyond the actual audio sent.
            audio_s = min(audio_s, sent_s[-1])
            i = min(bisect.bisect_left(sent_s, audio_s - 1e-6), len(sent_s) - 1)
            return sent_wall[i], self.start_time_offset + audio_s

        def send(event: stt.SpeechEvent) -> None:
            self._event_ch.send_nowait(event)

        def report_usage() -> None:
            nonlocal reported
            total = sent_s[-1] if sent_s else 0.0
            if total > reported:
                send(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.RECOGNITION_USAGE,
                        recognition_usage=stt.RecognitionUsage(audio_duration=total - reported),
                    )
                )
                reported = total

        def start() -> None:
            nonlocal speaking
            if not speaking:
                speaking = True
                send(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.START_OF_SPEECH, speech_start_time=first_word[0]
                    )
                )

        def speech_data(text: str, final: bool) -> list[stt.SpeechData]:
            return [
                stt.SpeechData(
                    language=self._language,
                    text=text,
                    start_time=first_word[1],
                    end_time=last_word[1],
                    confidence=1.0 if final else 0.0,
                )
            ]

        def on_transcript(m: dict) -> None:
            nonlocal speaking, last_interim, last_preflight, first_word, last_word
            text = (m.get("text") or "").strip()
            if not text:
                return
            if text != last_interim or not speaking:  # includes corrections of the same length
                when = at(m.get("audio_processed"))
                if not speaking:
                    first_word = when
                last_word = when
            start()
            if m.get("is_final"):
                send(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=speech_data(text, True),
                    )
                )
                speaking, last_interim, last_preflight = False, "", ""
                # Later transcripts start at or after this audio: drop older send times,
                # keeping the last entry (total audio sent, read by at() and report_usage()).
                audio = m.get("audio_processed")
                if audio is not None and sent_s:
                    i = min(bisect.bisect_left(sent_s, audio - 1e-6), len(sent_s) - 1)
                    del sent_s[:i], sent_wall[:i]
                send(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.END_OF_SPEECH, speech_end_time=last_word[0]
                    )
                )
                report_usage()
                self._stt.emit("end_of_turn", {"text": text})
                return
            if text != last_interim:
                last_interim = text
                send(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        alternatives=speech_data(text, False),
                    )
                )
            # p_end keeps rising in the silence after the words: check it on every partial
            threshold = self._opts.preflight_threshold
            frames = (m.get("turn") or {}).get("frames") or []
            # A response may cover several frames. An earlier peak doesn't mean the
            # speaker is still likely to stop: use the latest frame's probability.
            p_end = frames[-1].get("p_end") if frames else None
            if (
                threshold is not None
                and p_end is not None
                and p_end >= threshold
                and text != last_preflight
            ):
                last_preflight = text
                send(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.PREFLIGHT_TRANSCRIPT,
                        alternatives=speech_data(text, True),
                    )
                )

        async def send_task() -> None:
            nonlocal input_done
            chunker = utils.audio.AudioByteStream(RATE, 1, CHUNK)

            async def push(frames: list) -> None:
                for f in frames:
                    await ws.send_bytes(f.data.tobytes())
                    sent_s.append((sent_s[-1] if sent_s else 0.0) + f.samples_per_channel / RATE)
                    sent_wall.append(time.time())

            async for frame in self._input_ch:
                if isinstance(frame, self._FlushSentinel):
                    continue  # one continuous server stream; the server decides turns
                await push(chunker.write(frame.data.tobytes()))
            await push(chunker.flush())
            await ws.send_str(json.dumps({"type": "eof"}))
            input_done = True

        async def recv_task() -> None:
            async for msg in ws:  # the server closes after eof
                if msg.type == aiohttp.WSMsgType.ERROR:  # e.g. heartbeat: no pong
                    raise APIConnectionError(f"Nabrah STT connection lost: {ws.exception()}")
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                try:
                    m = json.loads(msg.data)
                except ValueError:
                    m = None
                if not isinstance(m, dict):
                    # No payload in the log: it can contain customer transcripts.
                    logger.warning("Nabrah STT sent a malformed message; ignored")
                    continue
                if m.get("type") == "transcript":
                    on_transcript(m)
                elif m.get("type") == "backchannel":
                    self._stt.emit("backchannel", {"text": last_interim})
                elif m.get("type") == "error":
                    raise _server_error(m.get("message") or "Unknown server error")
            if ws.close_code not in (None, aiohttp.WSCloseCode.OK):
                raise APIConnectionError(f"Nabrah STT closed the stream (code {ws.close_code})")
            if not input_done:
                raise APIConnectionError("Nabrah STT closed the stream before EOF")

        tasks = [asyncio.create_task(send_task()), asyncio.create_task(recv_task())]
        try:
            # recv ends when the server closes (after eof); send failing (closed socket) ends both
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for t in done:
                t.result()
            await tasks[1]
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Nabrah STT stream failed: {e}") from e
        finally:
            if speaking:  # stream ended mid-turn with no final: keep its words, close the turn
                send(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=speech_data(last_interim, True),
                    )
                )
                send(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.END_OF_SPEECH, speech_end_time=last_word[0]
                    )
                )
            report_usage()
            await utils.aio.cancel_and_wait(*tasks)
            await ws.close()
            logger.debug("Nabrah stream closed after %.1f s of audio", sent_s[-1] if sent_s else 0)
