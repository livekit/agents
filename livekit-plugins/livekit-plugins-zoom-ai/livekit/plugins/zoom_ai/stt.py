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

"""Zoom Scribe speech-to-text for LiveKit Agents.

* ``zoom_ai.STT()`` — **Scribe Live**: streaming recognition over WebSocket. Scribe's own
  VAD ends each turn and returns its final transcript; use
  ``turn_handling={"turn_detection": "stt"}``.
* ``zoom_ai.STT(mode="fast")`` — **Scribe Fast**: one HTTP request per utterance. Not
  streaming, so ``AgentSession`` wraps it with its VAD automatically.

Built on the public Scribe docs:
https://developers.zoom.us/docs/ai-services/scribe/live-mode/ and
https://developers.zoom.us/docs/ai-services/scribe/fast-mode/
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
import re
import time
import weakref
from dataclasses import dataclass
from typing import Any, Literal

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
from .version import __version__

BASE_URL = "https://api.zoom.us/v2/aiservices/scribe"
# Identifies this integration to Zoom (name and version only, no user data).
USER_AGENT = f"ZoomAI/1.0 (integration=LiveKit; plugin=livekit-plugins-zoom-ai/{__version__})"
SAMPLE_RATE = 16000  # Scribe Live accepts 16 kHz mono PCM16 only
KEEPALIVE_INTERVAL_S = 10.0  # Scribe closes a session after 30 s without audio
# Silence that ends a Scribe Live turn, tuned for voice agents.
DEFAULT_SILENCE_DURATION_MS = 350

SUPPORTED_LANGUAGES_URL = "https://developers.zoom.us/docs/ai-services/scribe/#supported-languages"
# Scribe's supported locales, as in its docs (SUPPORTED_LANGUAGES_URL), and the locale
# each base code maps to.
_LOCALES = (
    "en-US", "zh-CN", "ja-JP", "es-ES", "it-IT", "fr-FR",
    "de-DE", "ar-SA", "ar-AE", "pt-BR", "pt-PT",
)  # fmt: skip
_BASE_TO_LOCALE = {
    "en": "en-US", "zh": "zh-CN", "ja": "ja-JP", "es": "es-ES", "it": "it-IT",
    "fr": "fr-FR", "de": "de-DE", "ar": "ar-SA", "pt": "pt-BR",
}  # fmt: skip
_TAG = re.compile(r"\[(?:speaker[ _]?\d+|spk_\d+|channel\d+)\]\s*", re.IGNORECASE)


def _clean(text: str | None) -> str:
    """Remove the [Speaker N] / [channelN] tags Scribe adds when diarization is on."""
    return _TAG.sub("", text or "").strip()


def _is_config_error(code: object) -> bool:
    """A fatal Scribe error the same request would hit again, e.g. UNSUPPORTED_LANGUAGE."""
    code = str(code or "").upper()
    return "INVALID" in code or "UNSUPPORTED" in code


def _to_locale(language: str) -> str:
    """'en' -> 'en-US', 'en_us' -> 'en-US'. Raises ValueError for an unsupported language."""
    lang = language.replace("_", "-")
    if "-" in lang:
        base, region = lang.split("-", 1)
        locale = f"{base.lower()}-{region.upper()}"
    else:
        locale = _BASE_TO_LOCALE.get(lang.lower(), lang)
    if locale not in _LOCALES:
        raise ValueError(
            f"Language {language!r} isn't supported by this version of the Zoom AI plugin. "
            f"See Scribe's supported languages: {SUPPORTED_LANGUAGES_URL}"
        )
    return locale


@dataclass
class STTOptions:
    language: str
    diarization: bool
    silence_duration_ms: int | None
    word_time_offsets: bool


class STT(stt.STT):
    """Zoom Scribe speech-to-text: streaming (Scribe Live) or per-utterance (Scribe Fast).

    Scribe Live detects the end of each turn itself, so use STT turn detection::

        session = AgentSession(
            stt=zoom_ai.STT(),
            turn_handling={"turn_detection": "stt"},
            ...
        )

    Without it, LiveKit's VAD and turn detector add their own delay on top of Scribe's,
    and a turn can be committed before a later final arrives.

    In Live mode, ``recognize()`` (batch recognition) uses Scribe Fast.
    """

    def __init__(
        self,
        *,
        mode: Literal["live", "fast"] = "live",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        language: str = "en-US",
        silence_duration_ms: int | None = DEFAULT_SILENCE_DURATION_MS,
        diarization: bool = False,
        word_time_offsets: bool = False,
        base_url: str = BASE_URL,
        live_url: NotGivenOr[str] = NOT_GIVEN,
        fast_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a Zoom Scribe STT.

        Args:
            mode: ``"live"`` (streaming, default) or ``"fast"`` (one request per utterance).
            api_key: Zoom AI Services API key or JWT (Platform Studio → Credentials). Defaults to
                ``$ZOOM_SCRIBE_API_KEY``. It is sent on every request and reconnect, so it
                must stay valid for as long as the agent runs.
            language: A Scribe locale, e.g. ``"en-US"``; base codes such as ``"en"`` map to
                their locale. Raises ``ValueError`` for a language this version of the plugin
                doesn't support: https://developers.zoom.us/docs/ai-services/scribe/#supported-languages
            silence_duration_ms: Live only: milliseconds of silence that end a turn in
                Scribe's voice activity detection. Defaults to 350, tuned for voice agents;
                lower is faster, higher keeps sentences with longer pauses together. For
                transcription without an agent, a longer value such as 600 can improve
                accuracy. ``None`` uses Scribe's default (see the Scribe docs).
            diarization: Label each final transcript with a speaker. Off by default.
                With one speaker in the utterance, ``SpeechData.speaker_id`` is Scribe's
                label. With several, ``speaker_id`` is empty and
                ``metadata["speaker_segments"]`` has each one's span. In Fast mode labels
                are per request, so they aren't consistent across utterances.
                ``MultiSpeakerAdapter`` works with Live mode only.
            word_time_offsets: Fast only: fill ``SpeechData.words`` with word timings.
                Off by default.
            base_url: Scribe API base URL (``http`` or ``https``); Live uses
                ``<base>/live`` (as ``ws``/``wss``) and Fast uses ``<base>/transcribe``.
            live_url: Override the Scribe Live WebSocket URL.
            fast_url: Override the Scribe Fast HTTP URL.
            http_session: Optional aiohttp session; the job's session is used otherwise.
        """
        if mode not in ("live", "fast"):
            raise ValueError(f'mode must be "live" or "fast", got {mode!r}')
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=mode == "live",
                interim_results=False,  # not in the GA Scribe Live API
                diarization=diarization,
                offline_recognize=True,
            )
        )
        api_key = api_key if is_given(api_key) else os.environ.get("ZOOM_SCRIBE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Zoom Scribe API key is required: pass api_key or set ZOOM_SCRIBE_API_KEY"
            )
        self._api_key = api_key
        self._mode = mode
        base = base_url.rstrip("/")
        scheme, sep, rest = base.partition("://")
        if not sep or scheme.lower() not in ("http", "https"):
            raise ValueError(f"base_url must be an http(s) URL, got {base_url!r}")
        ws_scheme = "wss" if scheme.lower() == "https" else "ws"
        self._live_url = live_url if is_given(live_url) else f"{ws_scheme}://{rest}/live"
        self._fast_url = fast_url if is_given(fast_url) else base + "/transcribe"
        self._session = http_session
        self._opts = STTOptions(
            language=_to_locale(language),
            diarization=diarization,
            silence_duration_ms=silence_duration_ms,
            word_time_offsets=word_time_offsets,
        )
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        """The Scribe mode in use: ``"scribe-live"`` or ``"scribe-fast"``."""
        return f"scribe-{self._mode}"

    @property
    def provider(self) -> str:
        """The provider name, ``"Zoom"``."""
        return "Zoom"

    @property
    def _http(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "User-Agent": USER_AGENT}

    # ---------------------------------------------------------------- Scribe Fast
    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        locale = _to_locale(language) if is_given(language) else self._opts.language
        config: dict[str, Any] = {"language": locale}
        if self._opts.diarization:
            config["diarization"] = True
        if self._opts.word_time_offsets:
            config["word_time_offsets"] = True
        # Upload the utterance as multipart form data (`file` + `config`).
        form = aiohttp.FormData()
        form.add_field(
            "file",
            rtc.combine_audio_frames(buffer).to_wav_bytes(),
            filename="utterance.wav",
            content_type="audio/wav",
        )
        form.add_field("config", json.dumps(config))
        try:
            async with self._http.post(
                self._fast_url,
                data=form,
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=conn_options.timeout),
            ) as resp:
                body = await resp.text()
                request_id = resp.headers.get("x-zm-trackingid")
                if resp.status != 200:
                    raise APIStatusError(
                        "Zoom Scribe Fast request failed",
                        status_code=resp.status,
                        request_id=request_id,
                        body=body,
                    )
                try:
                    result = json.loads(body)
                except json.JSONDecodeError:
                    raise APIStatusError(
                        "Zoom Scribe Fast returned a non-JSON response",
                        status_code=resp.status,
                        request_id=request_id,
                    ) from None
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientError as e:
            raise APIConnectionError() from e

        text = _clean((result.get("result") or {}).get("text_display"))
        segments = (result.get("result") or {}).get("segments") or []
        speaker_id, metadata = None, None
        if self._opts.diarization:
            # One speaker: speaker_id. Several: speaker_id is empty and each one's span is
            # in metadata["speaker_segments"] (as Scribe Live does).
            speakers = {sg["speaker"] for sg in segments if sg.get("speaker")}
            if len(speakers) == 1:
                speaker_id = next(iter(speakers))
            elif len(speakers) > 1:
                metadata = {
                    "speaker_segments": [
                        {k: s.get(k) for k in ("speaker", "text", "start", "end")} for s in segments
                    ]
                }
        words = None
        if self._opts.word_time_offsets:
            words = [
                TimedString(text=w.get("word", ""), start_time=w["start"], end_time=w["end"])
                for s in segments
                for w in s.get("words") or []
                if "start" in w and "end" in w
            ] or None
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=result.get("request_id", ""),
            alternatives=[
                stt.SpeechData(
                    language=LanguageCode(locale),
                    text=text,
                    start_time=segments[0].get("start", 0.0) if segments else 0.0,
                    end_time=segments[-1].get("end", 0.0) if segments else 0.0,
                    speaker_id=speaker_id,
                    words=words,
                    metadata=metadata,
                )
            ],
        )

    # ---------------------------------------------------------------- Scribe Live
    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        """Open a Scribe Live streaming session.

        Args:
            language: Override the STT's language for this stream.
            conn_options: Connection timeout and retry options.

        Raises:
            NotImplementedError: When the STT was created with ``mode="fast"``.
        """
        if self._mode != "live":
            raise NotImplementedError(
                'mode="fast" is not streaming; AgentSession wraps it with a VAD'
            )
        opts = dataclasses.replace(self._opts)
        if is_given(language):
            opts.language = _to_locale(language)
        stream = SpeechStream(stt=self, opts=opts, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        silence_duration_ms: NotGivenOr[int | None] = NOT_GIVEN,
    ) -> None:
        """Change options; open Live streams reconnect with the new session config.

        Args:
            language: Scribe locale, e.g. ``"en-US"``.
            silence_duration_ms: Silence (ms) that ends a turn; ``None`` uses Scribe's
                default.

        An utterance in progress when the stream reconnects gets no final transcript; its
        turn is closed with ``END_OF_SPEECH``.
        """
        if not is_given(language) and not is_given(silence_duration_ms):
            return
        if is_given(language):
            self._opts.language = _to_locale(language)
        if is_given(silence_duration_ms):
            self._opts.silence_duration_ms = silence_duration_ms
        for stream in self._streams:
            stream.update_options(language=language, silence_duration_ms=silence_duration_ms)


class SpeechStream(stt.SpeechStream):
    """One Scribe Live session; reconnects on option changes and server-side session ends."""

    def __init__(self, *, stt: STT, opts: STTOptions, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._zoom = stt
        self._opts = opts
        self._reconnect_event = asyncio.Event()
        self._audio_duration = 0.0
        self._speech_started = False
        # Time between the start of this _run attempt and the current Scribe session.
        # The base class adds whole attempts to start_time_offset when it retries.
        self._session_offset = 0.0

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        silence_duration_ms: NotGivenOr[int | None] = NOT_GIVEN,
    ) -> None:
        """Change this stream's options; it reconnects with the new session config.

        Args:
            language: Scribe locale, e.g. ``"en-US"``.
            silence_duration_ms: Silence (ms) that ends a turn; ``None`` uses Scribe's
                default.

        An utterance in progress when the stream reconnects gets no final transcript; its
        turn is closed with ``END_OF_SPEECH``.
        """
        if not is_given(language) and not is_given(silence_duration_ms):
            return
        if is_given(language):
            self._opts.language = _to_locale(language)
        if is_given(silence_duration_ms):
            self._opts.silence_duration_ms = silence_duration_ms
        self._reconnect_event.set()

    def _session_update(self) -> dict[str, Any]:
        """The session.update from the Live docs."""
        update: dict[str, Any] = {
            "type": "session.update",
            "language": self._opts.language,
            "audio": {"format": "pcm16"},
        }
        config: dict[str, Any] = {}
        if self._opts.silence_duration_ms is not None:
            config["turn_detection"] = {
                "type": "acoustic_vad",
                "silence_duration_ms": self._opts.silence_duration_ms,
            }
        if self._opts.diarization:
            config["diarization"] = True
        if config:
            update["config"] = config
        return update

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        started_at = time.perf_counter()
        try:
            ws = await asyncio.wait_for(
                self._zoom._http.ws_connect(
                    self._zoom._live_url,
                    protocols=("live-asr",),
                    headers=self._zoom._headers(),
                    heartbeat=30.0,  # notice a dead connection; audio keeps Scribe's session alive
                ),
                self._conn_options.timeout,
            )
        except aiohttp.WSServerHandshakeError as e:
            # `from None`: the handshake error's repr keeps the request headers (the API key).
            raise APIStatusError(
                f"Zoom Scribe Live handshake failed: {e.message}", status_code=e.status
            ) from None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise APIConnectionError("failed to connect to Zoom Scribe Live") from e
        try:
            await ws.send_str(json.dumps(self._session_update()))
        except (aiohttp.ClientError, ConnectionError) as e:
            await ws.close()
            raise APIConnectionError("failed to configure the Zoom Scribe Live session") from e
        self._report_connection_acquired(time.perf_counter() - started_at, False)
        return ws

    def _end_open_utterance(self) -> None:
        """Close an utterance cut off by a reconnect so turn detection doesn't wait forever."""
        if self._speech_started:
            self._speech_started = False
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

    async def _run(self) -> None:
        self._session_offset = 0.0
        closing = False
        closed = asyncio.Event()  # Scribe confirmed our session.close
        last_audio_at = time.monotonic()
        session_audio_s = 0.0  # audio sent on the current Scribe session, keepalive included

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing, last_audio_at
            # ~50 ms chunks (Scribe recommends ~100 ms; smaller keeps end-of-speech latency low)
            bstream = utils.audio.AudioByteStream(
                sample_rate=SAMPLE_RATE, num_channels=1, samples_per_channel=SAMPLE_RATE // 20
            )
            nonlocal session_audio_s
            anchored = False
            try:
                async for data in self._input_ch:
                    if isinstance(data, self._FlushSentinel):
                        frames = bstream.flush()
                    else:
                        frames = bstream.write(data.data.tobytes())
                    for frame in frames:
                        self._audio_duration += frame.duration
                        await ws.send_bytes(frame.data.tobytes())
                        last_audio_at = time.monotonic()
                        # Scribe's audio_*_ms count from the session's first audio. Audio queued
                        # while connecting goes out in a burst, so anchor that 0 ms at the
                        # earliest "now minus audio sent", which is when that audio was captured.
                        session_audio_s += frame.duration
                        captured_at = time.time() - session_audio_s
                        if not anchored or captured_at < self.start_time:
                            self.start_time = captured_at
                            anchored = True
                    # On flush we only send the buffered audio: Scribe's own VAD ends the turn.
                closing = True
                await ws.send_str(json.dumps({"type": "session.close"}))
            except (aiohttp.ClientError, ConnectionError) as e:
                if closing or self._zoom._http.closed:
                    return
                raise APIConnectionError("Zoom Scribe Live connection closed unexpectedly") from e
            # Wait for the remaining final transcript and session.closed, but not forever.
            try:
                await asyncio.wait_for(closed.wait(), self._conn_options.timeout)
            except asyncio.TimeoutError:
                logger.warning("Zoom Scribe Live did not confirm session.close; closing")
                await ws.close()

        async def keepalive_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            # Scribe Live ends a session after 30 s without audio (e.g. the user muted), and it
            # has no keepalive message, so send a little silence when the input has gone quiet.
            nonlocal last_audio_at, session_audio_s
            silence = bytes(SAMPLE_RATE * 2 // 10)  # 100 ms
            try:
                while not closing:
                    await asyncio.sleep(KEEPALIVE_INTERVAL_S)
                    if not closing and time.monotonic() - last_audio_at >= KEEPALIVE_INTERVAL_S:
                        await ws.send_bytes(silence)
                        self._audio_duration += len(silence) / (2 * SAMPLE_RATE)
                        session_audio_s += len(silence) / (2 * SAMPLE_RATE)
                        last_audio_at = time.monotonic()
            except (aiohttp.ClientError, ConnectionError) as e:
                if closing or self._zoom._http.closed:
                    return
                raise APIConnectionError("Zoom Scribe Live connection closed unexpectedly") from e

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing:
                        closed.set()
                        return
                    raise APIStatusError(
                        "Zoom Scribe Live connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                try:
                    ev = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.warning("ignoring non-JSON message from Zoom Scribe")
                    continue
                if ev.get("type") == "session.closed" and not closing:
                    # Server-side end (session time limit, idle timeout):
                    # raise a retryable error so the base class reconnects with a new session.
                    raise APIStatusError(
                        f"Zoom Scribe Live {ev['type']} ({ev.get('reason')})",
                        status_code=-1,
                        body=ev,
                        retryable=True,
                    )
                try:
                    if self._process_event(ev):
                        closed.set()
                        return  # session.closed after our session.close
                except APIStatusError:
                    raise
                except Exception:
                    logger.exception("failed to process Zoom Scribe message")

        while True:
            self._end_open_utterance()
            connected_at = time.time()
            session_audio_s = 0.0
            self._reconnect_event.clear()  # this connect reads the current options
            ws = await self._connect_ws()
            tasks = [
                asyncio.create_task(send_task(ws)),
                asyncio.create_task(recv_task(ws)),
                asyncio.create_task(keepalive_task(ws)),
            ]
            io_done: asyncio.Future[list[None]] = asyncio.gather(*tasks[:2])
            wait_reconnect = asyncio.create_task(self._reconnect_event.wait())
            reconnect = False
            try:
                waiters: set[asyncio.Future[Any]] = {io_done, wait_reconnect, tasks[2]}
                while True:
                    done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        if task is wait_reconnect:
                            continue
                        if isinstance(task.exception(), APIConnectionError):
                            # A failed send usually means Scribe closed the socket; prefer the
                            # error event it sent first (e.g. an unsupported language).
                            await asyncio.wait({tasks[1]}, timeout=1.0)
                            if tasks[1].done() and isinstance(tasks[1].exception(), APIStatusError):
                                tasks[1].result()
                        task.result()  # surface errors from I/O and keepalive
                    if io_done in done:
                        break  # input ended and the session closed cleanly
                    if wait_reconnect in done:
                        reconnect = True  # options changed: reconnect with the new config
                        break
                    waiters.discard(tasks[2])  # keepalive stopped while closing; keep waiting
            finally:
                await utils.aio.gracefully_cancel(*tasks, wait_reconnect)
                io_done.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    io_done.exception()  # retrieve it so asyncio doesn't log it at shutdown
                await ws.close()
            if not reconnect:
                self._end_open_utterance()  # Scribe closed without a final for the last turn
                return
            # Scribe's audio_*_ms restart at 0 on the new session; keep stream-relative
            # timestamps moving forward, as the base class does when it retries.
            self._session_offset += time.time() - connected_at

    def _speech_end_time(self, end_ms: int) -> float | None:
        """Wall-clock end of speech for a final ending at ``end_ms``.

        Scribe ends a segment after ``silence_duration_ms`` of silence, so ``audio_end_ms``
        is that much later than the end of speech. With ``silence_duration_ms=None`` the
        window is Scribe's, so ``audio_end_ms`` is used as is.
        """
        if not end_ms:
            return None
        silence_ms = self._opts.silence_duration_ms or 0
        return self.start_time + max(0, end_ms - silence_ms) / 1000

    def _process_event(self, ev: dict[str, Any]) -> bool:
        kind = ev.get("type")
        if kind == "input_audio_buffer.speech_started":
            if not self._speech_started:
                self._speech_started = True
                start_ms = ev.get("audio_start_ms")
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.START_OF_SPEECH,
                        speech_start_time=self.start_time + start_ms / 1000
                        if start_ms is not None
                        else None,
                    )
                )
        elif kind == "transcription.completed":
            text = _clean(ev.get("transcript"))
            start_ms, end_ms = ev.get("audio_start_ms") or 0, ev.get("audio_end_ms") or 0
            speaker_id, metadata = None, None
            if self._opts.diarization:
                # One speaker: speaker_id. Several: Scribe sends `speaker_segments`, and
                # speaker_id is left empty (one field can't name several speakers).
                spans = ev.get("speaker_segments") or []
                if spans:
                    metadata = {"speaker_segments": spans}
                else:
                    speaker_id = ev.get("speaker") or None
            if text and not self._speech_started:
                # No speech_started for this transcript (e.g. it was lost in a reconnect):
                # open the turn so turn_detection="stt" still commits it.
                self._speech_started = True
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.START_OF_SPEECH,
                        speech_start_time=self.start_time + start_ms / 1000
                        if "audio_start_ms" in ev
                        else None,
                    )
                )
            if text:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        request_id=ev.get("item_id", ""),
                        alternatives=[
                            stt.SpeechData(
                                language=LanguageCode(self._opts.language),
                                text=text,
                                start_time=start_ms / 1000
                                + self.start_time_offset
                                + self._session_offset,
                                end_time=end_ms / 1000
                                + self.start_time_offset
                                + self._session_offset,
                                speaker_id=speaker_id,
                                metadata=metadata,
                            )
                        ],
                        speech_end_time=self._speech_end_time(end_ms),
                    )
                )
            # Scribe's final closes the utterance: end of speech comes after the transcript,
            # so turn_detection="stt" commits the turn with the text already in hand.
            if self._speech_started:
                self._speech_started = False
                # speech_end_time lets turn_detection="stt" count its endpointing delay from
                # when the user stopped speaking, not from when the final arrived.
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.END_OF_SPEECH,
                        speech_end_time=self._speech_end_time(end_ms),
                    )
                )
            self._report_usage()
        elif kind == "session.updated":
            # Scribe never rejects diarization; it reports whether labels will come.
            if self._opts.diarization and ev.get("diarization_enabled") is False:
                logger.warning("Zoom Scribe Live can't label speakers for this session")
        elif kind == "error":
            err = ev.get("error") or {}
            if err.get("fatal"):
                raise APIStatusError(
                    f"Zoom Scribe Live error: {err.get('message')}",
                    status_code=-1,
                    body=err,
                    retryable=not _is_config_error(err.get("code")),
                )
            logger.warning("Zoom Scribe Live: %s (%s)", err.get("message"), err.get("code"))
        elif kind == "session.closed":
            self._report_usage()  # audio sent after the last final transcript
            return True
        return False

    def _report_usage(self) -> None:
        if self._audio_duration > 0:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.RECOGNITION_USAGE,
                    recognition_usage=stt.RecognitionUsage(audio_duration=self._audio_duration),
                )
            )
            self._audio_duration = 0.0
