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
import contextlib
import json
import weakref
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, TypedDict, cast
from urllib.parse import urlencode

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.stt import SpeechEventType, STTCapabilities
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given
from livekit.agents.voice.io import TimedString

from . import _utils
from .log import logger
from .models import STTLanguages, STTModels, STTModes, STTSampleRates, STTStyles

DEFAULT_MODEL: STTModels = "chariot-1-asr-realtime"
SUPPORTED_LANGUAGES = frozenset(STTLanguages.__args__)  # type: ignore[attr-defined]
SUPPORTED_SAMPLE_RATES = frozenset(STTSampleRates.__args__)  # type: ignore[attr-defined]
# Audio is sent in chunks of this length
CHUNK_MS = 100
# The server closes a session that has heard nothing for 60 seconds
KEEPALIVE_INTERVAL = 15.0
# Recognition usage is reported for every this many seconds of audio sent
USAGE_INTERVAL = 5.0
# `recognize()` uploads audio as one file, which the API accepts up to this length
MAX_RECOGNIZE_SECONDS = 180.0

CLOSE_TIMEOUT = 1.0

if TYPE_CHECKING:
    # aiohttp narrows this type on Python 3.11+ only; `bool` holds on every supported version
    _WebSocket = aiohttp.ClientWebSocketResponse[bool]
else:
    # aiohttp before 3.14 does not make the class subscriptable at runtime
    _WebSocket = aiohttp.ClientWebSocketResponse


class VADOptions(TypedDict, total=False):
    """Server side voice activity detection. Unset fields use the server's defaults."""

    vad_threshold: float
    """Speech probability threshold, 0.05 to 0.95."""
    silence_duration_ms: int
    """Silence that ends an utterance, 100 to 10000 ms."""
    min_speech_duration_ms: int
    """Minimum speech length, 0 to 5000 ms."""
    prefix_padding_ms: int
    """Audio kept from before detected speech, 0 to 2000 ms."""


@dataclass
class STTOptions:
    model: STTModels | str
    language: LanguageCode | None
    mode: STTModes
    style: STTStyles
    include_timestamps: bool
    server_vad: VADOptions | None
    sample_rate: int
    api_key: str
    base_url: str


def _language(language: str | None) -> LanguageCode | None:
    """Normalize a language to the code the API takes; None (or "none") detects it."""
    if not language or language == "none":
        return None
    code = LanguageCode(language).language
    if code not in SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValueError(f"Chariot STT does not support {language!r}; supported: {supported}")
    return LanguageCode(code)


class STT(stt.STT):
    def __init__(
        self,
        *,
        language: NotGivenOr[STTLanguages | str | None] = NOT_GIVEN,
        model: STTModels | str = DEFAULT_MODEL,
        mode: STTModes = "balanced",
        style: STTStyles = "codemix",
        include_timestamps: bool = False,
        server_vad: VADOptions | None = None,
        sample_rate: STTSampleRates = 16000,
        api_key: str | None = None,
        base_url: str = _utils.BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Chariot STT.

        Args:
            language (STTLanguages | str): Language of the speech: English or one of 11 Indian
                languages. A regional code such as "hi-IN" is accepted. Defaults to detecting it.
            model (STTModels | str): STT model. Defaults to "chariot-1-asr-realtime".
            mode (STTModes): "balanced" (the default) favors latency; "high_accuracy" is more
                accurate, and its interim transcripts update less often. `recognize()` always
                uses "high_accuracy".
            style (STTStyles): "codemix" (the default) writes English words in Latin script
                within Indian language speech; "native" writes everything in the language's
                own script.
            include_timestamps (bool): Include word timestamps. Defaults to False.
            server_vad (VADOptions | None): None (the default) finalizes a transcript when the
                agent's turn detection ends the user's turn. Pass {} or VADOptions to let the
                server's voice detection split utterances instead.
            sample_rate (STTSampleRates): Rate of the audio sent to the API. Defaults to 16000.
            api_key (str): Chariot API key. Defaults to the CHARIOT_API_KEY environment variable.
            base_url (str): API base URL. Defaults to "https://api.chariot.in".
            http_session (aiohttp.ClientSession): Optional aiohttp session to use for requests.
        """
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"Chariot STT does not accept sample_rate={sample_rate}")
        super().__init__(
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript="word" if include_timestamps else False,
                offline_recognize=True,
                manual_flush=server_vad is None,
            )
        )
        self._opts = STTOptions(
            model=model,
            language=_language(language) if is_given(language) else None,
            mode=mode,
            style=style,
            include_timestamps=include_timestamps,
            server_vad=server_vad,
            sample_rate=sample_rate,
            api_key=_utils.resolve_api_key(api_key),
            base_url=base_url.rstrip("/"),
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Chariot"

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
        frame = rtc.combine_audio_frames(buffer)
        if frame.duration > MAX_RECOGNIZE_SECONDS:
            raise APIStatusError(
                f"recognize() accepts up to {MAX_RECOGNIZE_SECONDS:.0f} seconds of audio, got "
                f"{frame.duration:.1f}; use stream(), or Chariot's batch transcription API",
                status_code=413,
            )
        request_language = _language(language) if is_given(language) else self._opts.language

        # the API resamples any rate and mixes channels down, so the frames go as they are
        form = aiohttp.FormData()
        form.add_field(
            "input_audio_file", frame.to_wav_bytes(), filename="audio.wav", content_type="audio/wav"
        )
        form.add_field("model_type", self._opts.model)
        form.add_field("mode", "high_accuracy")  # the only mode file transcription offers
        form.add_field("style", self._opts.style)
        form.add_field("language", request_language or "none")
        form.add_field("include_timestamps", str(self._opts.include_timestamps).lower())

        try:
            async with self._ensure_session().post(
                f"{self._opts.base_url}/v1/asr/transcribe",
                data=form,
                headers=_utils.headers(self._opts.api_key),
                # transcription takes longer for longer audio
                timeout=aiohttp.ClientTimeout(total=conn_options.timeout + frame.duration),
            ) as resp:
                if resp.status != 200:
                    raise await _utils.http_error(resp)
                body = await resp.json()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e

        raw_words = body.get("words") or []
        return stt.SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            request_id=body.get("request_id", ""),
            alternatives=[
                stt.SpeechData(
                    language=request_language or LanguageCode(""),
                    text=body.get("text", ""),
                    start_time=float(raw_words[0]["start"]) if raw_words else 0.0,
                    end_time=(
                        float(raw_words[-1]["end"])
                        if raw_words
                        else float(body.get("duration") or 0.0)
                    ),
                    words=(
                        _timed_words(raw_words, offset=0.0)
                        if self._opts.include_timestamps
                        else None
                    ),
                )
            ],
        )

    def update_options(
        self,
        *,
        language: NotGivenOr[STTLanguages | str | None] = NOT_GIVEN,
        mode: NotGivenOr[STTModes] = NOT_GIVEN,
        style: NotGivenOr[STTStyles] = NOT_GIVEN,
        server_vad: NotGivenOr[VADOptions | None] = NOT_GIVEN,
    ) -> None:
        """Update STT options. Omitted options keep their current values.

        The options are fixed for a session, so open streams reconnect, which can discard a
        transcript in progress. Change them between turns.
        """
        if is_given(language):
            self._opts.language = _language(language)
        if is_given(mode):
            self._opts.mode = mode
        if is_given(style):
            self._opts.style = style
        if is_given(server_vad):
            self._opts.server_vad = server_vad
            self._capabilities.manual_flush = server_vad is None
        for stream in self._streams:
            stream.update_options(language=language, mode=mode, style=style, server_vad=server_vad)

    def stream(
        self,
        *,
        language: NotGivenOr[STTLanguages | str | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        opts = replace(self._opts)
        if is_given(language):
            opts.language = _language(language)
        stream = SpeechStream(
            stt=self, opts=opts, conn_options=conn_options, http_session=self._ensure_session()
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()


class SpeechStream(stt.SpeechStream):
    """Streams audio over `WSS /v1/asr/ws` and receives interim and final transcripts."""

    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._session = http_session
        self._reconnect_event = asyncio.Event()
        self._speaking = False
        self._partial_text = ""
        self._partial_words: list[TimedString] = []
        self._unreported_audio = 0.0
        self._pending_commits = 0  # commits the server has not answered yet
        self._committed_until = 0.0  # session audio clock at the end of the last commit

    def update_options(
        self,
        *,
        language: NotGivenOr[STTLanguages | str | None] = NOT_GIVEN,
        mode: NotGivenOr[STTModes] = NOT_GIVEN,
        style: NotGivenOr[STTStyles] = NOT_GIVEN,
        server_vad: NotGivenOr[VADOptions | None] = NOT_GIVEN,
    ) -> None:
        """Update stream options, reconnecting the stream. See `STT.update_options`."""
        if is_given(language):
            self._opts.language = _language(language)
        if is_given(mode):
            self._opts.mode = mode
        if is_given(style):
            self._opts.style = style
        if is_given(server_vad):
            self._opts.server_vad = server_vad
        if is_given(language) or is_given(mode) or is_given(style) or is_given(server_vad):
            self._reconnect_event.set()

    async def _run(self) -> None:
        try:
            await self._run_sessions()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e

    async def _run_sessions(self) -> None:
        closing = False  # every input was sent; the server ends the session after its last result

        async def send_task(ws: _WebSocket) -> None:
            nonlocal closing
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=self._opts.sample_rate * CHUNK_MS // 1000,
            )
            uncommitted = False  # audio sent since the last commit
            async for data in self._input_ch:
                flush = isinstance(data, self._FlushSentinel)
                if isinstance(data, rtc.AudioFrame):
                    frames = audio_bstream.write(data.data.tobytes())
                else:
                    frames = audio_bstream.flush()
                for frame in frames:
                    await ws.send_bytes(frame.data.tobytes())
                    uncommitted = True
                    self._report_usage(frame.duration)

                # end_input() flushes as it closes the input; input.done below transcribes that
                # audio as the session's tail, so the flush needs no commit of its own
                closing_flush = flush and self._input_ch.closed and self._input_ch.empty()
                if flush and self._opts.server_vad is None and not closing_flush:
                    if uncommitted:
                        self._pending_commits += 1
                        await ws.send_str(json.dumps({"type": "input.commit"}))
                        uncommitted = False
                    elif not self._pending_commits:
                        # nothing new to transcribe, so the server would not answer a commit;
                        # with a commit still pending, its answer covers this flush too
                        self._emit_final("", None, 0.0, 0.0)

            closing = True
            self._report_usage(0.0, flush=True)
            # transcribe the audio not yet committed; the server then ends the session
            await ws.send_str(json.dumps({"type": "input.done"}))

        async def recv_task(ws: _WebSocket) -> None:
            monitor = utils.aio.WebSocketMonitor(
                cast(aiohttp.ClientWebSocketResponse, ws),
                liveness_timeout=self._conn_options.liveness_timeout,
            )
            while True:
                msg = await monitor.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.CLOSED,
                ):
                    if closing and ws.close_code == 1000:
                        return
                    raise _utils.close_error(ws.close_code, str(msg.extra or ""))
                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Chariot STT message type %s", msg.type)
                    continue
                if self._process_event(json.loads(msg.data)):
                    return  # the session's last result

        async def keepalive_task(ws: _WebSocket) -> None:
            while True:
                await asyncio.sleep(KEEPALIVE_INTERVAL)
                await ws.send_str(json.dumps({"type": "keepalive"}))

        while True:
            ws: _WebSocket | None = None
            try:
                ws = await self._connect_ws()
                self._input_ch.restart_provider_clock()
                self._start_session()
                tasks = [asyncio.create_task(send_task(ws)), asyncio.create_task(recv_task(ws))]
                keepalive = asyncio.create_task(keepalive_task(ws))
                wait_reconnect = asyncio.create_task(self._reconnect_event.wait())
                session = asyncio.gather(*tasks)
                try:
                    done, _ = await asyncio.wait(
                        (session, wait_reconnect), return_when=asyncio.FIRST_COMPLETED
                    )
                    if session in done:
                        session.result()
                        return
                    self._reconnect_event.clear()
                finally:
                    # stop this connection's tasks before the next one reads the input, so no
                    # audio or flush is taken by a send loop whose connection has failed
                    await utils.aio.gracefully_cancel(*tasks, keepalive, wait_reconnect)
                    with contextlib.suppress(BaseException):
                        await session
            finally:
                if ws is not None:
                    await _close(ws)

    def _start_session(self) -> None:
        """Reset per-session state for a new connection."""
        if self._pending_commits:
            # the old connection never answered a commit; an agent waiting on that flush gets
            # what was heard so far rather than waiting out its timeout
            self._emit_final(self._partial_text.strip(), None, 0.0, 0.0)
        self._partial_text = ""
        self._partial_words = []
        self._pending_commits = 0
        self._committed_until = 0.0

    async def _connect_ws(self) -> _WebSocket:
        params: dict[str, Any] = {
            "model_type": self._opts.model,
            "audio_format": f"pcm_{self._opts.sample_rate}",
            "mode": self._opts.mode,
            "style": self._opts.style,
            "commit_strategy": "manual" if self._opts.server_vad is None else "vad",
        }
        if self._opts.language:
            params["language"] = self._opts.language
        if self._opts.include_timestamps:
            params["include_timestamps"] = "true"
        if self._opts.server_vad is not None:
            params.update(self._opts.server_vad)
            params["vad_events"] = "true"
        url = f"{_utils.ws_base_url(self._opts.base_url)}/v1/asr/ws?{urlencode(params)}"

        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    url, headers=_utils.headers(self._opts.api_key), autoping=False
                ),
                self._conn_options.timeout,
            )
        except aiohttp.WSServerHandshakeError as e:
            raise APIStatusError(e.message, status_code=e.status) from None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise APIConnectionError("failed to connect to Chariot STT") from e

        try:
            # a refused session (an invalid key or option, no credits, all sessions busy)
            # answers with an error instead of session.started
            msg = await ws.receive(timeout=self._conn_options.timeout)
            if msg.type != aiohttp.WSMsgType.TEXT:
                raise _utils.close_error(ws.close_code, str(msg.extra or ""))
            event = json.loads(msg.data)
            if event.get("type") == "error":
                raise _utils.error_event(event)
            if event.get("type") != "session.started":
                raise APIError(
                    f"unexpected Chariot message before session.started: {event['type']}"
                )
        except asyncio.TimeoutError:
            await _close(ws)
            raise APITimeoutError("Chariot STT did not start the session in time") from None
        except BaseException:
            await _close(ws)
            raise
        return ws

    def _report_usage(self, duration: float, *, flush: bool = False) -> None:
        self._unreported_audio += duration
        if self._unreported_audio > 0 and (flush or self._unreported_audio >= USAGE_INTERVAL):
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=SpeechEventType.RECOGNITION_USAGE,
                    recognition_usage=stt.RecognitionUsage(audio_duration=self._unreported_audio),
                )
            )
            self._unreported_audio = 0.0

    def _start_speech(self) -> None:
        if not self._speaking:
            self._speaking = True
            self._event_ch.send_nowait(stt.SpeechEvent(type=SpeechEventType.START_OF_SPEECH))

    def _end_speech(self) -> None:
        if self._speaking:
            self._speaking = False
            self._event_ch.send_nowait(stt.SpeechEvent(type=SpeechEventType.END_OF_SPEECH))

    def _speech_data(
        self, text: str, words: list[TimedString] | None, start: float, end: float
    ) -> stt.SpeechData:
        return stt.SpeechData(
            language=self._opts.language or LanguageCode(""),
            text=text,
            start_time=start + self.start_time_offset,
            end_time=end + self.start_time_offset,
            words=words if self._opts.include_timestamps else None,
        )

    def _emit_final(
        self, text: str, words: list[TimedString] | None, start: float, end: float
    ) -> None:
        # a final is sent even without text: an agent waiting on the flush it answers
        # can end the user's turn at once instead of timing out
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[self._speech_data(text, words, start, end)],
            )
        )

    def _process_event(self, event: dict[str, Any]) -> bool:
        """Handle one server message. Returns True after the session's last result."""
        event_type = event.get("type")
        if event_type == "transcript.delta":
            self._mark_provider_progress()
            self._partial_text += event.get("text", "")
            if (words := event.get("words")) is not None:
                # the delta replaces the segment's words from `words_from` on
                keep = event.get("words_from") or 0
                self._partial_words = self._partial_words[:keep] + _timed_words(
                    words, offset=self.start_time_offset
                )
            if self._partial_text.strip():
                self._start_speech()
                words_so_far = self._partial_words or None
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=SpeechEventType.INTERIM_TRANSCRIPT,
                        alternatives=[
                            self._speech_data(self._partial_text.strip(), words_so_far, 0.0, 0.0)
                        ],
                    )
                )
        elif event_type in ("transcript.committed", "transcript.done"):
            self._mark_provider_progress()
            # a final transcript replaces the interim text before it
            self._partial_text = ""
            self._partial_words = []
            text = event.get("text", "").strip()
            words = _timed_words(event.get("words"), offset=self.start_time_offset)
            if event_type == "transcript.committed":
                start = float(event.get("start") or 0.0)
                end = float(event.get("end") or 0.0)
            else:
                # the closing transcript carries no bounds: it covers the audio after the
                # last commit, up to the end of the session's audio
                start = self._committed_until
                end = max(start, float(event.get("audio_seconds") or 0.0))
            if text:
                self._start_speech()
            if event_type == "transcript.committed":
                self._pending_commits = max(0, self._pending_commits - 1)
                self._committed_until = max(self._committed_until, end)
            if event_type == "transcript.committed" or text:
                self._emit_final(text, words, start, end)
            if not text or self._opts.server_vad is not None or event_type == "transcript.done":
                self._end_speech()
            return bool(event_type == "transcript.done")
        elif event_type == "input_audio_buffer.speech_started":
            self._start_speech()
        elif event_type == "session.expiring":
            logger.info(
                "Chariot STT session ends in %s seconds and will reconnect",
                event.get("seconds_remaining"),
            )
        elif event_type == "error":
            error = _utils.error_event(event)
            if "code" in event:
                raise error  # the server ends the session after a coded error
            logger.warning("Chariot STT refused a message: %s", error.message)
        return False


async def _close(ws: _WebSocket) -> None:
    # a half-open peer never answers the closing handshake, so it is not waited on for long
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(ws.close(), CLOSE_TIMEOUT)


def _timed_words(words: list[dict[str, Any]] | None, *, offset: float) -> list[TimedString]:
    return [
        TimedString(
            text=word.get("text", ""),
            start_time=float(word.get("start", 0.0)) + offset,
            end_time=float(word.get("end", 0.0)) + offset,
            start_time_offset=offset,
        )
        for word in words or []
    ]
