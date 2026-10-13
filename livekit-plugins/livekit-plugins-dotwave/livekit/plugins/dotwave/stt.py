# Copyright 2026 Dollyglot, Inc.
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
import dataclasses
import json
import os
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr, TimedString
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger

DEFAULT_MODEL = "nemotron-asr-streaming"
DEFAULT_BASE_URL = "https://api.dotwave.ai/v1/listen"

SUPPORTED_SAMPLE_RATES = (16000, 24000)
UTTERANCE_END_MS_RANGE = (1000, 3200)

# close codes after which reconnecting cannot succeed: the request itself is wrong
# (4400) or the credential was refused (4401)
_NON_RETRYABLE_CLOSE_CODES = frozenset({4400, 4401})

_CLOSE_CODE_DESCRIPTIONS = {
    1000: "session closed",
    1011: "server error",
    1012: "service restarting",
    1013: "audio arrived faster than real time",
    4400: "protocol error",
    4401: "invalid API key",
    4413: "service not ready",
    4429: "no capacity available",
}


@dataclass
class STTOptions:
    model: str
    language: LanguageCode | None
    interim_results: bool
    sample_rate: int
    utterance_end_ms: int | None
    vad_events: bool
    base_url: str


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        language: str | None = None,
        interim_results: bool = True,
        sample_rate: int = 16000,
        utterance_end_ms: int | None = None,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        base_url: str = DEFAULT_BASE_URL,
        vad_events: bool = True,
    ) -> None:
        """Create a new instance of .wave STT.

        Args:
            model: The .wave speech-to-text model. Defaults to "nemotron-asr-streaming".
            language: A language tag such as "en-US" or "pt-BR". Defaults to None, which
                enables automatic language detection; each transcript then reports the
                language that was detected.
            interim_results: Whether to emit interim transcripts. Defaults to True.
            sample_rate: The sample rate of the audio sent to .wave, 16000 or 24000.
                Input audio at another rate is converted to it. Defaults to 16000.
            utterance_end_ms: Silence in milliseconds, between 1000 and 3200, after which
                .wave ends the utterance. Defaults to None, which leaves turn detection to
                the agent: the stream is finalized when the agent flushes it.
            api_key: Your .wave API key. If not provided, the ``DOTWAVE_API_KEY``
                environment variable is used.
            http_session: Optional aiohttp ClientSession to use for the connection.
            base_url: The .wave streaming endpoint. Defaults to
                "https://api.dotwave.ai/v1/listen".
            vad_events: Whether to emit START_OF_SPEECH as soon as .wave reports that speech
                started. When False, START_OF_SPEECH is emitted with the first transcript.
                Defaults to True.

        Raises:
            ValueError: If no API key is provided or found in the environment, or if
                ``sample_rate`` or ``utterance_end_ms`` is out of range.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=interim_results,
                aligned_transcript="word",
                keyterms=False,
                offline_recognize=False,
            )
        )

        dotwave_api_key = api_key if is_given(api_key) else os.environ.get("DOTWAVE_API_KEY")
        if not dotwave_api_key:
            raise ValueError(
                ".wave API key is required, either as argument or set"
                " DOTWAVE_API_KEY environment variable"
            )
        self._api_key = dotwave_api_key

        self._opts = STTOptions(
            model=_validate_model(model),
            language=_to_language(language),
            interim_results=interim_results,
            sample_rate=_validate_sample_rate(sample_rate),
            utterance_end_ms=_validate_utterance_end_ms(utterance_end_ms),
            vad_events=vad_events,
            base_url=base_url,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        """The .wave model used for recognition."""
        return self._opts.model

    @property
    def provider(self) -> str:
        """The provider name, ".wave"."""
        return ".wave"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError(
            ".wave STT only supports streaming recognition, use stream() instead of recognize()"
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        """Open a streaming recognition session.

        Args:
            language: Overrides the configured language for this stream. None enables
                automatic language detection.
            conn_options: Connection and retry options.

        Returns:
            A SpeechStream that accepts audio frames and yields SpeechEvents.
        """
        opts = dataclasses.replace(self._opts)
        if is_given(language):
            opts.language = _to_language(language)

        stream = SpeechStream(
            stt=self,
            opts=opts,
            conn_options=conn_options,
            api_key=self._api_key,
            http_session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
        utterance_end_ms: NotGivenOr[int | None] = NOT_GIVEN,
        interim_results: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        """Update options for this STT and every open stream.

        Open streams reconnect to apply the new options.

        Args:
            language: A language tag, or None for automatic language detection.
            model: The .wave model.
            utterance_end_ms: Silence in milliseconds (1000 to 3200) that ends an
                utterance, or None to disable it.
            interim_results: Whether to emit interim transcripts.
        """
        if is_given(language):
            self._opts.language = _to_language(language)
        if is_given(model):
            self._opts.model = _validate_model(model)
        if is_given(utterance_end_ms):
            self._opts.utterance_end_ms = _validate_utterance_end_ms(utterance_end_ms)
        if is_given(interim_results):
            self._opts.interim_results = interim_results

        for stream in self._streams:
            stream.update_options(
                language=language,
                model=model,
                utterance_end_ms=utterance_end_ms,
                interim_results=interim_results,
            )


class SpeechStream(stt.SpeechStream):
    _KEEPALIVE_MSG: str = json.dumps({"type": "KeepAlive"})
    _CLOSE_MSG: str = json.dumps({"type": "CloseStream"})
    _FINALIZE_MSG: str = json.dumps({"type": "Finalize"})

    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        http_session: aiohttp.ClientSession,
    ) -> None:
        """A streaming recognition session over the .wave WebSocket.

        Created by `STT.stream`; not meant to be constructed directly.
        """
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._api_key = api_key
        self._session = http_session
        self._speaking = False
        self._request_id = ""
        self._last_error: str | None = None
        self._reconnect_event = asyncio.Event()
        self._audio_duration_collector = _PeriodicCollector(
            callback=self._on_audio_duration_report,
            duration=5.0,
        )

    def update_options(
        self,
        *,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
        utterance_end_ms: NotGivenOr[int | None] = NOT_GIVEN,
        interim_results: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        """Update options for this stream and reconnect to apply them.

        Args:
            language: A language tag, or None for automatic language detection.
            model: The .wave model.
            utterance_end_ms: Silence in milliseconds (1000 to 3200) that ends an
                utterance, or None to disable it.
            interim_results: Whether to emit interim transcripts.
        """
        if is_given(language):
            self._opts.language = _to_language(language)
        if is_given(model):
            self._opts.model = _validate_model(model)
        if is_given(utterance_end_ms):
            self._opts.utterance_end_ms = _validate_utterance_end_ms(utterance_end_ms)
        if is_given(interim_results):
            self._opts.interim_results = interim_results

        self._reconnect_event.set()

    async def _run(self) -> None:
        closing_ws = False

        async def keepalive_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            # .wave accepts KeepAlive, but it is audio (silence included) that keeps a
            # session open: one that receives no audio for 30 s is closed. In a LiveKit
            # room audio flows continuously, so this only keeps the socket warm.
            try:
                while not closing_ws:
                    await ws.send_str(SpeechStream._KEEPALIVE_MSG)
                    await asyncio.sleep(5)
            except (aiohttp.ClientError, ConnectionError):
                # the socket is closing: recv_task sees the close and reports its reason
                return

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            # forward audio in chunks of 50ms
            samples_50ms = self._opts.sample_rate // 20
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_50ms,
            )

            # end_input() flushes too, so a Finalize goes out only when audio was sent
            # since the previous one
            sent_since_finalize = False
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
                        sent_since_finalize = True

                    if flushed and sent_since_finalize:
                        # ask .wave to finish decoding the audio sent so far; the answer is
                        # a final with from_finalize set
                        self._audio_duration_collector.flush()
                        await ws.send_str(SpeechStream._FINALIZE_MSG)
                        sent_since_finalize = False

                # no more input: tell .wave to close the session once it is done
                closing_ws = True
                await ws.send_str(SpeechStream._CLOSE_MSG)
            except (aiohttp.ClientError, ConnectionError):
                # a write fails when the socket is closing. recv_task sees the close and
                # raises with the reason .wave gave, which decides whether to reconnect;
                # raising here would hide that reason.
                return

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    # close is expected after CloseStream, or when the agent session ends
                    # and the http session is closed. After CloseStream there is no
                    # audio left to retry with, so a close that is not the normal one
                    # is reported rather than raised: the final it may have cut off
                    # cannot be recovered by reconnecting.
                    if closing_ws or self._session.closed:
                        if closing_ws and ws.close_code not in (None, 1000):
                            logger.warning(
                                ".wave closed the session with code %s after CloseStream: %s",
                                ws.close_code,
                                msg.extra or "",
                            )
                        return

                    # raising here makes the base class reconnect when the error is
                    # retryable
                    raise self._close_error(ws.close_code, msg.extra)

                if msg.type == aiohttp.WSMsgType.ERROR:
                    if closing_ws or self._session.closed:
                        if closing_ws:
                            logger.warning(
                                ".wave connection lost after CloseStream: %s", ws.exception()
                            )
                        return

                    # the heartbeat closes the socket when a ping goes unanswered, and
                    # that surfaces here rather than as a close frame
                    raise APIConnectionError(".wave connection lost") from ws.exception()

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected .wave message type %s", msg.type)
                    continue

                try:
                    self._process_stream_event(json.loads(msg.data))
                except Exception:
                    logger.exception("failed to process .wave message")

        # A retry after a dropped connection starts a fresh session with the audio
        # still unread in the input channel; audio already sent is not replayed.
        # .wave accepts at most 640 ms of audio ahead of real time, so replaying a
        # buffer would close the new session, and a repeated transcript would be
        # worse than a short gap. This matches the framework's other streaming
        # plugins.
        while True:
            ws: aiohttp.ClientWebSocketResponse | None = None
            try:
                ws = await self._connect_ws()
                send = asyncio.create_task(send_task(ws), name="dotwave.send")
                recv = asyncio.create_task(recv_task(ws), name="dotwave.recv")
                keepalive = asyncio.create_task(keepalive_task(ws), name="dotwave.keepalive")
                wait_reconnect = asyncio.create_task(self._reconnect_event.wait())
                pending: set[asyncio.Task[Any]] = {send, recv, keepalive}
                reconnect = False
                try:
                    while recv in pending:
                        done, _ = await asyncio.wait(
                            pending | {wait_reconnect},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        # propagate exceptions from completed tasks
                        for task in done:
                            if task is not wait_reconnect:
                                task.result()

                        if wait_reconnect in done:
                            reconnect = True
                            break

                        # the session is over once the receiver returns: .wave closed
                        # the socket after CloseStream
                        pending -= done
                finally:
                    await utils.aio.gracefully_cancel(send, recv, keepalive, wait_reconnect)

                if not reconnect:
                    break

                self._reconnect_event.clear()
            finally:
                self._audio_duration_collector.flush()
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        query: dict[str, Any] = {
            "model": self._opts.model,
            "encoding": "linear16",
            "sample_rate": self._opts.sample_rate,
            "channels": 1,
            "interim_results": self._opts.interim_results,
            # .wave does not take a silence-based endpointing value; turns end on
            # Finalize or utterance_end_ms
            "endpointing": False,
        }
        if self._opts.language:
            query["language"] = self._opts.language
        if self._opts.utterance_end_ms is not None:
            query["utterance_end_ms"] = self._opts.utterance_end_ms

        self._last_error = None
        t0 = time.perf_counter()
        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    _to_ws_url(query, self._opts.base_url),
                    headers={"Authorization": f"Token {self._api_key}"},
                    # without a heartbeat a silently dropped socket is never noticed:
                    # recv_task would wait on ws.receive() forever
                    heartbeat=30.0,
                ),
                self._conn_options.timeout,
            )
        except asyncio.TimeoutError:
            raise APIConnectionError("failed to connect to .wave") from None
        except aiohttp.ClientResponseError as e:
            # RequestInfo carries the request headers, so chaining this error or
            # formatting it would put the API key in the exception repr.
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError(f"failed to connect to .wave ({type(e).__name__})") from None

        self._report_connection_acquired(time.perf_counter() - t0, False)
        logger.debug("established .wave STT WebSocket connection")
        return ws

    def _close_error(self, close_code: int | None, reason: object) -> APIStatusError:
        code = close_code if close_code is not None else -1
        detail = self._last_error or (reason if isinstance(reason, str) and reason else None)
        message = f".wave closed the connection: {_CLOSE_CODE_DESCRIPTIONS.get(code, 'closed')}"
        if detail:
            message = f"{message} ({detail})"
        return APIStatusError(
            message=message,
            status_code=code,
            request_id=self._request_id or None,
            body=None,
            retryable=code not in _NON_RETRYABLE_CLOSE_CODES,
        )

    def _on_audio_duration_report(self, duration: float) -> None:
        usage_event = stt.SpeechEvent(
            type=stt.SpeechEventType.RECOGNITION_USAGE,
            request_id=self._request_id,
            alternatives=[],
            recognition_usage=stt.RecognitionUsage(audio_duration=duration),
        )
        self._event_ch.send_nowait(usage_event)

    def _start_speaking(self) -> None:
        if self._speaking:
            return
        self._speaking = True
        self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))

    def _end_speaking(self) -> None:
        if not self._speaking:
            return
        self._speaking = False
        self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

    def _process_stream_event(self, data: dict[str, Any]) -> None:
        msg_type = data.get("type")

        if msg_type == "Metadata":
            self._request_id = data.get("request_id", self._request_id)

        elif msg_type == "SpeechStarted":
            if self._opts.vad_events:
                self._start_speaking()

        elif msg_type == "Results":
            request_id = data.get("metadata", {}).get("request_id") or self._request_id
            self._request_id = request_id
            is_final = bool(data.get("is_final"))

            alts = live_transcription_to_speech_data(
                self._opts.language,
                data,
                start_time_offset=self.start_time_offset,
            )
            # every final carries the whole words since the previous final, so each
            # non-empty one is its own FINAL_TRANSCRIPT; an interim replaces the last one.
            # A Finalize answer can be empty and emits nothing.
            if alts and alts[0].text:
                self._start_speaking()
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT
                        if is_final
                        else stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        request_id=request_id,
                        alternatives=alts,
                    )
                )

            if data.get("speech_final"):
                self._end_speaking()

        elif msg_type == "UtteranceEnd":
            self._end_speaking()

        elif msg_type == "Error":
            # .wave sends an Error just before a failing close; recv_task raises with it
            self._last_error = str(data.get("message") or "")
            logger.warning(".wave reported an error: %s", self._last_error)

        else:
            logger.debug("received unexpected message from .wave", extra={"type": msg_type})


def live_transcription_to_speech_data(
    language: str | None,
    data: dict[str, Any],
    *,
    start_time_offset: float,
) -> list[stt.SpeechData]:
    """Convert a .wave ``Results`` message to SpeechData alternatives.

    Args:
        language: The configured language, or None for automatic detection.
        data: The decoded ``Results`` message.
        start_time_offset: Offset in seconds added to every timestamp, so that times stay
            monotonic across reconnections.

    Returns:
        One SpeechData per alternative.
    """
    fallback_language = language if language and language != "multi" else ""
    speech_data = []
    for alt in data.get("channel", {}).get("alternatives", []):
        words = alt.get("words") or []
        languages = alt.get("languages") or []
        sd = stt.SpeechData(
            language=LanguageCode(languages[0] if languages else fallback_language),
            text=alt.get("transcript", ""),
            start_time=(words[0].get("start", 0.0) if words else data.get("start", 0.0))
            + start_time_offset,
            end_time=(
                words[-1].get("end", 0.0)
                if words
                else data.get("start", 0.0) + data.get("duration", 0.0)
            )
            + start_time_offset,
            confidence=alt.get("confidence", 0.0),
            words=[
                TimedString(
                    text=word.get("punctuated_word") or word.get("word", ""),
                    start_time=word.get("start", 0.0) + start_time_offset,
                    end_time=word.get("end", 0.0) + start_time_offset,
                    confidence=word.get("confidence", NOT_GIVEN),
                    start_time_offset=start_time_offset,
                )
                for word in words
            ]
            if words
            else None,
        )
        speech_data.append(sd)
    return speech_data


class _PeriodicCollector:
    """Accumulates audio durations and reports the total at most every ``duration`` s."""

    def __init__(self, callback: Callable[[float], None], *, duration: float) -> None:
        self._duration = duration
        self._callback = callback
        self._last_flush_time = time.monotonic()
        self._total: float | None = None

    def push(self, value: float) -> None:
        self._total = value if self._total is None else self._total + value
        if time.monotonic() - self._last_flush_time >= self._duration:
            self.flush()

    def flush(self) -> None:
        if self._total is not None:
            self._callback(self._total)
            self._total = None
        self._last_flush_time = time.monotonic()


def _to_ws_url(query: dict[str, Any], base_url: str) -> str:
    # lowercase bools, as the Deepgram protocol expects
    params = {k: str(v).lower() if isinstance(v, bool) else v for k, v in query.items()}
    if base_url.startswith("http"):
        base_url = base_url.replace("http", "ws", 1)
    return f"{base_url}?{urlencode(params)}"


def _to_language(language: str | None) -> LanguageCode | None:
    return LanguageCode(language) if language else None


def _validate_model(model: str) -> str:
    if not model:
        raise ValueError("model must not be empty")
    return model


def _validate_sample_rate(sample_rate: int) -> int:
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise ValueError(f"sample_rate must be one of {SUPPORTED_SAMPLE_RATES}, got {sample_rate}")
    return sample_rate


def _validate_utterance_end_ms(utterance_end_ms: int | None) -> int | None:
    if utterance_end_ms is None:
        return None
    low, high = UTTERANCE_END_MS_RANGE
    if not low <= utterance_end_ms <= high:
        raise ValueError(
            f"utterance_end_ms must be between {low} and {high}, got {utterance_end_ms}"
        )
    return utterance_end_ms
