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
import collections
import dataclasses
import json
import os
import weakref
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, get_args

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
from livekit.agents.utils import AudioBuffer, is_given

from .api import ws_headers
from .log import log_server_message, logger
from .models import LogLevel
from .tts import _close_error

STTEncoding = Literal["pcm_s16le"]

# Define bytes per frame for different encoding types
bytes_per_frame = {
    "pcm_s16le": 2,
}


SUPPORTED_SAMPLE_RATE = 24000


PreflightTrigger: TypeAlias = Literal["vad_trip"]


def _api_error(msg_data: dict[str, Any]) -> APIStatusError:
    """Build the error for a `{"type": "error"}` frame.

    The server sends it right before closing the socket, so the message is the only
    place the actual failure reason (unsupported language, no credits, ...) shows up.
    Its `code` is the websocket close code that follows; client errors are not retried.
    """
    message = msg_data.get("message") or "unknown error"
    return _close_error(
        f"Gradium returned error: {message}", code=msg_data.get("code"), body=msg_data
    )


@dataclass
class STTOptions:
    sample_rate: int = SUPPORTED_SAMPLE_RATE
    buffer_size_seconds: float = 0.08
    encoding: str = "pcm_s16le"
    temperature: float | None = None
    language: LanguageCode = LanguageCode("en")
    vad_threshold: float = 0.9
    vad_bucket: int | None = 2
    # Positive VAD steps (80 ms each) since the trip, or since the last new word, that the
    # turn needs besides the server's flush reply before it ends; 0 ends it on the reply.
    vad_confirm_steps: int = 0
    # When set, a VAD inactivity probability below this threshold signals the start of
    # speech before any text is transcribed, which allows a faster barge-in.
    vad_start_threshold: float | None = None
    # When to preview the words of the turn as a PREFLIGHT_TRANSCRIPT so the session can
    # start generating before the turn is finalized; None disables the preview.
    preflight_trigger: PreflightTrigger | None = "vad_trip"
    # Words or phrases to favor in the transcript, such as product names or jargon.
    keywords: list[str] | None = None
    # How strongly to favor the keywords, the server defaults to 2.0.
    keyword_boost: float | None = None
    # Lowest level of the server's log messages to receive; None asks for none.
    log_level: LogLevel | None = None


# how long a failed end_of_stream write waits for the server's own close
_STOP_WRITE_GRACE = 0.1


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_endpoint: str | None = None,
        model_name: str = "default",
        sample_rate: int = SUPPORTED_SAMPLE_RATE,
        encoding: NotGivenOr[STTEncoding] = NOT_GIVEN,
        buffer_size_seconds: float = 0.08,
        http_session: aiohttp.ClientSession | None = None,
        vad_threshold: float = 0.9,
        vad_bucket: int | None = 2,
        vad_confirm_steps: int = 0,
        vad_flush: NotGivenOr[bool] = NOT_GIVEN,
        vad_start_threshold: float | None = None,
        preflight_trigger: PreflightTrigger | None = "vad_trip",
        temperature: float | None = None,
        language: str = "en",
        keywords: list[str] | None = None,
        keyword_boost: float | None = None,
        log_level: LogLevel | None = None,
    ):
        """
        Create a Gradium STT.

        When the VAD trips (the inactivity probability of `vad_bucket` goes above
        `vad_threshold`), the turn ends: the STT asks the server to flush the words the
        model still holds and emits the final transcript on the server's reply.

        Args:
            vad_confirm_steps: How many more steps, 80 ms each, above `vad_threshold` the
                turn needs after the trip before it ends, on top of the server's flush
                reply, so the turn ends at least `N × 80 ms` after the trip. Steps below
                the threshold don't count but don't reset the count either; new words after
                the flush reply do, as the user is talking again: the turn stays open and
                the next trip flushes again. 0, the default, ends the turn on the reply: the
                lowest latency, but a pause in the middle of a sentence ends the turn.
            vad_flush: Deprecated and ignored: a VAD trip always flushes.
            preflight_trigger: When to emit a `PREFLIGHT_TRANSCRIPT` with the words heard
                so far, or None to never emit one. The only trigger at the moment is the
                default, `"vad_trip"`: when the VAD trips, again only if the words changed. The session starts
                the LLM request from the preview and only restarts it if the final
                transcript differs: a turn where words were still held by the model delay at
                the trip costs one cancelled request.
            log_level: Opt in to the server's log messages, such as a warning about an
                unknown or deprecated setup option: the lowest level to receive ("debug",
                "info", "warning" or "error"). They are written to the
                `livekit.plugins.gradium` logger at their own level. Off by default.
        """
        if preflight_trigger is not None and preflight_trigger not in get_args(PreflightTrigger):
            raise ValueError(
                f"unknown preflight_trigger {preflight_trigger!r}, "
                f"expected one of {get_args(PreflightTrigger)} or None"
            )

        if vad_confirm_steps < 0:
            raise ValueError(f"vad_confirm_steps must be >= 0, got {vad_confirm_steps}")

        if is_given(vad_flush):
            logger.warning("'vad_flush' is deprecated and ignored: a VAD trip always flushes")

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript=False,  # only chunk start times are available
                offline_recognize=False,
                manual_flush=True,
            ),
        )

        api_key = api_key or os.environ.get("GRADIUM_API_KEY")

        if sample_rate != SUPPORTED_SAMPLE_RATE:
            raise ValueError(f"Only {SUPPORTED_SAMPLE_RATE}Hz sample rate is supported")

        if not api_key:
            raise ValueError(
                "Gradium API key is required. "
                "Pass one in via the `api_key` parameter, "
                "or set it as the `GRADIUM_API_KEY` environment variable"
            )

        self._api_key = api_key

        model_endpoint = (
            model_endpoint
            or os.environ.get("GRADIUM_MODEL_ENDPOINT")
            or "wss://api.gradium.ai/api/speech/asr"
        )

        self._model_endpoint = model_endpoint
        self._model_name = model_name

        self._opts = STTOptions(
            sample_rate=sample_rate,
            buffer_size_seconds=buffer_size_seconds,
            vad_threshold=vad_threshold,
            vad_bucket=vad_bucket,
            vad_confirm_steps=vad_confirm_steps,
            vad_start_threshold=vad_start_threshold,
            preflight_trigger=preflight_trigger,
            temperature=temperature,
            language=LanguageCode(language),
            keywords=list(keywords) if keywords is not None else None,
            keyword_boost=keyword_boost,
            log_level=log_level,
        )

        if is_given(encoding):
            self._opts.encoding = encoding

        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "Gradium"

    @property
    def session(self) -> aiohttp.ClientSession:
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
        raise NotImplementedError("Not implemented")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        config = dataclasses.replace(self._opts)
        if is_given(language):
            config.language = LanguageCode(language)
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            opts=config,
            api_key=self._api_key,
            model_endpoint=self._model_endpoint,
            model_name=self._model_name,
            http_session=self.session,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        buffer_size_seconds: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(buffer_size_seconds):
            self._opts.buffer_size_seconds = buffer_size_seconds

        for stream in self._streams:
            stream.update_options(
                buffer_size_seconds=buffer_size_seconds,
            )


class SpeechStream(stt.SpeechStream):
    @dataclass
    class _VADFlush(stt.SpeechStream._FlushSentinel):
        """Ask the server to process the audio it holds and answer with `flushed`."""

        flush_id: int

    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        model_endpoint: str,
        model_name: str,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)

        self._opts = opts
        self._api_key = api_key
        self._model_endpoint = model_endpoint
        self._model_name = model_name
        self._session = http_session
        self._speech_duration: float = 0

        self._reconnect_event = asyncio.Event()
        # counts the flush requests of the stream, so an id is never reused after a reconnect.
        # Both kinds of flush take their id when queued on the input channel, so the ids
        # follow the order the server receives the flushes in.
        self._flush_count = 0
        # ids of the caller flushes still in the input channel, not yet sent to the server
        self._queued_manual_flushes: collections.deque[int] = collections.deque()

    def flush(self) -> None:
        super().flush()
        self._flush_count += 1
        self._queued_manual_flushes.append(self._flush_count)

    def update_options(
        self,
        *,
        buffer_size_seconds: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(buffer_size_seconds):
            self._opts.buffer_size_seconds = buffer_size_seconds

        self._reconnect_event.set()

    async def _run(self) -> None:
        """
        Run a single websocket connection to Gradium and make sure to reconnect
        when something went wrong.
        """

        drain_started = asyncio.Event()
        # whether this connection produced words; a failed end_of_stream then ends with them
        words_received = False
        stop_write_error: BaseException | None = None
        peer_closed = asyncio.Event()
        # ids of the flushes the caller asked for, whose reply must always yield a final
        manual_flushes: set[int] = set()

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            samples_per_buffer = 1920

            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_per_buffer,
            )

            async for data in self._input_ch:
                flush_id: int | None = None
                if isinstance(data, self._VADFlush):
                    # the flush covers the audio sent so far, so push the partial frame
                    # still buffered first
                    frames = audio_bstream.flush()
                    flush_id = data.flush_id
                elif isinstance(data, self._FlushSentinel):
                    frames = audio_bstream.flush()
                    flush_id = self._queued_manual_flushes.popleft()
                    manual_flushes.add(flush_id)
                else:
                    frames = audio_bstream.write(data.data.tobytes())

                for frame in frames:
                    if len(frame.data) % 2 != 0:
                        logger.warning("Frame data size not aligned to int16 (multiple of 2)")

                    audio_data = base64.b64encode(frame.data.tobytes()).decode("utf-8")
                    audio_msg = {
                        "type": "audio",
                        "audio": audio_data,
                    }
                    try:
                        await ws.send_str(json.dumps(audio_msg))
                    except (aiohttp.ClientError, ConnectionError):
                        # the server closed first; recv_task raises the retryable error
                        # within the final-response deadline
                        return

                if flush_id is not None:
                    try:
                        await ws.send_str(json.dumps({"type": "flush", "flush_id": flush_id}))
                    except (aiohttp.ClientError, ConnectionError):
                        # the server closed first; recv_task raises the retryable error
                        # within the final-response deadline
                        return

            # Input ended: tell the server, then let recv_task drain the model's last
            # words until the server replies with its own end_of_stream.
            drain_started.set()
            try:
                await ws.send_str(json.dumps({"type": "end_of_stream"}))
            except (aiohttp.ClientError, ConnectionError) as e:
                # the server may have closed first with the words it had: if its close
                # follows promptly, recv_task decides; a silent peer means the write failed
                nonlocal stop_write_error
                stop_write_error = e
                try:
                    await asyncio.wait_for(peer_closed.wait(), _STOP_WRITE_GRACE)
                except asyncio.TimeoutError:
                    raise APIConnectionError("Gradium STT end-of-stream send failed") from e

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal words_received
            buffered_text: list[str] = []
            speech_end = 0.0
            speaking = False
            # id of the flush whose `flushed` reply ends the current turn
            pending_flush: int | None = None
            flush_replied = False
            # steps above the VAD threshold since the trip that sent the flush; a dip below it
            # doesn't reset the count, new words after the flush reply do
            confirmed_steps = 0
            # the words of the last preview, so a re-trip in the same turn does not repeat it
            preflight_text = ""
            # when to stop waiting for that reply: event-loop time, in seconds
            flush_deadline: float | None = None
            loop = asyncio.get_running_loop()
            monitor = utils.aio.WebSocketMonitor(
                ws, liveness_timeout=self._conn_options.liveness_timeout
            )

            def _end_time() -> float:
                # the `end_text` timing when the server already sent it, else unknown
                if speech_end > 0:
                    return speech_end + self.start_time_offset
                return 0.0

            def _finalize() -> None:
                nonlocal speaking, pending_flush, flush_deadline, buffered_text, speech_end
                nonlocal flush_replied, confirmed_steps, preflight_text
                if not speaking:
                    return
                speaking = False
                pending_flush = None
                flush_deadline = None
                flush_replied = False
                confirmed_steps = 0
                preflight_text = ""
                event = stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(
                            text=" ".join(buffered_text),
                            language=self._opts.language,
                            end_time=_end_time(),
                        )
                    ],
                )
                self._event_ch.send_nowait(event)
                buffered_text = []
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.END_OF_SPEECH,
                        alternatives=event.alternatives if speech_end > 0 else [],
                    )
                )
                speech_end = 0.0

            def _commit_after_pending_flush() -> bool:
                # A caller flush queued after the VAD flush: words spoken in between may still
                # be held by the model, and a final now would answer the commit without them.
                # One sent before the VAD flush does not count: its reply was due first, and
                # one the server never sent would otherwise hold back every later turn.
                assert pending_flush is not None
                return any(
                    manual_id > pending_flush
                    for manual_id in (*manual_flushes, *self._queued_manual_flushes)
                )

            def _end_turn_if_confirmed() -> None:
                nonlocal flush_deadline
                if not flush_replied or confirmed_steps < self._opts.vad_confirm_steps:
                    return
                if _commit_after_pending_flush():
                    # the caller's reply ends the turn instead; bound the wait for it
                    if flush_deadline is None:
                        flush_deadline = loop.time() + self._conn_options.timeout
                    return
                _finalize()

            def _send_preflight() -> None:
                nonlocal preflight_text
                text = " ".join(buffered_text)
                if not text or text == preflight_text:
                    return
                preflight_text = text
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.PREFLIGHT_TRANSCRIPT,
                        alternatives=[
                            stt.SpeechData(
                                text=text, language=self._opts.language, end_time=_end_time()
                            )
                        ],
                    )
                )

            while True:
                wait: float | None = None
                if flush_deadline is not None:
                    wait = flush_deadline - loop.time()
                    if wait <= 0:
                        # the server never answered the flush: keep what the model already
                        # produced rather than hold the turn open until the stream closes
                        logger.warning(
                            "Gradium did not answer a flush within %.1fs, ending the turn",
                            self._conn_options.timeout,
                        )
                        flush_deadline = None
                        _finalize()
                        continue
                try:
                    # the final-response deadline bounds the end_of_stream reply
                    msg = await asyncio.wait_for(monitor.receive(), timeout=wait)
                except asyncio.TimeoutError:
                    continue

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    peer_closed.set()
                    if drain_started.is_set():
                        # a close instead of the server's end_of_stream: same thing
                        _finalize()
                        if stop_write_error is not None and not words_received:
                            raise APIConnectionError(
                                "Gradium STT end-of-stream send failed"
                            ) from stop_write_error
                        return
                    raise APIStatusError(
                        "Gradium connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.error("Unexpected Gradium message type: %s", msg.type)
                    continue

                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.exception("Failed to decode message from Gradium")
                    continue
                if not isinstance(data, dict):
                    logger.error("Unexpected Gradium message: %s", msg.data)
                    continue

                type_ = data.get("type", "")
                if type_ == "error":
                    # the words the model already produced are still valid
                    _finalize()
                    raise _api_error(data)
                if type_ == "end_of_stream":
                    # The model has consumed all the audio: whatever is still buffered
                    # is the end of the last utterance.
                    _finalize()
                    if drain_started.is_set():
                        return
                    # The server ended the session while audio is still coming (a
                    # duration limit, for instance) and will close the socket; raise
                    # so the base class reconnects instead of the next audio write
                    # failing with a raw connection error.
                    raise APIStatusError(
                        "Gradium ended the session while input was still open",
                        status_code=ws.close_code or -1,
                        body=data,
                    )

                try:
                    if type_ == "text":
                        words_received = True
                        speech_end = 0.0
                        if flush_replied:
                            # words after the flush reply are new speech, not the words the
                            # flush recovered: the pause was not the end of the turn, and the
                            # next trip flushes again for the words the model holds by then
                            pending_flush = None
                            flush_deadline = None
                            flush_replied = False
                            confirmed_steps = 0
                        if speaking is False:
                            speaking = True
                            start_event = stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                            self._event_ch.send_nowait(start_event)
                        buffered_text.append(data["text"])
                        event = stt.SpeechEvent(
                            type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                            alternatives=[
                                stt.SpeechData(
                                    text=data["text"],
                                    language=self._opts.language,
                                    start_time=data["start_s"] + self.start_time_offset,
                                )
                            ],
                        )
                        self._event_ch.send_nowait(event)

                    elif type_ == "step":
                        if not speaking:
                            start_threshold = self._opts.vad_start_threshold
                            vad_bucket = self._opts.vad_bucket
                            # steps sent before any audio was processed carry a warm-up
                            # estimate that reads as activity, ignore them
                            if (
                                vad_bucket is not None
                                and start_threshold is not None
                                and data.get("total_duration_s", 0.0) > 0
                                and data["vad"][vad_bucket]["inactivity_prob"] < start_threshold
                            ):
                                speaking = True
                                self._event_ch.send_nowait(
                                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                                )
                            self._mark_provider_progress()
                            continue
                        # A trip ends the turn: flush, and the `flushed` reply finalizes
                        # once `vad_confirm_steps` steps confirmed the pause. Once the input
                        # has ended no flush can be sent, and the server's end_of_stream
                        # finalizes instead.
                        vad_bucket = self._opts.vad_bucket
                        tripped = vad_bucket is not None and (
                            data["vad"][vad_bucket]["inactivity_prob"] > self._opts.vad_threshold
                        )
                        if pending_flush is None:
                            if tripped and not self._input_ch.closed:
                                if self._opts.preflight_trigger == "vad_trip":
                                    # the words so far are what the turn will most likely end
                                    # with, so start generating while the flush recovers the rest
                                    _send_preflight()
                                self._flush_count += 1
                                pending_flush = self._flush_count
                                flush_deadline = loop.time() + self._conn_options.timeout
                                await self._input_ch.send(self._VADFlush(pending_flush))
                        elif self._opts.vad_confirm_steps > 0 and tripped:
                            # a dip doesn't count but doesn't reset the count either: the
                            # VAD can hover around the threshold for seconds after the user
                            # stopped, and only new words tell speech from that
                            confirmed_steps += 1
                            _end_turn_if_confirmed()
                    elif type_ == "ready":
                        pass
                    elif type_ == "end_text":
                        if speaking:
                            speech_end = data.get("stop_s", 0.0)
                    elif type_ == "flushed":
                        # the server returned every word the model delay held at the flush
                        flush_id = data.get("flush_id")
                        if flush_id in manual_flushes:
                            manual_flushes.discard(flush_id)
                            if speaking:
                                _finalize()
                            else:
                                # nothing was heard, but the flush still needs an answer or
                                # the commit waits for its transcript timeout
                                self._event_ch.send_nowait(
                                    stt.SpeechEvent(
                                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                        alternatives=[
                                            stt.SpeechData(text="", language=self._opts.language)
                                        ],
                                    )
                                )
                        elif pending_flush is not None and flush_id == pending_flush:
                            # When the caller committed after this flush, its reply ends the
                            # turn instead. The deadline stays, so a lost reply can't hold the
                            # turn open.
                            if not _commit_after_pending_flush():
                                # the server answered: only `vad_confirm_steps` can end the
                                # turn now, not the missing-reply timeout
                                flush_deadline = None
                                flush_replied = True
                                _end_turn_if_confirmed()
                    elif type_ == "log":
                        log_server_message(data)
                    else:
                        logger.warning(f"Unknown message type from Gradium {type_}")

                    if type_ in ("text", "step", "end_text"):
                        self._mark_provider_progress()

                except Exception:
                    logger.exception("Failed to process message from Gradium")

        ws: aiohttp.ClientWebSocketResponse | None = None

        while True:
            try:
                ws = await self._connect_ws()
                self._input_ch.restart_provider_clock()
                manual_flushes.clear()
                drain_started.clear()
                reconnect = await self._run_connection_tasks(
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                    reconnect=self._reconnect_event,
                    drain_started=drain_started,
                )
                if not reconnect:
                    break
                self._reconnect_event.clear()
            finally:
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        ws = await self._session.ws_connect(
            self._model_endpoint, headers=ws_headers(self._api_key), autoping=False
        )

        # Build and send the setup payload as the first message
        setup_msg: dict[str, Any] = {
            "type": "setup",
            "model_name": self._model_name,
            "input_format": "pcm",
        }
        json_config: dict[str, Any] = {"language": self._opts.language.language}
        if self._opts.temperature is not None:
            json_config["temp"] = self._opts.temperature
        if self._opts.keywords:
            keywords: dict[str, Any] = {"words": self._opts.keywords}
            if self._opts.keyword_boost is not None:
                keywords["boost"] = self._opts.keyword_boost
            json_config["keywords"] = keywords
        setup_msg["json_config"] = json_config
        if self._opts.log_level is not None:
            setup_msg["log_level"] = self._opts.log_level

        await ws.send_str(json.dumps(setup_msg))
        return ws
