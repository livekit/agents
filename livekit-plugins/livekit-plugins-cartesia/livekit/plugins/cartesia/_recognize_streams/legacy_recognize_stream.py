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
import json
from typing import TYPE_CHECKING
from urllib.parse import urlencode

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, TimedString

from ..constants import (
    API_AUTH_HEADER,
    API_VERSION,
    API_VERSION_HEADER,
    REQUEST_ID_HEADER,
    USER_AGENT,
)
from ..log import logger
from .cartesia_recognize_stream import CartesiaRecognizeStream

if TYPE_CHECKING:
    from typing import Literal

    from typing_extensions import NotRequired, TypedDict

    from livekit.agents import APIConnectOptions
    from livekit.agents.types import NotGivenOr

    from ..models import STTEncoding, STTLanguages, STTModels

    class STTWord(TypedDict):
        """Word-level timestamp from Cartesia STT.

        Attributes:
            word: The transcribed word.
            start: Start time in seconds.
            end: End time in seconds.
        """

        word: str
        start: float
        end: float

    class STTTranscriptEvent(TypedDict):
        """Transcript chunk for the current connection.

        Each event is a delta from the last chunk with ``is_final=True``, not the
        cumulative transcript.

        Attributes:
            type: Event discriminator.
            is_final: Whether ``text`` is finalized.
            request_id: Unique identifier for this WebSocket connection.
            text: Transcribed text delta.
            duration: Duration of the audio in seconds.
            words: Optional word-level timestamps.
        """

        type: Literal["transcript"]
        is_final: bool
        request_id: str
        text: str
        duration: NotRequired[float]
        words: NotRequired[list[STTWord]]

    class STTFlushDoneEvent(TypedDict):
        """Acknowledgment for the ``finalize`` command.

        Attributes:
            type: Event discriminator.
            request_id: Unique identifier for this WebSocket connection.
        """

        type: Literal["flush_done"]
        request_id: str

    class STTDoneEvent(TypedDict):
        """Acknowledgment for the ``close`` command; session is closing.

        Attributes:
            type: Event discriminator.
            request_id: Unique identifier for this WebSocket connection.
        """

        type: Literal["done"]
        request_id: str

    class STTErrorEvent(TypedDict):
        """Error event sent by the server.

        Attributes:
            type: Event discriminator.
            code: HTTP-style status code; values >= 500 are treated as retryable.
            message: Human-readable error message.
            request_id: Unique identifier for this WebSocket connection.
        """

        type: Literal["error"]
        code: NotRequired[int]
        message: NotRequired[str]
        request_id: NotRequired[str]

    STTEventMessage = STTTranscriptEvent | STTFlushDoneEvent | STTDoneEvent | STTErrorEvent
    """Server-sent message on the ``/stt/websocket`` endpoint."""


def _get_api_language_param_from_language_code(language_code: LanguageCode) -> str:
    """API expects an ISO 639-1 language code (without region)"""
    return language_code.language


class LegacyRecognizeStream(CartesiaRecognizeStream):
    """Cartesia STT stream implementation for ``ink-whisper``.

    See also:
        https://docs.cartesia.ai/api-reference/stt/stt
    """

    def __init__(
        self,
        *,
        stt: stt.STT,
        conn_options: APIConnectOptions,
        sample_rate: int,
        encoding: STTEncoding,
        audio_chunk_duration_ms: int,
        model: STTModels | str,
        api_key: str,
        ws_base_url: str,
        session: aiohttp.ClientSession,
        language: LanguageCode | None,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=sample_rate)
        self._encoding = encoding
        self._sample_rate = sample_rate
        self._audio_chunk_duration_ms = audio_chunk_duration_ms
        self._model = model
        self._api_key = api_key
        self._ws_base_url = ws_base_url
        self._session = session
        self._language = language
        self._request_id = ""
        self._reconnect_event = asyncio.Event()
        self._speaking = False
        self._speech_duration: float = 0
        self._last_speech_end_time: float = 0

    async def _run(self) -> None:
        if self._input_ch.closed:
            return

        closing_ws = False
        # Reset per-connection state so a transport-error retry (a new _run
        # invocation by the base class) starts fresh.
        self._speaking = False
        self._speech_duration = 0
        self._last_speech_end_time = 0

        async def keepalive_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            try:
                while True:
                    await ws.ping()
                    await asyncio.sleep(30)
            except Exception:
                return

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            samples_per_chunk = self._sample_rate * self._audio_chunk_duration_ms // 1000
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._sample_rate,
                num_channels=1,
                samples_per_channel=samples_per_chunk,
            )

            try:
                async for data in self._input_ch:
                    frames: list[rtc.AudioFrame | LegacyRecognizeStream._FlushSentinel] = []
                    if isinstance(data, rtc.AudioFrame):
                        frames.extend(audio_bstream.write(data.data.tobytes()))
                    elif isinstance(data, self._FlushSentinel):
                        frames.extend(audio_bstream.flush())
                        frames.append(data)

                    for frame in frames:
                        if isinstance(frame, self._FlushSentinel):
                            await ws.send_str("finalize")
                        else:
                            self._speech_duration += frame.duration
                            await ws.send_bytes(frame.data.tobytes())

                for frame in audio_bstream.flush():
                    self._speech_duration += frame.duration
                    await ws.send_bytes(frame.data.tobytes())

                closing_ws = True
                await ws.send_str("close")
            except (aiohttp.ClientError, ConnectionError) as e:
                if closing_ws or self._session.closed:
                    return
                raise APIConnectionError(
                    message="Cartesia STT connection closed unexpectedly",
                    retryable=True,
                ) from e

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
                    raise APIConnectionError(
                        message=(
                            "Cartesia STT connection closed unexpectedly "
                            f"(close_code={ws.close_code}, "
                            f"data={msg.data!r}, extra={msg.extra!r})"
                        ),
                        retryable=True,
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Cartesia STT message type %s", msg.type)
                    continue

                try:
                    data: STTEventMessage = json.loads(msg.data)
                except Exception:
                    logger.exception("failed to parse Cartesia STT message")
                else:
                    self._process_stream_event(data)

        ws: aiohttp.ClientWebSocketResponse | None = None

        while True:
            try:
                ws = await self._connect_ws()

                send = asyncio.create_task(send_task(ws))
                recv = asyncio.create_task(recv_task(ws))
                keepalive = asyncio.create_task(keepalive_task(ws))
                wait_reconnect = asyncio.create_task(self._reconnect_event.wait())
                tasks = [send, recv, keepalive, wait_reconnect]

                # Only race send/recv against the reconnect signal. keepalive sleeps up to 30s
                # between pings, so including it here would block teardown for up to 30s after
                # send/recv have already finished.
                send_recv_group = asyncio.gather(send, recv)

                try:
                    done, _ = await asyncio.wait(
                        (send_recv_group, wait_reconnect),
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for task in done:
                        if task != wait_reconnect:
                            task.result()

                    if wait_reconnect not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks)
                    send_recv_group.cancel()
                    send_recv_group.exception()  # retrieve the exception
            finally:
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        params = {
            "model": self._model,
            "sample_rate": str(self._sample_rate),
            "encoding": self._encoding,
        }

        if self._language is not None:
            params["language"] = _get_api_language_param_from_language_code(
                language_code=self._language
            )

        ws_url = f"{self._ws_base_url}/stt/websocket?{urlencode(params)}"

        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    ws_url,
                    headers={
                        API_VERSION_HEADER: API_VERSION,
                        API_AUTH_HEADER: self._api_key,
                        "User-Agent": USER_AGENT,
                    },
                ),
                self._conn_options.timeout,
            )
            self._request_id = ws._response.headers.get(REQUEST_ID_HEADER) or ""
            logger.debug(
                "Established new Cartesia STT WebSocket connection",
                extra={"cartesia_request_id": self._request_id},
            )
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            raise APIConnectionError("failed to connect to cartesia") from e
        return ws

    def _process_stream_event(self, data: STTEventMessage) -> None:
        """Process incoming WebSocket messages.

        See https://docs.cartesia.ai/api-reference/stt/stt.
        """
        if request_id := data.get("request_id"):
            self._request_id = request_id

        if data["type"] == "transcript":
            if self._event_ch.closed:
                return

            text = data["text"]
            words = data.get("words", [])
            timed_words: list[TimedString] = [
                TimedString(
                    text=word.get("word", ""),
                    start_time=word.get("start", 0) + self.start_time_offset,
                    end_time=word.get("end", 0) + self.start_time_offset,
                    start_time_offset=self.start_time_offset,
                )
                for word in words
            ]
            # word timestamps are often within the audio window, so we track time separately
            if self._last_speech_end_time == 0.0:
                self._last_speech_end_time = self.start_time_offset
            start_time = self._last_speech_end_time
            end_time = start_time + data.get("duration", 0)
            self._last_speech_end_time = end_time
            is_final = data["is_final"]

            if not text and not is_final:
                return

            # we don't have a super accurate way of detecting when speech started.
            # this is typically the job of the VAD, but perfoming it here just in case something's
            # relying on STT to perform this task.
            if not self._speaking:
                self._speaking = True
                start_event = stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                self._event_ch.send_nowait(start_event)

            speech_data = stt.SpeechData(
                language=self._language or LanguageCode("en"),
                start_time=start_time,
                end_time=end_time,
                text=text,
                words=timed_words,
            )

            if is_final:
                if self._speech_duration > 0:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.RECOGNITION_USAGE,
                            request_id=self._request_id,
                            recognition_usage=stt.RecognitionUsage(
                                audio_duration=self._speech_duration,
                            ),
                        )
                    )
                    self._speech_duration = 0

                event = stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=[speech_data],
                )
                self._event_ch.send_nowait(event)

                if self._speaking:
                    self._speaking = False
                    end_event = stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                    self._event_ch.send_nowait(end_event)
            else:
                event = stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=[speech_data],
                )
                self._event_ch.send_nowait(event)

        elif data["type"] == "flush_done":
            logger.debug("Received flush_done acknowledgment from Cartesia STT")

        elif data["type"] == "done":
            logger.debug("Received done acknowledgment from Cartesia STT - session closing")

        elif data["type"] == "error":
            message = data.get("message") or "unknown error from cartesia"
            status_code = data.get("code") or 500
            logger.warning("cartesia sent an error", extra={"data": data})
            if status_code >= 500:
                raise APIConnectionError(message=message, retryable=True)
        else:
            logger.warning("received unexpected message from Cartesia STT: %s", data)

    def update_options(
        self,
        *,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Change Cartesia STT options. Use sparingly; this will interrupt transcription.

        Args:
            language: Used to change the language to match what the user is speaking.
            model: Deprecated. This is a no-op. Construct a new STT instance to change the model.
        """
        # not changing the model and reconnecting since that is likely unexpected behavior
        if utils.is_given(model) and model != self._model:
            logger.warning(
                "Cartesia STT update_options() ignores the model kwarg. Construct a new STT instance to change the model."
            )

        if utils.is_given(language):
            language_code = LanguageCode(language)

            current_api_language_param = (
                _get_api_language_param_from_language_code(language_code=self._language)
                if self._language is not None
                else None
            )
            api_language_param = _get_api_language_param_from_language_code(
                language_code=language_code
            )

            self._language = language_code

            if current_api_language_param != api_language_param:
                self._reconnect_event.set()
