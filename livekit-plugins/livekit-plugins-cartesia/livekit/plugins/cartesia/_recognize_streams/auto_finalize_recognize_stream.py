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
from livekit.agents.types import NOT_GIVEN

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

    from ..models import STTEncoding, STTLanguages, TurnDetectingSTTModel

    class STTConnectedEvent(TypedDict):
        """Fires once when the WebSocket connection is established.

        You do not need to wait for this event before sending audio.

        Attributes:
            type: Event discriminator.
            request_id: Unique identifier for this connection. Does not change between turns.

        See also:
            https://docs.cartesia.ai/api-reference/stt/turns/websocket
        """

        type: Literal["connected"]
        request_id: str

    class STTTurnStartEvent(TypedDict):
        """Model predicts the start of a user turn.

        Attributes:
            type: Event discriminator.
            request_id: Unique identifier for this connection. Does not change between turns.

        See also:
            https://docs.cartesia.ai/api-reference/stt/turns/websocket
        """

        type: Literal["turn.start"]
        request_id: str

    class STTTurnUpdateEvent(TypedDict):
        """Fires repeatedly as the model transcribes the current user turn.

        Used for interim transcript events.

        Attributes:
            type: Event discriminator.
            transcript: Cumulative text for the current turn, i.e. the full text transcribed
                so far in this turn, not a delta.
            request_id: Unique identifier for this connection. Does not change between turns.

        See also:
            https://docs.cartesia.ai/api-reference/stt/turns/websocket
        """

        type: Literal["turn.update"]
        transcript: str
        request_id: str

    class STTTurnEagerEndEvent(TypedDict):
        """Fires when the model predicts the user might be done speaking.

        Used for preflight transcript events.

        Attributes:
            type: Event discriminator.
            transcript: Cumulative text for the current turn, i.e. the full text transcribed
                so far in this turn, not a delta.
            request_id: Unique identifier for this connection. Does not change between turns.

        See also:
            https://docs.cartesia.ai/api-reference/stt/turns/websocket
        """

        type: Literal["turn.eager_end"]
        transcript: str
        request_id: str

    class STTTurnResumeEvent(TypedDict):
        """Fires after ``turn.eager_end`` if the user turn has not actually ended.

        Attributes:
            type: Event discriminator.
            request_id: Unique identifier for this connection. Does not change between turns.

        See also:
            https://docs.cartesia.ai/api-reference/stt/turns/websocket
        """

        type: Literal["turn.resume"]
        request_id: str

    class STTTurnEndEvent(TypedDict):
        """Marks the end of a user turn.

        Used for end-of-speech and final transcript events.

        Attributes:
            type: Event discriminator.
            transcript: Cumulative text for the current turn, i.e. the full text transcribed
                so far in this turn, not a delta.
            request_id: Unique identifier for this connection. Does not change between turns.

        See also:
            https://docs.cartesia.ai/api-reference/stt/turns/websocket
        """

        type: Literal["turn.end"]
        transcript: str
        request_id: str

    class STTErrorEvent(TypedDict):
        """Error event sent by the server.

        Attributes:
            type: Event discriminator.
            error_code: Stable code identifying the error.
            status_code: HTTP-style status code; values >= 500 are treated as retryable.
            title: Short human-readable error title.
            message: Detailed human-readable error message.
            doc_url: URL to documentation describing this error.
            request_id: Unique identifier for this connection. Does not change between turns.

        See also:
            https://docs.cartesia.ai/api-reference/stt/turns/websocket
        """

        type: Literal["error"]
        error_code: NotRequired[str]
        status_code: NotRequired[int]
        title: NotRequired[str]
        message: NotRequired[str]
        doc_url: NotRequired[str]
        request_id: NotRequired[str]

    STTEventMessage = (
        STTConnectedEvent
        | STTTurnStartEvent
        | STTTurnUpdateEvent
        | STTTurnEagerEndEvent
        | STTTurnResumeEvent
        | STTTurnEndEvent
        | STTErrorEvent
    )
    """Server-sent message on the ``/stt/turns/websocket`` endpoint.

    See also:
        https://docs.cartesia.ai/api-reference/stt/turns/websocket
    """


class AutoFinalizeRecognizeStream(CartesiaRecognizeStream):
    """
    Cartesia STT stream implementation with turn detection.

    This implementation ignores :meth:`flush`.
    Final transcripts are emitted when the STT model detects :class:`~stt.SpeechEventType.END_OF_SPEECH`.

    See also:
        - [API Reference](https://docs.cartesia.ai/api-reference/stt/turns/websocket)
        - [Compare STT Endpoints](https://docs.cartesia.ai/use-the-api/compare-stt-endpoints)
    """

    def __init__(
        self,
        *,
        stt: stt.STT,
        conn_options: APIConnectOptions,
        sample_rate: int,
        encoding: STTEncoding,
        audio_chunk_duration_ms: int,
        model: TurnDetectingSTTModel | str,
        api_key: str,
        ws_base_url: str,
        session: aiohttp.ClientSession,
        language: LanguageCode,
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
        self._speaking = False
        self._speech_duration: float = 0.0
        self._closing_ws = False
        # cumulative transcript for the current turn; used to re-emit on turn.resume
        self._current_transcript = ""

    async def _run(self) -> None:
        if self._input_ch.closed:
            return

        async def keepalive_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            try:
                while not self._closing_ws:
                    await ws.ping()
                    await asyncio.sleep(30)
            except Exception:
                return

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            samples_per_chunk = self._sample_rate * self._audio_chunk_duration_ms // 1000
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._sample_rate,
                num_channels=1,
                samples_per_channel=samples_per_chunk,
            )

            try:
                async for data in self._input_ch:
                    if isinstance(data, rtc.AudioFrame):
                        for frame in audio_bstream.write(data.data.tobytes()):
                            self._speech_duration += frame.duration
                            await ws.send_bytes(frame.data.tobytes())
                    elif isinstance(data, self._FlushSentinel):
                        if not self._input_ch.closed:
                            logger.warning(
                                "Cartesia STT stream.flush() was ignored. See https://docs.cartesia.ai/use-the-api/compare-stt-endpoints for details."
                            )

                for frame in audio_bstream.flush():
                    self._speech_duration += frame.duration
                    await ws.send_bytes(frame.data.tobytes())

                self._closing_ws = True
                await ws.send_str('{"type":"close"}')
            except (aiohttp.ClientError, ConnectionError) as e:
                if self._closing_ws or self._session.closed:
                    return
                raise APIConnectionError(
                    message="Cartesia STT connection closed unexpectedly",
                    retryable=True,
                ) from e

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if self._closing_ws or self._session.closed:
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

        # Reset per-connection state so a transport-error retry (a new _run
        # invocation by the base class) starts fresh. Without this, stale
        # `_speaking=True` silently drops the next turn.start,
        # `_current_transcript` can leak into turn.eager_end / turn.end
        # transcript fallbacks, and a stale `_closing_ws=True` makes recv_task
        # suppress legitimate unexpected-close failures on the new connection.
        self._speaking = False
        self._current_transcript = ""
        self._speech_duration = 0.0
        self._closing_ws = False

        ws: aiohttp.ClientWebSocketResponse | None = None

        try:
            ws = await self._connect_ws()
            send = asyncio.create_task(send_task(ws))
            recv = asyncio.create_task(recv_task(ws))
            keepalive = asyncio.create_task(keepalive_task(ws))
            tasks = [send, recv, keepalive]
            try:
                # Only wait on send/recv. keepalive_task sleeps up to 30s
                # between pings, so awaiting it here would keep gather() (and
                # the teardown/finalization below) blocked for up to 30s after
                # send/recv have already finished.
                await asyncio.gather(send, recv)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
                self._send_recognition_usage_event()
                # If the websocket dropped mid-turn, flush the partial
                # transcript as a FINAL_TRANSCRIPT so the consumer can finalize
                # the turn instead of losing it. In the normal close path the
                # server sends turn.end first and _speaking is already False.
                if self._speaking:
                    if not self._event_ch.closed:
                        if self._current_transcript:
                            self._send_transcript_event(
                                stt.SpeechEventType.FINAL_TRANSCRIPT, self._current_transcript
                            )
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                        )
                    self._speaking = False
                    self._current_transcript = ""
        finally:
            if ws is not None:
                await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        params = {
            "model": self._model,
            "sample_rate": str(self._sample_rate),
            "encoding": self._encoding,
        }

        ws_url = f"{self._ws_base_url}/stt/turns/websocket?{urlencode(params)}"

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
            c_request_id = ws._response.headers.get(REQUEST_ID_HEADER)
            logger.debug(
                "Established new Cartesia STT WebSocket connection",
                extra={"cartesia_request_id": c_request_id},
            )
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            raise APIConnectionError("failed to connect to cartesia", retryable=True) from e
        return ws

    def _send_transcript_event(self, event_type: stt.SpeechEventType, transcript: str) -> None:
        if self._event_ch.closed:
            return

        speech_data = stt.SpeechData(
            text=transcript,
            language=self._language,
        )
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=event_type,
                request_id=self._request_id,
                alternatives=[speech_data],
            )
        )

    def _send_recognition_usage_event(self) -> None:
        if self._speech_duration > 0 and not self._event_ch.closed:
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

    def _process_stream_event(self, data: STTEventMessage) -> None:
        if request_id := data.get("request_id"):
            self._request_id = request_id

        if data["type"] == "connected":
            return

        if data["type"] == "turn.start":
            if self._speaking:
                return
            self._speaking = True
            self._current_transcript = ""
            if not self._event_ch.closed:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                )
            return

        if data["type"] == "turn.update":
            if not self._speaking:
                return
            transcript = data["transcript"]
            if not transcript:
                return
            # only send interim transcript events if there's a change
            # this avoids canceling the preflight transcript if there's no change
            if self._current_transcript == transcript:
                return
            self._current_transcript = transcript
            self._send_transcript_event(stt.SpeechEventType.INTERIM_TRANSCRIPT, transcript)
            return

        if data["type"] == "turn.eager_end":
            if not self._speaking:
                return
            transcript = data["transcript"] or self._current_transcript
            if not transcript:
                return
            self._current_transcript = transcript
            self._send_transcript_event(stt.SpeechEventType.PREFLIGHT_TRANSCRIPT, transcript)
            return

        if data["type"] == "turn.resume":
            # turn.resume has no transcript; re-emit the most recent cumulative
            # transcript as an interim so the pipeline cancels the preflight.
            if not self._speaking or not self._current_transcript:
                return
            self._send_transcript_event(
                stt.SpeechEventType.INTERIM_TRANSCRIPT, self._current_transcript
            )
            return

        if data["type"] == "turn.end":
            if not self._speaking:
                return
            transcript = data["transcript"] or self._current_transcript

            self._send_recognition_usage_event()
            self._send_transcript_event(stt.SpeechEventType.FINAL_TRANSCRIPT, transcript)
            if not self._event_ch.closed:
                self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

            self._speaking = False
            self._current_transcript = ""
            return

        if data["type"] == "error":
            message = data.get("message") or data.get("title") or "unknown error from cartesia"
            status_code = data.get("status_code") or 500
            logger.warning("cartesia sent an error", extra={"data": data})
            if status_code >= 500:
                raise APIConnectionError(message=message, retryable=True)
            return

        logger.warning("received unexpected message from Cartesia STT: %s", data)

    def update_options(
        self,
        *,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Change Cartesia STT options.

        Args:
            language: Changes the language emitted in :class:`~stt.SpeechData`.
                Ink 2 does not have multi-lingual support yet and only works with English.
            model: Deprecated. This is a no-op. Construct a new STT instance to change the model.
        """
        if utils.is_given(language):
            self._language = LanguageCode(language)

        # not changing the model and reconnecting since that is likely unexpected behavior
        logger.warning(
            f"Cartesia STT model={self._model} does not currently support update_options()."
        )
