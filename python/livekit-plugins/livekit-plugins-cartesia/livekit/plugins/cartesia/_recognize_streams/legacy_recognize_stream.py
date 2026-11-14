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

from typing import Annotated, Literal, NotRequired, TypedDict
from urllib.parse import urlencode

from pydantic import Field

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError, LanguageCode, http, stt, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr, TranscriptChunk

from ..constants import API_AUTH_HEADER, API_VERSION, API_VERSION_HEADER, AUDIO_ENCODING, USER_AGENT
from ..log import logger
from ..models import STTLanguages, STTModels


class STTWord(TypedDict):
    """Word-level timestamp from Cartesia STT, in seconds."""

    word: NotRequired[str]
    start: NotRequired[float]
    end: NotRequired[float]


class STTTranscriptEvent(TypedDict):
    """Transcript chunk for the current connection.

    Each event is a delta from the last chunk with ``is_final=True``, not the
    cumulative transcript.
    """

    type: Literal["transcript"]
    is_final: bool
    text: str
    request_id: NotRequired[str]
    duration: NotRequired[float]
    words: NotRequired[list[STTWord]]


class STTAckEvent(TypedDict):
    """Acknowledgment of ``finalize`` (``flush_done``) or ``close`` (``done``)."""

    type: Literal["flush_done", "done"]
    request_id: NotRequired[str]


class STTErrorEvent(TypedDict):
    """Error event sent by the server; ``code`` is HTTP-style."""

    type: Literal["error"]
    code: NotRequired[int]
    message: NotRequired[str]
    request_id: NotRequired[str]


STTEventMessage = Annotated[
    STTTranscriptEvent | STTAckEvent | STTErrorEvent, Field(discriminator="type")
]
"""Server-sent message on the ``/stt/websocket`` endpoint."""


def _get_api_language_param_from_language_code(language_code: LanguageCode) -> str:
    """API expects an ISO 639-1 language code (without region)"""
    return language_code.language


class LegacyRecognizeStream(stt.SocketStream[STTEventMessage, None]):
    """Cartesia STT stream implementation for ``ink-whisper``.

    See also:
        https://docs.cartesia.ai/api-reference/stt/stt
    """

    goodbye = "close"

    def __init__(
        self,
        *,
        stt: stt.STT,
        conn_options: APIConnectOptions,
        sample_rate: int,
        audio_chunk_duration_ms: int,
        model: STTModels | str,
        api_key: str,
        ws_base_url: str,
        language: LanguageCode | None,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=sample_rate)
        self.frame_ms = audio_chunk_duration_ms
        self._sample_rate = sample_rate
        self._model = model
        self._api_key = api_key
        self._ws_base_url = ws_base_url
        self._language = language
        self._request_id = ""
        self._last_speech_end_time: float = 0

    async def connect(self, conn_options: APIConnectOptions) -> http.SocketTarget:
        self._last_speech_end_time = 0
        params = {
            "model": self._model,
            "sample_rate": str(self._sample_rate),
            "encoding": AUDIO_ENCODING,
        }

        if self._language is not None:
            params["language"] = _get_api_language_param_from_language_code(
                language_code=self._language
            )

        return http.SocketTarget(
            f"{self._ws_base_url}/stt/websocket?{urlencode(params)}",
            headers={
                API_VERSION_HEADER: API_VERSION,
                API_AUTH_HEADER: self._api_key,
                "User-Agent": USER_AGENT,
            },
        )

    def encode(
        self, item: rtc.AudioFrame | stt.RecognizeStream.Flush | http.EndOfInput
    ) -> http.Encoded[None]:
        if isinstance(item, rtc.AudioFrame):
            return bytes(item.data)
        if isinstance(item, stt.RecognizeStream.Flush):
            return "finalize"
        return None

    def on_message(self, message: STTEventMessage) -> stt.Report | None:
        """Process incoming WebSocket messages.

        See https://docs.cartesia.ai/api-reference/stt/stt.
        """
        if request_id := message.get("request_id"):
            self._request_id = request_id

        if message["type"] == "flush_done":
            logger.debug("Received flush_done acknowledgment from Cartesia STT")
            return None
        if message["type"] == "done":
            logger.debug("Received done acknowledgment from Cartesia STT - session closing")
            return None
        if message["type"] == "error":
            # an HTTP-style status: a 5xx is retried, a rejected request is not
            raise APIStatusError(
                "Cartesia STT returned an error",
                status_code=message.get("code") or 500,
                body=dict(message),
                detail=message.get("message"),
            )

        offset = self.start_time_offset
        # word timestamps are often within the audio window, so we track time separately
        if self._last_speech_end_time == 0.0:
            self._last_speech_end_time = offset
        start_time = self._last_speech_end_time
        end_time = start_time + message.get("duration", 0)
        self._last_speech_end_time = end_time

        return stt.Transcript(
            message["text"],
            self._language or LanguageCode("en"),
            final=message["is_final"],
            request_id=self._request_id,
            start_time=start_time,
            end_time=end_time,
            words=[
                TranscriptChunk(
                    text=word.get("word", ""),
                    start_time=word.get("start", 0) + offset,
                    end_time=word.get("end", 0) + offset,
                    start_time_offset=offset,
                )
                for word in message.get("words", [])
            ],
        )

    def update_options(
        self,
        *,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
    ) -> None:
        """Change Cartesia STT options. Use sparingly; this will interrupt transcription.

        Args:
            language: Used to change the language to match what the user is speaking.
        """
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
                self.reconnect("language changed")
