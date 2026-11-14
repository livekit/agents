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
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from ..constants import API_AUTH_HEADER, API_VERSION, API_VERSION_HEADER, AUDIO_ENCODING, USER_AGENT
from ..log import logger
from ..models import STTLanguages, TurnDetectingSTTModel


class STTConnectedEvent(TypedDict):
    """Fires once when the WebSocket connection is established; audio need not wait for it."""

    type: Literal["connected"]
    request_id: NotRequired[str]


class STTTurnEvent(TypedDict):
    """A user turn's progress.

    ``turn.start`` and ``turn.resume`` carry no transcript; ``turn.update``, ``turn.eager_end``
    and ``turn.end`` carry the cumulative text of the turn so far, not a delta.

    See also:
        https://docs.cartesia.ai/api-reference/stt/turns/websocket
    """

    type: Literal["turn.start", "turn.update", "turn.eager_end", "turn.resume", "turn.end"]
    transcript: NotRequired[str]
    request_id: NotRequired[str]


class STTErrorEvent(TypedDict):
    """Error event sent by the server; ``status_code`` is HTTP-style."""

    type: Literal["error"]
    status_code: NotRequired[int]
    title: NotRequired[str]
    message: NotRequired[str]
    request_id: NotRequired[str]


STTEventMessage = Annotated[
    STTConnectedEvent | STTTurnEvent | STTErrorEvent, Field(discriminator="type")
]
"""Server-sent message on the ``/stt/turns/websocket`` endpoint."""


class AutoFinalizeRecognizeStream(stt.SocketStream[STTEventMessage, None]):
    """
    Cartesia STT stream implementation with turn detection.

    This implementation ignores :meth:`flush`.
    Final transcripts are emitted when the STT model detects :class:`~stt.SpeechEventType.END_OF_SPEECH`.

    See also:
        - [API Reference](https://docs.cartesia.ai/api-reference/stt/turns/websocket)
        - [Compare STT Endpoints](https://docs.cartesia.ai/use-the-api/compare-stt-endpoints)
    """

    goodbye = '{"type":"close"}'
    # turn.start and turn.end are Cartesia's own turn boundaries
    provider_vad = True

    def __init__(
        self,
        *,
        stt: stt.STT,
        conn_options: APIConnectOptions,
        sample_rate: int,
        audio_chunk_duration_ms: int,
        model: TurnDetectingSTTModel | str,
        api_key: str,
        ws_base_url: str,
        language: LanguageCode,
        turn_start_threshold: NotGivenOr[float] = NOT_GIVEN,
        turn_eager_end_threshold: NotGivenOr[float] = NOT_GIVEN,
        turn_end_threshold: NotGivenOr[float] = NOT_GIVEN,
        turn_end_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        keyterm: NotGivenOr[list[str]] = NOT_GIVEN,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=sample_rate)
        self.frame_ms = audio_chunk_duration_ms
        self._sample_rate = sample_rate
        self._model = model
        self._api_key = api_key
        self._ws_base_url = ws_base_url
        self._language = language
        self._turn_start_threshold = turn_start_threshold
        self._turn_eager_end_threshold = turn_eager_end_threshold
        self._turn_end_threshold = turn_end_threshold
        self._turn_end_timeout_ms = turn_end_timeout_ms
        self._keyterm = keyterm
        self._request_id = ""
        # cumulative transcript for the current turn; used to re-emit on turn.resume
        self._current_transcript = ""

    async def connect(self, conn_options: APIConnectOptions) -> http.SocketTarget:
        self._current_transcript = ""
        # keyterm may repeat, so params is a list of pairs rather than a dict
        params: list[tuple[str, str]] = [
            ("model", str(self._model)),
            ("sample_rate", str(self._sample_rate)),
            ("encoding", AUDIO_ENCODING),
        ]
        if utils.is_given(self._turn_start_threshold):
            params.append(("turn_start_threshold", str(self._turn_start_threshold)))
        if utils.is_given(self._turn_eager_end_threshold):
            params.append(("turn_eager_end_threshold", str(self._turn_eager_end_threshold)))
        if utils.is_given(self._turn_end_threshold):
            params.append(("turn_end_threshold", str(self._turn_end_threshold)))
        if utils.is_given(self._turn_end_timeout_ms):
            params.append(("turn_end_timeout_ms", str(self._turn_end_timeout_ms)))
        if utils.is_given(self._keyterm):
            params.extend(("keyterm", term) for term in self._keyterm)

        return http.SocketTarget(
            f"{self._ws_base_url}/stt/turns/websocket?{urlencode(params)}",
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
        if isinstance(item, stt.RecognizeStream.Flush) and not self.input.closed:
            logger.warning(
                "Cartesia STT stream.flush() was ignored. See https://docs.cartesia.ai/use-the-api/compare-stt-endpoints for details."
            )
        return None

    def _transcript(
        self, text: str, *, final: bool = False, preflight: bool = False
    ) -> stt.Transcript:
        return stt.Transcript(
            text, self._language, final=final, preflight=preflight, request_id=self._request_id
        )

    def on_message(self, message: STTEventMessage) -> stt.Report | list[stt.Report] | None:
        if request_id := message.get("request_id"):
            self._request_id = request_id

        if message["type"] == "connected":
            return None

        if message["type"] == "error":
            # an HTTP-style status: a 5xx is retried, a rejected request is not
            raise APIStatusError(
                "Cartesia STT returned an error",
                status_code=message.get("status_code") or 500,
                body=dict(message),
                detail=message.get("message") or message.get("title"),
            )

        if message["type"] == "turn.start":
            if self.speaking:
                return None
            self._current_transcript = ""
            return stt.SpeechStarted()

        if not self.speaking:
            return None
        transcript = message.get("transcript", "")

        if message["type"] == "turn.update":
            # only an interim that changed: an unchanged one would cancel the preflight
            if not transcript or transcript == self._current_transcript:
                return None
            self._current_transcript = transcript
            return self._transcript(transcript)

        if message["type"] == "turn.eager_end":
            transcript = transcript or self._current_transcript
            if not transcript:
                return None
            self._current_transcript = transcript
            return self._transcript(transcript, preflight=True)

        if message["type"] == "turn.resume":
            # no transcript: the last one again as an interim, so the preflight is cancelled
            return self._transcript(self._current_transcript) if self._current_transcript else None

        # turn.end
        transcript = transcript or self._current_transcript
        self._current_transcript = ""
        return [self._transcript(transcript, final=True), stt.SpeechEnded()]

    def update_options(
        self,
        *,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
    ) -> None:
        """Change Cartesia STT options.

        Args:
            language: Changes the language emitted in :class:`~stt.SpeechData`.
                Ink 2 does not have multi-lingual support yet and only works with English.
        """
        if utils.is_given(language):
            self._language = LanguageCode(language)

        logger.warning(
            f"Cartesia STT model={self._model} does not currently support update_options()."
        )
