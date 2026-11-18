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
import os
import weakref
from typing import Annotated, ClassVar, Literal, NotRequired, TypedDict, assert_never
from urllib.parse import urlencode

from pydantic import Field

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIStatusError,
    LanguageCode,
    http,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr, TranscriptChunk

from .constants import API_AUTH_HEADER, API_VERSION, API_VERSION_HEADER, AUDIO_ENCODING, USER_AGENT
from .log import logger
from .models import STTLanguages, STTModels


def _ws_base_url(base_url: str) -> str:
    if base_url.startswith(("http://", "https://")):
        return base_url.replace("http", "ws", 1)
    return f"wss://{base_url}"


class STT(stt.STT):
    """Cartesia speech to text.

    Model ``ink-2`` streams on the turns endpoint, which detects turns itself and sends
    interim results. Model ``ink-whisper`` streams on the manual endpoint, where a flush
    finalizes the transcript, with word-aligned transcripts.

    See also:
        https://docs.cartesia.ai/build-with-cartesia/stt-models/latest
        https://docs.cartesia.ai/use-the-api/compare-stt-endpoints

    Examples:

        ```# Turn detecting
        from livekit.agents import VoiceSession
        from livekit.plugins import cartesia

        session = VoiceSession(
            stt=cartesia.STT(),
            llm=LLM(),  # choose your favorite LLM
            tts=cartesia.TTS(),
            turn_handling={
                "turn_detection": "stt",
            },
        )
        ```
    """

    def __init__(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        sample_rate: int = 16000,
        api_key: str | None = None,
        audio_chunk_duration_ms: int = 160,
        http_session: http.HttpSession | None = None,
        base_url: str = "https://api.cartesia.ai",
        language: STTLanguages | str | None = None,
        turn_start_threshold: NotGivenOr[float] = NOT_GIVEN,
        turn_eager_end_threshold: NotGivenOr[float] = NOT_GIVEN,
        turn_end_threshold: NotGivenOr[float] = NOT_GIVEN,
        turn_end_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        keyterm: NotGivenOr[list[str]] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of Cartesia STT.

        Args:
            model: The Cartesia STT model to use.
                Defaults to ``ink-2`` if language is ``en``.
                Defaults to ``ink-whisper`` for other languages.
            sample_rate: The sample rate of the audio in Hz. Defaults to 16 kHz.
            api_key: The Cartesia API key. If not provided, it will be read from
                the ``CARTESIA_API_KEY`` environment variable.
            audio_chunk_duration_ms: Duration in milliseconds of each audio chunk
                sent to the Cartesia STT websocket. Defaults to 160 ms.
            http_session: Optional HttpSession to use for requests.
            base_url: The base URL for the Cartesia API.
                Defaults to ``https://api.cartesia.ai``.
            language: The language code for recognition.
                This plugin only supports ``en`` for ``ink-2``.
            turn_start_threshold: Likelihood above which the model starts a turn.
                Range 0.5-0.9, Cartesia default 0.8. Must stay above
                ``turn_eager_end_threshold``. Turn-detecting models (e.g. ``ink-2``) only.
            turn_eager_end_threshold: Likelihood below which the model emits
                ``turn.eager_end``, the early might-be-done signal. Range 0.3-0.6,
                Cartesia default 0.4. Must stay between the end and start thresholds.
                Turn-detecting models only.
            turn_end_threshold: Likelihood below which the model ends the turn.
                Range 0.05-0.5, Cartesia default 0.2. Must stay below
                ``turn_eager_end_threshold``. Turn-detecting models only.
            turn_end_timeout_ms: Maximum time in milliseconds to wait after the user
                stops speaking before ending the turn even if the likelihood never
                falls below ``turn_end_threshold``. Range 640-11200, Cartesia default
                5600. Turn-detecting models only.
            keyterm: Key terms to improve recall of specific words and phrases
                (up to 100 terms totaling 1200 characters). Turn-detecting models only.

        Raises:
            ValueError: If no API key is provided or found in environment variables,
                or if a turn-detection parameter is passed with a model that does not
                support turn detection (e.g. ``ink-whisper``).
        """
        resolved_api_key = api_key or os.environ.get("CARTESIA_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Cartesia API key is required, either as argument or set"
                " CARTESIA_API_KEY environment variable"
            )

        language_code = None if language is None else LanguageCode(language)

        # TODO: default all languages to ink-2 once they are supported
        if utils.is_given(model):
            resolved_model = model
        elif language_code is None or language_code.language == "en":
            resolved_model = "ink-2"
        else:
            resolved_model = "ink-whisper"

        # ink-whisper streams on the manual endpoint, every other model on the turns endpoint
        self._turns = not resolved_model.startswith("ink-whisper")
        if not self._turns:
            turns_only = {
                "turn_start_threshold": turn_start_threshold,
                "turn_eager_end_threshold": turn_eager_end_threshold,
                "turn_end_threshold": turn_end_threshold,
                "turn_end_timeout_ms": turn_end_timeout_ms,
                "keyterm": keyterm,
            }
            for name, value in turns_only.items():
                if utils.is_given(value):
                    raise ValueError(
                        f"The {name!r} parameter is only supported by turn-detecting"
                        f" models (e.g. ink-2); model {resolved_model!r} does not support it."
                    )

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=self._turns,
                aligned_transcript=False if self._turns else "word",
                offline_recognize=False,
                diarization=False,
            )
        )

        self._api_key = resolved_api_key
        self._audio_chunk_duration_ms = audio_chunk_duration_ms
        self._language = language_code
        self._model = resolved_model
        self._sample_rate = sample_rate
        self._session = http_session
        self._ws_base_url = _ws_base_url(base_url)
        self._turn_start_threshold = turn_start_threshold
        self._turn_eager_end_threshold = turn_eager_end_threshold
        self._turn_end_threshold = turn_end_threshold
        self._turn_end_timeout_ms = turn_end_timeout_ms
        self._keyterm = keyterm

        self._streams = weakref.WeakSet[_TurnsStream | _ManualStream]()

        self._warn_on_unexpected_args(language=self._language)

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "Cartesia"

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError(
            "Cartesia STT does not support batch recognition, use stream() instead"
        )

    def stream(
        self,
        *,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        resolved_language = LanguageCode(language) if utils.is_given(language) else self._language
        self._warn_on_unexpected_args(language=resolved_language)

        stream: _TurnsStream | _ManualStream
        if self._turns:
            stream = _TurnsStream(
                stt=self,
                conn_options=conn_options,
                language=resolved_language or LanguageCode("en"),
            )
        else:
            stream = _ManualStream(stt=self, conn_options=conn_options, language=resolved_language)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """Close every stream created by :meth:`stream`.

        The HTTP session is left open: it is either supplied by the caller or
        owned by the shared HTTP context, so it is not ours to close.
        """
        streams = list(self._streams)
        self._streams.clear()
        # return_exceptions=True so one stream failing to close doesn't abandon the rest
        await asyncio.gather(*(stream.aclose() for stream in streams), return_exceptions=True)

    def update_options(
        self,
        *,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
    ) -> None:
        """Change Cartesia STT options, for open streams too.

        Args:
            language: The language the user speaks. ``ink-2`` only supports English; on
                ``ink-whisper`` a new language reconnects open streams.
        """
        if not utils.is_given(language):
            return
        self._language = LanguageCode(language)
        self._warn_on_unexpected_args(language=self._language)
        for stream in self._streams:
            stream.set_language(self._language)

    def _target(self, path: str, params: list[tuple[str, str]]) -> stt.WebSocketTarget:
        params = [
            ("model", self._model),
            ("sample_rate", str(self._sample_rate)),
            ("encoding", AUDIO_ENCODING),
            *params,
        ]
        return stt.WebSocketTarget(
            f"{self._ws_base_url}{path}?{urlencode(params)}",
            headers={
                API_VERSION_HEADER: API_VERSION,
                API_AUTH_HEADER: self._api_key,
                "User-Agent": USER_AGENT,
            },
        )

    def _warn_on_unexpected_args(self, language: LanguageCode | None) -> None:
        """Warn about a language ink-2 does not support yet: not an error, since the API may
        support it in the future."""
        if self._turns and language and language.language != "en":
            logger.warning(
                f'Cartesia STT model="{self._model}" currently only supports English. '
                f"You provided {language}, which may produce unexpected results."
            )


def _status_error(
    message: _TurnsError | _ManualError, status_code: int | None, detail: str | None
) -> APIStatusError:
    # an HTTP-style status: a 5xx is retried, a rejected request is not
    return APIStatusError(
        "Cartesia STT returned an error",
        status_code=status_code or 500,
        body=message,
        detail=detail,
    )


# -- turns endpoint (/stt/turns/websocket)


class _Connected(TypedDict):
    """Fires once when the WebSocket connection is established; audio need not wait for it."""

    type: Literal["connected"]
    request_id: NotRequired[str]


class _Turn(TypedDict):
    """A user turn's progress.

    ``turn.start`` and ``turn.resume`` carry no transcript; ``turn.update``, ``turn.eager_end``
    and ``turn.end`` carry the cumulative text of the turn so far, not a delta.
    """

    transcript: NotRequired[str]
    request_id: NotRequired[str]


class _TurnStart(_Turn):
    type: Literal["turn.start"]


class _TurnUpdate(_Turn):
    type: Literal["turn.update"]


class _TurnEagerEnd(_Turn):
    type: Literal["turn.eager_end"]


class _TurnResume(_Turn):
    type: Literal["turn.resume"]


class _TurnEnd(_Turn):
    type: Literal["turn.end"]


class _TurnsError(TypedDict):
    type: Literal["error"]
    status_code: NotRequired[int]
    title: NotRequired[str]
    message: NotRequired[str]
    request_id: NotRequired[str]


_TurnsMessage = Annotated[
    _Connected | _TurnStart | _TurnUpdate | _TurnEagerEnd | _TurnResume | _TurnEnd | _TurnsError,
    Field(discriminator="type"),
]


class _Close(TypedDict):
    type: Literal["close"]


class _TurnsStream(stt.WebSocketStream[_TurnsMessage, _Close]):
    """A recognition on the turns endpoint: the model detects the turns and finalizes each one
    at its end, so a flush sends nothing.

    See https://docs.cartesia.ai/api-reference/stt/turns/websocket
    """

    completion: ClassVar[stt.Completion] = "peer_closed"

    # turn.start and turn.end are Cartesia's own turn boundaries
    provider_vad = True

    def __init__(self, *, stt: STT, conn_options: APIConnectOptions, language: LanguageCode):
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._sample_rate)
        self.frame_ms = stt._audio_chunk_duration_ms
        self._owner = stt
        self._language = language
        self._request_id = ""
        # cumulative transcript for the current turn; used to re-emit on turn.resume
        self._current_transcript = ""

    def set_language(self, language: LanguageCode) -> None:
        # only the label: ink-2 takes no language
        self._language = language

    async def connect(self, conn_options: APIConnectOptions) -> stt.WebSocketTarget:
        self._current_transcript = ""
        owner = self._owner
        params: list[tuple[str, str]] = []
        if utils.is_given(owner._turn_start_threshold):
            params.append(("turn_start_threshold", str(owner._turn_start_threshold)))
        if utils.is_given(owner._turn_eager_end_threshold):
            params.append(("turn_eager_end_threshold", str(owner._turn_eager_end_threshold)))
        if utils.is_given(owner._turn_end_threshold):
            params.append(("turn_end_threshold", str(owner._turn_end_threshold)))
        if utils.is_given(owner._turn_end_timeout_ms):
            params.append(("turn_end_timeout_ms", str(owner._turn_end_timeout_ms)))
        if utils.is_given(owner._keyterm):
            # keyterm repeats
            params.extend(("keyterm", term) for term in owner._keyterm)
        return owner._target("/stt/turns/websocket", params)

    def encode(self, item: rtc.AudioFrame | stt.Flush | stt.EndOfAudio) -> bytes | _Close | None:
        match item:
            case rtc.AudioFrame():
                return bytes(item.data)
            case stt.Flush():
                if not self.input.closed:
                    logger.warning(
                        "Cartesia STT stream.flush() was ignored: the turns endpoint "
                        "finalizes each turn itself."
                    )
                return None
            case stt.EndOfAudio():
                return _Close(type="close")
            case _:
                assert_never(item)

    def _transcript(
        self, text: str, *, final: bool = False, preflight: bool = False
    ) -> stt.Transcript:
        return stt.Transcript(
            text, self._language, final=final, preflight=preflight, request_id=self._request_id
        )

    def on_message(
        self,
        message: _Connected
        | _TurnStart
        | _TurnUpdate
        | _TurnEagerEnd
        | _TurnResume
        | _TurnEnd
        | _TurnsError,
    ) -> None:
        if request_id := message.get("request_id"):
            self._request_id = request_id

        if message["type"] == "connected":
            return
        if message["type"] == "error":
            raise _status_error(
                message, message.get("status_code"), message.get("message") or message.get("title")
            )
        if message["type"] == "turn.start":
            if not self.speaking:
                self._current_transcript = ""
                self.report(stt.SpeechStarted())
            return
        if not self.speaking:
            return
        transcript = message.get("transcript", "")

        if message["type"] == "turn.update":
            # only an interim that changed: an unchanged one would cancel the preflight
            if transcript and transcript != self._current_transcript:
                self._current_transcript = transcript
                self.report(self._transcript(transcript))
        elif message["type"] == "turn.eager_end":
            transcript = transcript or self._current_transcript
            if transcript:
                self._current_transcript = transcript
                self.report(self._transcript(transcript, preflight=True))
        elif message["type"] == "turn.resume":
            # no transcript: the last one again as an interim, so the preflight is cancelled
            if self._current_transcript:
                self.report(self._transcript(self._current_transcript))
        elif message["type"] == "turn.end":
            transcript = transcript or self._current_transcript
            self._current_transcript = ""
            self.report(self._transcript(transcript, final=True))
            self.report(stt.SpeechEnded())
        else:
            assert_never(message)


# -- manual endpoint (/stt/websocket)


class _Word(TypedDict):
    """Word-level timestamp, in seconds."""

    word: NotRequired[str]
    start: NotRequired[float]
    end: NotRequired[float]


class _Transcript(TypedDict):
    """A transcript chunk: a delta from the last final one, not the cumulative text."""

    type: Literal["transcript"]
    is_final: bool
    text: str
    request_id: NotRequired[str]
    duration: NotRequired[float]
    words: NotRequired[list[_Word]]


class _FlushDone(TypedDict):
    """Acknowledgment of ``finalize``."""

    type: Literal["flush_done"]
    request_id: NotRequired[str]


class _Done(TypedDict):
    """Acknowledgment of ``close``."""

    type: Literal["done"]
    request_id: NotRequired[str]


class _ManualError(TypedDict):
    type: Literal["error"]
    code: NotRequired[int]
    message: NotRequired[str]
    request_id: NotRequired[str]


_ManualMessage = Annotated[
    _Transcript | _FlushDone | _Done | _ManualError, Field(discriminator="type")
]


class _ManualStream(stt.WebSocketStream[_ManualMessage, str]):
    """A recognition on the manual endpoint: a flush finalizes the transcript so far.

    See https://docs.cartesia.ai/api-reference/stt/stt
    """

    completion: ClassVar[stt.Completion] = "session_finished"

    def __init__(
        self, *, stt: STT, conn_options: APIConnectOptions, language: LanguageCode | None
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._sample_rate)
        self.frame_ms = stt._audio_chunk_duration_ms
        self._owner = stt
        self._language = language
        self._request_id = ""
        self._last_speech_end_time = 0.0

    def set_language(self, language: LanguageCode) -> None:
        # the endpoint takes the ISO 639-1 code: only a new one needs a new connection
        changed = self._language is None or self._language.language != language.language
        self._language = language
        if changed:
            self.reconnect("language changed")

    async def connect(self, conn_options: APIConnectOptions) -> stt.WebSocketTarget:
        self._last_speech_end_time = 0.0
        params = [] if self._language is None else [("language", self._language.language)]
        return self._owner._target("/stt/websocket", params)

    def encode(self, item: rtc.AudioFrame | stt.Flush | stt.EndOfAudio) -> bytes | str:
        match item:
            case rtc.AudioFrame():
                return bytes(item.data)
            case stt.Flush():
                return "finalize"
            case stt.EndOfAudio():
                return "close"
            case _:
                assert_never(item)

    def on_message(self, message: _Transcript | _FlushDone | _Done | _ManualError) -> None:
        if request_id := message.get("request_id"):
            self._request_id = request_id

        if message["type"] == "flush_done":
            logger.debug("Received flush_done acknowledgment from Cartesia STT")
        elif message["type"] == "done":
            # the answer to "close": every transcript came before it
            self.report(stt.SessionFinished())
        elif message["type"] == "error":
            raise _status_error(message, message.get("code"), message.get("message"))
        elif message["type"] == "transcript":
            self._report_transcript(message)
        else:
            assert_never(message)

    def _report_transcript(self, message: _Transcript) -> None:
        offset = self.start_time_offset
        # word timestamps are often within the audio window, so we track time separately
        if self._last_speech_end_time == 0.0:
            self._last_speech_end_time = offset
        start_time = self._last_speech_end_time
        end_time = start_time + message.get("duration", 0)
        self._last_speech_end_time = end_time

        self.report(
            stt.Transcript(
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
        )
