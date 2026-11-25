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
#
# Adapted from the OpenAI STT plugin: Microsoft AI transcription only, with client-owned
# VAD boundaries and explicit commits.

from __future__ import annotations

import asyncio
import base64
import weakref
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, NotRequired, TypedDict, assert_never

from pydantic import Field

from livekit import rtc
from livekit.agents import (
    VAD,
    APIConnectOptions,
    APIError,
    LanguageCode,
    http,
    stt,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr

from ._http import Configuration, HTTPClient, status_error

SAMPLE_RATE = 16000
# the provider's error codes that name an HTTP status
_ERROR_STATUS = {
    "invalid_api_key": 401,
    "rate_limit_exceeded": 429,
    "content_filter": 403,
    "safety_violation": 403,
}


class STT(stt.STT):
    """Native streaming Microsoft AI transcription with explicit client commits.

    See the package README for deployment-specific contract and validation limits.

    Args:
        vad: A VAD whose end of speech commits the audio as one turn (e.g. LiveKit's Silero
            VAD); only its speech, led by the 300 ms before it, is uploaded, so silence is not
            billed. Required explicitly: pass None only when driving flush/end_input yourself
            (every frame is then uploaded). VoiceSession's separate VAD does not commit a native
            STT stream.
        url: Full WebSocket URL, or MICROSOFT_AI_STT_URL. No paths/query parameters
            are added automatically.
        model: Deployment model ID, or MICROSOFT_AI_STT_MODEL. No model is assumed.
        api_key: Credential, or MICROSOFT_AI_STT_API_KEY, sent using auth_header.
        auth_header: Authorization (the default, with Bearer prefix) or api-key
            (raw credential), or MICROSOFT_AI_STT_AUTH_HEADER. No auth fallback occurs.
        headers: Explicit authentication headers instead of api_key/auth_header
            environment lookup. Cannot be combined with either constructor argument.
        language: Optional transcription language hint, or MICROSOFT_AI_STT_LANGUAGE.
        http_session: Optional caller-owned HttpSession.
        env_file: Explicit dotenv file, or MICROSOFT_AI_ENV_FILE. Constructor
            arguments override environment variables, which override this file.
    """

    def __init__(
        self,
        *,
        vad: VAD | None,
        url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        auth_header: Literal["Authorization", "api-key"] | None = None,
        headers: Mapping[str, str] | None = None,
        language: str | None = None,
        http_session: http.HttpSession | None = None,
        env_file: str | Path | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                offline_recognize=False,
                # a turn is transcribed once committed: its final follows the commit
                transcribes_after_speech_ends=True,
            )
        )
        config = Configuration(env_file)
        if language is None:
            language = config.get("MICROSOFT_AI_STT_LANGUAGE") or None
        if language is not None and not language.strip():
            raise ValueError("language must be nonempty when supplied")
        self._model = config.required(model, "MICROSOFT_AI_STT_MODEL")
        self._client = HTTPClient(
            config=config,
            service="STT",
            url=url,
            api_key=api_key,
            headers=headers,
            http_session=http_session,
            auth_header=auth_header,
        )
        self._vad = vad
        self._language = language
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "microsoft_ai"

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Microsoft AI STT supports stream(), not batch recognize()")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        """Open a transcription stream; a flush commits a turn, end_input also drains finals."""
        selected_language = language if utils.is_given(language) else self._language
        if selected_language is not None and not selected_language.strip():
            raise ValueError("language must be nonempty when supplied")
        stream = SpeechStream(stt=self, language=selected_language, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        await asyncio.gather(*(stream.aclose() for stream in list(self._streams)))
        await self._client.aclose()


class _ErrorBody(TypedDict):
    code: NotRequired[str]
    status_code: NotRequired[int]


class _SessionUpdated(TypedDict):
    type: Literal["session.updated"]


class _Committed(TypedDict):
    type: Literal["input_audio_buffer.committed"]
    item_id: str
    event_id: NotRequired[str]


class _Intermediate(TypedDict):
    type: Literal["conversation.item.input_audio_transcription.intermediate"]
    item_id: str
    intermediate: str
    event_id: NotRequired[str]


class _Delta(TypedDict):
    type: Literal["conversation.item.input_audio_transcription.delta"]
    item_id: str
    delta: str
    event_id: NotRequired[str]


class _Completed(TypedDict):
    type: Literal["conversation.item.input_audio_transcription.completed"]
    item_id: str
    transcript: str
    event_id: NotRequired[str]


class _Failed(TypedDict):
    type: Literal["conversation.item.input_audio_transcription.failed"]
    error: NotRequired[_ErrorBody]


class _Error(TypedDict):
    type: Literal["error"]
    error: NotRequired[_ErrorBody]


_Message = Annotated[
    _SessionUpdated | _Committed | _Intermediate | _Delta | _Completed | _Failed | _Error,
    Field(discriminator="type"),
]


class _AudioFormat(TypedDict):
    type: Literal["audio/pcm"]
    rate: int


class _Transcription(TypedDict):
    model: str
    language: NotRequired[str]


class _AudioInput(TypedDict):
    format: _AudioFormat
    transcription: _Transcription
    # null: the client commits each turn
    turn_detection: None
    noise_reduction: None


class _Audio(TypedDict):
    input: _AudioInput


class _Session(TypedDict):
    type: Literal["transcription"]
    audio: _Audio


class _SessionUpdate(TypedDict):
    type: Literal["session.update"]
    session: _Session


class _Append(TypedDict):
    type: Literal["input_audio_buffer.append"]
    audio: str


class _Commit(TypedDict):
    type: Literal["input_audio_buffer.commit"]


_Command = _SessionUpdate | _Append | _Commit


@dataclass
class _Item:
    finalized: str = ""
    hypothesis: str = ""
    last_interim: str = ""


def _provider_error(error: _ErrorBody) -> APIError:
    status = error.get("status_code")
    if status is None:
        status = _ERROR_STATUS.get(error.get("code", ""))
    if status is not None and 400 <= status < 600:
        return status_error("STT", status)
    # no provider message or body in the error: it can echo audio, text, URLs or keys
    return APIError("Microsoft AI STT rejected the transcription request", kind="invalid_request")


class SpeechStream(stt.WebSocketStream[_Message, _Command]):
    """Audio appended in 50 ms frames, each turn committed at a flush or at the VAD's end of
    speech; a turn's final transcript arrives on its item once committed. With a VAD only the
    speech is appended: the service bills the audio it is sent."""

    frame_ms = 50
    truncate_silence = True
    # nothing but the session.update goes until session.updated
    awaits_ready = True
    # a commit ends the turn; the core holds it open until its final
    provider_vad = True

    def __init__(self, *, stt: STT, language: str | None, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE, vad=stt._vad)
        self._owner = stt
        self._language_hint = language
        self._language = LanguageCode(language or "")
        # audio appended since the last commit: an empty commit is an error
        self._uncommitted = False
        self._pending_commits = 0
        self._audio_ended = False
        self._items: dict[str, _Item] = {}
        self._finished_items: deque[str] = deque(maxlen=128)
        self._event_ids: deque[str] = deque(maxlen=256)

    async def connect(self, conn_options: APIConnectOptions) -> stt.WebSocketTarget:
        self._uncommitted = False
        self._pending_commits = 0
        self._items.clear()
        return stt.WebSocketTarget(self._owner._client.url, headers=self._owner._client.headers)

    def setup(self) -> _SessionUpdate:
        transcription = _Transcription(model=self._owner.model)
        if self._language_hint is not None:
            transcription["language"] = self._language_hint
        return _SessionUpdate(
            type="session.update",
            session=_Session(
                type="transcription",
                audio=_Audio(
                    input=_AudioInput(
                        format=_AudioFormat(type="audio/pcm", rate=SAMPLE_RATE),
                        transcription=transcription,
                        turn_detection=None,
                        noise_reduction=None,
                    )
                ),
            ),
        )

    def _commit(self) -> _Commit | None:
        if not self._uncommitted:
            return None
        self._uncommitted = False
        self._pending_commits += 1
        # the committed audio is a turn that owes its final, even before any words of it
        if not self.speaking:
            self.report(stt.SpeechStarted())
        self.report(stt.SpeechEnded())
        return _Commit(type="input_audio_buffer.commit")

    def encode(self, item: rtc.AudioFrame | stt.Flush | stt.EndOfAudio) -> _Command | None:
        match item:
            case rtc.AudioFrame():
                # with a VAD, audio goes only during speech: its first frame starts the turn
                if self._owner._vad is not None and not self._uncommitted:
                    self.report(stt.SpeechStarted())
                self._uncommitted = True
                return _Append(
                    type="input_audio_buffer.append",
                    audio=base64.b64encode(item.data).decode("ascii"),
                )
            case stt.Flush():
                return self._commit()
            case stt.EndOfAudio():
                self._audio_ended = True
                commit = self._commit()
                if commit is None and not self._pending_commits:
                    # every turn was transcribed: nothing more will come
                    self.report(stt.SessionFinished())
                return commit
            case _:
                assert_never(item)

    def on_message(
        self,
        message: _SessionUpdated
        | _Committed
        | _Intermediate
        | _Delta
        | _Completed
        | _Failed
        | _Error,
    ) -> None:
        if message["type"] == "session.updated":
            self.report(stt.Ready())
            return
        if (
            message["type"] == "error"
            or message["type"] == "conversation.item.input_audio_transcription.failed"
        ):
            raise _provider_error(message.get("error", _ErrorBody()))
        # a resent event counts once
        if event_id := message.get("event_id"):
            if event_id in self._event_ids:
                return
            self._event_ids.append(event_id)
        item_id = message["item_id"]
        if item_id in self._finished_items:
            return
        item = self._items.setdefault(item_id, _Item())
        if message["type"] == "input_audio_buffer.committed":
            pass
        elif message["type"] == "conversation.item.input_audio_transcription.intermediate":
            # replaces the revisable hypothesis
            item.hypothesis = message["intermediate"]
            self._interim(item_id, item)
        elif message["type"] == "conversation.item.input_audio_transcription.delta":
            # finalized text: appended, and it clears the hypothesis
            item.finalized += message["delta"]
            item.hypothesis = ""
            self._interim(item_id, item)
        elif message["type"] == "conversation.item.input_audio_transcription.completed":
            self._complete(item_id, message["transcript"])
        else:
            self._skip_unknown_message(message)

    def _interim(self, item_id: str, item: _Item) -> None:
        text = item.finalized + item.hypothesis
        if text != item.last_interim:
            item.last_interim = text
            self.report(stt.Transcript(text, self._language, request_id=item_id))

    def _complete(self, item_id: str, transcript: str) -> None:
        del self._items[item_id]
        self._finished_items.append(item_id)
        # the completion is authoritative: an empty one after an interim retracts it
        self.report(stt.Transcript(transcript, self._language, final=True, request_id=item_id))
        self._pending_commits = max(0, self._pending_commits - 1)
        if self._audio_ended and not self._pending_commits:
            self.report(stt.SessionFinished())
