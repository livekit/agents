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

import os
from dataclasses import dataclass, replace
from typing import Annotated, ClassVar, Literal, NotRequired, TypedDict, assert_never
from urllib.parse import urlencode

from aiobotocore.session import AioSession
from botocore.auth import SigV4QueryAuth
from botocore.awsrequest import AWSRequest
from botocore.credentials import ReadOnlyCredentials
from pydantic import Field, JsonValue, TypeAdapter, ValidationError

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIError,
    LanguageCode,
    stt,
)
from livekit.agents.types import NOT_GIVEN, ErrorKind, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TranscriptChunk

from . import _eventstream
from .utils import DEFAULT_REGION, resolve_session

# a presigned URL is valid for at most 5 minutes; each connection signs a fresh one
_URL_EXPIRES = 300
_AUDIO_HEADERS = {
    ":content-type": "application/octet-stream",
    ":event-type": "AudioEvent",
    ":message-type": "event",
}
_EXCEPTION_KINDS: dict[str, ErrorKind] = {
    "BadRequestException": "invalid_request",
    "ConflictException": "invalid_request",
    "LimitExceededException": "rate_limit",
    "InternalFailureException": "server",
    "ServiceUnavailableException": "server",
}


@dataclass
class Credentials:
    access_key_id: str
    secret_access_key: str
    session_token: str | None = None


@dataclass
class STTOptions:
    sample_rate: int
    language: LanguageCode | None
    encoding: str
    vocabulary_name: NotGivenOr[str]
    session_id: NotGivenOr[str]
    vocab_filter_method: NotGivenOr[str]
    vocab_filter_name: NotGivenOr[str]
    show_speaker_label: NotGivenOr[bool]
    enable_channel_identification: NotGivenOr[bool]
    number_of_channels: NotGivenOr[int]
    enable_partial_results_stabilization: NotGivenOr[bool]
    partial_results_stability: NotGivenOr[str]
    language_model_name: NotGivenOr[str]
    region: str
    identify_language: bool
    identify_multiple_languages: bool
    language_options: NotGivenOr[str]
    preferred_language: NotGivenOr[str]
    vocabulary_names: NotGivenOr[str]
    vocabulary_filter_names: NotGivenOr[str]
    base_url: str

    def query(self) -> dict[str, str]:
        """The StartStreamTranscription parameters of the websocket URL."""
        params: dict[str, object] = {
            "media-encoding": self.encoding,
            "sample-rate": self.sample_rate,
            "vocabulary-name": self.vocabulary_name,
            "session-id": self.session_id,
            "vocabulary-filter-method": self.vocab_filter_method,
            "vocabulary-filter-name": self.vocab_filter_name,
            "show-speaker-label": self.show_speaker_label,
            "enable-channel-identification": self.enable_channel_identification,
            "number-of-channels": self.number_of_channels,
            "enable-partial-results-stabilization": self.enable_partial_results_stabilization,
            "partial-results-stability": self.partial_results_stability,
            "language-model-name": self.language_model_name,
        }
        # automatic language identification excludes language-code
        if self.identify_language or self.identify_multiple_languages:
            mode = "identify-language" if self.identify_language else "identify-multiple-languages"
            params[mode] = True
            params["language-options"] = self.language_options
            params["preferred-language"] = self.preferred_language
            params["vocabulary-names"] = self.vocabulary_names
            params["vocabulary-filter-names"] = self.vocabulary_filter_names
        elif self.language:
            params["language-code"] = self.language
        return {
            name: str(value).lower() if isinstance(value, bool) else str(value)
            for name, value in params.items()
            if is_given(value)
        }


class STT(stt.STT):
    def __init__(
        self,
        *,
        region: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: int = 24000,
        language: str | None = "en-US",
        encoding: str = "pcm",
        vocabulary_name: NotGivenOr[str] = NOT_GIVEN,
        session_id: NotGivenOr[str] = NOT_GIVEN,
        vocab_filter_method: NotGivenOr[str] = NOT_GIVEN,
        vocab_filter_name: NotGivenOr[str] = NOT_GIVEN,
        show_speaker_label: NotGivenOr[bool] = NOT_GIVEN,
        enable_channel_identification: NotGivenOr[bool] = NOT_GIVEN,
        number_of_channels: NotGivenOr[int] = NOT_GIVEN,
        enable_partial_results_stabilization: NotGivenOr[bool] = NOT_GIVEN,
        partial_results_stability: NotGivenOr[str] = NOT_GIVEN,
        language_model_name: NotGivenOr[str] = NOT_GIVEN,
        credentials: NotGivenOr[Credentials] = NOT_GIVEN,
        identify_language: bool = False,
        identify_multiple_languages: bool = False,
        language_options: NotGivenOr[str] = NOT_GIVEN,
        preferred_language: NotGivenOr[str] = NOT_GIVEN,
        vocabulary_names: NotGivenOr[str] = NOT_GIVEN,
        vocabulary_filter_names: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        session: AioSession | None = None,
    ):
        """Amazon Transcribe streaming over its websocket API.

        Args:
            credentials: Signs the requests; the default AWS credential chain when not given.
            base_url: The websocket endpoint; the region's Transcribe endpoint when not given.
            session: The aiobotocore session whose credential chain signs the requests.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript="word",
                offline_recognize=False,
            )
        )

        if not is_given(region):
            region = os.getenv("AWS_REGION") or DEFAULT_REGION

        if identify_language and identify_multiple_languages:
            raise ValueError(
                "identify_language and identify_multiple_languages are mutually exclusive. "
                "Set only one to True."
            )

        # When auto language detection is enabled, language_code must not be set
        lang: LanguageCode | None = None
        if not identify_language and not identify_multiple_languages:
            lang = LanguageCode(language) if language else LanguageCode("en-US")

        self._config = STTOptions(
            language=lang,
            sample_rate=sample_rate,
            encoding=encoding,
            vocabulary_name=vocabulary_name,
            session_id=session_id,
            vocab_filter_method=vocab_filter_method,
            vocab_filter_name=vocab_filter_name,
            show_speaker_label=show_speaker_label,
            enable_channel_identification=enable_channel_identification,
            number_of_channels=number_of_channels,
            enable_partial_results_stabilization=enable_partial_results_stabilization,
            partial_results_stability=partial_results_stability,
            language_model_name=language_model_name,
            region=region,
            identify_language=identify_language,
            identify_multiple_languages=identify_multiple_languages,
            language_options=language_options,
            preferred_language=preferred_language,
            vocabulary_names=vocabulary_names,
            vocabulary_filter_names=vocabulary_filter_names,
            base_url=base_url
            if is_given(base_url)
            else f"wss://transcribestreaming.{region}.amazonaws.com:8443",
        )
        self._credentials = credentials if is_given(credentials) else None
        self._session = resolve_session(session)

    @property
    def model(self) -> str:
        return (
            self._config.language_model_name
            if is_given(self._config.language_model_name)
            else "unknown"
        )

    @property
    def provider(self) -> str:
        return "amazon_transcribe"

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Amazon Transcribe does not support single frame recognition")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        opts = replace(self._config)
        if is_given(language):
            opts.language = LanguageCode(language)
        return SpeechStream(stt=self, opts=opts, conn_options=conn_options)

    async def _signing_credentials(self) -> ReadOnlyCredentials:
        if (explicit := self._credentials) is not None:
            return ReadOnlyCredentials(
                explicit.access_key_id, explicit.secret_access_key, explicit.session_token
            )
        found = await self._session.get_credentials()
        if found is None:
            raise APIError("no AWS credentials found to sign Transcribe requests", kind="auth")
        return await found.get_frozen_credentials()


class _Item(TypedDict):
    Content: str
    StartTime: float
    EndTime: float
    Confidence: NotRequired[float]


class _Alternative(TypedDict):
    Transcript: str
    Items: NotRequired[list[_Item]]


class _Result(TypedDict):
    StartTime: float
    EndTime: float
    IsPartial: bool
    Alternatives: list[_Alternative]
    LanguageCode: NotRequired[str]


class _Transcript(TypedDict):
    Results: list[_Result]


class _TranscriptEvent(TypedDict):
    type: Literal["TranscriptEvent"]
    Transcript: _Transcript


class _Exception(TypedDict):
    """An exception or error message: ``name`` is its type, ``Message`` its text."""

    type: Literal["exception"]
    name: str
    Message: NotRequired[str]


_Event = Annotated[_TranscriptEvent | _Exception, Field(discriminator="type")]
_EVENT: TypeAdapter[_TranscriptEvent | _Exception] = TypeAdapter(_Event)
_JSON = TypeAdapter(dict[str, JsonValue])


class SpeechStream(stt.WebSocketStream[_Event, bytes]):
    completion: ClassVar[stt.Completion] = "peer_closed"
    frame_ms = 50
    # Transcribe fails a stream that received no audio for 15 s: a new one opens before
    audio_idle = 14.0

    def __init__(self, *, stt: STT, opts: STTOptions, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._transcribe = stt
        self._opts = opts

    async def connect(self, conn_options: APIConnectOptions) -> stt.WebSocketTarget:
        credentials = await self._transcribe._signing_credentials()
        request = AWSRequest(
            method="GET",
            url=f"{self._opts.base_url}/stream-transcription-websocket?"
            + urlencode(self._opts.query()),
        )
        SigV4QueryAuth(credentials, "transcribe", self._opts.region, _URL_EXPIRES).add_auth(request)
        assert request.url is not None, "signing keeps the URL"
        return stt.WebSocketTarget(request.url, request_id_header="x-amzn-request-id")

    def encode(self, item: rtc.AudioFrame | stt.Flush | stt.EndOfAudio) -> bytes | None:
        match item:
            case rtc.AudioFrame():
                return _eventstream.encode(_AUDIO_HEADERS, bytes(item.data))
            case stt.Flush():
                return None
            case stt.EndOfAudio():
                # an empty audio event ends the stream: Transcribe sends its last results and
                # closes
                return _eventstream.encode(_AUDIO_HEADERS, b"")
            case _:
                assert_never(item)

    def on_audio_idle(self) -> None:
        self.reconnect("no audio before Transcribe's idle timeout")

    def decode(self, frame: bytes | str) -> _TranscriptEvent | _Exception | None:
        if isinstance(frame, str):
            raise stt.protocol_error(self, f"an unexpected text frame: {frame}")
        try:
            headers, payload = _eventstream.decode(frame)
            match headers.get(":message-type"), headers.get(":event-type"):
                case "event", "TranscriptEvent":
                    body = {**_JSON.validate_json(payload), "type": "TranscriptEvent"}
                case "exception", _:
                    # the exception's text is "Message" in its JSON payload
                    body = {
                        **_JSON.validate_json(payload or b"{}"),
                        "type": "exception",
                        "name": headers.get(":exception-type", ""),
                    }
                case "error", _:
                    body = {
                        "type": "exception",
                        "name": headers.get(":error-code", ""),
                        "Message": headers.get(":error-message", ""),
                    }
                case message_type, event_type:
                    # an event Transcribe added since: skipped, named once
                    stt.log_skipped("Amazon Transcribe", f"{message_type}/{event_type}")
                    return None
            return _EVENT.validate_python(body)
        except (ValueError, ValidationError) as e:
            raise stt.protocol_error(self, f"{e}; frame: {frame!r}") from e

    def on_message(self, message: _TranscriptEvent | _Exception) -> None:
        if message["type"] == "exception":
            raise APIError(
                f"Amazon Transcribe {message['name']}",
                kind=_EXCEPTION_KINDS.get(message["name"], "server"),
                detail=message.get("Message") or None,
            )
        elif message["type"] == "TranscriptEvent":
            for result in message["Transcript"]["Results"]:
                if result["Alternatives"]:
                    self.report(self._transcript(result))
        else:
            self._skip_unknown_message(message)

    def _transcript(self, result: _Result) -> stt.Transcript:
        alternative = result["Alternatives"][0]
        items = alternative.get("Items", [])
        offset = self.start_time_offset
        detected = result.get("LanguageCode")
        identifying = self._opts.identify_language or self._opts.identify_multiple_languages
        return stt.Transcript(
            alternative["Transcript"],
            LanguageCode(detected or self._opts.language or "en-US"),
            final=not result["IsPartial"],
            start_time=result["StartTime"] + offset,
            end_time=result["EndTime"] + offset,
            confidence=items[0].get("Confidence", 0.0) if items else 0.0,
            source_languages=[LanguageCode(detected)] if identifying and detected else None,
            words=[
                TranscriptChunk(
                    text=item["Content"],
                    start_time=item["StartTime"] + offset,
                    end_time=item["EndTime"] + offset,
                    start_time_offset=offset,
                    confidence=item.get("Confidence", 0.0),
                )
                for item in items
            ]
            or None,
        )
