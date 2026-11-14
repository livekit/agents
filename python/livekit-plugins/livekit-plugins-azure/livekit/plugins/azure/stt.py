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

import json
import os
import struct
import uuid
import weakref
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Annotated, Literal, NotRequired, TypedDict
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import Field, JsonValue, TypeAdapter, ValidationError

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIError,
    LanguageCode,
    http,
    stt,
)
from livekit.agents.types import NOT_GIVEN, ErrorKind, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger

ProfanityOption = Literal["masked", "removed", "raw"]

# what speech.config tells the service about the client, as the JS Speech SDK sends it
_SPEECH_CONFIG = {
    "context": {
        "system": {
            "name": "SpeechSDK",
            "version": "1.44.0",
            "build": "JavaScript",
            "lang": "JavaScript",
        }
    }
}
# speech.phrase statuses that recognized nothing; any other one but Success cancels
_NO_MATCH = {"NoMatch", "InitialSilenceTimeout", "BabbleTimeout", "EndOfDictation"}
_CANCEL_KINDS: dict[str, ErrorKind] = {
    "BadRequest": "invalid_request",
    "Forbidden": "auth",
    "TooManyRequests": "rate_limit",
}
_TICKS = 10**7  # offsets and durations are in 100 ns ticks


@dataclass
class STTOptions:
    speech_key: NotGivenOr[str]
    speech_region: NotGivenOr[str]
    # a speech container's host, ws://host:port
    speech_host: NotGivenOr[str]
    # a Microsoft Entra token
    speech_auth_token: NotGivenOr[str]
    sample_rate: int
    num_channels: int
    segmentation_silence_timeout_ms: NotGivenOr[int]
    segmentation_max_time_ms: NotGivenOr[int]
    segmentation_strategy: NotGivenOr[str]
    language: list[str]
    speech_endpoint: NotGivenOr[str] = NOT_GIVEN
    profanity: NotGivenOr[ProfanityOption] = NOT_GIVEN
    phrase_list: NotGivenOr[list[str] | None] = NOT_GIVEN
    explicit_punctuation: bool = False
    true_text_post_processing: bool = False


class STT(stt.STT):
    def __init__(
        self,
        *,
        speech_key: NotGivenOr[str] = NOT_GIVEN,
        speech_region: NotGivenOr[str] = NOT_GIVEN,
        speech_host: NotGivenOr[str] = NOT_GIVEN,
        speech_auth_token: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: int = 16000,
        num_channels: int = 1,
        segmentation_silence_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        segmentation_max_time_ms: NotGivenOr[int] = NOT_GIVEN,
        segmentation_strategy: NotGivenOr[str] = NOT_GIVEN,
        # Azure handles multiple languages and can auto-detect the language used. It requires the candidate set to be set.  # noqa: E501
        language: NotGivenOr[str | list[str] | None] = NOT_GIVEN,
        profanity: NotGivenOr[ProfanityOption] = NOT_GIVEN,
        speech_endpoint: NotGivenOr[str] = NOT_GIVEN,
        phrase_list: NotGivenOr[list[str] | None] = NOT_GIVEN,
        explicit_punctuation: bool = False,
        true_text_post_processing: bool = False,
    ):
        """
        Create a new instance of Azure STT.

        Either ``speech_host`` or ``speech_key`` and ``speech_region`` or
        ``speech_auth_token`` and ``speech_region`` or
        ``speech_key`` and ``speech_endpoint``
        must be set using arguments.
         Alternatively,  set the ``AZURE_SPEECH_HOST``, ``AZURE_SPEECH_KEY``
        and ``AZURE_SPEECH_REGION`` environmental variables, respectively.
        ``speech_auth_token`` must be set using the arguments as it's an ephemeral token.

        Args:
            profanity: How profanity is returned: masked, removed or raw.
            phrase_list: List of words or phrases to boost recognition accuracy.
                        Azure will give higher priority to these phrases during recognition.
            explicit_punctuation: Controls punctuation behavior. If True, enables explicit punctuation mode
                        where punctuation marks are added explicitly. If False (default), uses Azure's
                        default punctuation behavior.
            true_text_post_processing: Enables Azure "TrueText" post-processing in the recognition result.
        """  # noqa: E501

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript="chunk",
                offline_recognize=False,
            )
        )
        if not language or not is_given(language):
            language = ["en-US"]

        if isinstance(language, str):
            language = [LanguageCode(language)]
        else:
            language = [LanguageCode(lg) for lg in language]

        if not is_given(speech_host):
            speech_host = os.environ.get("AZURE_SPEECH_HOST") or NOT_GIVEN

        if not is_given(speech_key):
            speech_key = os.environ.get("AZURE_SPEECH_KEY") or NOT_GIVEN

        if not is_given(speech_region):
            speech_region = os.environ.get("AZURE_SPEECH_REGION") or NOT_GIVEN

        if not (
            is_given(speech_host)
            or (is_given(speech_key) and is_given(speech_region))
            or (is_given(speech_auth_token) and is_given(speech_region))
            or (is_given(speech_key) and is_given(speech_endpoint))
        ):
            raise ValueError(
                "AZURE_SPEECH_HOST or AZURE_SPEECH_KEY and AZURE_SPEECH_REGION or speech_auth_token and AZURE_SPEECH_REGION or AZURE_SPEECH_KEY and speech_endpoint must be set"  # noqa: E501
            )

        if speech_region and speech_endpoint:
            logger.warning("speech_region and speech_endpoint both are set, using speech_endpoint")
            speech_region = NOT_GIVEN

        self._config = STTOptions(
            speech_key=speech_key,
            speech_region=speech_region,
            speech_host=speech_host,
            speech_auth_token=speech_auth_token,
            language=language,
            sample_rate=sample_rate,
            num_channels=num_channels,
            segmentation_silence_timeout_ms=segmentation_silence_timeout_ms,
            segmentation_max_time_ms=segmentation_max_time_ms,
            segmentation_strategy=segmentation_strategy,
            profanity=profanity,
            speech_endpoint=speech_endpoint,
            phrase_list=phrase_list,
            explicit_punctuation=explicit_punctuation,
            true_text_post_processing=true_text_post_processing,
        )
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return "unknown"

    @property
    def provider(self) -> str:
        return "Azure STT"

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Azure STT does not support single frame recognition")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        config = replace(self._config, language=list(self._config.language))
        if is_given(language):
            config.language = [LanguageCode(language)]
        stream = SpeechStream(stt=self, opts=config, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        language: NotGivenOr[list[str] | str] = NOT_GIVEN,
        segmentation_silence_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        segmentation_max_time_ms: NotGivenOr[int] = NOT_GIVEN,
        segmentation_strategy: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(language):
            if isinstance(language, str):
                language = [LanguageCode(language)]
            else:
                language = [LanguageCode(lg) for lg in language]
            self._config.language = language
        if is_given(segmentation_silence_timeout_ms):
            self._config.segmentation_silence_timeout_ms = segmentation_silence_timeout_ms
        if is_given(segmentation_max_time_ms):
            self._config.segmentation_max_time_ms = segmentation_max_time_ms
        if is_given(segmentation_strategy):
            self._config.segmentation_strategy = segmentation_strategy

        for stream in self._streams:
            stream.update_options(
                language=language,
                segmentation_silence_timeout_ms=segmentation_silence_timeout_ms,
                segmentation_max_time_ms=segmentation_max_time_ms,
                segmentation_strategy=segmentation_strategy,
            )


class _Language(TypedDict):
    Language: str


class _Hypothesis(TypedDict):
    path: Literal["speech.hypothesis", "speech.fragment"]
    request_id: str
    Text: str
    Offset: int
    Duration: int
    PrimaryLanguage: NotRequired[_Language]


class _Phrase(TypedDict):
    path: Literal["speech.phrase"]
    request_id: str
    RecognitionStatus: str
    DisplayText: NotRequired[str]
    Offset: NotRequired[int]
    Duration: NotRequired[int]
    PrimaryLanguage: NotRequired[_Language]


class _TurnEnd(TypedDict):
    path: Literal["turn.end"]
    request_id: str


_Message = Annotated[_Hypothesis | _Phrase | _TurnEnd, Field(discriminator="path")]
_MESSAGE: TypeAdapter[_Hypothesis | _Phrase | _TurnEnd] = TypeAdapter(_Message)
_PATHS = {"speech.hypothesis", "speech.fragment", "speech.phrase", "turn.end"}
_BODY = TypeAdapter(dict[str, JsonValue])


class SpeechStream(stt.SocketStream[_Message, str]):
    """Continuous recognition over the Speech service's websocket: each connection sends
    speech.config, then one turn's speech.context and wave header, then the audio; a turn the
    service ends while audio still flows is followed by a new one."""

    frame_ms = 50

    def __init__(self, *, stt: STT, opts: STTOptions, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._request_id = ""
        # the turns this stream ended: a message of one is late
        self._ended_turns: set[str] = set()
        self._audio_ended = False
        # seconds of audio sent on this connection, and where the current turn started
        self._sent = 0.0
        self._turn_start = 0.0

    def update_options(
        self,
        *,
        language: NotGivenOr[list[str]] = NOT_GIVEN,
        segmentation_silence_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        segmentation_max_time_ms: NotGivenOr[int] = NOT_GIVEN,
        segmentation_strategy: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(language):
            self._opts.language = language
        if is_given(segmentation_silence_timeout_ms):
            self._opts.segmentation_silence_timeout_ms = segmentation_silence_timeout_ms
        if is_given(segmentation_max_time_ms):
            self._opts.segmentation_max_time_ms = segmentation_max_time_ms
        if is_given(segmentation_strategy):
            self._opts.segmentation_strategy = segmentation_strategy
        # the options are read at each connection
        self.reconnect("options changed")

    @property
    def _identifies_language(self) -> bool:
        return len(self._opts.language) > 1

    @property
    def _universal(self) -> bool:
        # containers have no /stt/speech/universal/v2 route
        return not is_given(self._opts.speech_host) or is_given(self._opts.speech_endpoint)

    async def connect(self, conn_options: APIConnectOptions) -> http.SocketTarget:
        opts = self._opts
        if is_given(opts.speech_endpoint):
            url = opts.speech_endpoint
            if urlsplit(url).path in ("", "/"):
                url = url.rstrip("/") + "/stt/speech/universal/v2"
        elif is_given(opts.speech_host):
            url = (
                opts.speech_host.rstrip("/")
                + "/speech/recognition/conversation/cognitiveservices/v1"
            )
        else:
            region = str(opts.speech_region)
            suffix = (
                ".azure.cn"
                if region.lower().startswith("china")
                else ".azure.us"
                if region.lower().startswith("usgov")
                else ".microsoft.com"
            )
            url = f"wss://{region}.stt.speech{suffix}/stt/speech/universal/v2"

        parts = urlsplit(url)
        params = dict(parse_qsl(parts.query))
        if not self._identifies_language:
            params.setdefault("language", opts.language[0])
        params.setdefault("format", "simple")
        if self._identifies_language:
            params["lidEnabled"] = "true"
        if is_given(opts.segmentation_silence_timeout_ms):
            params.setdefault(
                "segmentationSilenceTimeoutMs", str(opts.segmentation_silence_timeout_ms)
            )
        if is_given(opts.profanity):
            params.setdefault("profanity", opts.profanity)
        if opts.explicit_punctuation:
            params["punctuation"] = "explicit"
        scheme = {"https": "wss", "http": "ws"}.get(parts.scheme, parts.scheme)

        connection_id = uuid.uuid4().hex.upper()
        headers = {"X-ConnectionId": connection_id}
        if is_given(opts.speech_auth_token):
            headers["Authorization"] = f"Bearer {opts.speech_auth_token}"
        elif is_given(opts.speech_key):
            headers["Ocp-Apim-Subscription-Key"] = opts.speech_key
        return http.SocketTarget(
            urlunsplit((scheme, parts.netloc, parts.path, urlencode(params), "")),
            headers=headers,
            request_id_header="X-RequestId",
        )

    def setup(self) -> list[str | bytes]:
        self._sent = self._turn_start = 0.0
        self._ended_turns.clear()
        self._request_id = uuid.uuid4().hex.upper()
        return [
            _text("speech.config", self._request_id, json.dumps(_SPEECH_CONFIG)),
            *self._turn(),
        ]

    def _turn(self) -> list[str | bytes]:
        """A turn's speech.context and wave header, under the current request id."""
        rate, channels = self._opts.sample_rate, self._opts.num_channels
        wave = struct.pack(
            "<4sI8sIHHIIHH4sI",
            b"RIFF",
            0,
            b"WAVEfmt ",
            16,
            1,
            channels,
            rate,
            rate * channels * 2,
            channels * 2,
            16,
            b"data",
            0,
        )
        return [
            _text("speech.context", self._request_id, json.dumps(self._speech_context())),
            _audio(self._request_id, wave, "audio/x-wav"),
        ]

    def _speech_context(self) -> dict[str, JsonValue]:
        opts = self._opts
        detection: dict[str, JsonValue] = {}
        if self._universal:
            detection["mode"] = "Conversation"
        context: dict[str, JsonValue] = {"phraseDetection": detection}
        if self._identifies_language:
            context["languageId"] = {
                "languages": list(opts.language),
                "mode": "DetectContinuous",
                "onSuccess": {"action": "Recognize"},
                "onUnknown": {"action": "None"},
                "priority": "PrioritizeLatency",
            }
            context["phraseOutput"] = {
                "interimResults": {"resultType": "Auto"},
                "phraseResults": {"resultType": "Always"},
            }
        else:
            detection["language"] = opts.language[0]

        segmentation: dict[str, JsonValue] = {}
        if is_given(opts.segmentation_strategy):
            strategy = opts.segmentation_strategy.lower()
            segmentation["mode"] = {"time": "Custom", "semantic": "Semantic"}.get(
                strategy, "Normal"
            )
        if is_given(opts.segmentation_silence_timeout_ms):
            segmentation["mode"] = "Custom"
            segmentation["segmentationSilenceTimeoutMs"] = opts.segmentation_silence_timeout_ms
        if is_given(opts.segmentation_max_time_ms):
            segmentation["mode"] = "Custom"
            segmentation["segmentationForcedTimeoutMs"] = opts.segmentation_max_time_ms
        if segmentation:
            detection["mode"] = "Conversation"
            detection["conversation"] = {"segmentation": segmentation}

        enrichment: dict[str, JsonValue] = {}
        if is_given(opts.profanity):
            enrichment["profanity"] = opts.profanity.capitalize()
        if opts.true_text_post_processing:
            enrichment["conversation"] = {
                "punctuationMode": "Implicit",
                "disfluencyMode": "Removed",
                "intermediatePunctuationMode": "Implicit",
                "intermediatedisfluencymode": "Removed",
            }
        if enrichment:
            detection["enrichment"] = enrichment

        if is_given(opts.phrase_list) and opts.phrase_list:
            context["dgi"] = {
                "groups": [{"type": "Generic", "items": [{"text": p} for p in opts.phrase_list]}],
                "bias": 1.0,
            }
        return context

    def encode(
        self, item: rtc.AudioFrame | stt.RecognizeStream.Flush | http.EndOfInput
    ) -> http.Encoded[str]:
        if isinstance(item, rtc.AudioFrame):
            self._sent += item.duration
            return _audio(self._request_id, bytes(item.data))
        if isinstance(item, http.EndOfInput):
            # an empty audio message ends the audio: the service sends its last phrases, then
            # turn.end
            self._audio_ended = True
            return _audio(self._request_id, b"")
        return None

    def decode(self, frame: bytes | str) -> _Hypothesis | _Phrase | _TurnEnd | None:
        if isinstance(frame, bytes):
            return None
        head, separator, body = frame.partition("\r\n\r\n")
        if not separator:
            raise http.protocol_error(self, f"a message without a header block: {frame}")
        headers = {
            name.strip().lower(): value.strip()
            for name, _, value in (line.partition(":") for line in head.split("\r\n") if line)
        }
        path = headers.get("path", "").lower()
        if path not in _PATHS:
            return None
        try:
            fields = _BODY.validate_json(body) if body.strip() else {}
            return _MESSAGE.validate_python(
                {**fields, "path": path, "request_id": headers.get("x-requestid", "")}
            )
        except ValidationError as e:
            raise http.protocol_error(self, f"{e}; frame: {frame}") from e

    def on_message(
        self, message: _Hypothesis | _Phrase | _TurnEnd
    ) -> stt.Transcript | http.Done | None:
        if message["request_id"].upper() in self._ended_turns:
            return None
        if message["path"] == "turn.end":
            if self._audio_ended:
                return http.DONE
            # the service ended the turn while audio still flows: the next turn starts here
            self._ended_turns.add(self._request_id)
            self._request_id = uuid.uuid4().hex.upper()
            self._turn_start = self._sent
            for frame in self._turn():
                self.send_command(frame)
            return None
        if message["path"] == "speech.phrase":
            status = message["RecognitionStatus"]
            if status != "Success" and status not in _NO_MATCH:
                raise APIError(
                    f"Azure STT canceled the recognition: {status}",
                    kind=_CANCEL_KINDS.get(status, "server"),
                )
            # a phrase that recognized nothing retracts the interim it follows
            text = message.get("DisplayText", "") if status == "Success" else ""
            return self._transcript(message, text, final=True)
        if not message["Text"].strip():
            return None
        return self._transcript(message, message["Text"], final=False)

    def _transcript(
        self, message: _Hypothesis | _Phrase, text: str, *, final: bool
    ) -> stt.Transcript:
        start = self.start_time_offset + self._turn_start + message.get("Offset", 0) / _TICKS
        detected = message.get("PrimaryLanguage", {"Language": ""})["Language"]
        return stt.Transcript(
            text if text.strip() else "",
            LanguageCode(detected or self._opts.language[0]),
            final=final,
            request_id=self._request_id,
            start_time=start,
            end_time=start + message.get("Duration", 0) / _TICKS,
            confidence=1.0 if final else 0.0,
        )


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _text(path: str, request_id: str, body: str) -> str:
    return (
        f"Path: {path}\r\nX-RequestId: {request_id}\r\nX-Timestamp: {_timestamp()}\r\n"
        f"Content-Type: application/json\r\n\r\n{body}"
    )


def _audio(request_id: str, body: bytes, content_type: str | None = None) -> bytes:
    """An audio message: a 2-byte header length, the header block, then the audio."""
    headers = f"Path: audio\r\nX-RequestId: {request_id}\r\nX-Timestamp: {_timestamp()}\r\n"
    if content_type is not None:
        headers += f"Content-Type: {content_type}\r\n"
    encoded = headers.encode("ascii")
    return len(encoded).to_bytes(2, "big") + encoded + body
