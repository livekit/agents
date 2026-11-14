# Copyright 2025 LiveKit, Inc.
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
import os
import weakref
from collections.abc import Sequence
from enum import Enum, StrEnum
from typing import Annotated, Any, Literal, NotRequired, TypedDict, TypeVar, assert_never
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from pydantic import Field, JsonValue, TypeAdapter, ValidationError

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    VAD,
    APIConnectOptions,
    APIError,
    LanguageCode,
    VADEventType,
    VADStream,
    stt,
    utils,
)
from livekit.agents.types import (
    NOT_GIVEN,
    ErrorKind,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger
from .version import __version__ as lk_version

_OptionT = TypeVar("_OptionT")


class Model(StrEnum):
    """Models available on the Agent STT service."""

    LINDEN_1 = "linden-1"


DEFAULT_MODEL = Model.LINDEN_1


class AudioEncoding(StrEnum):
    """Audio encodings of the real-time service."""

    PCM_F32LE = "pcm_f32le"
    PCM_S16LE = "pcm_s16le"
    MULAW = "mulaw"


@dataclasses.dataclass
class AdditionalVocabEntry:
    """A word to bias the engine towards, optionally with pronunciation hints."""

    content: str
    sounds_like: list[str] | None = None


@dataclasses.dataclass
class SpeakerIdentifier:
    """Known identifiers of one speaker, used to attribute words to it across sessions."""

    label: str = ""
    speaker_identifiers: list[str] = dataclasses.field(default_factory=list[str])


_SPEAKERS = TypeAdapter(list[SpeakerIdentifier])
_VOCAB_ENTRY = TypeAdapter(AdditionalVocabEntry)

# An `Error` message's type; any other type refused the session.
_ERROR_KINDS: dict[str, ErrorKind] = {
    "not_authorised": "auth",
    "insufficient_funds": "quota",
    "quota_exceeded": "quota",
    "job_error": "server",
    "unknown_error": "server",
    "idle_timeout": "timeout",
    "session_timeout": "timeout",
}

# Endpoint resolution. The default is the EU real-time Agent STT host; the env var
# overrides it (e.g. to target another region or a self-hosted endpoint), and an explicit
# `base_url` argument overrides both. Every Agent STT endpoint ends in `AGENT_URL_PATH`;
# anything else speaks a protocol this plugin cannot read.
DEFAULT_BASE_URL = "wss://eu2.rt.speechmatics.com/v2/agent"
BASE_URL_ENV_VAR = "SPEECHMATICS_RT_URL"
AGENT_URL_PATH = "/v2/agent"

# Audio format we can actually send. The service fixes the sample rate at 16 kHz, and
# LiveKit frames are 16-bit PCM which we forward unconverted, so declaring any other
# encoding would describe the bytes wrongly rather than change them.
SUPPORTED_SAMPLE_RATE = 16000
SUPPORTED_AUDIO_ENCODINGS = (AudioEncoding.PCM_S16LE,)


class TurnDetectionMode(StrEnum):
    """How turn boundaries (end of speech) are detected.

    `VAD`: the STT service runs its own VAD and closes turns itself.

    `EXTERNAL`: turn boundaries are controlled by the caller — the service does not
    endpoint on its own, and the caller drives turns by calling `finalize()` (for
    example from an external VAD).

    The values are the service's `turn_config.turn_detection_mode`.
    """

    VAD = "vad"
    EXTERNAL = "external"


# The mode a caller gets when they ask for none. `EXTERNAL` for parity with the plugin's
# pre-Agent-STT default.
DEFAULT_TURN_DETECTION_MODE = TurnDetectionMode.EXTERNAL


@dataclasses.dataclass
class STTOptions:
    """Configuration parameters for Speechmatics STT service."""

    # Service configuration
    language: LanguageCode = LanguageCode("en")
    output_locale: str | None = None
    domain: str | None = None

    # Endpointing mode
    turn_detection_mode: TurnDetectionMode = DEFAULT_TURN_DETECTION_MODE

    # Output formatting
    speaker_format: str | None = None

    # Speakers
    known_speakers: list[SpeakerIdentifier] = dataclasses.field(
        default_factory=list[SpeakerIdentifier]
    )

    # Custom dictionary
    additional_vocab: list[AdditionalVocabEntry] = dataclasses.field(
        default_factory=list[AdditionalVocabEntry]
    )

    # -------------------
    # Advanced features
    # -------------------

    # Features
    # The resolved model name (operating point). See `_resolve_model`.
    model: str = DEFAULT_MODEL.value
    include_partials: bool | None = None

    # Diarization
    enable_diarization: bool | None = None
    speaker_sensitivity: float | None = None
    max_speakers: int | None = None
    prefer_current_speaker: bool | None = None


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        turn_detection_mode: TurnDetectionMode = DEFAULT_TURN_DETECTION_MODE,
        model: NotGivenOr[Model | str] = NOT_GIVEN,
        domain: NotGivenOr[str] = NOT_GIVEN,
        language: str = "en",
        output_locale: NotGivenOr[str] = NOT_GIVEN,
        include_partials: NotGivenOr[bool] = NOT_GIVEN,
        enable_diarization: NotGivenOr[bool] = NOT_GIVEN,
        additional_vocab: NotGivenOr[list[AdditionalVocabEntry]] = NOT_GIVEN,
        speaker_sensitivity: NotGivenOr[float] = NOT_GIVEN,
        max_speakers: NotGivenOr[int] = NOT_GIVEN,
        speaker_format: NotGivenOr[str] = NOT_GIVEN,
        prefer_current_speaker: NotGivenOr[bool] = NOT_GIVEN,
        known_speakers: NotGivenOr[list[SpeakerIdentifier]] = NOT_GIVEN,
        sample_rate: int = 16000,
        audio_encoding: AudioEncoding = AudioEncoding.PCM_S16LE,
        vad: NotGivenOr[VAD | None] = NOT_GIVEN,
    ):
        """Create a new instance of Speechmatics STT using Agent STT SDK.

        Args:
            api_key: Speechmatics API key. Can be set via `api_key` argument
                or `SPEECHMATICS_API_KEY` environment variable.

            base_url: Custom base URL for the API. Use this to target a specific
                Speechmatics real-time host (e.g. another region or a self-hosted
                endpoint). Can be set via `base_url` argument or `SPEECHMATICS_RT_URL`
                environment variable. Falls back to `DEFAULT_BASE_URL` (the EU real-time
                Agent STT host, `wss://eu2.rt.speechmatics.com/v2/agent`) when neither is
                set.

            turn_detection_mode: How end-of-speech turns are detected. `EXTERNAL` (the
                default) hands turn control to the caller, who drives it via `finalize()`
                — in practice from the `vad` passed below, since LiveKit does not call
                `finalize()` itself. With no `vad` and none loadable, `EXTERNAL` never
                closes a turn, so nothing is finalized. `VAD` instead lets the service
                and close turns itself; pair it with
                `turn_handling={"turn_detection": "stt"}` on the `VoiceSession`, which
                otherwise ignores the end-of-speech events this plugin emits. Defaults to `DEFAULT_TURN_DETECTION_MODE`.

            model: The transcription model to use, e.g. `"linden-1"`. A name agent-STT
                does not accept falls back to the default model with a warning. Defaults
                to the SDK's default model.

            domain: Domain to use. Optional.

            language: Language code for the STT model. Defaults to `en`.

            output_locale: Output locale for the STT model, e.g. `en-GB`. Optional.

            include_partials: Whether the service sends partial segments, emitted as
                interim transcripts. Defaults to True.

            enable_diarization: Attribute words to distinct speakers. Defaults to True.

            additional_vocab: Vocabulary entries that bias the model toward specific
                words. Defaults to [].

            speaker_sensitivity: Diarization sensitivity between 0.0 and 1.0; higher
                values separate similar-sounding speakers more aggressively. Optional.

            max_speakers: Upper bound on the number of distinct speakers (2–100).
                Optional.

            speaker_format: Formatter for speaker-attributed text, with `text` and
                `speaker_id` placeholders, e.g. `@{speaker_id}: {text}`. Defaults to the raw
                transcript.

            prefer_current_speaker: Bias closely-spaced words toward the same speaker.
                Optional.

            known_speakers: Known speaker identifiers, used to attribute words to the
                same speakers across sessions. Defaults to [].

            sample_rate: Audio sample rate in Hz. The service accepts 16000 only, so any
                other value raises a `ValueError`. Defaults to 16000.

            audio_encoding: Audio encoding format. LiveKit frames are 16-bit PCM and are
                forwarded unconverted, so `AudioEncoding.PCM_S16LE` is the only value that
                describes what is actually sent; anything else raises a `ValueError`.
                Defaults to `AudioEncoding.PCM_S16LE`.

            vad: External Voice Activity Detector, used only in `EXTERNAL` turn-detection
                mode where its end-of-speech drives `finalize()`. Ignored in `VAD` mode.
                When omitted in `EXTERNAL` mode, `livekit-plugins-silero` is loaded if it
                is installed, so a bare `STT()` still closes turns. Pass `vad=None` to opt
                out and drive `finalize()` yourself. Defaults to NOT_GIVEN.
        """

        # `EXTERNAL` is the default and the service does not endpoint in it, so a bare
        # `STT()` has nothing to close a turn with. Load a local VAD to drive `finalize()`,
        # as the plugin did before Agent STT, so zero-config still transcribes. An explicit
        # `vad=None` opts out: that caller drives `finalize()` themselves.
        if turn_detection_mode == TurnDetectionMode.EXTERNAL and not is_given(vad):
            vad = _load_default_vad()

        self._vad = vad if is_given(vad) else None

        # EXTERNAL mode needs something to close turns. The service does not endpoint on its own,
        # and LiveKit never calls STT.finalize() itself, so with no `vad` turns close only if the
        # caller drives finalize() by hand — otherwise nothing is ever finalized. Warn loudly
        # instead of silently producing no transcripts.
        if turn_detection_mode == TurnDetectionMode.EXTERNAL and self._vad is None:
            logger.warning(
                "Speechmatics STT is in EXTERNAL turn-detection mode with no `vad`: the service "
                "will not endpoint on its own and LiveKit does not call finalize() for you, so "
                "turns close only if you call STT.finalize() yourself. Pass a `vad` to drive "
                "finalize() from end-of-speech, or use turn_detection_mode=VAD for service-side "
                "endpointing."
            )

        # Set STT options
        def _set(value: NotGivenOr[_OptionT]) -> _OptionT | None:
            return value if is_given(value) else None

        # Create STT options from parameters
        opts = STTOptions(
            language=LanguageCode(language),
            output_locale=_set(output_locale),
            domain=_set(domain),
            turn_detection_mode=turn_detection_mode,
            speaker_format=_set(speaker_format),
            known_speakers=_set(known_speakers) or [],
            additional_vocab=_normalize_additional_vocab(_set(additional_vocab) or []),
            model=_resolve_model(model),
            include_partials=_set(include_partials),
            enable_diarization=enable_diarization if is_given(enable_diarization) else True,
            speaker_sensitivity=_set(speaker_sensitivity),
            max_speakers=_set(max_speakers),
            prefer_current_speaker=_set(prefer_current_speaker),
        )

        # Capabilities mirror the resolved options. Unset partials means the service default,
        # which is on.
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=opts.include_partials is not False,
                diarization=opts.enable_diarization is not False,
                aligned_transcript="chunk",
                offline_recognize=False,
            ),
        )

        self._stt_options = opts

        # Validate config options
        errors = self._validate_stt_options()
        if errors:
            raise ValueError("Invalid STT options: " + ", ".join(errors))

        # Set API key
        self._api_key: str = api_key if is_given(api_key) else os.getenv("SPEECHMATICS_API_KEY", "")

        # Set base URL
        self._base_url: str = _resolve_base_url(base_url)

        # Validate API key and base URL
        if not self._api_key:
            raise ValueError("Missing Speechmatics API key")
        if not self._base_url:
            raise ValueError("Missing Speechmatics base URL")

        # Set audio parameters. Neither is a free choice: an unsupported rate is rejected
        # by the service, and an encoding we do not actually send would mislabel the audio.
        if sample_rate != SUPPORTED_SAMPLE_RATE:
            raise ValueError(f"sample_rate must be {SUPPORTED_SAMPLE_RATE}")
        if audio_encoding not in SUPPORTED_AUDIO_ENCODINGS:
            supported = ", ".join(e.value for e in SUPPORTED_AUDIO_ENCODINGS)
            raise ValueError(f"audio_encoding must be one of: {supported}")

        self._sample_rate = sample_rate
        self._audio_encoding = audio_encoding

        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def provider(self) -> str:
        return "Speechmatics"

    @property
    def model(self) -> str:
        return self._stt_options.model

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Not implemented")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        """Create a new SpeechStream."""
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            config=self._prepare_config(language),
            turn_detection_mode=self._stt_options.turn_detection_mode,
            vad_instance=self._vad,
            api_key=self._api_key,
            url=_session_url(self._base_url),
            audio_encoding=self._audio_encoding,
            sample_rate=self._sample_rate,
            options=self._stt_options,
        )
        self._streams.add(stream)
        return stream

    def _validate_stt_options(self) -> list[str]:
        """Validate options in STTOptions."""
        errors: list[str] = []
        opts = self._stt_options

        # server rejects speaker counts outside 2–100
        if opts.max_speakers is not None and not (1 < opts.max_speakers <= 100):
            errors.append("max_speakers must be between 2 and 100")

        # diarization sensitivity range enforced by the engine
        if opts.speaker_sensitivity is not None and not (0.0 <= opts.speaker_sensitivity <= 1.0):
            errors.append("speaker_sensitivity must be between 0.0 and 1.0")

        return errors

    def _prepare_config(self, language: NotGivenOr[str] = NOT_GIVEN) -> _TranscriptionConfig:
        """The session's `transcription_config`: only the fields agent-STT accepts, and none
        left unset. Turn detection is the `turn_config` beside it."""
        opts = self._stt_options
        config = _TranscriptionConfig(
            language=language if is_given(language) else opts.language, model=opts.model
        )
        if opts.enable_diarization:
            config["diarization"] = "speaker"
        if (diarization := _build_diarization_config(opts)) is not None:
            config["speaker_diarization_config"] = diarization
        if opts.additional_vocab:
            config["additional_vocab"] = [_vocab_entry(e) for e in opts.additional_vocab]
        if opts.output_locale is not None:
            config["output_locale"] = opts.output_locale
        if opts.domain is not None:
            config["domain"] = opts.domain
        if opts.include_partials is not None:
            config["enable_partials"] = opts.include_partials
        return config

    def finalize(self) -> None:
        """Force the current turn to end, flushing buffered words as final segments.

        Only takes effect in `EXTERNAL` turn-detection mode; in `VAD` mode the
        service endpoints on its own and this is a no-op.
        """

        for stream in list(self._streams):
            stream.finalize()

    async def get_speaker_ids(
        self,
    ) -> list[SpeakerIdentifier] | list[list[SpeakerIdentifier]]:
        """Get the list of speakers from the current STT session.

        If diarization is enabled, then this will use the GET_SPEAKERS message
        to retrieve the list of speakers for the current session. This should
        be used once speakers have said at least 5 words to improve the results.

        Returns:
            list[SpeakerIdentifier]: List of speakers in the session.
        """

        results = [await stream.get_speaker_ids() for stream in list(self._streams)]
        if len(results) == 1:
            return results[0]
        return results


class _TranscriptionConfig(TypedDict):
    language: str
    model: str
    diarization: NotRequired[str]
    speaker_diarization_config: NotRequired[dict[str, Any]]
    additional_vocab: NotRequired[list[dict[str, Any]]]
    output_locale: NotRequired[str]
    domain: NotRequired[str]
    enable_partials: NotRequired[bool]


class _RecognitionStarted(TypedDict):
    message: Literal["RecognitionStarted"]
    id: NotRequired[str]


class _Logged(TypedDict):
    message: Literal["Info", "Warning"]


class _Error(TypedDict):
    message: Literal["Error"]
    type: NotRequired[str]
    reason: NotRequired[str]


class _SegmentBody(TypedDict):
    transcript: NotRequired[str]
    speaker: NotRequired[str | None]


class _SegmentTiming(TypedDict):
    start_time: NotRequired[float]
    end_time: NotRequired[float]


class _Segment(TypedDict):
    message: Literal["AddSegment", "AddPartialSegment"]
    segment: NotRequired[_SegmentBody | None]
    metadata: NotRequired[_SegmentTiming | None]


class _Turn(TypedDict):
    message: Literal["StartOfTurn", "EndOfTurn", "EndOfTranscript"]


class _SpeakersResult(TypedDict):
    message: Literal["SpeakersResult"]
    # read on its own, so a bad reply fails only get_speaker_ids()
    speakers: NotRequired[JsonValue]


_Message = Annotated[
    _RecognitionStarted | _Logged | _Error | _Segment | _Turn | _SpeakersResult,
    Field(discriminator="message"),
]


class _StartRecognition(TypedDict):
    message: Literal["StartRecognition"]
    audio_format: dict[str, Any]
    transcription_config: _TranscriptionConfig
    turn_config: dict[str, str]


class _EndOfStream(TypedDict):
    message: Literal["EndOfStream"]
    last_seq_no: int


class _ForceEndOfUtterance(TypedDict):
    message: Literal["ForceEndOfUtterance"]
    timestamp: float


class _GetSpeakers(TypedDict):
    message: Literal["GetSpeakers"]


_Command = _StartRecognition | _EndOfStream | _ForceEndOfUtterance | _GetSpeakers


class SpeechStream(stt.WebSocketStream[_Message, _Command]):
    frame_ms = 100
    # audio waits for RecognitionStarted
    awaits_ready = True
    # StartOfTurn and EndOfTurn are the service's own turns
    provider_vad = True

    def __init__(
        self,
        stt: STT,
        conn_options: APIConnectOptions,
        config: _TranscriptionConfig,
        turn_detection_mode: TurnDetectionMode,
        vad_instance: VAD | None = None,
        *,
        api_key: str,
        url: str,
        audio_encoding: AudioEncoding,
        sample_rate: int,
        options: STTOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=sample_rate)
        self._api_key = api_key
        self._url = url
        self._audio_encoding = audio_encoding
        self._sample_rate = sample_rate
        self._opts = options
        self._config = config
        self._turn_detection_mode = turn_detection_mode
        self._vad = vad_instance
        self._vad_stream: VADStream | None = None
        self._session_id: str | None = None
        # audio chunks and bytes sent on this connection: EndOfStream names the last chunk,
        # ForceEndOfUtterance the audio time it ends at
        self._seq_no = 0
        self._audio_bytes = 0
        self._speakers: asyncio.Future[list[SpeakerIdentifier]] | None = None

    @property
    def session_id(self) -> str | None:
        """The service-assigned session id, set once `RecognitionStarted` arrives."""
        return self._session_id

    def finalize(self) -> None:
        """End the current turn when external endpointing controls this stream."""
        if self._turn_detection_mode == TurnDetectionMode.EXTERNAL:
            seconds = self._audio_bytes / (2 * self._sample_rate)
            self.send_now(_ForceEndOfUtterance(message="ForceEndOfUtterance", timestamp=seconds))

    async def get_speaker_ids(self) -> list[SpeakerIdentifier]:
        """Request the current session's speaker identifiers."""
        if self._session_id is None:
            logger.warning("Speechmatics stream is not connected")
            return []
        if "diarization" not in self._config:
            logger.warning("Diarization is not enabled in this Speechmatics stream")
            return []

        speakers: asyncio.Future[list[SpeakerIdentifier]]
        speakers = self._speakers = asyncio.get_running_loop().create_future()
        self.send_now(_GetSpeakers(message="GetSpeakers"))
        try:
            async with asyncio.timeout(5.0):
                return await speakers
        except TimeoutError:
            logger.warning("GetSpeakers timed out")
            return []
        finally:
            self._speakers = None

    async def _run(self) -> None:
        if self._vad is None or self._turn_detection_mode != TurnDetectionMode.EXTERNAL:
            await super()._run()
            return
        vad_stream = self._vad_stream = self._vad.stream()
        try:
            async with utils.aio.TaskSet() as tasks:
                await tasks.first_completed(super()._run(), self._finalize_on_vad(vad_stream))
        finally:
            self._vad_stream = None
            await vad_stream.aclose()

    async def _finalize_on_vad(self, vad_stream: VADStream) -> None:
        """Call `finalize()` whenever the external VAD reports end of speech."""
        async for ev in vad_stream:
            if ev.type == VADEventType.END_OF_SPEECH:
                self.finalize()

    async def connect(self, conn_options: APIConnectOptions) -> stt.WebSocketTarget:
        self._session_id = None
        self._seq_no = self._audio_bytes = 0
        return stt.WebSocketTarget(
            self._url,
            headers={"Authorization": f"Bearer {self._api_key}", "X-Request-Id": utils.shortuuid()},
        )

    def setup(self) -> _StartRecognition:
        return _StartRecognition(
            message="StartRecognition",
            audio_format={
                "type": "raw",
                "encoding": self._audio_encoding.value,
                "sample_rate": self._sample_rate,
            },
            transcription_config=self._config,
            turn_config={"turn_detection_mode": self._turn_detection_mode.value},
        )

    def encode(self, item: rtc.AudioFrame | stt.Flush | stt.EndOfAudio) -> bytes | _Command | None:
        match item:
            case rtc.AudioFrame():
                if self._vad_stream is not None:
                    self._vad_stream.push_frame(item)
                self._seq_no += 1
                self._audio_bytes += len(item.data) * 2
                return bytes(item.data)
            case stt.Flush():
                return None
            case stt.EndOfAudio():
                return _EndOfStream(message="EndOfStream", last_seq_no=self._seq_no)
            case _:
                assert_never(item)

    def on_message(
        self,
        message: _RecognitionStarted | _Logged | _Error | _Segment | _Turn | _SpeakersResult,
    ) -> None:
        if message["message"] == "RecognitionStarted":
            self._session_id = message.get("id")
            logger.info("Speechmatics session started", extra={"session_id": self._session_id})
            self.report(stt.Ready())
        elif message["message"] == "Info":
            logger.info("received Info message", extra={"lk.pii.message": message})
        elif message["message"] == "Warning":
            logger.warning("received Warning message", extra={"lk.pii.message": message})
        elif message["message"] == "Error":
            # the session is over
            error_type = message.get("type", "error")
            raise APIError(
                f"Speechmatics returned an error [{error_type}]",
                kind=_ERROR_KINDS.get(error_type, "invalid_request"),
                body=message,
                detail=message.get("reason", "unknown"),
            )
        elif message["message"] == "StartOfTurn":
            self.report(stt.SpeechStarted())
        elif message["message"] == "EndOfTurn":
            self.report(stt.SpeechEnded())
        elif message["message"] == "EndOfTranscript":
            self.report(stt.SessionFinished())
        elif message["message"] == "SpeakersResult":
            self._speakers_result(message)
        elif message["message"] == "AddSegment" or message["message"] == "AddPartialSegment":
            transcript = self._transcript(message)
            self.report(transcript)
            # EXTERNAL mode sends no turns: each final segment ends its speech
            if transcript.final and self._turn_detection_mode == TurnDetectionMode.EXTERNAL:
                self.report(stt.SpeechEnded())
        else:
            assert_never(message)

    def _transcript(self, message: _Segment) -> stt.Transcript:
        segment = message.get("segment") or _SegmentBody()
        timing = message.get("metadata") or _SegmentTiming()
        # segments the service did not attribute are labelled UU
        speaker_id = segment.get("speaker") or "UU"
        text = (self._opts.speaker_format or "{text}").format(
            speaker_id=speaker_id, text=segment.get("transcript", "")
        )
        offset = self.start_time_offset
        return stt.Transcript(
            text,
            # the stream's own language, which honors a stream(language=...) override
            LanguageCode(self._config["language"]),
            final=message["message"] == "AddSegment",
            speaker_id=speaker_id,
            start_time=timing.get("start_time", 0.0) + offset,
            end_time=timing.get("end_time", 0.0) + offset,
        )

    def _speakers_result(self, message: _SpeakersResult) -> None:
        if (speakers := self._speakers) is None or speakers.done():
            return
        try:
            speakers.set_result(_SPEAKERS.validate_python(message.get("speakers", [])))
        except ValidationError as e:
            error = APIError(
                "Speechmatics returned invalid speaker identifiers",
                kind="protocol",
                detail=str(e),
            )
            error.__cause__ = e
            speakers.set_exception(error)


def _vocab_entry(entry: AdditionalVocabEntry) -> dict[str, Any]:
    if entry.sounds_like is None:
        return {"content": entry.content}
    return {"content": entry.content, "sounds_like": entry.sounds_like}


def _session_url(base_url: str) -> str:
    """The Agent STT endpoint: `/agent` appended unless present, the app named as `sm-app`."""
    parsed = urlparse(base_url)
    segments = [segment for segment in parsed.path.split("/") if segment]
    if "agent" not in segments:
        segments.append("agent")
    params = parse_qs(parsed.query, keep_blank_values=True)
    params["sm-app"] = [f"livekit/{lk_version}"]
    return urlunparse(
        parsed._replace(path="/" + "/".join(segments), query=urlencode(params, doseq=True))
    )


def _build_diarization_config(opts: STTOptions) -> dict[str, Any] | None:
    """Build the wire `speaker_diarization_config` from the diarization options.

    Returns `None` when diarization is off or no diarization knob was set, so an empty
    config is never sent. Only the fields the caller actually set are included.
    """
    if not opts.enable_diarization:
        return None

    config: dict[str, Any] = {}
    if opts.max_speakers is not None:
        config["max_speakers"] = opts.max_speakers
    if opts.speaker_sensitivity is not None:
        config["speaker_sensitivity"] = opts.speaker_sensitivity
    if opts.prefer_current_speaker is not None:
        config["prefer_current_speaker"] = opts.prefer_current_speaker
    if opts.known_speakers:
        config["speakers"] = [dataclasses.asdict(s) for s in opts.known_speakers]
    return config or None


def _load_default_vad() -> VAD | None:
    """Load a local VAD to drive `finalize()` in `EXTERNAL` mode, or `None` if unavailable.

    The pre-Agent-STT plugin loaded Silero here and raised `ImportError` when it was
    missing, which made the plugin unusable wherever the optional package was not
    installed. Returning `None` instead lets construction succeed and leaves the caller
    with the `EXTERNAL`-without-a-`vad` warning, which says what to do about it.

    Deliberately not `livekit.agents.inference.VAD`: that one is bundled, so it would load
    for everybody and run a second VAD over the same audio as the one `VoiceSession` fills
    in for itself, with independent thresholds.
    """
    try:
        from livekit.plugins.silero import VAD as SileroVAD
    except ImportError:
        return None

    return SileroVAD.load()


def _normalize_additional_vocab(entries: Sequence[object]) -> list[AdditionalVocabEntry]:
    """Accept vocab entries from either SDK, returning entries agent-STT can serialize.

    The pre-Agent-STT plugin took `speechmatics.voice`'s pydantic `AdditionalVocabEntry`.
    The Agent STT config carries an unrecognised object straight into
    `transcription_config.additional_vocab`, where the JSON encode then refuses it — so
    without this the old class fails when the session starts rather than at construction.
    Both classes have the same two fields, so the value survives the copy.

    Entries are duck-typed on `content` instead of matched against the voice class, which
    keeps the voice SDK off the runtime path. Dicts are decoded into the same SDK record
    before entering the session config.

    No warning: the two classes hold the same two fields, so the entry is copied without
    losing anything and the word reaches the engine exactly as asked. There is nothing for
    the caller to do differently.

    Raises:
        ValueError: if an entry is neither a mapping nor an object with `content`.
    """
    normalized: list[AdditionalVocabEntry] = []

    for entry in entries:
        if isinstance(entry, AdditionalVocabEntry):
            normalized.append(entry)
            continue
        if isinstance(entry, dict):
            normalized.append(_VOCAB_ENTRY.validate_python(entry))
            continue

        content = getattr(entry, "content", None)
        if content is None:
            raise ValueError(
                f"`additional_vocab` entry {entry!r} is not an `AdditionalVocabEntry`: no "
                "`content` to read"
            )

        normalized.append(
            _VOCAB_ENTRY.validate_python(
                {"content": content, "sounds_like": getattr(entry, "sounds_like", None)}
            )
        )

    return normalized


def _resolve_base_url(base_url: NotGivenOr[str]) -> str:
    """Resolve the STT endpoint URL.

    Precedence (highest first):
        1. the explicit `base_url` argument
        2. the ``SPEECHMATICS_RT_URL`` environment variable
        3. ``DEFAULT_BASE_URL``

    Warns if the result is not an Agent STT endpoint. The check covers all three sources,
    which matters most for the environment variable: it survives an upgrade untouched, so
    a value left pointing at `/v2` silently keeps the caller on the old protocol.
    """
    resolved = base_url if is_given(base_url) else os.getenv(BASE_URL_ENV_VAR, DEFAULT_BASE_URL)

    if not urlparse(resolved).path.rstrip("/").endswith(AGENT_URL_PATH):
        logger.warning(
            f"{resolved!r} is not an Agent STT endpoint: those end in {AGENT_URL_PATH!r}. This "
            "plugin only reads the Agent STT protocol, so the session will connect and then "
            "produce no transcripts — if your transcripts are missing, this is why."
        )

    return resolved


def _model_name(value: Model | str) -> str:
    """Normalize a model value to its wire string.

    Accepts an enum member (`Model`, or any future `str` enum) or a plain string,
    and returns the string the service reads.
    """
    return value.value if isinstance(value, Enum) else str(value)


def _resolve_model(model: NotGivenOr[Model | str]) -> str:
    """Resolve `model` to a model name agent-STT accepts; the SDK's default when not given.

    This never raises. A name agent-STT does not accept — a typo, or one of the `enhanced` /
    `standard` operating points the plugin took before Agent STT — falls back to the default
    model with a warning, so old code keeps transcribing instead of dying at construction.

    Returns:
        The resolved model name as a string.
    """
    resolved = _model_name(model) if is_given(model) else None
    if not resolved:
        return DEFAULT_MODEL.value

    supported = [m.value for m in Model]
    if resolved in supported:
        return resolved

    logger.warning(
        f"{resolved!r} is not a model agent-STT accepts ({', '.join(supported)}); using "
        f"{DEFAULT_MODEL.value!r} instead. The old `enhanced` and `standard` operating points "
        f"belong to the previous service and have no Agent STT equivalent."
    )
    return DEFAULT_MODEL.value
