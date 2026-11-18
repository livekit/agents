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

"""Speech-to-Text for Sarvam.ai: realtime streaming (``saaras:v3-realtime``) and batch
recognition (``saaras:v3``, ``saaras:v4``), the model picking the endpoint."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, replace
from typing import ClassVar, Literal, NotRequired, TypedDict, assert_never
from urllib.parse import urlencode

import numpy as np
from pydantic import JsonValue

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIError,
    APIStatusError,
    LanguageCode,
    __version__ as livekit_version,
    http,
    stt,
)
from livekit.agents.types import NOT_GIVEN, ErrorKind, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import TranscriptionData

USER_AGENT = f"Livekit/{livekit_version} Python/{platform.python_version()}"

SARVAM_STT_BASE_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_STT_REALTIME_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
REALTIME_MODEL = "saaras:v3-realtime"

SarvamSTTModels = Literal["saaras:v3-realtime", "saaras:v4", "saaras:v3"]
SarvamSTTModes = Literal["transcribe", "translate", "verbatim", "translit", "codemix"]
RealtimeStreamType = Literal["fast", "balanced", "simulated"]
RealtimeEndpointing = Literal["vad", "manual"]
RealtimeEncoding = Literal["linear16", "linear32", "mulaw", "alaw"]

_MODES = frozenset({"transcribe", "translate", "verbatim", "translit", "codemix"})
_STREAM_TYPES = frozenset({"fast", "balanced", "simulated"})
_ENDPOINTING = frozenset({"vad", "manual"})
_ENCODINGS = frozenset({"linear16", "linear32", "mulaw", "alaw"})
_REALTIME_SAMPLE_RATES = frozenset({8000, 16000})
_BATCH_LANGUAGES = frozenset(
    {
        "unknown",
        "hi-IN",
        "bn-IN",
        "kn-IN",
        "ml-IN",
        "mr-IN",
        "od-IN",
        "pa-IN",
        "ta-IN",
        "te-IN",
        "en-IN",
        "gu-IN",
        "as-IN",
        "ur-IN",
        "ne-IN",
        "kok-IN",
        "ks-IN",
        "sd-IN",
        "sa-IN",
        "sat-IN",
        "mni-IN",
        "brx-IN",
        "mai-IN",
        "doi-IN",
    }
)
_REALTIME_LANGUAGES = (_BATCH_LANGUAGES - {"unknown", "od-IN"}) | {"or-IN", "auto"}


@dataclass
class _Options:
    model: str
    language: str
    mode: str
    sample_rate: int
    prompt: str | None
    stream_type: str
    endpointing: str
    encoding: str
    return_timestamps: bool
    vad_sot_threshold: float | None
    vad_min_speech_ms: int | None
    vad_min_silence_ms: int | None
    vad_prefix_padding_ms: int | None

    @property
    def realtime(self) -> bool:
        return self.model == REALTIME_MODEL

    def validate(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"mode must be one of {', '.join(sorted(_MODES))}")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")
        if not self.realtime:
            if self.language not in _BATCH_LANGUAGES:
                raise ValueError(f"language {self.language} is not supported for {self.model}")
            return
        if self.language not in _REALTIME_LANGUAGES:
            raise ValueError(f"language {self.language} is not supported for {self.model}")
        if self.stream_type not in _STREAM_TYPES:
            raise ValueError(f"stream_type must be one of {', '.join(sorted(_STREAM_TYPES))}")
        if self.endpointing not in _ENDPOINTING:
            raise ValueError(f"endpointing must be one of {', '.join(sorted(_ENDPOINTING))}")
        if self.encoding not in _ENCODINGS:
            raise ValueError(f"encoding must be one of {', '.join(sorted(_ENCODINGS))}")
        if self.sample_rate not in _REALTIME_SAMPLE_RATES:
            raise ValueError(f"sample_rate must be 8000 or 16000 for {self.model}")
        if self.vad_sot_threshold is not None and not 0.0 <= self.vad_sot_threshold <= 1.0:
            raise ValueError("vad_sot_threshold must be between 0.0 and 1.0")
        for name, value in (
            ("vad_min_speech_ms", self.vad_min_speech_ms),
            ("vad_min_silence_ms", self.vad_min_silence_ms),
            ("vad_prefix_padding_ms", self.vad_prefix_padding_ms),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be greater than or equal to 0")


class STT(stt.STT):
    """Sarvam.ai speech-to-text, specialized in Indian languages.

    ``saaras:v3-realtime`` streams over Sarvam's realtime WebSocket; ``saaras:v3`` and
    ``saaras:v4`` recognize whole utterances over HTTP (a session streams them through its VAD).
    """

    def __init__(
        self,
        *,
        model: SarvamSTTModels | str = REALTIME_MODEL,
        language: str = "en-IN",
        mode: SarvamSTTModes | str = "transcribe",
        prompt: str | None = None,
        sample_rate: int = 16000,
        stream_type: NotGivenOr[RealtimeStreamType | str] = NOT_GIVEN,
        endpointing: NotGivenOr[RealtimeEndpointing | str] = NOT_GIVEN,
        encoding: NotGivenOr[RealtimeEncoding | str] = NOT_GIVEN,
        return_timestamps: NotGivenOr[bool] = NOT_GIVEN,
        vad_sot_threshold: NotGivenOr[float] = NOT_GIVEN,
        vad_min_speech_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_min_silence_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_prefix_padding_ms: NotGivenOr[int] = NOT_GIVEN,
        api_key: str | None = None,
        base_url: str | None = None,
        http_session: http.HttpSession | None = None,
    ) -> None:
        """Create a Sarvam STT.

        Args:
            model: ``saaras:v3-realtime`` streams; ``saaras:v3`` and ``saaras:v4`` recognize
                whole utterances.
            language: BCP-47 language code; ``auto`` (realtime) identifies it.
            mode: Task applied to the transcript: ``transcribe``, ``translate``, ``verbatim``,
                ``translit`` or ``codemix``.
            prompt: Context or terminology hint (realtime only).
            sample_rate: Audio sample rate in Hz; ``8000`` or ``16000`` for realtime.
            stream_type: Realtime latency profile: ``fast``, ``balanced`` or ``simulated``.
            endpointing: Realtime turn boundaries: ``vad`` (Sarvam's), or ``manual`` (the
                stream's flushes).
            encoding: Realtime wire encoding: ``linear16``, ``linear32``, ``mulaw`` or ``alaw``.
            return_timestamps: Realtime finals carry their start and end times.
            vad_sot_threshold: Realtime VAD activation threshold (``vad`` endpointing).
            vad_min_speech_ms: Realtime minimum speech duration in ms (``vad`` endpointing).
            vad_min_silence_ms: Realtime end-of-turn silence in ms (``vad`` endpointing).
            vad_prefix_padding_ms: Realtime audio kept before speech onset in ms.
            api_key: Sarvam API key; ``SARVAM_API_KEY`` when not given.
            base_url: The model's endpoint; Sarvam's for its family when not given.
            http_session: The session batch recognition posts with.

        Raises:
            ValueError: No API key, a realtime-only option with a batch model, or an option
                the endpoint rejects.
        """
        realtime_only = {
            "prompt": prompt is not None,
            "stream_type": is_given(stream_type),
            "endpointing": is_given(endpointing),
            "encoding": is_given(encoding),
            "return_timestamps": is_given(return_timestamps),
            "vad_sot_threshold": is_given(vad_sot_threshold),
            "vad_min_speech_ms": is_given(vad_min_speech_ms),
            "vad_min_silence_ms": is_given(vad_min_silence_ms),
            "vad_prefix_padding_ms": is_given(vad_prefix_padding_ms),
        }
        if model != REALTIME_MODEL and (given := [n for n, set_ in realtime_only.items() if set_]):
            raise ValueError(f"{', '.join(given)} only apply to {REALTIME_MODEL}, not {model}")
        self._opts = _Options(
            model=model,
            language=language,
            mode=mode,
            sample_rate=sample_rate,
            prompt=prompt,
            stream_type=stream_type if is_given(stream_type) else "balanced",
            endpointing=endpointing if is_given(endpointing) else "vad",
            encoding=encoding if is_given(encoding) else "linear16",
            return_timestamps=return_timestamps if is_given(return_timestamps) else False,
            vad_sot_threshold=vad_sot_threshold if is_given(vad_sot_threshold) else None,
            vad_min_speech_ms=vad_min_speech_ms if is_given(vad_min_speech_ms) else None,
            vad_min_silence_ms=vad_min_silence_ms if is_given(vad_min_silence_ms) else None,
            vad_prefix_padding_ms=vad_prefix_padding_ms
            if is_given(vad_prefix_padding_ms)
            else None,
        )
        self._opts.validate()
        realtime = self._opts.realtime
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=realtime,
                interim_results=realtime,
                aligned_transcript=False,
                offline_recognize=not realtime,
            )
        )
        resolved_key = api_key or os.environ.get("SARVAM_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Sarvam API key is required. "
                "Provide it directly or set SARVAM_API_KEY environment variable."
            )
        self._api_key = resolved_key
        self._base_url = base_url or (SARVAM_STT_REALTIME_URL if realtime else SARVAM_STT_BASE_URL)
        self._session = http_session
        self._streams: set[SpeechStream] = set()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Sarvam"

    async def recognize(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        # each request is billed: never sent twice
        return await super().recognize(
            buffer, language=language, conn_options=replace(conn_options, max_retry=0)
        )

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        if self._opts.realtime:
            raise NotImplementedError(f"{REALTIME_MODEL} only streams")
        opts = replace(self._opts, language=language) if is_given(language) else self._opts
        opts.validate()
        audio = rtc.combine_audio_frames(buffer)
        form_data = http.FormData()
        form_data.add_field(
            "file", audio.to_wav_bytes(), filename="audio.wav", content_type="audio/wav"
        )
        form_data.add_field("language_code", opts.language)
        form_data.add_field("model", opts.model)
        form_data.add_field("mode", opts.mode)
        session = self._session or http.http_session()
        async with session.post(
            self._base_url,
            data=form_data,
            headers={"api-subscription-key": self._api_key, "User-Agent": USER_AGENT},
            conn_options=conn_options,
        ) as res:
            if res.status != 200:
                error_text = await res.text()
                raise APIStatusError(
                    "Sarvam STT request failed",
                    status_code=res.status,
                    body=error_text,
                    detail=error_text,
                )
            response = TranscriptionData.model_validate(await res.json())

        start_time = end_time = 0.0
        if (timestamps := response.timestamps) is not None:
            if timestamps.start_time_seconds:
                start_time = timestamps.start_time_seconds[0]
            if timestamps.end_time_seconds:
                end_time = timestamps.end_time_seconds[-1]
        if start_time == 0.0 and end_time == 0.0:
            end_time = audio.duration
        probability = response.language_probability
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=response.request_id or "",
            alternatives=[
                stt.SpeechData(
                    language=LanguageCode(response.language_code or opts.language),
                    text=response.transcript,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=1.0 if probability is None else probability,
                )
            ],
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        """A realtime recognition. It connects once: Sarvam bills each connection, so a failure
        is never retried."""
        if not self._opts.realtime:
            raise NotImplementedError(
                f"{self._opts.model} recognizes whole utterances: stream it through "
                "stt.StreamAdapter"
            )
        opts = replace(self._opts, language=language) if is_given(language) else self._opts
        opts.validate()
        stream_class = _VadStream if opts.endpointing == "vad" else _ManualStream
        stream = stream_class(
            stt=self,
            opts=replace(opts),
            conn_options=replace(conn_options, max_retry=0),
            api_key=self._api_key,
            base_url=self._base_url,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        mode: NotGivenOr[SarvamSTTModes | str] = NOT_GIVEN,
        stream_type: NotGivenOr[RealtimeStreamType | str] = NOT_GIVEN,
        prompt: NotGivenOr[str | None] = NOT_GIVEN,
        vad_sot_threshold: NotGivenOr[float | None] = NOT_GIVEN,
        vad_min_speech_ms: NotGivenOr[int | None] = NOT_GIVEN,
        vad_min_silence_ms: NotGivenOr[int | None] = NOT_GIVEN,
    ) -> None:
        """Update the options every stream applies from now on: open realtime streams get them
        as Sarvam's ``config.update``. Options fixed by the connection (sample rate, encoding,
        endpointing, timestamps, prefix padding) are set at construction."""
        opts = replace(
            self._opts,
            language=language if is_given(language) else self._opts.language,
            mode=mode if is_given(mode) else self._opts.mode,
            stream_type=stream_type if is_given(stream_type) else self._opts.stream_type,
            prompt=prompt if is_given(prompt) else self._opts.prompt,
            vad_sot_threshold=vad_sot_threshold
            if is_given(vad_sot_threshold)
            else self._opts.vad_sot_threshold,
            vad_min_speech_ms=vad_min_speech_ms
            if is_given(vad_min_speech_ms)
            else self._opts.vad_min_speech_ms,
            vad_min_silence_ms=vad_min_silence_ms
            if is_given(vad_min_silence_ms)
            else self._opts.vad_min_silence_ms,
        )
        opts.validate()
        self._opts = opts
        for stream in self._streams:
            stream.update_options(
                language=language,
                mode=mode,
                stream_type=stream_type,
                prompt=prompt,
                vad_sot_threshold=vad_sot_threshold,
                vad_min_speech_ms=vad_min_speech_ms,
                vad_min_silence_ms=vad_min_silence_ms,
            )

    async def aclose(self) -> None:
        """Close every stream this instance created."""
        streams = list(self._streams)
        for stream in streams:
            await stream.aclose()
        self._streams.difference_update(streams)


def _encode_pcm(encoding: str, pcm: bytes) -> bytes:
    """Little-endian signed 16-bit PCM in Sarvam's realtime wire encoding."""
    if encoding == "linear16":
        return pcm
    # NumPy's Buffer/ArrayLike overloads contain Unknown.
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.int32)  # pyright: ignore[reportUnknownMemberType]
    if encoding == "linear32":
        return (samples.astype(np.int64) * (1 << 16)).astype("<i4").tobytes()
    if encoding == "mulaw":
        # ITU-T G.711 mu-law
        magnitude = np.minimum(np.abs(samples), 32635) + 0x84
        # NumPy's Buffer/ArrayLike overloads contain Unknown.
        exponent = np.clip(np.floor(np.log2(magnitude)).astype(np.int32) - 7, 0, 7)  # pyright: ignore[reportUnknownMemberType]
        mantissa = (magnitude >> (exponent + 3)) & 0x0F
        # NumPy's Buffer/ArrayLike overloads contain Unknown.
        sign = np.where(samples < 0, 0x80, 0)  # pyright: ignore[reportUnknownMemberType]
        return bytes((~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8).tobytes())
    # ITU-T G.711 A-law
    magnitude = np.minimum(np.abs(samples), 32767)
    # NumPy's Buffer/ArrayLike overloads contain Unknown.
    exponent = np.clip(np.floor(np.log2(np.maximum(magnitude, 1))).astype(np.int32) - 7, 0, 7)  # pyright: ignore[reportUnknownMemberType]
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    # NumPy's Buffer/ArrayLike overloads contain Unknown.
    encoded = np.where(magnitude < 256, magnitude >> 4, (exponent << 4) | mantissa)  # pyright: ignore[reportUnknownMemberType]
    # NumPy's Buffer/ArrayLike overloads contain Unknown.
    xor_mask = np.where(samples >= 0, 0xD5, 0x55)  # pyright: ignore[reportUnknownMemberType]
    return bytes((encoded ^ xor_mask).astype(np.uint8).tobytes())


class _SessionBegin(TypedDict):
    event: Literal["session.begin"]
    request_id: NotRequired[str]


class _SpeechStart(TypedDict):
    event: Literal["vad.speech_start"]


class _SpeechEnd(TypedDict):
    event: Literal["vad.speech_end"]


class _TranscriptFields(TypedDict):
    text: NotRequired[str | None]
    language: NotRequired[str | None]
    confidence: NotRequired[float | None]
    language_confidence: NotRequired[float | None]
    utterance_idx: NotRequired[int | None]
    start_s: NotRequired[float | None]
    end_s: NotRequired[float | None]


class _PartialTranscript(_TranscriptFields):
    event: Literal["transcript.partial"]


class _FinalTranscript(_TranscriptFields):
    event: Literal["transcript.final"]


class _SessionEnd(TypedDict):
    event: Literal["session.end"]


class _Error(TypedDict):
    event: Literal["error"]
    code: NotRequired[str]
    message: NotRequired[str]
    is_fatal: NotRequired[bool]
    status_code: NotRequired[int | None]


_Message = (
    _SessionBegin
    | _SpeechStart
    | _SpeechEnd
    | _PartialTranscript
    | _FinalTranscript
    | _SessionEnd
    | _Error
)


class _Control(TypedDict):
    event: Literal["speech_start", "speech_end", "end"]


class _ConfigUpdate(TypedDict, total=False):
    event: Literal["config.update"]
    language_code: str
    stream_type: str
    mode: str
    prompt: str
    threshold: float | None
    min_speech_duration_ms: int | None
    silence_duration_ms: int | None


class SpeechStream(stt.WebSocketStream[_Message, _Control | _ConfigUpdate]):
    """A realtime recognition on Sarvam's ``saaras:v3-realtime`` socket."""

    completion: ClassVar[stt.Completion] = "session_finished"

    frame_ms = 50

    def __init__(
        self,
        *,
        stt: STT,
        opts: _Options,
        conn_options: APIConnectOptions,
        api_key: str,
        base_url: str,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._sarvam = stt
        self._opts = opts
        self._api_key = api_key
        self._base_url = base_url
        self._request_id = ""
        # seconds of audio sent: where a final without timestamps ends
        self._audio_sent = 0.0

    async def aclose(self) -> None:
        await super().aclose()
        self._sarvam._streams.discard(self)

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        mode: NotGivenOr[SarvamSTTModes | str] = NOT_GIVEN,
        stream_type: NotGivenOr[RealtimeStreamType | str] = NOT_GIVEN,
        prompt: NotGivenOr[str | None] = NOT_GIVEN,
        vad_sot_threshold: NotGivenOr[float | None] = NOT_GIVEN,
        vad_min_speech_ms: NotGivenOr[int | None] = NOT_GIVEN,
        vad_min_silence_ms: NotGivenOr[int | None] = NOT_GIVEN,
    ) -> None:
        update = _ConfigUpdate(event="config.update")
        if is_given(language):
            self._opts.language = update["language_code"] = language
        if is_given(mode):
            self._opts.mode = update["mode"] = mode
        if is_given(stream_type):
            self._opts.stream_type = update["stream_type"] = stream_type
        if is_given(prompt):
            self._opts.prompt = prompt
            update["prompt"] = prompt or ""
        if is_given(vad_sot_threshold):
            self._opts.vad_sot_threshold = update["threshold"] = vad_sot_threshold
        if is_given(vad_min_speech_ms):
            self._opts.vad_min_speech_ms = update["min_speech_duration_ms"] = vad_min_speech_ms
        if is_given(vad_min_silence_ms):
            self._opts.vad_min_silence_ms = update["silence_duration_ms"] = vad_min_silence_ms
        if len(update) > 1:
            self.send_now(update)

    async def connect(self, conn_options: APIConnectOptions) -> stt.WebSocketTarget:
        opts = self._opts
        params = {
            "language_code": opts.language,
            "stream_type": opts.stream_type,
            "endpointing": opts.endpointing,
            "encoding": opts.encoding,
            "sample_rate": str(opts.sample_rate),
            "model": opts.model,
            "mode": opts.mode,
            "return_timestamps": str(opts.return_timestamps).lower(),
        }
        if opts.prompt is not None:
            params["prompt"] = opts.prompt
        if opts.endpointing == "vad":
            for key, value in (
                ("threshold", opts.vad_sot_threshold),
                ("min_speech_duration_ms", opts.vad_min_speech_ms),
                ("silence_duration_ms", opts.vad_min_silence_ms),
                ("prefix_padding_ms", opts.vad_prefix_padding_ms),
            ):
                if value is not None:
                    params[key] = str(value)
        return stt.WebSocketTarget(
            f"{self._base_url}?{urlencode(params)}",
            headers={"API-SUBSCRIPTION-KEY": self._api_key, "User-Agent": USER_AGENT},
        )

    def encode(self, item: rtc.AudioFrame | stt.Flush | stt.EndOfAudio) -> bytes | _Control | None:
        match item:
            case rtc.AudioFrame():
                self._audio_sent += item.duration
                return _encode_pcm(self._opts.encoding, bytes(item.data))
            case stt.Flush():
                return None
            case stt.EndOfAudio():
                return _Control(event="end")
            case _:
                assert_never(item)

    def on_message(self, message: _Message) -> None:
        if message["event"] == "session.begin":
            self._request_id = message.get("request_id", "")
        elif message["event"] == "vad.speech_start":
            self.report(stt.SpeechStarted())
        elif message["event"] == "vad.speech_end":
            self.report(stt.SpeechEnded())
        elif message["event"] == "transcript.partial" or message["event"] == "transcript.final":
            self.report(self._transcript(message))
        elif message["event"] == "session.end":
            self.report(stt.SessionFinished())
        elif message["event"] == "error":
            self._error(message)
        else:
            self._skip_unknown_message(message)

    def _transcript(self, message: _PartialTranscript | _FinalTranscript) -> stt.Transcript:
        final = message["event"] == "transcript.final"
        offset = self.start_time_offset
        start, end = message.get("start_s"), message.get("end_s")
        # recognition confidence only: language_confidence scores the language identification
        metadata: dict[str, JsonValue] = {}
        if (utterance := message.get("utterance_idx")) is not None:
            metadata["utterance_idx"] = utterance
        if (language_confidence := message.get("language_confidence")) is not None:
            metadata["language_confidence"] = language_confidence
        confidence = message.get("confidence")
        return stt.Transcript(
            # whitespace carries no words
            (message.get("text") or "").strip(),
            LanguageCode(message.get("language") or self._opts.language),
            final=final,
            request_id=self._request_id,
            start_time=max(start, 0.0) + offset if final and start is not None else 0.0,
            end_time=(max(end, 0.0) if end is not None else self._audio_sent) + offset
            if final
            else 0.0,
            confidence=1.0 if confidence is None else confidence,
            metadata=metadata or None,
        )

    def _error(self, message: _Error) -> None:
        code = message.get("code", "unknown")
        if not message.get("is_fatal", False):
            logger.warning(
                "Sarvam realtime STT reported a non-fatal error",
                extra={"error_code": code, "status_code": message.get("status_code")},
            )
            return
        # an unavailable model may come back; without a status the frame refused the request
        kind: ErrorKind | None = "server" if code == "model_unavailable" else None
        detail = message.get("message", code)
        if (status := message.get("status_code")) is None:
            raise APIError(
                "Sarvam realtime STT error",
                kind=kind or "invalid_request",
                body=message,
                detail=detail,
            )
        raise APIStatusError(
            "Sarvam realtime STT error",
            status_code=status,
            kind=kind,
            request_id=self._request_id or None,
            body=message,
            detail=detail,
        )


class _VadStream(SpeechStream):
    # Sarvam's VAD starts and ends each speech
    provider_vad = True


class _ManualStream(SpeechStream):
    """The stream's flushes delimit the speech Sarvam transcribes."""

    _speaking = False

    def encode(self, item: rtc.AudioFrame | stt.Flush | stt.EndOfAudio) -> bytes | _Control | None:
        if isinstance(item, rtc.AudioFrame) and not self._speaking:
            self._speaking = True
            self.send_now(_Control(event="speech_start"))
        elif isinstance(item, stt.Flush) and self._speaking:
            self._speaking = False
            return _Control(event="speech_end")
        return super().encode(item)
