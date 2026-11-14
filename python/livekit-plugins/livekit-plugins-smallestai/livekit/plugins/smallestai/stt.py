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

"""Smallest AI (Pulse) speech-to-text for LiveKit Agents.

Wire contract of the Pulse streaming API, as this plugin relies on it:

* Audio goes up as raw PCM binary frames; control messages are JSON text frames:
  ``{"type": "finalize"}`` (return everything buffered as a final now, keep the
  socket) and ``{"type": "close_stream"}`` (flush, answer with ``is_last`` and close).
* Transcript messages carry ``transcript`` (the gateway also duplicates it as
  ``transcription``), ``is_final``, ``is_last``, ``from_finalize`` on the answer to a
  finalize, optional ``speech_final``, and ``words`` on finals when word timestamps or
  diarization are on.
* With ``vad_events`` the server sends standalone ``speech_started`` /
  ``speech_ended`` messages from its acoustic VAD. They are the end-of-turn signal:
  a final on its own is not, because the server also cuts finals inside an utterance
  (word-count rollover, blank-token endpoint at a chunk boundary).
* Errors arrive as ``{"error": "..."}``.
"""

from __future__ import annotations

import json
import math
import os
import re
import weakref
from collections import Counter
from dataclasses import dataclass, replace
from typing import Literal, TypedDict
from urllib.parse import urlencode

from pydantic import JsonValue, TypeAdapter

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    http,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr, TranscriptChunk
from livekit.agents.utils import is_given

from .models import STTEncoding, STTModels
from .version import __version__

NUM_CHANNELS = 1
# Base URL for the Smallest AI API.
# Streaming: wss://api.smallest.ai/waves/v1/stt/live?model={model}
# Batch:     https://api.smallest.ai/waves/v1/stt/?model={model}
SMALLEST_STT_BASE_URL = "https://api.smallest.ai/waves/v1"
# Streaming path on the public API. A directly reachable inference server uses "/transcribe".
DEFAULT_STREAM_PATH = "/stt/live"

# Models that support real-time streaming. All others are batch-only and will be
# wrapped with a StreamAdapter by the agent framework automatically.
_STREAMING_MODELS: frozenset[str] = frozenset({"pulse"})

# Any letter or digit in any script; a transcript with none is punctuation only.
_WORD_RE = re.compile(r"\w")


class _Word(TypedDict, total=False):
    word: str
    start: float
    end: float
    confidence: float
    speaker: str | int


class _Message(TypedDict, total=False):
    """Every Pulse message: transcripts carry no ``type``, and a server reached directly reports
    errors as ``{"error": ...}``."""

    type: str
    error: JsonValue
    status: str
    session_id: str
    transcript: str
    transcription: str
    is_final: bool
    is_last: bool
    from_finalize: bool
    speech_final: bool
    words: list[_Word]
    utterances: list[dict[str, JsonValue]]
    language: str
    redacted_entities: list[str]


class _Command(TypedDict):
    type: Literal["finalize", "close_stream", "keepalive"]


_MESSAGE = TypeAdapter(_Message)


@dataclass
class _STTOptions:
    model: STTModels | str
    api_key: str
    language: str  # BCP-47 code, e.g. "en", "hi"; use "multi" for auto-detection
    sample_rate: int
    encoding: STTEncoding | str
    word_timestamps: bool
    diarize: bool
    format: bool  # punctuation/capitalization; streaming only
    itn_normalize: bool  # spoken -> written numbers ("twenty five" -> "25"); streaming only
    numerals: bool | None  # legacy digits post-pass; None = server default
    keywords: list[tuple[str, float]]  # (keyword, intensifier) pairs; streaming only
    sentence_timestamps: bool  # include sentence-level "utterances"; streaming only
    redact_pii: bool
    redact_pci: bool
    endpointing: bool  # server VAD finalizes the open segment on trailing silence
    endpointing_timeout_ms: int | None  # the VAD silence window; None = server default (600)
    eou_timeout_ms: int | None  # blank-token endpoint; None = server default (800)
    vad_events: bool  # speech_started / speech_ended messages -> START/END_OF_SPEECH
    finalize_on_flush: bool  # {"type":"finalize"} when the framework flushes the stream
    finalize_on_words: bool | None  # word-count rollover of long segments; None = server default
    max_words: int | None  # rollover length; None = server default
    keepalive_interval: float | None  # seconds between {"type":"keepalive"} messages; None = off
    base_url: str
    stream_path: str


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: STTModels | str = "pulse",
        language: str = "en",
        sample_rate: int = 16000,
        encoding: STTEncoding | str = "linear16",
        word_timestamps: bool = True,
        diarize: bool = False,
        format: bool = True,
        itn_normalize: bool = False,
        numerals: NotGivenOr[bool] = NOT_GIVEN,
        keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN,
        sentence_timestamps: bool = False,
        redact_pii: bool = False,
        redact_pci: bool = False,
        endpointing: bool = True,
        endpointing_timeout_ms: int | None = 300,
        eou_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_events: bool = True,
        finalize_on_flush: bool = True,
        finalize_on_words: NotGivenOr[bool] = NOT_GIVEN,
        max_words: NotGivenOr[int] = NOT_GIVEN,
        keepalive_interval: NotGivenOr[float | None] = NOT_GIVEN,
        api_key: str | None = None,
        http_session: http.HttpSession | None = None,
        base_url: str = SMALLEST_STT_BASE_URL,
        stream_path: str = DEFAULT_STREAM_PATH,
    ) -> None:
        """Create a new instance of Smallest AI STT.

        Args:
            model: STT model to use. ``"pulse"`` supports streaming and batch
                transcription. Other models are batch-only; ``stream()`` raises
                ``ValueError`` for them.
            language: BCP-47 language code (e.g. "en", "hi", "fr"). Use "multi" for
                automatic language detection.
            sample_rate: Audio sample rate in Hz. Supported: 8000, 16000, 22050,
                24000, 44100, 48000. Defaults to 16000. Telephony audio should be sent
                at its native 8000 Hz rather than upsampled.
            encoding: PCM encoding of the audio stream; "linear16" is the default.
            word_timestamps: Include per-word start/end timestamps and confidence
                scores on final transcripts. Defaults to True.
            diarize: Enable speaker diarization; each word carries a speaker id.
                Defaults to False.
            format: Apply punctuation and capitalization to streaming transcripts.
                Defaults to True.
            itn_normalize: Run the server's inverse text normalization so spoken
                numbers come back as digits ("twenty five" -> "25"). Independent of
                ``format``. Defaults to False.
            numerals: Legacy digits post-pass; leave unset and use ``itn_normalize``.
            keywords: Boost recognition via ``(keyword, intensifier)`` tuples, e.g.
                ``[("NVIDIA", 2.0)]``. Intensifier ~1.0 (mild) to ~5.0 (strong).
            sentence_timestamps: Include sentence-level timing as an ``"utterances"``
                list in ``SpeechData.metadata``. Defaults to False.
            redact_pii: Mask names, addresses and phone numbers with placeholder
                tokens; matches are listed in ``SpeechData.metadata["redacted_entities"]``.
                Reliable for ``language="en"`` and ``"hi"``. Defaults to False.
            redact_pci: Mask card numbers, CVVs, ZIP codes and account numbers.
                Reliable for ``language="en"`` and ``"hi"``. Defaults to False.
            endpointing: Let the server's VAD finalize the open segment once
                ``endpointing_timeout_ms`` of trailing silence has passed, so the
                last words of a turn arrive as a final without waiting for the
                blank-token endpoint at the next decode boundary. Defaults to True.
            endpointing_timeout_ms: The server VAD silence window in ms, shared by
                ``endpointing`` and ``vad_events``. Defaults to 300 so the server's final
                lands before a VAD-driven agent commits the turn (LiveKit commits about
                0.55 s after speech with its default VAD): measured against 600 on 19 call
                recordings this cut the words leaking into the next turn by 60 % at the
                same accuracy. Shorter windows start cutting inside words. ``None`` uses
                the server default (600).
            eou_timeout_ms: Silence (ms) for the transducer's blank-token endpoint.
                Unset uses the server default (800). Values below the ~1.1 s decode
                chunk make the server cut finals inside utterances; prefer
                ``endpointing`` for low latency.
            vad_events: Ask the server for ``speech_started`` / ``speech_ended``
                messages and map them to START_OF_SPEECH / END_OF_SPEECH. This is the
                only reliable end-of-speech signal: finals alone are also emitted
                mid-utterance. With it off, every final closes the turn. Defaults to True.
            finalize_on_flush: Send ``{"type": "finalize"}`` when the framework flushes
                the stream, so buffered speech is returned as a final immediately. Defaults to True.
            finalize_on_words: Cut a final every ``max_words`` words during long
                continuous speech. Unset uses the server default (on). The cut is not a
                turn end; END_OF_SPEECH still follows the VAD.
            max_words: Word count for ``finalize_on_words``. Unset uses the server default.
            keepalive_interval: Seconds between ``{"type": "keepalive"}`` messages, which
                reset the public API's inactivity timer during long pauses (it answers with
                ``{"type": "pong"}``). Defaults to 5 s on the public API and off for a
                direct server, which does not accept the message. ``None`` disables it.
            api_key: Smallest AI API key. Falls back to the SMALLEST_API_KEY
                environment variable if not provided.
            http_session: An existing HttpSession to reuse.
            base_url: API base URL. For a directly reachable inference server pass its
                ``http://host:port`` and ``stream_path="/transcribe"``.
            stream_path: Path of the streaming endpoint under ``base_url``.
                ``"/stt/live"`` on the public API, ``"/transcribe"`` on a server.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=model in _STREAMING_MODELS,
                interim_results=True,
                diarization=diarize,
                aligned_transcript="word" if word_timestamps else False,
            )
        )

        api_key = api_key or os.environ.get("SMALLEST_API_KEY")
        if not api_key:
            raise ValueError(
                "Smallest AI API key is required, either as argument or set "
                "SMALLEST_API_KEY environment variable"
            )

        self._opts = _STTOptions(
            model=model,
            api_key=api_key,
            language=language,
            sample_rate=sample_rate,
            encoding=encoding,
            word_timestamps=word_timestamps,
            diarize=diarize,
            format=format,
            itn_normalize=itn_normalize,
            numerals=numerals if is_given(numerals) else None,
            keywords=list(keywords) if is_given(keywords) else [],
            sentence_timestamps=sentence_timestamps,
            redact_pii=redact_pii,
            redact_pci=redact_pci,
            endpointing=endpointing,
            endpointing_timeout_ms=endpointing_timeout_ms
            if is_given(endpointing_timeout_ms)
            else None,
            eou_timeout_ms=eou_timeout_ms if is_given(eou_timeout_ms) else None,
            vad_events=vad_events,
            finalize_on_flush=finalize_on_flush,
            finalize_on_words=finalize_on_words if is_given(finalize_on_words) else None,
            max_words=max_words if is_given(max_words) else None,
            keepalive_interval=(
                keepalive_interval
                if is_given(keepalive_interval)
                else (5.0 if stream_path == DEFAULT_STREAM_PATH else None)
            ),
            base_url=base_url,
            stream_path=stream_path,
        )
        self._session = http_session
        self._streams: weakref.WeakSet[SpeechStream] = weakref.WeakSet()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "SmallestAI"

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        config = self._sanitize_options(language=language)
        params: dict[str, JsonValue] = {
            "model": config.model,
            "language": config.language,
            "encoding": config.encoding,
            "sample_rate": config.sample_rate,
            "word_timestamps": str(config.word_timestamps).lower(),
            "diarize": str(config.diarize).lower(),
        }

        try:
            async with self._ensure_session().post(
                f"{config.base_url}/stt/",
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/octet-stream",
                    "X-Source": "livekit",
                    "X-LiveKit-Version": __version__,
                },
                params=params,
                # to_wav_bytes() produces a valid WAV file; the server auto-detects format.
                data=rtc.combine_audio_frames(buffer).to_wav_bytes(),
                conn_options=conn_options,
            ) as resp:
                await resp.raise_for_status()
                data = _MESSAGE.validate_python(await resp.json())
                return _batch_transcription_to_speech_event(config.language, data)

        except TimeoutError as e:
            raise APITimeoutError() from e
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        if not self.capabilities.streaming:
            raise ValueError(
                f"{self._opts.model} does not support streaming; use recognize() for batch transcription"
            )
        config = self._sanitize_options(language=language)
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            opts=config,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        encoding: NotGivenOr[STTEncoding | str] = NOT_GIVEN,
        format: NotGivenOr[bool] = NOT_GIVEN,
        itn_normalize: NotGivenOr[bool] = NOT_GIVEN,
        keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN,
        sentence_timestamps: NotGivenOr[bool] = NOT_GIVEN,
        redact_pii: NotGivenOr[bool] = NOT_GIVEN,
        redact_pci: NotGivenOr[bool] = NOT_GIVEN,
        endpointing: NotGivenOr[bool] = NOT_GIVEN,
        endpointing_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        eou_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_events: NotGivenOr[bool] = NOT_GIVEN,
        finalize_on_flush: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        """Update STT options; propagates to all active streams (triggers reconnect)."""
        # capabilities are fixed at construction: a session wraps a non-streaming STT once
        if is_given(model) and (model in _STREAMING_MODELS) != self.capabilities.streaming:
            raise ValueError(
                f"{model} {'streams' if model in _STREAMING_MODELS else 'does not stream'}, "
                "unlike the model this STT was created with; create a new STT for it"
            )
        if is_given(model):
            self._opts.model = model
        self._opts = _apply_updates(
            self._opts,
            language=language,
            sample_rate=sample_rate,
            encoding=encoding,
            format=format,
            itn_normalize=itn_normalize,
            keywords=keywords,
            sentence_timestamps=sentence_timestamps,
            redact_pii=redact_pii,
            redact_pci=redact_pci,
            endpointing=endpointing,
            endpointing_timeout_ms=endpointing_timeout_ms,
            eou_timeout_ms=eou_timeout_ms,
            vad_events=vad_events,
            finalize_on_flush=finalize_on_flush,
        )
        for stream in self._streams:
            stream.update_options(
                model=model,
                language=language,
                sample_rate=sample_rate,
                encoding=encoding,
                format=format,
                itn_normalize=itn_normalize,
                keywords=keywords,
                sentence_timestamps=sentence_timestamps,
                redact_pii=redact_pii,
                redact_pci=redact_pci,
                endpointing=endpointing,
                endpointing_timeout_ms=endpointing_timeout_ms,
                eou_timeout_ms=eou_timeout_ms,
                vad_events=vad_events,
                finalize_on_flush=finalize_on_flush,
            )

    def _sanitize_options(self, *, language: NotGivenOr[str] = NOT_GIVEN) -> _STTOptions:
        config = replace(self._opts)
        if is_given(language):
            config.language = language
        return config


def _apply_updates(
    opts: _STTOptions,
    *,
    model: NotGivenOr[STTModels | str] = NOT_GIVEN,
    language: NotGivenOr[str] = NOT_GIVEN,
    sample_rate: NotGivenOr[int] = NOT_GIVEN,
    encoding: NotGivenOr[STTEncoding | str] = NOT_GIVEN,
    format: NotGivenOr[bool] = NOT_GIVEN,
    itn_normalize: NotGivenOr[bool] = NOT_GIVEN,
    keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN,
    sentence_timestamps: NotGivenOr[bool] = NOT_GIVEN,
    redact_pii: NotGivenOr[bool] = NOT_GIVEN,
    redact_pci: NotGivenOr[bool] = NOT_GIVEN,
    endpointing: NotGivenOr[bool] = NOT_GIVEN,
    endpointing_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
    eou_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
    vad_events: NotGivenOr[bool] = NOT_GIVEN,
    finalize_on_flush: NotGivenOr[bool] = NOT_GIVEN,
) -> _STTOptions:
    return replace(
        opts,
        model=model if is_given(model) else opts.model,
        language=language if is_given(language) else opts.language,
        sample_rate=sample_rate if is_given(sample_rate) else opts.sample_rate,
        encoding=encoding if is_given(encoding) else opts.encoding,
        format=format if is_given(format) else opts.format,
        itn_normalize=itn_normalize if is_given(itn_normalize) else opts.itn_normalize,
        keywords=list(keywords) if is_given(keywords) else list(opts.keywords),
        sentence_timestamps=sentence_timestamps
        if is_given(sentence_timestamps)
        else opts.sentence_timestamps,
        redact_pii=redact_pii if is_given(redact_pii) else opts.redact_pii,
        redact_pci=redact_pci if is_given(redact_pci) else opts.redact_pci,
        endpointing=endpointing if is_given(endpointing) else opts.endpointing,
        endpointing_timeout_ms=endpointing_timeout_ms
        if is_given(endpointing_timeout_ms)
        else opts.endpointing_timeout_ms,
        eou_timeout_ms=eou_timeout_ms if is_given(eou_timeout_ms) else opts.eou_timeout_ms,
        vad_events=vad_events if is_given(vad_events) else opts.vad_events,
        finalize_on_flush=finalize_on_flush
        if is_given(finalize_on_flush)
        else opts.finalize_on_flush,
    )


class SpeechStream(stt.SocketStream[_Message, _Command]):
    goodbye = _Command(type="close_stream")
    # 50 ms packets, the cadence the Pulse docs recommend
    frame_ms = 50
    # speech_started and speech_ended come from the server's acoustic VAD; without them each
    # final ends the turn
    provider_vad = True

    def __init__(self, *, stt: STT, opts: _STTOptions, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._session_id = ""
        if opts.keepalive_interval:
            # resets the public API's inactivity timer while no audio flows
            self.keepalive = http.Keepalive(opts.keepalive_interval, _Command(type="keepalive"))

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        encoding: NotGivenOr[STTEncoding | str] = NOT_GIVEN,
        format: NotGivenOr[bool] = NOT_GIVEN,
        itn_normalize: NotGivenOr[bool] = NOT_GIVEN,
        keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN,
        sentence_timestamps: NotGivenOr[bool] = NOT_GIVEN,
        redact_pii: NotGivenOr[bool] = NOT_GIVEN,
        redact_pci: NotGivenOr[bool] = NOT_GIVEN,
        endpointing: NotGivenOr[bool] = NOT_GIVEN,
        endpointing_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        eou_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_events: NotGivenOr[bool] = NOT_GIVEN,
        finalize_on_flush: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        self._opts = _apply_updates(
            self._opts,
            model=model,
            language=language,
            sample_rate=sample_rate,
            encoding=encoding,
            format=format,
            itn_normalize=itn_normalize,
            keywords=keywords,
            sentence_timestamps=sentence_timestamps,
            redact_pii=redact_pii,
            redact_pci=redact_pci,
            endpointing=endpointing,
            endpointing_timeout_ms=endpointing_timeout_ms,
            eou_timeout_ms=eou_timeout_ms,
            vad_events=vad_events,
            finalize_on_flush=finalize_on_flush,
        )
        self.reconnect("options changed")

    def _stream_params(self) -> dict[str, JsonValue]:
        """Query parameters for the streaming connection; unset knobs are left to the server."""
        o = self._opts
        params: dict[str, JsonValue] = {
            "model": o.model,
            "language": o.language,
            "encoding": o.encoding,
            "sample_rate": o.sample_rate,
            "word_timestamps": str(o.word_timestamps).lower(),
            "diarize": str(o.diarize).lower(),
            "format": str(o.format).lower(),
            "itn_normalize": str(o.itn_normalize).lower(),
            "sentence_timestamps": str(o.sentence_timestamps).lower(),
            "redact_pii": str(o.redact_pii).lower(),
            "redact_pci": str(o.redact_pci).lower(),
            "endpointing": str(o.endpointing).lower(),
            "vad_events": str(o.vad_events).lower(),
        }
        if o.numerals is not None:
            params["numerals"] = str(o.numerals).lower()
        if o.endpointing_timeout_ms is not None:
            params["endpointing_timeout"] = o.endpointing_timeout_ms
        if o.eou_timeout_ms is not None:
            params["eou_timeout_ms"] = o.eou_timeout_ms
        if o.finalize_on_words is not None:
            params["finalize_on_words"] = str(o.finalize_on_words).lower()
        if o.max_words is not None:
            params["max_words"] = o.max_words
        if o.keywords:
            params["keywords"] = ",".join(f"{kw}:{boost:g}" for kw, boost in o.keywords)
        return params

    async def connect(self, conn_options: APIConnectOptions) -> http.SocketTarget:
        base = self._opts.base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        return http.SocketTarget(
            f"{base.rstrip('/')}{self._opts.stream_path}?{urlencode(self._stream_params())}",
            headers={
                "Authorization": f"Bearer {self._opts.api_key}",
                "X-Source": "livekit",
                "X-LiveKit-Version": __version__,
            },
        )

    def encode(
        self, item: rtc.AudioFrame | stt.RecognizeStream.Flush | http.EndOfInput
    ) -> http.Encoded[_Command]:
        if isinstance(item, rtc.AudioFrame):
            return bytes(item.data)
        if isinstance(item, stt.RecognizeStream.Flush) and self._opts.finalize_on_flush:
            # return the buffered tail as a final now instead of at the server's endpointing
            return _Command(type="finalize")
        return None

    def on_message(self, message: _Message) -> http.Done | None:
        # errors: {"type": "error", "status", "message"} from the public API, {"error": "..."}
        # from a server reached directly
        if message.get("type") == "error" or (
            "error" in message and "transcript" not in message and "transcription" not in message
        ):
            status = message.get("status")
            detail = json.dumps(message)
            error = f"Smallest AI STT error ({status or 'error'})"
            if status is not None and status.isdigit():
                raise APIStatusError(
                    error,
                    status_code=int(status),
                    request_id=self._session_id or None,
                    body=message,
                    detail=detail,
                )
            raise APIError(error, kind="invalid_request", body=message, detail=detail)

        if session_id := message.get("session_id"):
            self._session_id = session_id
        kind = message.get("type")
        if kind == "speech_started":
            self.report(stt.SpeechStarted())
            return None
        if kind == "speech_ended":
            self.report(stt.SpeechEnded())
            return None

        # the gateway duplicates the server's `transcript` into `transcription`
        text = message.get("transcript") or message.get("transcription", "")
        final = message.get("is_final", False)
        # punctuation-only text ("." or a lone danda) is not a user turn
        if _WORD_RE.search(text):
            self.report(
                _transcript(
                    self._opts.language,
                    message,
                    text=text,
                    final=final,
                    request_id=self._session_id,
                    offset=self.start_time_offset,
                    diarize=self._opts.diarize,
                )
            )
        # a final ends the turn when it answers a finalize, ends the stream, or the server says
        # the speaker paused; other finals are cuts inside an utterance (word rollover,
        # blank-token endpoint), so only without vad_events does every final end it
        last = message.get("is_last", False)
        if final and (
            last
            or message.get("from_finalize", False)
            or message.get("speech_final", False)
            or not self._opts.vad_events
        ):
            self.report(stt.SpeechEnded())
        # the session is fully flushed: nothing else arrives
        return http.DONE if last else None


def _aggregate_confidence(words: list[_Word]) -> float:
    """Geometric mean of the per-word confidences (each already a geometric mean over its pieces)."""
    probs = [w["confidence"] for w in words if "confidence" in w]
    if not probs:
        return 0.0
    return min(1.0, max(0.0, math.exp(sum(math.log(max(p, 1e-10)) for p in probs) / len(probs))))


def _shifted(utterance: dict[str, JsonValue], offset: float) -> dict[str, JsonValue]:
    start, end = utterance.get("start"), utterance.get("end")
    return {
        **utterance,
        "start": (start if isinstance(start, int | float) else 0.0) + offset,
        "end": (end if isinstance(end, int | float) else 0.0) + offset,
    }


def _transcript(
    language: str,
    message: _Message,
    *,
    text: str,
    final: bool,
    request_id: str,
    offset: float,
    diarize: bool,
) -> stt.Transcript:
    words = message.get("words", [])
    # streaming diarization: per-word speaker ids; the utterance takes the majority
    speakers = Counter(w["speaker"] for w in words if "speaker" in w) if diarize else Counter()
    metadata: dict[str, JsonValue] = {}
    if utterances := message.get("utterances"):
        metadata["utterances"] = [_shifted(u, offset) for u in utterances]
    if redacted := message.get("redacted_entities"):
        metadata["redacted_entities"] = list[JsonValue](redacted)
    return stt.Transcript(
        text.strip(),
        # with language="multi" the server reports the detected language on finals
        LanguageCode(message.get("language") or language),
        final=final,
        request_id=request_id,
        start_time=words[0].get("start", 0.0) + offset if words else 0.0,
        end_time=words[-1].get("end", 0.0) + offset if words else 0.0,
        confidence=_aggregate_confidence(words),
        speaker_id=f"S{speakers.most_common(1)[0][0]}" if speakers else None,
        words=[
            TranscriptChunk(
                text=w.get("word", ""),
                start_time=w.get("start", 0.0) + offset,
                end_time=w.get("end", 0.0) + offset,
                start_time_offset=offset,
            )
            for w in words
        ]
        or None,
        metadata=metadata or None,
    )


def _batch_transcription_to_speech_event(language: str, data: _Message) -> stt.SpeechEvent:
    # Batch HTTP response: {"status", "transcription", "audio_length", "words": [...],
    # "utterances": [...] (when word_timestamps=True), "language", "metadata"}.
    words = data.get("words", [])
    utterances = data.get("utterances", [])
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        request_id=utils.shortuuid(),
        alternatives=[
            stt.SpeechData(
                language=LanguageCode(data.get("language") or language),
                text=data.get("transcription", ""),
                start_time=words[0].get("start", 0.0) if words else 0.0,
                end_time=words[-1].get("end", 0.0) if words else 0.0,
                confidence=_aggregate_confidence(words),
                words=[
                    TranscriptChunk(
                        text=w.get("word", ""),
                        start_time=w.get("start", 0.0),
                        end_time=w.get("end", 0.0),
                    )
                    for w in words
                ]
                or None,
                metadata={"utterances": list[JsonValue](utterances)} if utterances else None,
            )
        ],
    )
