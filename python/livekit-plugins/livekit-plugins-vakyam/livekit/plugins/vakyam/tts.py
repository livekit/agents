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

"""Text-to-Speech implementation for Vakyam AI (Raaga 1).

Streaming uses the realtime WebSocket API, one ``text`` message per prepared
chunk because Vakyam expects one complete utterance per message. One-shot
``synthesize()`` uses HTTP ``POST /v1/tts/stream`` for PCM bytes.
"""

from __future__ import annotations

import os
import weakref
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, NotRequired, TypedDict, assert_never

from pydantic import JsonValue

from livekit.agents import (
    APIConnectOptions,
    APIError,
    APIStatusError,
    __version__ as livekit_version,
    http,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from ._utils import (
    WS_ERROR_KINDS,
    http_stream_url,
    normalize_base_url,
    normalize_voice,
    raise_http_error,
    speech_payload,
    validate_tts_options,
    websocket_url,
)
from .models import (
    DEFAULT_BASE_URL,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    KEEPALIVE_INTERVAL_SECONDS,
    MAX_TEXT_CHARACTERS,
    TTSLanguages,
    TTSModels,
    TTSSampleRates,
)
from .version import __version__

NUM_CHANNELS = 1
USER_AGENT = f"LiveKit-Agents-Vakyam/{__version__} livekit-agents/{livekit_version}"


@dataclass
class _TTSOptions:
    model: TTSModels | str
    voice: str
    language: TTSLanguages | str
    sample_rate: TTSSampleRates | int
    speed: float
    api_key: str
    base_url: str
    allow_insecure_base_url: bool


class TTS(tts.TTS):
    """Vakyam AI text-to-speech for LiveKit Agents.

    Uses the realtime WebSocket API (``WS /v1/tts/websocket``) for low-latency
    streaming. Each prepared text chunk is one synthesis request because Vakyam
    expects one complete utterance per request.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: TTSModels | str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        language: TTSLanguages | str = DEFAULT_LANGUAGE,
        sample_rate: TTSSampleRates | int = DEFAULT_SAMPLE_RATE,
        speed: float = DEFAULT_SPEED,
        base_url: str | None = None,
        allow_insecure_base_url: bool = False,
        http_session: http.HttpSession | None = None,
    ) -> None:
        """Create a Vakyam TTS instance.

        Args:
            api_key: Vakyam API key. Defaults to ``VAKYAM_API_KEY``.
            model: TTS model id. Currently ``raaga-v1``.
            voice: Voice selector — a preset name from ``GET /v1/voices``, or a
                custom voice ID beginning with ``vc_``.
            language: BCP-47 language code (for example ``ta-IN``).
            sample_rate: Output PCM sample rate in Hz (8000, 16000, 24000, 48000).
            speed: Speech rate multiplier (``0.5``–``2.0``).
            base_url: API base URL. Defaults to ``https://api.vakyam.ai``.
            allow_insecure_base_url: Allow non-localhost ``http://`` base URLs.
            http_session: Optional HttpSession for HTTP ``synthesize()``.
        """
        resolved_key = api_key or os.environ.get("VAKYAM_API_KEY")
        if not resolved_key:
            raise ValueError("Vakyam API key is required, either as api_key= or VAKYAM_API_KEY")

        validate_tts_options(
            model=str(model),
            language=str(language),
            sample_rate=int(sample_rate),
            speed=speed,
            voice=voice,
        )

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True, aligned_transcript=False, max_text_len=MAX_TEXT_CHARACTERS
            ),
            sample_rate=int(sample_rate),
            num_channels=NUM_CHANNELS,
        )

        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            language=language,
            sample_rate=sample_rate,
            speed=speed,
            api_key=resolved_key,
            base_url=normalize_base_url(
                base_url or DEFAULT_BASE_URL, allow_insecure_base_url=allow_insecure_base_url
            ),
            allow_insecure_base_url=allow_insecure_base_url,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._mux = _Pool(self)

    @property
    def model(self) -> str:
        return str(self._opts.model)

    @property
    def provider(self) -> str:
        return "vakyam"

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()
        return self._session

    def update_options(
        self,
        *,
        model: TTSModels | str | None = None,
        voice: str | None = None,
        language: TTSLanguages | str | None = None,
        sample_rate: TTSSampleRates | int | None = None,
        speed: float | None = None,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """Update synthesis options for streams created after this call."""
        with self._updating_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing):
            next_opts = replace(self._opts)
            next_sample_rate = self._sample_rate
            if model is not None:
                next_opts.model = model
            if voice is not None:
                next_opts.voice = voice
            if language is not None:
                next_opts.language = language
            if sample_rate is not None:
                next_opts.sample_rate = sample_rate
                next_sample_rate = int(sample_rate)
            if speed is not None:
                next_opts.speed = speed

            validate_tts_options(
                model=str(next_opts.model),
                language=str(next_opts.language),
                sample_rate=int(next_opts.sample_rate),
                speed=next_opts.speed,
                voice=next_opts.voice,
            )
        # a socket is configured once, when it opens: a change reaches the sockets opened next
        if next_opts != self._opts:
            self._mux.invalidate()
        self._opts = next_opts
        self._sample_rate = next_sample_rate

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            opts=replace(self._opts),
            http_session=self._ensure_session,
            input_text=chunk.text,
            conn_options=conn_options,
        )

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        self._mux.prewarm()

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._mux.aclose()


class ChunkedStream(tts.ChunkedStream):
    """One-shot synthesis over HTTP ``POST /v1/tts/stream``."""

    def __init__(
        self,
        *,
        tts: TTS,
        opts: _TTSOptions,
        http_session: Callable[[], http.HttpSession],
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._http_session = http_session

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        payload = speech_payload(
            text=self._input_text,
            model=str(self._opts.model),
            voice=self._opts.voice,
            language=str(self._opts.language),
            sample_rate=int(self._opts.sample_rate),
            speed=self._opts.speed,
            output_format="pcm",
        )
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"LiveKit-Agents-Vakyam/{__version__}",
        }
        async with self._http_session().post(
            http_stream_url(self._opts.base_url),
            json=payload,
            headers=headers,
            conn_options=self._conn_options,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise_http_error(resp.status, body)

            content_type = resp.headers.get("Content-Type", "").lower()
            if content_type and not (
                content_type.startswith("audio/")
                or content_type.startswith("application/octet-stream")
            ):
                body = await resp.text()
                raise APIStatusError(
                    "Vakyam TTS returned a non-audio response",
                    status_code=502,
                    body=body,
                )

            output_emitter.initialize(
                request_id=request_id,
                sample_rate=int(self._opts.sample_rate),
                num_channels=NUM_CHANNELS,
                mime_type="audio/pcm",
            )
            async for chunk in resp.iter_bytes():
                if chunk:
                    output_emitter.push(chunk)
            output_emitter.flush()


# Vakyam's socket: connected, then its config answered by configured; then one utterance at a
# time, a text answered by binary audio and end_of_utterance. Each message declares only the
# fields read.


class _Connected(TypedDict):
    type: Literal["connected"]


class _Configured(TypedDict):
    type: Literal["configured"]


class _Pong(TypedDict):
    type: Literal["pong"]


class _Error(TypedDict):
    type: Literal["error"]
    error: NotRequired[JsonValue]


class _EndOfUtterance(TypedDict):
    type: Literal["end_of_utterance"]
    characters_used: int
    truncated: NotRequired[bool]


class _Cancellation(TypedDict):
    type: Literal["cancellation"]


_Message = bytes | _Connected | _Configured | _Pong | _Error | _EndOfUtterance | _Cancellation


class _Config(TypedDict):
    type: Literal["config"]
    model_id: str
    voice: str
    language: str
    output_format: str
    sample_rate: int
    speed: float


class _Text(TypedDict):
    type: Literal["text"]
    text: str


class _Control(TypedDict):
    type: Literal["ping", "cancel"]


_Command = _Config | _Text | _Control


class _Pool(tts.WebSocketMultiplexer[_Message, _Command]):
    """Vakyam sockets, one stream at a time each, configured with the TTS's options when they
    open and reused by the next stream."""

    routed = False
    awaits_ready = True
    keepalive = tts.Keepalive(every=KEEPALIVE_INTERVAL_SECONDS, frame=_Control(type="ping"))
    max_session_duration = 3600

    def __init__(self, tts: TTS) -> None:
        self._tts = tts

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        opts = self._tts._opts
        return tts.WebSocketTarget(
            websocket_url(opts.base_url),
            headers={"Authorization": f"Bearer {opts.api_key}", "User-Agent": USER_AGENT},
        )

    def setup(self) -> _Config:
        opts = self._tts._opts
        return _Config(
            type="config",
            model_id=str(opts.model),
            voice=normalize_voice(opts.voice),
            language=str(opts.language),
            output_format="pcm",
            sample_rate=int(opts.sample_rate),
            speed=opts.speed,
        )

    def ready(self, message: _Message) -> bool:
        return not isinstance(message, bytes) and message["type"] == "configured"

    def route(self, message: _Message) -> None:
        # a socket's own message: its config refused
        if not isinstance(message, bytes) and message["type"] == "error":
            raise _error(message)

    def close_context(self, context_id: str) -> _Control:
        return _Control(type="cancel")


class SynthesizeStream(tts.WebSocketStream[_Message, _Command]):
    """A stream on a pooled Vakyam socket. Vakyam speaks one utterance at a time, so each text
    waits for the end of the one before."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self.multiplexer = tts._mux
        # the texts not yet sent, the utterance being spoken, and the segments whose text ended
        self._queued = deque[tuple[str, str]]()
        self._speaking: str | None = None
        self._ended: set[str] = set()

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> _Text | None:
        match item:
            case tts.SegmentStart() | tts.EndOfText():
                return None
            case tts.SegmentText():
                self._queued.append((item.segment_id, item.chunk.text))
                return self._next() if self._speaking is None else None
            case tts.SegmentEnd():
                self._ended.add(item.segment_id)
                self._complete(item.segment_id)
                return None
            case _:
                assert_never(item)

    def _next(self) -> _Text | None:
        if not self._queued:
            return None
        self._speaking, text = self._queued.popleft()
        return _Text(type="text", text=text)

    def _complete(self, segment_id: str) -> None:
        """The segment is done once its text ended and each of its utterances was spoken."""
        if (
            segment_id in self._ended
            and self._speaking != segment_id
            and all(queued != segment_id for queued, _ in self._queued)
        ):
            self._ended.discard(segment_id)
            self.report(tts.SegmentDone(segment_id))

    def on_message(self, message: _Message) -> None:
        segment = self._speaking or ""
        if isinstance(message, bytes):
            if message:
                self.report(tts.SegmentAudio(segment, message))
        elif message["type"] == "end_of_utterance":
            if message.get("truncated"):
                raise APIStatusError("Vakyam TTS utterance was truncated", status_code=500)
            self.report(tts.SegmentUsage(segment, characters_count=message["characters_used"]))
            self._speaking = None
            if (text := self._next()) is not None:
                self.send_now(text)
            self._complete(segment)
        elif message["type"] == "error":
            raise _error(message)
        elif message["type"] == "cancellation":
            raise APIError("Vakyam TTS cancelled an utterance", kind="server")
        elif (
            message["type"] == "connected"
            or message["type"] == "configured"
            or message["type"] == "pong"
        ):
            pass
        else:
            self._skip_unknown_message(message)


def _error(message: _Error) -> APIError:
    detail = message.get("error")
    code = detail.get("code") if isinstance(detail, dict) else None
    error_code = None if code is None else str(code)
    return APIError(
        "Vakyam TTS WebSocket request failed"
        + (f" (error code: {error_code})" if error_code else ""),
        kind=WS_ERROR_KINDS.get(error_code or "", "invalid_request"),
        body={"type": "error", "code": error_code} if error_code else {"type": "error"},
    )
