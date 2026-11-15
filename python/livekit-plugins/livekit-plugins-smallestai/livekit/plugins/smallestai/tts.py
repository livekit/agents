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

import base64
import os
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, NotRequired, TypedDict, assert_never

from pydantic import JsonValue

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APITimeoutError,
    LanguageCode,
    http,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TranscriptChunk

from .models import TTSEncoding, TTSModels
from .version import __version__

NUM_CHANNELS = 1
SMALLEST_BASE_URL = "https://api.smallest.ai/waves/v1"
SMALLEST_WS_URL = "wss://api.smallest.ai/waves/v1/tts/live"
# G.711 output_format values and the mime types the audio emitter decodes them as
_OUTPUT_FORMAT_MIME = {"ulaw": "audio/pcmu", "alaw": "audio/pcma"}

# Continuations gives no terminal marker for a segment's last fragment, so completion is
# inferred after both the closing fragment and this much silence in received audio.
_CONTINUATIONS_IDLE_TIMEOUT = 0.6


@dataclass
class _TTSOptions:
    model: TTSModels | str
    api_key: str
    voice_id: str
    sample_rate: int
    speed: float
    language: LanguageCode
    output_format: TTSEncoding | str
    word_timestamps: bool
    max_buffer_flush_ms: int
    use_continuations: bool
    max_buffer_delay_ms: int
    base_url: str
    ws_url: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: TTSModels | str = "lightning_v3.1_pro",
        voice_id: str | None = None,
        sample_rate: int = 24000,
        speed: float = 1.0,
        language: str = "en",
        output_format: TTSEncoding | str = "pcm",
        word_timestamps: bool = False,
        max_buffer_flush_ms: int = 0,
        use_continuations: bool = True,
        max_buffer_delay_ms: int = 3000,
        base_url: str = SMALLEST_BASE_URL,
        ws_url: str = SMALLEST_WS_URL,
        http_session: http.HttpSession | None = None,
    ) -> None:
        """
        Create a new instance of Smallest AI Lightning TTS.

        Args:
            api_key: Your Smallest AI API key.
            model: The TTS model to use. Use "lightning_v3.1" for the standard model with
                217 voices across 12 languages, or "lightning_v3.1_pro" (default) for the
                premium pool with curated American, British, and Indian voices at 44.1 kHz.
            voice_id: The voice ID to use for synthesis. Defaults to "meher" for
                "lightning_v3.1_pro" and "sophia" for all other models. Pro voices must be
                paired with "lightning_v3.1_pro"; standard voices with "lightning_v3.1".
            sample_rate: Sample rate for the audio output. Both models are natively 44.1 kHz;
                supported rates are 8000, 16000, 24000, and 44100.
            speed: Speed of the speech synthesis (0.5–2.0).
            language: Language of the text to be synthesized. Use "auto" for automatic
                detection and code-switching. Pro supports "en", "hi", and "auto" only.
            output_format: Output format for HTTP synthesize() calls ("pcm", "mp3", "wav",
                "ulaw", "alaw"). WebSocket streaming always returns PCM.
            word_timestamps: Request per-word timing events from the server and emit them
                as timed transcript entries alongside audio. Applies to WebSocket streaming
                only; HTTP synthesize() returns raw audio without word events. Disabled by
                default. Supported on base-queue English + Hindi voices (meher, devansh,
                kartik, maithili, liam, avery); other voices silently emit no word events.
            max_buffer_flush_ms: Server-side buffer bound (milliseconds) for the legacy
                ``continue``/``flush`` WebSocket protocol (used when ``use_continuations``
                is False). As text tokens are streamed in with ``continue: true``, the
                server accumulates them and forces partial audio output once this many
                milliseconds of text have buffered, without waiting for an explicit flush.
                ``0`` (default) disables time-based forced flushing, so audio for a segment
                is produced when the segment's end-of-input flush is sent. Ignored when
                ``use_continuations`` is True.
            use_continuations: Use the ``context_id``-based continuations protocol instead
                of the legacy ``continue``/``flush`` fields. Continuations let the server
                release buffered text as soon as it hits a natural sentence boundary (or
                ``max_buffer_delay_ms`` elapses, whichever comes first) instead of holding
                everything until an explicit flush, and carry prosody across fragments by
                priming each release with the audio of the previous one. Defaults to True;
                set False to fall back to the legacy protocol.
            max_buffer_delay_ms: Upper bound (0-5000ms) on how long the server holds a
                buffered fragment waiting for a clean sentence boundary before speaking it
                anyway. Only applies when ``use_continuations`` is True. Defaults to 3000ms
                (the server default).
            base_url: Base URL for the Smallest AI HTTP API.
            ws_url: WebSocket URL for low-latency streaming synthesis.
            http_session: An existing HttpSession to use.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=word_timestamps,
            ),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        api_key = api_key or os.environ.get("SMALLEST_API_KEY")
        if not api_key:
            raise ValueError(
                "Smallest.ai API key is required, either as argument or set"
                " SMALLEST_API_KEY environment variable"
            )

        if voice_id is None:
            voice_id = "meher" if model == "lightning_v3.1_pro" else "sophia"

        self._opts = _TTSOptions(
            model=model,
            api_key=api_key,
            voice_id=voice_id,
            sample_rate=sample_rate,
            speed=speed,
            language=LanguageCode(language),
            output_format=output_format,
            word_timestamps=word_timestamps,
            max_buffer_flush_ms=max_buffer_flush_ms,
            use_continuations=use_continuations,
            max_buffer_delay_ms=max_buffer_delay_ms,
            base_url=base_url,
            ws_url=ws_url,
        )
        self._session = http_session
        self._pool = _Pool(self._opts)

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

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        output_format: NotGivenOr[TTSEncoding | str] = NOT_GIVEN,
        word_timestamps: NotGivenOr[bool] = NOT_GIVEN,
        max_buffer_flush_ms: NotGivenOr[int] = NOT_GIVEN,
        use_continuations: NotGivenOr[bool] = NOT_GIVEN,
        max_buffer_delay_ms: NotGivenOr[int] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """Update TTS options."""
        with self._updating_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing):
            # capabilities are fixed at construction: word timestamps decide `aligned_transcript`
            if is_given(word_timestamps) and (
                word_timestamps != self.capabilities.aligned_transcript
            ):
                raise ValueError(
                    "word_timestamps is fixed at construction: it sets the aligned_transcript "
                    "capability; create a new TTS to change it"
                )
        if is_given(model):
            self._opts.model = model
        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(speed):
            self._opts.speed = speed
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
            # keep the base class property in sync; the framework reads tts.sample_rate
            self._sample_rate = sample_rate
        if is_given(language):
            self._opts.language = LanguageCode(language)
        if is_given(output_format):
            self._opts.output_format = output_format
        if is_given(max_buffer_flush_ms):
            self._opts.max_buffer_flush_ms = max_buffer_flush_ms
        if is_given(use_continuations):
            self._opts.use_continuations = use_continuations
        if is_given(max_buffer_delay_ms):
            self._opts.max_buffer_delay_ms = max_buffer_delay_ms

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
        return SynthesizeStream(
            tts=self, pool=self._pool, opts=replace(self._opts), conn_options=conn_options
        )

    def prewarm(self) -> None:
        self._pool.prewarm()

    async def aclose(self) -> None:
        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    """HTTP-based synthesis — used when synthesize() is called directly."""

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
        try:
            data: dict[str, JsonValue] = {**_to_smallest_options(self._opts)}
            data["text"] = self._input_text

            headers = {
                "Authorization": f"Bearer {self._opts.api_key}",
                "Content-Type": "application/json",
                "X-Source": "livekit",
                "X-LiveKit-Version": __version__,
            }
            async with self._http_session().post(
                f"{self._opts.base_url}/tts",
                headers=headers,
                json=data,
                conn_options=self._conn_options,
            ) as resp:
                await resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type=_OUTPUT_FORMAT_MIME.get(
                        self._opts.output_format, f"audio/{self._opts.output_format}"
                    ),
                )

                async for chunk in resp.iter_bytes():
                    output_emitter.push(chunk)

                output_emitter.flush()

        except TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


# Smallest's streaming socket: each segment is a context (continuations) or a run of text closed
# by a flush (legacy). Each message declares only the fields read.


class _Audio(TypedDict):
    audio: NotRequired[str | None]


class _Chunk(TypedDict):
    status: Literal["chunk"]
    data: NotRequired[_Audio]


class _Word(TypedDict, total=False):
    word: str | None
    start: float | None
    end: float | None


class _WordTimestamp(TypedDict):
    status: Literal["word_timestamp"]
    data: _Word


class _Complete(TypedDict):
    status: Literal["complete"]


class _Error(TypedDict):
    status: Literal["error"]
    message: NotRequired[JsonValue]


_Message = _Chunk | _WordTimestamp | _Complete | _Error


class _Speak(TypedDict):
    model: str
    voice_id: str
    sample_rate: int
    speed: float
    language: str
    text: str
    word_timestamps: NotRequired[bool]
    context_id: NotRequired[str]
    max_buffer_delay_ms: NotRequired[int]
    max_buffer_flush_ms: NotRequired[int]
    flush: NotRequired[bool]


_Continue = TypedDict("_Continue", {"continue": bool})  # a keyword: the functional form


class _Packet(_Speak, _Continue):
    pass


class _Pool(tts.WebSocketMultiplexer[_Message, _Packet]):
    """Smallest's sockets, one stream each and never reused: a continuation sends no end of a
    segment's audio, so a used socket may still carry the last one's. The next stream's socket is
    kept open ahead, so it pays no connect."""

    routed = False
    reuses_sockets = False
    keep_warm = True
    max_session_duration = 3600

    def __init__(self, opts: _TTSOptions) -> None:
        self._opts = opts

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        return tts.WebSocketTarget(
            self._opts.ws_url,
            headers={
                "Authorization": f"Bearer {self._opts.api_key}",
                "X-Source": "livekit",
                "X-LiveKit-Version": __version__,
            },
        )


class SynthesizeStream(tts.WebSocketStream[_Message, _Packet]):
    """Streaming synthesis over a socket of Smallest's pool, the stream's alone.

    Defaults to the continuations protocol: each segment's text goes under its own
    ``context_id`` and the server releases audio at natural sentence boundaries (or
    ``max_buffer_delay_ms``), priming each release with the previous one's audio so prosody
    stays continuous. It sends no end of a segment's audio: a segment whose closing fragment
    went and whose audio came is done once received audio stays idle. The legacy
    ``continue``/``flush`` protocol (``use_continuations=False``) ends each with ``complete``.
    """

    audio_idle = _CONTINUATIONS_IDLE_TIMEOUT
    # Responses carry no context id: each segment drains on its own socket.
    connection_per_segment = True

    def __init__(
        self, *, tts: TTS, pool: _Pool, opts: _TTSOptions, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self.multiplexer = pool
        self._opts = opts
        # segments in the order they went, those whose closing fragment went, and whether audio
        # came since the last closing fragment
        self._open = deque[str]()
        self._closed: set[str] = set()
        self._answered = False

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> _Packet | None:
        match item:
            case tts.SegmentStart():
                self._open.append(item.segment_id)
                self._answered = False
                return None
            case tts.SegmentText():
                return self._packet(item.segment_id, item.chunk.text + " ", more=True)
            case tts.SegmentEnd():
                self._closed.add(item.segment_id)
                return self._packet(item.segment_id, "", more=False)
            case tts.EndOfText():
                return None
            case _:
                assert_never(item)

    def on_audio_idle(self) -> None:
        if self._opts.use_continuations and self._answered:
            while self._open and self._open[0] in self._closed:
                segment = self._open.popleft()
                self._closed.discard(segment)
                self.report(tts.SegmentDone(segment))

    def on_message(self, message: _Message) -> None:
        segment = self._open[0] if self._open else ""
        if message["status"] == "chunk":
            if audio := message.get("data", _Audio()).get("audio"):
                self._answered = True
                self.report(tts.SegmentAudio(segment, base64.b64decode(audio)))
        elif message["status"] == "word_timestamp":
            data = message["data"]
            word, start, end = data.get("word"), data.get("start"), data.get("end")
            if word is not None and start is not None and end is not None:
                chunk = TranscriptChunk(text=word, start_time=start, end_time=end)
                self.report(tts.SegmentWords(segment, [chunk]))
        elif message["status"] == "complete":
            # the legacy protocol's end of a segment; a continuation's release ends no segment
            if not self._opts.use_continuations and self._open:
                segment = self._open.popleft()
                self._closed.discard(segment)
                self.report(tts.SegmentDone(segment))
        elif message["status"] == "error":
            raise APIError(
                "SmallestAI TTS error", kind="invalid_request", detail=str(message.get("message"))
            )
        else:
            assert_never(message)

    def _packet(self, segment_id: str, text: str, *, more: bool) -> _Packet:
        opts = self._opts
        packet: _Packet = {
            "model": opts.model,
            "voice_id": opts.voice_id,
            "sample_rate": opts.sample_rate,
            "speed": opts.speed,
            "language": opts.language.language,
            "text": text,
            "continue": more,
        }
        if opts.word_timestamps:
            packet["word_timestamps"] = True
        if opts.use_continuations:
            packet["context_id"] = segment_id
            packet["max_buffer_delay_ms"] = opts.max_buffer_delay_ms
        else:
            packet["max_buffer_flush_ms"] = opts.max_buffer_flush_ms
            packet["flush"] = not more
        return packet


def _to_smallest_options(opts: _TTSOptions) -> dict[str, JsonValue]:
    return {
        "model": opts.model,
        "voice_id": opts.voice_id,
        "sample_rate": opts.sample_rate,
        "speed": opts.speed,
        "language": opts.language.language,
        "output_format": opts.output_format,
    }
