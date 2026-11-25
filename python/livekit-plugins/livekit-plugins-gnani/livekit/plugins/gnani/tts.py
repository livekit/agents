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

"""Text-to-Speech for Gnani Vachana: ``synthesize()`` over REST, ``stream()`` over the
WebSocket API, one connection per segment."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, replace
from typing import Literal, NotRequired, TypedDict, assert_never

from livekit.agents import (
    APIConnectOptions,
    APIError,
    http,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from .models import (
    DEFAULT_MODEL,
    GnaniTTSBitrates,
    GnaniTTSContainers,
    GnaniTTSEncodings,
    GnaniTTSLanguages,
    GnaniTTSModels,
    GnaniTTSVoices,
)

GNANI_TTS_BASE_URL = "https://api.vachana.ai"

SUPPORTED_SAMPLE_RATES = (8000, 16000, 22050, 44100)


@dataclass
class GnaniTTSOptions:
    api_key: str
    voice: str = "Pranav"
    model: str = DEFAULT_MODEL
    language: str | None = None
    sample_rate: int = 16000
    encoding: str = "linear_pcm"
    container: str = "wav"
    num_channels: int = 1
    sample_width: int = 2
    bitrate: str | None = None
    base_url: str = GNANI_TTS_BASE_URL


class _AudioConfig(TypedDict):
    sample_rate: int
    encoding: str
    num_channels: int
    sample_width: int
    container: str
    bitrate: NotRequired[str]


class _Request(TypedDict):
    text: str
    voice: str
    model: str
    audio_config: _AudioConfig
    language: NotRequired[str]


# Gnani's websocket: one request a connection, answered with audio (binary frames or base64
# messages), then `complete`. Each message declares only the fields read.


class _AudioData(TypedDict):
    audio: NotRequired[str]


class _Audio(TypedDict):
    type: Literal["audio"]
    data: NotRequired[_AudioData | None]


class _Complete(TypedDict):
    type: Literal["complete"]
    data: NotRequired[_AudioData | None]


class _Error(TypedDict):
    type: Literal["error"]
    message: NotRequired[str]


_Message = bytes | _Audio | _Complete | _Error


class TTS(tts.TTS):
    """Gnani Vachana Text-to-Speech.

    Args:
        voice: Voice to use for synthesis
            (see https://docs.gnani.ai/api/TTS/tts-sse#available-voices).
        model: TTS model name (default: timbre-v2.0; also: timbre-v2.5).
        language: BCP-47 language code for timbre-v2.5 only (e.g. "hi-IN").
        sample_rate: Audio output sample rate (8000-44100).
        encoding: Audio encoding (linear_pcm or oggopus).
        container: Audio container format (raw, mp3, wav, ogg).
        api_key: Gnani API key (falls back to GNANI_API_KEY env var).
        base_url: Vachana API base URL.
    """

    def __init__(
        self,
        *,
        voice: GnaniTTSVoices | str = "Pranav",
        model: GnaniTTSModels | str = DEFAULT_MODEL,
        language: GnaniTTSLanguages | str | None = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        encoding: GnaniTTSEncodings | str = "linear_pcm",
        container: GnaniTTSContainers | str = "wav",
        bitrate: GnaniTTSBitrates | str | None = None,
        api_key: str | None = None,
        base_url: str = GNANI_TTS_BASE_URL,
    ) -> None:
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate must be one of {SUPPORTED_SAMPLE_RATES}, got {sample_rate}"
            )

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        api_key = api_key or os.environ.get("GNANI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gnani API key is required. "
                "Provide it directly or set GNANI_API_KEY environment variable."
            )

        self._opts = GnaniTTSOptions(
            api_key=api_key,
            voice=voice,
            model=model,
            language=language,
            sample_rate=sample_rate,
            encoding=encoding,
            container=container,
            num_channels=num_channels,
            bitrate=bitrate,
            base_url=base_url,
        )
        self._session: http.HttpSession | None = None

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "gnani"

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()
        return self._session

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self, opts=replace(self._opts), input_text=chunk.text, conn_options=conn_options
        )

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        return SynthesizeStream(tts=self, opts=replace(self._opts), conn_options=conn_options)

    def update_options(
        self,
        *,
        voice: GnaniTTSVoices | str | None = None,
        model: GnaniTTSModels | str | None = None,
        language: GnaniTTSLanguages | str | None = None,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        with self._updating_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing):
            if voice is not None:
                self._opts.voice = voice
            if model is not None:
                self._opts.model = model
            if language is not None:
                self._opts.language = language


def _request(opts: GnaniTTSOptions, text: str) -> _Request:
    audio_config = _AudioConfig(
        sample_rate=opts.sample_rate,
        encoding=opts.encoding,
        num_channels=opts.num_channels,
        sample_width=opts.sample_width,
        container=opts.container,
    )
    if opts.bitrate is not None:
        audio_config["bitrate"] = opts.bitrate
    request = _Request(text=text, voice=opts.voice, model=opts.model, audio_config=audio_config)
    if opts.model == "timbre-v2.5" and opts.language is not None:
        request["language"] = opts.language
    return request


def _headers(opts: GnaniTTSOptions) -> dict[str, str]:
    return {"X-API-Key-ID": opts.api_key, "Content-Type": "application/json"}


def _mime_type(opts: GnaniTTSOptions) -> str:
    return "audio/pcm" if opts.container == "raw" else f"audio/{opts.container}"


class ChunkedStream(tts.ChunkedStream):
    """A whole text over REST: POST /api/v1/tts/inference."""

    def __init__(
        self, *, tts: TTS, opts: GnaniTTSOptions, input_text: str, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        async with self._tts._ensure_session().post(
            f"{self._opts.base_url}/api/v1/tts/inference",
            json=_request(self._opts, self._input_text),
            headers=_headers(self._opts),
            conn_options=self._conn_options,
        ) as res:
            await res.raise_for_status()
            audio = await res.read()
            output_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=self._tts.sample_rate,
                num_channels=self._tts.num_channels,
                mime_type=_mime_type(self._opts),
            )
            output_emitter.push(audio)
            output_emitter.flush()


class SynthesizeStream(tts.WebSocketStream[_Message, _Request]):
    """Each segment is one request on its own connection to wss://api.vachana.ai/api/v1/tts:
    its whole text at its end."""

    connection_per_segment = True

    def __init__(self, *, tts: TTS, opts: GnaniTTSOptions, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts
        self.mime_type = _mime_type(opts)
        self._segment = ""
        self._text: list[str] = []

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        base = self._opts.base_url
        ws_base = base.replace("http", "ws", 1) if base.startswith("http") else f"wss://{base}"
        return tts.WebSocketTarget(f"{ws_base}/api/v1/tts", headers=_headers(self._opts))

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> _Request | None:
        match item:
            case tts.SegmentStart():
                self._segment, self._text = item.segment_id, []
                return None
            case tts.SegmentText():
                self._text.append(item.chunk.text)
                return None
            case tts.SegmentEnd():
                text = " ".join(self._text).strip()
                if not text:
                    # nothing to say: the segment is done without a request
                    self.report(tts.SegmentDone(item.segment_id))
                    return None
                return _request(self._opts, text)
            case tts.EndOfText():
                return None
            case _:
                assert_never(item)

    def on_message(self, message: _Message) -> None:
        if isinstance(message, bytes):
            self.report(tts.SegmentAudio(self._segment, message))
        elif message["type"] == "error":
            raise APIError(
                "Gnani TTS stream error",
                kind="invalid_request",
                body=message,
                detail=message.get("message"),
            )
        elif message["type"] == "audio" or message["type"] == "complete":
            if audio := (message.get("data") or {}).get("audio"):
                self.report(tts.SegmentAudio(self._segment, base64.b64decode(audio)))
            if message["type"] == "complete":
                self.report(tts.SegmentDone(self._segment))
        else:
            self._skip_unknown_message(message)


__all__ = ["TTS", "SynthesizeStream"]
