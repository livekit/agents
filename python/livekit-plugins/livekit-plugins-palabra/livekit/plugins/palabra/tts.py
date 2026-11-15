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

"""Palabra realtime text-to-speech plugin for LiveKit Agents, over Palabra's TTS socket."""

from __future__ import annotations

import base64
import os
import weakref
from dataclasses import dataclass, replace
from typing import Literal, NotRequired, TypedDict, assert_never

from livekit.agents import APIConnectOptions, APIError, tokenize, tts
from livekit.agents.types import NOT_GIVEN, ErrorKind, NotGivenOr
from livekit.agents.utils import is_given

from .models import (
    DEFAULT_DEACCENT_STRENGTH,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_VOICE_ID,
    TTSModels,
)

# Palabra's realtime TTS endpoint in each region
TTS_URLS: dict[str, str] = {
    "eu": "wss://stream.palabra.ai/tts-api/v1/text-to-speech/stream",
    "us": "wss://stream.us.palabra.ai/tts-api/v1/text-to-speech/stream",
}

# A server error code's kind; any other code refused the request.
_ERROR_KINDS: dict[str, ErrorKind] = {
    "SERVICE_UNAVAILABLE": "server",
    "RATE_LIMIT_EXCEEDED": "rate_limit",
}

# The server accepts at most 20 text messages per second per identity; above that it refuses
# the text with RATE_LIMIT_EXCEEDED. Every stream of a TTS spaces its texts by this interval
# (~16 messages per second, one step below the cap).
_MIN_SEND_INTERVAL = 0.06


@dataclass
class _TTSOptions:
    language: str
    voice_id: str
    speed: float | None  # None: the server default is used
    deaccent_strength: float
    model: TTSModels | str
    sample_rate: int


# Palabra's TTS socket: its init, then each text a generation (`is_eos`) answered with audio
# chunks, the last one marked. Each message declares only the fields read.


class _ChunkData(TypedDict):
    audio: NotRequired[str | None]
    generation_id: NotRequired[str]
    last_chunk: NotRequired[bool]


class _AudioChunk(TypedDict):
    message_type: Literal["audio_chunk"]
    data: _ChunkData


class _ErrorData(TypedDict):
    code: NotRequired[str]
    desc: NotRequired[str]


class _Error(TypedDict):
    message_type: Literal["error"]
    data: NotRequired[_ErrorData | None]


_Message = _AudioChunk | _Error


class _VoiceOptions(TypedDict):
    voice_id: str
    deaccent_strength: float
    speed: NotRequired[float]


class _Output(TypedDict):
    format: Literal["pcm"]
    sample_rate: int


class _Init(TypedDict):
    type: Literal["init"]
    language: str
    model: str
    voice_options: _VoiceOptions
    output: _Output


class _Text(TypedDict):
    type: Literal["text"]
    text: str
    is_eos: bool
    generation_id: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        language: str = DEFAULT_LANGUAGE,
        voice_id: str = DEFAULT_VOICE_ID,
        speed: NotGivenOr[float] = NOT_GIVEN,
        deaccent_strength: float = DEFAULT_DEACCENT_STRENGTH,
        model: TTSModels | str = DEFAULT_MODEL,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        region: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Create a new instance of Palabra realtime TTS.

        Args:
            api_key: Palabra API Key (create one at https://platform.palabra.ai/api-keys).
                Falls back to the ``PALABRA_API_KEY`` environment variable.
            language: BCP-47 language code for synthesis. Defaults to ``"en"``.
            voice_id: Voice identifier; ``"default_low"`` / ``"default_high"`` pick the
                language default voice. Defaults to ``"default_low"``.
            speed: Speech rate in the ``0.0``-``1.0`` range (``0.5`` is natural pace).
                Unset by default, so the server default is used.
            deaccent_strength: Foreign-accent reduction in the ``0.0``-``1.0`` range.
            model: Synthesis model. Defaults to ``"auto"`` (server picks).
            sample_rate: Output sample rate in Hz (``8000``-``48000``). Defaults to ``24000``.
            region: Palabra region (``"eu"`` / ``"us"``). Falls back to the
                ``PALABRA_REGION`` environment variable, then ``"eu"``.
            base_url: The TTS socket's URL, in place of the region's.

        Raises:
            ValueError: no ``api_key`` argument and no ``PALABRA_API_KEY`` in the environment,
                or an unknown region.
        """
        super().__init__(
            # the server caps a single text message at 1024 characters
            capabilities=tts.TTSCapabilities(streaming=True, max_text_len=1024),
            sample_rate=sample_rate,
            num_channels=1,
        )
        api_key = api_key or os.environ.get("PALABRA_API_KEY")
        if not api_key:
            raise ValueError(
                "Palabra api_key is required, either as an argument or set the"
                " PALABRA_API_KEY environment variable"
            )
        region = (region or os.environ.get("PALABRA_REGION") or "eu").lower()
        if base_url is None and region not in TTS_URLS:
            raise ValueError(f"Unknown Palabra region {region!r}; one of {', '.join(TTS_URLS)}")

        self._api_key = api_key
        self._url = base_url or TTS_URLS[region]
        self._opts = _TTSOptions(
            language=language,
            voice_id=voice_id,
            speed=speed if is_given(speed) else None,
            deaccent_strength=deaccent_strength,
            model=model,
            sample_rate=sample_rate,
        )
        self._pacing = tts.WebSocketPacing(min_interval=_MIN_SEND_INTERVAL)
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def model(self) -> str:
        """Active synthesis model id."""
        return str(self._opts.model)

    @property
    def provider(self) -> str:
        """Provider display name used in metrics."""
        return "Palabra"

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        deaccent_strength: NotGivenOr[float] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """Update synthesis options; unset arguments keep their current value.

        Each new stream receives the new settings. Streams already running keep
        the options they were created with.
        """
        with self._updating_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing):
            if is_given(language):
                self._opts.language = language
            if is_given(voice_id):
                self._opts.voice_id = voice_id
            if is_given(speed):
                self._opts.speed = speed
            if is_given(deaccent_strength):
                self._opts.deaccent_strength = deaccent_strength
            if is_given(model):
                self._opts.model = model

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(chunk, conn_options=conn_options)

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, opts=replace(self._opts), conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """Close all active streams and their realtime TTS sessions."""
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()


class SynthesizeStream(tts.WebSocketStream[_Message, _Init | _Text]):
    """A stream on its own Palabra session: each text is a generation, and a segment is done
    once its text ended and each of its generations sent its last chunk."""

    def __init__(self, *, tts: TTS, opts: _TTSOptions, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._palabra = tts
        self._opts = opts
        self.pacing = tts._pacing
        # the segment of each generation not yet finished, and the segments whose text ended
        self._generations: dict[str, str] = {}
        self._ended: set[str] = set()
        self._count = 0

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        self._generations.clear()
        self._ended.clear()
        # the key authorizes the socket as its token
        return tts.WebSocketTarget(self._palabra._url, params={"token": self._palabra._api_key})

    def setup(self) -> _Init:
        opts = self._opts
        voice = _VoiceOptions(voice_id=opts.voice_id, deaccent_strength=opts.deaccent_strength)
        if opts.speed is not None:
            voice["speed"] = opts.speed
        return _Init(
            type="init",
            language=opts.language,
            model=str(opts.model),
            voice_options=voice,
            output=_Output(format="pcm", sample_rate=opts.sample_rate),
        )

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> _Text | None:
        match item:
            case tts.SegmentStart() | tts.EndOfText():
                return None
            case tts.SegmentText():
                self._count += 1
                generation = f"{item.segment_id}-{self._count}"
                self._generations[generation] = item.segment_id
                return _Text(
                    type="text", text=item.chunk.text, is_eos=True, generation_id=generation
                )
            case tts.SegmentEnd():
                self._ended.add(item.segment_id)
                self._complete(item.segment_id)
                return None
            case _:
                assert_never(item)

    def on_message(self, message: _Message) -> None:
        if message["message_type"] == "audio_chunk":
            data = message["data"]
            # a generation not of this session (or already finished) is not played
            segment = self._generations.get(data.get("generation_id", ""))
            if segment is None:
                return
            if audio := data.get("audio"):
                self.report(tts.SegmentAudio(segment, base64.b64decode(audio)))
            if data.get("last_chunk"):
                del self._generations[data.get("generation_id", "")]
                self._complete(segment)
        elif message["message_type"] == "error":
            error = message.get("data") or _ErrorData()
            code = error.get("code", "UNKNOWN_ERROR")
            raise APIError(
                f"palabra tts error: {code}",
                kind=_ERROR_KINDS.get(code, "invalid_request"),
                detail=error.get("desc"),
            )
        else:
            assert_never(message)

    def _complete(self, segment_id: str) -> None:
        if segment_id in self._ended and segment_id not in self._generations.values():
            self._ended.discard(segment_id)
            self.report(tts.SegmentDone(segment_id))
