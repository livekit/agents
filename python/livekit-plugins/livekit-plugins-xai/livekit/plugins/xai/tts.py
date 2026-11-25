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
import weakref
from collections import deque
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict, assert_never

from livekit.agents import APIConnectOptions, APIError, tokenize, tts
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .types import GrokVoices, TTSLanguages

SAMPLE_RATE = 24000
NUM_CHANNELS = 1

XAI_WEBSOCKET_URL = "wss://api.x.ai/v1/tts"
DEFAULT_VOICE = "ara"


@dataclass
class _TTSOptions:
    voice: GrokVoices | str
    language: TTSLanguages | str
    optimize_streaming_latency: NotGivenOr[int]
    speed: NotGivenOr[float]
    text_normalization: NotGivenOr[bool]


# xAI's TTS socket: each segment is its text deltas and `text.done`, answered with audio
# deltas and `audio.done`. Each message declares only the fields read.


class _AudioDelta(TypedDict):
    type: Literal["audio.delta"]
    delta: str


class _AudioDone(TypedDict):
    type: Literal["audio.done"]


class _Error(TypedDict):
    type: Literal["error"]
    message: NotRequired[str]


_Message = _AudioDelta | _AudioDone | _Error


class _TextDelta(TypedDict):
    type: Literal["text.delta"]
    delta: str


class _TextDone(TypedDict):
    type: Literal["text.done"]


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        voice: GrokVoices | str = DEFAULT_VOICE,
        language: TTSLanguages | str = "auto",
        optimize_streaming_latency: NotGivenOr[int] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        text_normalization: NotGivenOr[bool] = NOT_GIVEN,
        websocket_url: str = XAI_WEBSOCKET_URL,
    ) -> None:
        """
        Create a new instance of the xAI TTS.

        Args:
            voice (str, optional): The voice ID for the desired voice. Defaults to "ara".
            language (TTSLanguages | str, optional): Language code for synthesis (e.g., "en", "fr", "ja"). Defaults to "auto".
            optimize_streaming_latency (int, optional): Latency optimization level for the xAI TTS websocket.
            speed (float, optional): Speaking-rate multiplier for the generated audio.
            text_normalization (bool, optional): Whether to normalize text before synthesis.
            api_key (str | None, optional): The xAI API key. If not provided, it will be read from the xAI environment variable.
            websocket_url (str, optional): The xAI TTS websocket endpoint.
        """  # noqa: E501
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, text_unit="words"),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        resolved_key: str | None = api_key if is_given(api_key) else os.environ.get("XAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "xAI API key is required, either as argument or set XAI_API_KEY"
                " environment variable"
            )
        self._api_key = resolved_key
        self._opts = _TTSOptions(
            voice=voice,
            language=language,
            optimize_streaming_latency=optimize_streaming_latency,
            speed=speed,
            text_normalization=text_normalization,
        )

        self._websocket_url = websocket_url
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._mux = _Pool(self)

    class Markup(tts.TTS.Markup):
        # markup delegation lives in the base class, keyed on _provider_key()
        def _provider_key(self) -> str:
            return "xai"

    @property
    def model(self) -> str:
        return "unknown"

    @property
    def provider(self) -> str:
        return "xai"

    def update_options(
        self,
        *,
        voice: str | None = None,
        language: TTSLanguages | str | None = None,
        optimize_streaming_latency: NotGivenOr[int] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        text_normalization: NotGivenOr[bool] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """
        Update the Text-to-Speech (TTS) configuration options.

        Args:
            voice (str, optional): The voice ID for the desired voice.
            language (TTSLanguages | str, optional): Language code for synthesis (e.g., "en", "fr", "ja").
            optimize_streaming_latency (int, optional): Latency optimization level for the xAI TTS websocket.
            speed (float, optional): Speaking-rate multiplier for the generated audio.
            text_normalization (bool, optional): Whether to normalize text before synthesis.
            sentence_tokenizer (SentenceTokenizer, optional): What splits the text into the chunks
                the TTS is sent.
            pacing (SentenceStreamPacer | None, optional): A ``SentenceStreamPacer``, or ``None`` to
                send each chunk as soon as it is split.
        """  # noqa: E501
        with self._updating_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing):
            connection_options_before = (
                self._opts.voice,
                self._opts.language,
                self._opts.optimize_streaming_latency,
                self._opts.speed,
                self._opts.text_normalization,
            )

            self._opts.voice = voice or self._opts.voice
            self._opts.language = language or self._opts.language
            if is_given(optimize_streaming_latency):
                self._opts.optimize_streaming_latency = optimize_streaming_latency
            if is_given(speed):
                self._opts.speed = speed
            if is_given(text_normalization):
                self._opts.text_normalization = text_normalization

            connection_options_after = (
                self._opts.voice,
                self._opts.language,
                self._opts.optimize_streaming_latency,
                self._opts.speed,
                self._opts.text_normalization,
            )
        if connection_options_after != connection_options_before:
            self._mux.invalidate()

    def prewarm(self) -> None:
        self._mux.prewarm()

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(chunk, conn_options=conn_options)

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._mux.aclose()


class _Pool(tts.WebSocketMultiplexer[_Message, _TextDelta | _TextDone]):
    """xAI sockets, one stream at a time each, opened with the TTS's voice and options and
    reused by the next stream."""

    routed = False
    # xAI's TTS server enforces an undocumented ~2100s deadline per websocket connection; stay
    # below it so connections rotate before the server kills them
    max_session_duration = 1800

    def __init__(self, tts: TTS) -> None:
        self._tts = tts

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        opts = self._tts._opts
        params: dict[str, str] = {
            "voice": opts.voice,
            "language": opts.language,
            "codec": "pcm",
            "sample_rate": str(SAMPLE_RATE),
        }
        if is_given(opts.optimize_streaming_latency):
            params["optimize_streaming_latency"] = str(opts.optimize_streaming_latency)
        if is_given(opts.speed):
            params["speed"] = str(opts.speed)
        if is_given(opts.text_normalization):
            params["text_normalization"] = str(opts.text_normalization).lower()
        return tts.WebSocketTarget(
            self._tts._websocket_url,
            headers={"Authorization": f"Bearer {self._tts._api_key}"},
            params=params,
        )


class SynthesizeStream(tts.WebSocketStream[_Message, _TextDelta | _TextDone]):
    """A stream on a pooled xAI socket; its segments are answered in the order they went."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self.multiplexer = tts._mux
        # the segments awaiting their audio in order, and those whose text.done went
        self._open = deque[str]()
        self._ended: set[str] = set()

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> _TextDelta | _TextDone | None:
        match item:
            case tts.SegmentStart():
                self._open.append(item.segment_id)
                return None
            case tts.SegmentText():
                # the deltas are concatenated, so each ends with a space
                return _TextDelta(type="text.delta", delta=item.chunk.text + " ")
            case tts.SegmentEnd():
                self._ended.add(item.segment_id)
                return _TextDone(type="text.done")
            case tts.EndOfText():
                return None
            case _:
                assert_never(item)

    def on_message(self, message: _Message) -> None:
        segment = self._open[0] if self._open else ""
        if message["type"] == "audio.delta":
            self.report(tts.SegmentAudio(segment, base64.b64decode(message["delta"])))
        elif message["type"] == "audio.done":
            # only the end of a segment's whole text ends its audio
            if segment in self._ended:
                self._open.popleft()
                self._ended.discard(segment)
                self.report(tts.SegmentDone(segment))
        elif message["type"] == "error":
            raise APIError(
                "xAI TTS error", kind="invalid_request", body=message, detail=message.get("message")
            )
        else:
            self._skip_unknown_message(message)
