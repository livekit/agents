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

import dataclasses
import os
import weakref
from dataclasses import dataclass
from typing import NotRequired, TypedDict, assert_never
from urllib.parse import urlencode

from pydantic import JsonValue

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    LanguageCode,
    http,
    stt,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

_STREAMING_PATH = "/audio/transcriptions/streaming"


class _Word(TypedDict, total=False):
    word: str
    is_final: bool


class _Segment(TypedDict):
    id: int
    text: str
    words: NotRequired[list[_Word]]


class _StreamEvent(TypedDict, total=False):
    segments: list[_Segment]


class _Checkpoint(TypedDict):
    checkpoint_id: str


@dataclass
class STTOptions:
    model: NotGivenOr[str]
    sample_rate: int
    language: NotGivenOr[LanguageCode] = NOT_GIVEN
    prompt: NotGivenOr[str] = NOT_GIVEN
    temperature: NotGivenOr[float] = NOT_GIVEN
    skip_vad: NotGivenOr[bool] = NOT_GIVEN
    vad_kwargs: NotGivenOr[dict[str, JsonValue]] = NOT_GIVEN
    text_timeout_seconds: float = 1.0
    response_format: str = "verbose_json"
    timestamp_granularities: NotGivenOr[list[str]] = NOT_GIVEN
    base_url: NotGivenOr[str] = NOT_GIVEN


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: int = 16000,
        language: NotGivenOr[str] = NOT_GIVEN,
        prompt: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        skip_vad: NotGivenOr[bool] = NOT_GIVEN,
        vad_kwargs: NotGivenOr[dict[str, JsonValue]] = NOT_GIVEN,
        text_timeout_seconds: float = 1.0,
        timestamp_granularities: NotGivenOr[list[str]] = NOT_GIVEN,
        response_format: str = "verbose_json",
        http_session: http.HttpSession | None = None,
        base_url: str = "wss://audio-streaming.us-virginia-1.direct.fireworks.ai/v1",
    ):
        """
        Create a new instance of Fireworks AI STT.

        Args:
            model: The Fireworks AI STT model to use. Defaults to NOT_GIVEN (server uses default model).
            language: The target language for transcription. Defaults to NOT_GIVEN (server detects language automatically).
                Full list: https://fireworks.ai/docs/api-reference/audio-streaming-transcriptions#supported-languages
            prompt: The input prompt that the model will use when generating the transcription. Defaults to NOT_GIVEN.
            temperature: Sampling temperature to use when decoding text tokens during transcription. Defaults to NOT_GIVEN.
            skip_vad: Whether to skip server-side VAD. Defaults to NOT_GIVEN.
            vad_kwargs: The optional kwargs to pass to the VAD model.
                Defaults to NOT_GIVEN. Example: Set to {"threshold": 0.15} to adjust the VAD threshold.
            text_timeout_seconds: Duration of silence before marking transcript as final. Defaults to 1.0.
            timestamp_granularities: The timestamp granularities to populate for this streaming transcription.
                Defaults to NOT_GIVEN. Set to "word,segment" to enable timestamp granularities.
            response_format: The format in which to return the response. Default to "verbose_json".
            base_url: The base URL for the Fireworks AI STT.
                Defaults to "wss://audio-streaming.us-virginia-1.direct.fireworks.ai/v1".
            api_key: The Fireworks AI API key. If not provided, it will be read from
                the FIREWORKS_API_KEY environment variable.
            http_session: Optional HttpSession to use for requests.

        Raises:
            ValueError: If no API key is provided, found in environment variables, or if a parameter is invalid.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript=False,
                offline_recognize=False,
            ),
        )
        if sample_rate != 16000:
            raise ValueError("FireworksAI STT only supports a sample rate of 16000")

        if not 1.0 <= text_timeout_seconds <= 29.0:
            raise ValueError("text_timeout_seconds must be between 1.0 and 29.0")

        fireworks_api_key = api_key if is_given(api_key) else os.environ.get("FIREWORKS_API_KEY")
        if fireworks_api_key is None:
            raise ValueError(
                "Fireworks API key is required. "
                "Pass one in via the `api_key` parameter, "
                "or set it as the `FIREWORKS_API_KEY` environment variable"
            )
        self._api_key = fireworks_api_key
        self._opts = STTOptions(
            model=model,
            sample_rate=sample_rate,
            language=LanguageCode(language) if isinstance(language, str) else language,
            prompt=prompt,
            temperature=temperature,
            skip_vad=skip_vad,
            vad_kwargs=vad_kwargs,
            text_timeout_seconds=text_timeout_seconds,
            response_format=response_format,
            timestamp_granularities=timestamp_granularities,
            base_url=base_url,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return self._opts.model if is_given(self._opts.model) else "unknown"

    @property
    def provider(self) -> str:
        return "FireworksAI"

    @property
    def session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError(
            "FireworksAI STT does not support batch recognition, use stream() instead"
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        config = dataclasses.replace(self._opts)
        if is_given(language):
            config.language = LanguageCode(language)
        stream = SpeechStream(
            stt=self, opts=config, conn_options=conn_options, api_key=self._api_key
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        prompt: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        skip_vad: NotGivenOr[bool] = NOT_GIVEN,
        vad_kwargs: NotGivenOr[dict[str, JsonValue]] = NOT_GIVEN,
        text_timeout_seconds: NotGivenOr[float] = NOT_GIVEN,
        timestamp_granularities: NotGivenOr[list[str]] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = LanguageCode(language)
        if is_given(prompt):
            self._opts.prompt = prompt
        if is_given(temperature):
            self._opts.temperature = temperature
        if is_given(skip_vad):
            self._opts.skip_vad = skip_vad
        if is_given(vad_kwargs):
            self._opts.vad_kwargs = vad_kwargs
        if is_given(text_timeout_seconds):
            if not 1.0 <= text_timeout_seconds <= 29.0:
                raise ValueError("text_timeout_seconds must be between 1.0 and 29.0")
            self._opts.text_timeout_seconds = text_timeout_seconds
        if is_given(timestamp_granularities):
            self._opts.timestamp_granularities = timestamp_granularities

        for stream in self._streams:
            stream.update_options(
                model=model,
                language=language,
                prompt=prompt,
                temperature=temperature,
                skip_vad=skip_vad,
                vad_kwargs=vad_kwargs,
                text_timeout_seconds=text_timeout_seconds,
                timestamp_granularities=timestamp_granularities,
            )


class SpeechStream(stt.WebSocketStream[_StreamEvent, _Checkpoint]):
    frame_ms = 50

    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)

        self._opts = opts
        self._api_key = api_key
        self._transcript_state: dict[int, str] = {}
        self._final_segments_length: dict[int, int] = {}
        self._last_final_segment_id = -1

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        prompt: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        skip_vad: NotGivenOr[bool] = NOT_GIVEN,
        vad_kwargs: NotGivenOr[dict[str, JsonValue]] = NOT_GIVEN,
        text_timeout_seconds: NotGivenOr[float] = NOT_GIVEN,
        timestamp_granularities: NotGivenOr[list[str]] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = LanguageCode(language)
        if is_given(prompt):
            self._opts.prompt = prompt
        if is_given(temperature):
            self._opts.temperature = temperature
        if is_given(skip_vad):
            self._opts.skip_vad = skip_vad
        if is_given(vad_kwargs):
            self._opts.vad_kwargs = vad_kwargs
        if is_given(text_timeout_seconds):
            self._opts.text_timeout_seconds = text_timeout_seconds
        if is_given(timestamp_granularities):
            self._opts.timestamp_granularities = timestamp_granularities

        # the options are read at each connection
        self.reconnect("options changed")

    async def connect(self, conn_options: APIConnectOptions) -> stt.WebSocketTarget:
        # segment ids restart with each connection
        self._transcript_state.clear()
        self._final_segments_length.clear()
        self._last_final_segment_id = -1
        live_config = {
            "model": self._opts.model if is_given(self._opts.model) else None,
            "language": self._opts.language if is_given(self._opts.language) else None,
            "prompt": self._opts.prompt if is_given(self._opts.prompt) else None,
            "temperature": self._opts.temperature if is_given(self._opts.temperature) else None,
            "skip_vad": self._opts.skip_vad if is_given(self._opts.skip_vad) else None,
            "vad_kwargs": self._opts.vad_kwargs if is_given(self._opts.vad_kwargs) else None,
            "text_timeout_seconds": self._opts.text_timeout_seconds,
            "response_format": self._opts.response_format,
            "timestamp_granularities": (
                self._opts.timestamp_granularities
                if is_given(self._opts.timestamp_granularities)
                else None
            ),
        }
        ws_url = str(self._opts.base_url).rstrip("/") + _STREAMING_PATH
        filtered_config = {k: v for k, v in live_config.items() if v is not None}
        return stt.WebSocketTarget(
            f"{ws_url}?{urlencode(filtered_config, doseq=True)}",
            headers={"User-Agent": "LiveKit Agents", "Authorization": self._api_key},
        )

    def encode(
        self, item: rtc.AudioFrame | stt.Flush | stt.EndOfAudio
    ) -> bytes | _Checkpoint | None:
        match item:
            case rtc.AudioFrame():
                return bytes(item.data)
            case stt.Flush():
                return None
            case stt.EndOfAudio():
                return _Checkpoint(checkpoint_id="final")
            case _:
                assert_never(item)

    def on_message(self, message: _StreamEvent) -> None:
        segments = message.get("segments", [])
        if not segments:
            return
        latest_segment = max(segments, key=lambda segment: segment["id"])
        max_segment_id = latest_segment["id"]

        for segment in segments:
            segment_id = segment["id"]
            if segment_id < self._last_final_segment_id:
                continue

            if segment_id == self._last_final_segment_id:
                finalized_word_count = self._final_segments_length.get(segment_id, 0)
                words = segment.get("words", [])
                if finalized_word_count < len(words):
                    new_words = words[finalized_word_count:]
                    new_text = " ".join(word.get("word", "") for word in new_words).strip()
                    self._transcript_state[segment_id] = new_text
                elif segment_id in self._transcript_state:
                    del self._transcript_state[segment_id]
            else:
                self._transcript_state[segment_id] = segment["text"]

        for local_segment_id in list(self._transcript_state.keys()):
            if local_segment_id > max_segment_id:
                del self._transcript_state[local_segment_id]

        # the state is not sorted: join the text by segment id
        sorted_segments = sorted(self._transcript_state.items(), key=lambda item: item[0])
        full_transcript = " ".join([text for _, text in sorted_segments])
        if not full_transcript:
            return

        words = latest_segment.get("words", [])
        is_final = bool(words) and words[-1].get("is_final", False)
        if is_final:
            self._transcript_state.clear()
            self._last_final_segment_id = max_segment_id
            self._final_segments_length[max_segment_id] = len(words)
        # a final ends the speech: turn_detection="stt" commits on END_OF_SPEECH
        self.report(
            stt.Transcript(full_transcript, LanguageCode(self._opts.language or ""), final=is_final)
        )
