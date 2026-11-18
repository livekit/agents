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
from dataclasses import dataclass
from typing import TypedDict, assert_never

from pydantic import JsonValue

from livekit import rtc
from livekit.agents import LanguageCode, http, stt, utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from .log import logger

DEFAULT_MODEL = "inworld/inworld-stt-1"
DEFAULT_LANGUAGE = "en-US"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_NUM_CHANNELS = 1
DEFAULT_API_URL = "https://api.inworld.ai/"
WS_ENDPOINT = "stt/v1/transcribe:streamBidirectional"

# Inworld supports multiple STT models (e.g. inworld/inworld-stt-1,
# assemblyai/universal-streaming-multilingual, soniox/stt-rt-v4, etc.).
# No validation is done here — pass any model string through so new models work
# without a plugin update.


class _Transcription(TypedDict, total=False):
    transcript: str
    isFinal: bool
    voiceProfile: JsonValue


class _Result(TypedDict, total=False):
    speechStarted: JsonValue
    transcription: _Transcription


class _Event(TypedDict, total=False):
    result: _Result


class _Command(TypedDict, total=False):
    transcribeConfig: dict[str, object]
    audioChunk: dict[str, str]
    endTurn: dict[str, object]
    closeStream: dict[str, object]


@dataclass
class _STTOptions:
    model: str
    language: str
    sample_rate: int
    num_channels: int
    enable_voice_profile: bool
    voice_profile_top_n: int
    vad_threshold: NotGivenOr[float]
    min_end_of_turn_silence_when_confident: int
    end_of_turn_confidence_threshold: float


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        num_channels: NotGivenOr[int] = NOT_GIVEN,
        enable_voice_profile: bool = True,
        voice_profile_top_n: int = 1,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_end_of_turn_silence_when_confident: int = 200,
        end_of_turn_confidence_threshold: float = 0.3,
        base_url: str = DEFAULT_API_URL,
        http_session: http.HttpSession | None = None,
    ) -> None:
        """Create a new instance of Inworld STT.

        Args:
            api_key: Inworld API key. If not provided, reads from INWORLD_API_KEY env var.
            model: STT model identifier. Any model string supported by the Inworld STT API
                is accepted (e.g. "inworld/inworld-stt-1",
                "assemblyai/universal-streaming-multilingual", "soniox/stt-rt-v4").
                Defaults to "inworld/inworld-stt-1".
            language: Language code. Defaults to "en-US".
            sample_rate: Audio sample rate in Hz. Defaults to 16000.
            num_channels: Number of audio channels. Defaults to 1.
            enable_voice_profile: Enable voice profiling (age, gender, emotion, accent).
                Defaults to True.
            voice_profile_top_n: Number of top voice profile results per category.
            vad_threshold: VAD sensitivity threshold.
            min_end_of_turn_silence_when_confident: Minimum silence (ms) before end-of-turn
                when confidence is high. Defaults to 200.
            end_of_turn_confidence_threshold: Confidence threshold for end-of-turn detection.
                Lower values trigger end-of-turn more eagerly. Defaults to 0.3.
            base_url: Base URL for the Inworld API. Defaults to "https://api.inworld.ai/".
            http_session: Optional HttpSession to use.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True, interim_results=True, offline_recognize=False
            ),
        )

        api_key = api_key if utils.is_given(api_key) else os.getenv("INWORLD_API_KEY", "")
        if not api_key:
            raise ValueError("Inworld API key required. Set INWORLD_API_KEY or provide api_key.")

        self._authorization = f"Basic {api_key}"
        self._base_url = base_url
        self._http_session = http_session
        self._streams: weakref.WeakSet[SpeechStream] = weakref.WeakSet()

        self._opts = _STTOptions(
            model=model if utils.is_given(model) else DEFAULT_MODEL,
            language=language if utils.is_given(language) else DEFAULT_LANGUAGE,
            sample_rate=sample_rate if utils.is_given(sample_rate) else DEFAULT_SAMPLE_RATE,
            num_channels=num_channels if utils.is_given(num_channels) else DEFAULT_NUM_CHANNELS,
            enable_voice_profile=enable_voice_profile,
            voice_profile_top_n=voice_profile_top_n,
            vad_threshold=vad_threshold,
            min_end_of_turn_silence_when_confident=min_end_of_turn_silence_when_confident,
            end_of_turn_confidence_threshold=end_of_turn_confidence_threshold,
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Inworld"

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        enable_voice_profile: NotGivenOr[bool] = NOT_GIVEN,
        voice_profile_top_n: NotGivenOr[int] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_end_of_turn_silence_when_confident: NotGivenOr[int] = NOT_GIVEN,
        end_of_turn_confidence_threshold: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """Update STT options. Changes apply to new streams only.

        Args:
            model: STT model identifier (e.g. "inworld/inworld-stt-1",
                "assemblyai/universal-streaming-multilingual"). Any model string is accepted.
            language: Language code (e.g. "en-US").
            enable_voice_profile: Enable voice profiling.
            voice_profile_top_n: Number of top voice profile results.
            vad_threshold: VAD sensitivity threshold.
            min_end_of_turn_silence_when_confident: Min silence (ms) for end-of-turn.
            end_of_turn_confidence_threshold: Confidence threshold for end-of-turn.
        """
        if utils.is_given(model):
            self._opts.model = model
        if utils.is_given(language):
            self._opts.language = language
        if utils.is_given(enable_voice_profile):
            self._opts.enable_voice_profile = enable_voice_profile
        if utils.is_given(voice_profile_top_n):
            self._opts.voice_profile_top_n = voice_profile_top_n
        if utils.is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold
        if utils.is_given(min_end_of_turn_silence_when_confident):
            self._opts.min_end_of_turn_silence_when_confident = (
                min_end_of_turn_silence_when_confident
            )
        if utils.is_given(end_of_turn_confidence_threshold):
            self._opts.end_of_turn_confidence_threshold = end_of_turn_confidence_threshold

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError(
            "Inworld STT does not support batch recognition — use streaming via stream()"
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        stream = SpeechStream(
            stt=self,
            opts=self._opts,
            base_url=self._base_url,
            authorization=self._authorization,
            conn_options=conn_options,
            language=language if utils.is_given(language) else self._opts.language,
        )
        self._streams.add(stream)
        return stream


class SpeechStream(stt.WebSocketStream[_Event, _Command]):
    # speechStarted and the final transcription are Inworld's own speech boundaries
    provider_vad = True

    def __init__(
        self,
        *,
        stt: STT,
        opts: _STTOptions,
        base_url: str,
        authorization: str,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        language: str = DEFAULT_LANGUAGE,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._base_url = base_url
        self._authorization = authorization
        self._language = language
        self._request_id = ""

    def _build_transcribe_config(self) -> dict[str, object]:
        opts = self._opts
        config: dict[str, object] = {
            "modelId": opts.model,
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": opts.sample_rate,
            "numberOfChannels": opts.num_channels,
            "language": self._language,
        }

        if opts.enable_voice_profile:
            config["voiceProfileConfig"] = {
                "enableVoiceProfile": True,
                "topN": opts.voice_profile_top_n,
            }

        config["endOfTurnConfidenceThreshold"] = opts.end_of_turn_confidence_threshold

        inworld_v1_config: dict[str, object] = {
            "minEndOfTurnSilenceWhenConfident": opts.min_end_of_turn_silence_when_confident,
        }
        if utils.is_given(opts.vad_threshold):
            inworld_v1_config["vadThreshold"] = opts.vad_threshold
        config["inworldSttV1Config"] = inworld_v1_config

        return config

    async def connect(self, conn_options: APIConnectOptions) -> stt.WebSocketTarget:
        ws_url = self._base_url.replace("https://", "wss://").replace("http://", "ws://")
        self._request_id = utils.shortuuid()
        return stt.WebSocketTarget(
            ws_url.rstrip("/") + "/" + WS_ENDPOINT,
            headers={"Authorization": self._authorization},
        )

    def setup(self) -> _Command:
        return _Command(transcribeConfig=self._build_transcribe_config())

    def encode(self, item: rtc.AudioFrame | stt.Flush | stt.EndOfAudio) -> _Command | None:
        match item:
            case rtc.AudioFrame():
                return _Command(audioChunk={"content": base64.b64encode(item.data).decode()})
            case stt.Flush():
                if self.input.closed:
                    return None
                return _Command(endTurn={})
            case stt.EndOfAudio():
                return _Command(closeStream={})
            case _:
                assert_never(item)

    def on_message(self, message: _Event) -> None:
        result = message.get("result", {})
        if "speechStarted" in result:
            self.report(stt.SpeechStarted())
            return
        if (t := result.get("transcription")) is None:
            return
        text = t.get("transcript", "")
        is_final = t.get("isFinal", False)
        if text:
            metadata: dict[str, JsonValue] | None = None
            if voice_profile := t.get("voiceProfile"):
                metadata = {"voice_profile": voice_profile}
                if is_final:
                    logger.info(f"Inworld voice profile: {voice_profile}")
            self.report(
                stt.Transcript(
                    text,
                    LanguageCode(self._language),
                    final=is_final,
                    request_id=self._request_id,
                    metadata=metadata,
                )
            )
        # an empty final (a false start, unrecognizable noise) still ends the speech
        if is_final:
            self.report(stt.SpeechEnded())
