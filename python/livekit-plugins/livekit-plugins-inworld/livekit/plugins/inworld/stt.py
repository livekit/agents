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
import base64
import json
import os
import weakref
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    http,
    stt,
    utils,
)
from livekit.agents.stt import SpeechEventType
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


class _Transcription(BaseModel):
    model_config = ConfigDict(strict=True)

    transcript: str = ""
    is_final: bool = Field(default=False, alias="isFinal")
    voice_profile: JsonValue = Field(default=None, alias="voiceProfile")


class _TranscriptionResult(BaseModel):
    model_config = ConfigDict(strict=True)

    speech_started: object = Field(default=None, alias="speechStarted")
    transcription: _Transcription | None = None


class _StreamEvent(BaseModel):
    model_config = ConfigDict(strict=True)

    result: _TranscriptionResult = Field(default_factory=_TranscriptionResult)


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

    def _ensure_session(self) -> http.HttpSession:
        if not self._http_session:
            self._http_session = http.http_session()
        return self._http_session

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
            get_session=self._ensure_session,
            base_url=self._base_url,
            authorization=self._authorization,
            conn_options=conn_options,
            language=language if utils.is_given(language) else self._opts.language,
        )
        self._streams.add(stream)
        return stream


class SpeechStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt: STT,
        opts: _STTOptions,
        get_session: Callable[[], http.HttpSession],
        base_url: str,
        authorization: str,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        language: str = DEFAULT_LANGUAGE,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._stt: STT = stt
        self._opts = opts
        self._get_session = get_session
        self._base_url = base_url
        self._authorization = authorization
        self._language = language
        self._ws: http.WebSocket | None = None
        self._reconnect_event = asyncio.Event()
        self._speaking = False
        self._input_closed = False
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

    async def _connect_ws(self) -> http.WebSocket:
        ws_url = self._base_url.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = ws_url.rstrip("/") + "/" + WS_ENDPOINT

        ws = await self._get_session().ws_connect(
            ws_url,
            headers={"Authorization": self._authorization},
            conn_options=self._conn_options,
        )

        self._request_id = utils.shortuuid()
        await ws.send_str(json.dumps({"transcribeConfig": self._build_transcribe_config()}))
        logger.debug("Inworld STT WebSocket connection established")
        return ws

    async def _run(self) -> None:
        while True:
            self._input_closed = False
            try:
                ws = await self._connect_ws()
                self._ws = ws

                async with utils.aio.TaskSet() as tasks:
                    io = asyncio.gather(
                        tasks.spawn(self._send_audio_task()),
                        tasks.spawn(self._recv_messages_task()),
                    )
                    reconnect = tasks.spawn(self._reconnect_event.wait())
                    if await tasks.first_completed(io, reconnect) is not reconnect:
                        break

                    self._reconnect_event.clear()

            except TimeoutError as e:
                raise APITimeoutError("Timeout connecting to Inworld STT API") from e
            except (APIConnectionError, APIStatusError):
                raise
            except Exception as e:
                raise APIConnectionError(f"Unexpected error: {e}") from e
            finally:
                if self._speaking:
                    self._speaking = False
                    if not self.closed:
                        self.send(
                            stt.SpeechEvent(
                                type=SpeechEventType.END_OF_SPEECH, request_id=self._request_id
                            )
                        )
                if self._ws is not None:
                    await self._ws.close()
                    self._ws = None

    async def _send_audio_task(self) -> None:
        if not self._ws:
            return

        async for data in self.input:
            if isinstance(data, self.Flush):
                await self._ws.send_str(json.dumps({"endTurn": {}}))
            else:
                pcm_bytes = data.data.tobytes()
                audio_b64 = base64.b64encode(pcm_bytes).decode()
                await self._ws.send_str(json.dumps({"audioChunk": {"content": audio_b64}}))

        # Input channel closed: tell the server to close the stream
        self._input_closed = True
        await self._ws.send_str(json.dumps({"closeStream": {}}))

    async def _recv_messages_task(self) -> None:
        if not self._ws:
            return

        async for msg in self._ws:
            if msg.type == http.WebSocketMessageType.TEXT:
                try:
                    event = _StreamEvent.model_validate_json(msg.data)
                except ValidationError as exc:
                    raise APIConnectionError(
                        "Inworld STT sent an invalid event", retryable=False
                    ) from exc
                self._process_stream_event(event)
        if not self._input_closed:
            raise self._ws.close_error("Inworld STT connection closed unexpectedly")

    def _process_stream_event(self, event: _StreamEvent) -> None:
        result = event.result

        if "speech_started" in result.model_fields_set and not self._speaking:
            self._speaking = True
            self.send(
                stt.SpeechEvent(
                    type=SpeechEventType.START_OF_SPEECH,
                    request_id=self._request_id,
                )
            )
            return

        t = result.transcription
        if not t:
            return

        text = t.transcript
        is_final = t.is_final
        voice_profile = t.voice_profile

        # An empty-text final (VAD false positive, unrecognizable noise) must still
        # reach the END_OF_SPEECH emission below — otherwise _speaking stays True
        # and subsequent speechStarted events are ignored, wedging the stream.
        if not text and not is_final:
            return

        if text:
            event_type = (
                SpeechEventType.FINAL_TRANSCRIPT if is_final else SpeechEventType.INTERIM_TRANSCRIPT
            )

            metadata: dict[str, JsonValue] | None = None
            if voice_profile:
                metadata = {"voice_profile": voice_profile}
                if is_final:
                    logger.info(f"Inworld voice profile: {voice_profile}")

            self.send(
                stt.SpeechEvent(
                    type=event_type,
                    request_id=self._request_id,
                    alternatives=[
                        stt.SpeechData(
                            text=text,
                            language=LanguageCode(self._language),
                            metadata=metadata,
                        )
                    ],
                )
            )

        if is_final and self._speaking:
            self._speaking = False
            self.send(
                stt.SpeechEvent(
                    type=SpeechEventType.END_OF_SPEECH,
                    request_id=self._request_id,
                )
            )
