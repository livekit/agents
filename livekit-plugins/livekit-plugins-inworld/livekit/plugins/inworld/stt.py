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
from dataclasses import dataclass
from typing import Any

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
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

from ._utils import PeriodicCollector
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
        http_session: aiohttp.ClientSession | None = None,
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
            http_session: Optional aiohttp.ClientSession to use.
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

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._http_session:
            self._http_session = utils.http_context.http_session()
        return self._http_session

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
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
            conn_options=conn_options,
            language=language if utils.is_given(language) else self._opts.language,
        )
        self._streams.add(stream)
        return stream


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        language: str = DEFAULT_LANGUAGE,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._opts.sample_rate)
        self._stt: STT = stt
        self._language = language
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._reconnect_event = asyncio.Event()
        self._speaking = False
        self._request_id = ""
        self._audio_duration_collector: PeriodicCollector[float] = PeriodicCollector(
            callback=self._on_audio_duration_report,
            duration=5.0,
        )

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._stt._http_session:
            self._stt._http_session = utils.http_context.http_session()
        return self._stt._http_session

    def _build_transcribe_config(self) -> dict:
        opts = self._stt._opts
        config: dict = {
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

        inworld_v1_config: dict = {
            "minEndOfTurnSilenceWhenConfident": opts.min_end_of_turn_silence_when_confident,
        }
        if utils.is_given(opts.vad_threshold):
            inworld_v1_config["vadThreshold"] = opts.vad_threshold
        config["inworldSttV1Config"] = inworld_v1_config

        return config

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        ws_url = self._stt._base_url.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = ws_url.rstrip("/") + "/" + WS_ENDPOINT

        ws = await asyncio.wait_for(
            self._ensure_session().ws_connect(
                ws_url,
                headers={"Authorization": self._stt._authorization},
            ),
            timeout=self._conn_options.timeout,
        )

        self._request_id = utils.shortuuid()
        await ws.send_str(json.dumps({"transcribeConfig": self._build_transcribe_config()}))
        logger.debug("Inworld STT WebSocket connection established")
        return ws

    async def _run(self) -> None:
        while True:
            try:
                ws = await self._connect_ws()
                self._ws = ws

                tasks = [
                    asyncio.create_task(self._send_audio_task()),
                    asyncio.create_task(self._recv_messages_task()),
                ]
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())

                tasks_group: asyncio.Future[Any] = asyncio.gather(*tasks)
                try:
                    done, _ = await asyncio.wait(
                        [tasks_group, wait_reconnect_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
                    tasks_group.cancel()
                    tasks_group.exception()

            except asyncio.TimeoutError as e:
                logger.error(f"Timeout during Inworld STT connection: {e}")
                raise APITimeoutError("Timeout connecting to Inworld STT API") from e
            except aiohttp.ClientResponseError as e:
                logger.error(f"Inworld STT status error: {e.status} {e.message}")
                raise APIStatusError(
                    message=e.message, status_code=e.status, request_id=None, body=None
                ) from e
            except aiohttp.ClientError as e:
                logger.error(f"Inworld STT connection error: {e}")
                raise APIConnectionError(f"Inworld STT connection error: {e}") from e
            except Exception as e:
                logger.exception(f"Unexpected error in Inworld STT: {e}")
                raise APIConnectionError(f"Unexpected error: {e}") from e
            finally:
                if self._ws is not None:
                    await self._ws.close()
                    self._ws = None

    async def _send_audio_task(self) -> None:
        if not self._ws:
            return

        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                self._audio_duration_collector.flush()
                try:
                    await self._ws.send_str(json.dumps({"endTurn": {}}))
                except Exception as e:
                    logger.error(f"Error sending endTurn: {e}")
                    break
            elif isinstance(data, rtc.AudioFrame):
                self._audio_duration_collector.push(data.duration)
                pcm_bytes = data.data.tobytes()
                audio_b64 = base64.b64encode(pcm_bytes).decode()
                try:
                    await self._ws.send_str(json.dumps({"audioChunk": {"content": audio_b64}}))
                except Exception as e:
                    logger.error(f"Error sending audio chunk: {e}")
                    break

        self._audio_duration_collector.flush()
        # Input channel closed — tell the server to close the stream
        if self._ws:
            try:
                await self._ws.send_str(json.dumps({"closeStream": {}}))
            except Exception:
                pass

    def _on_audio_duration_report(self, duration: float) -> None:
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=SpeechEventType.RECOGNITION_USAGE,
                request_id=self._request_id,
                alternatives=[],
                recognition_usage=stt.RecognitionUsage(audio_duration=duration),
            )
        )

    async def _recv_messages_task(self) -> None:
        if not self._ws:
            return

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    self._process_stream_event(data)
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    return
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"Inworld STT WebSocket error: {self._ws.exception()}")
                    return
        except aiohttp.ClientError as e:
            logger.error(f"WebSocket error while receiving: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error receiving messages: {e}")
            raise

    def _process_stream_event(self, data: dict) -> None:
        result = data.get("result", {})

        if "speechStarted" in result and not self._speaking:
            self._speaking = True
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=SpeechEventType.START_OF_SPEECH,
                    request_id=self._request_id,
                )
            )
            return

        t = result.get("transcription", {})
        if not t:
            return

        text = t.get("transcript", "")
        is_final = t.get("isFinal", False)
        voice_profile = t.get("voiceProfile")

        # An empty-text final (VAD false positive, unrecognizable noise) must still
        # reach the END_OF_SPEECH emission below — otherwise _speaking stays True
        # and subsequent speechStarted events are ignored, wedging the stream.
        if not text and not is_final:
            return

        if text:
            event_type = (
                SpeechEventType.FINAL_TRANSCRIPT if is_final else SpeechEventType.INTERIM_TRANSCRIPT
            )

            metadata = None
            if voice_profile:
                metadata = {"voice_profile": voice_profile}
                if is_final:
                    logger.info(f"Inworld voice profile: {voice_profile}")

            self._event_ch.send_nowait(
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
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=SpeechEventType.END_OF_SPEECH,
                    request_id=self._request_id,
                )
            )
