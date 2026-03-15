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
import contextlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Literal

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
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
    NotGivenOr,
)

from .log import logger
from .models import SonioxLanguages, SonioxRTModels

BASE_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
KEEPALIVE_MSG = '{"type": "keepalive"}'
FINALIZE_MSG = '{"type": "finalize"}'
_END_TOKENS = frozenset(("<end>", "<fin>"))


@dataclass
class ContextGeneralItem:
    key: str
    value: str


@dataclass
class ContextTranslationTerm:
    source: str
    target: str


@dataclass
class ContextObject:
    """Context object for models with context_version 2, for Soniox stt-rt-v3-preview and higher.

    Learn more about context in the documentation:
    https://soniox.com/docs/stt/concepts/context
    """

    general: list[ContextGeneralItem] | None = None
    text: str | None = None
    terms: list[str] | None = None
    translation_terms: list[ContextTranslationTerm] | None = None


@dataclass
class TranslationConfig:
    """Configuration for Soniox real-time translation.

    See: https://soniox.com/docs/stt/rt/real-time-translation
    """

    type: Literal["one_way", "two_way"]
    target_language: LgType | None = None
    """Required for one_way: translate all speech into this language (e.g. ``"es"``)."""
    language_a: LgType | None = None
    """Required for two_way: first language (e.g. ``"en"``)."""
    language_b: LgType | None = None
    """Required for two_way: second language (e.g. ``"de"``)."""


LgType = SonioxLanguages | str


@dataclass
class STTOptions:
    """Configuration options for Soniox Speech-to-Text service."""

    model: SonioxRTModels | str = "stt-rt-v4"

    language_hints: list[LgType] | None = None
    language_hints_strict: bool = False
    context: ContextObject | str | None = None

    num_channels: int = 1
    sample_rate: int = 16000

    enable_speaker_diarization: bool = False
    enable_language_identification: bool = True

    client_reference_id: str | None = None

    max_endpoint_delay_ms: int | None = None
    """Must be between 500 and 3000 when set. None uses the API default (2000)."""
    translation: TranslationConfig | None = None


class STT(stt.STT):
    """Speech-to-Text service using Soniox Speech-to-Text API.

    This service connects to Soniox Speech-to-Text API for real-time transcription
    with support for multiple languages, custom context, speaker diarization,
    real-time translation, and more.

    For complete API documentation, see: https://soniox.com/docs/stt/api-reference/websocket-api
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
        params: STTOptions | None = None,
    ):
        """Initialize instance of Soniox Speech-to-Text API service.

        Args:
            api_key: Soniox API key, if not provided, will look for SONIOX_API_KEY env variable.
            base_url: Base URL for Soniox Speech-to-Text API, default to BASE_URL defined in this
                module.
            http_session: Optional aiohttp.ClientSession to use for requests.
            params: Additional configuration parameters, such as model, language hints, context and
                speaker diarization.
        """
        params = params or STTOptions()
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript="chunk",
                offline_recognize=False,
                diarization=params.enable_speaker_diarization,
            )
        )

        self._api_key = api_key or os.getenv("SONIOX_API_KEY")
        if not self._api_key:
            raise ValueError("Soniox API key is required. Set SONIOX_API_KEY or pass api_key")
        self._base_url = base_url
        self._http_session = http_session
        self._params = params

    @property
    def model(self) -> str:
        return self._params.model

    @property
    def provider(self) -> str:
        return "Soniox"

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Raise error since single-frame recognition is not supported
        by Soniox Speech-to-Text API."""
        raise NotImplementedError(
            "Soniox Speech-to-Text API does not support single frame recognition"
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        """Return a new LiveKit streaming speech-to-text session."""
        return SpeechStream(
            stt=self,
            conn_options=conn_options,
        )


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        stt: STT,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._params.sample_rate)
        self._stt: STT = stt
        self._reported_duration_ms: float = 0

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        params = self._stt._params
        config: dict[str, Any] = {
            "api_key": self._stt._api_key,
            "model": params.model,
            "audio_format": "pcm_s16le",
            "num_channels": params.num_channels or 1,
            "sample_rate": params.sample_rate,
            "enable_endpoint_detection": True,
            "max_endpoint_delay_ms": params.max_endpoint_delay_ms,
            "language_hints": params.language_hints,
            "language_hints_strict": params.language_hints_strict,
            "context": (
                asdict(params.context)
                if isinstance(params.context, ContextObject)
                else params.context
            ),
            "enable_speaker_diarization": params.enable_speaker_diarization,
            "enable_language_identification": params.enable_language_identification,
            "client_reference_id": params.client_reference_id,
            "translation": (
                {k: v for k, v in asdict(params.translation).items() if v is not None}
                if params.translation is not None
                else None
            ),
        }
        config = {k: v for k, v in config.items() if v is not None}

        session = self._stt._http_session or utils.http_context.http_session()
        self._stt._http_session = session
        ws = await asyncio.wait_for(
            session.ws_connect(self._stt._base_url),
            timeout=self._conn_options.timeout,
        )
        await ws.send_str(json.dumps(config))
        self._reported_duration_ms = 0
        return ws

    def _report_processed_audio_duration(self, total_audio_proc_ms: float) -> None:
        to_report_ms = total_audio_proc_ms - self._reported_duration_ms
        if to_report_ms <= 0:
            return

        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                alternatives=[],
                recognition_usage=stt.RecognitionUsage(
                    audio_duration=to_report_ms / 1000,
                ),
            )
        )
        self._reported_duration_ms = int(total_audio_proc_ms)

    async def _run(self) -> None:
        try:
            ws = await self._connect_ws()
            try:
                send = asyncio.create_task(self._send_task(ws), name="soniox-send")
                recv = asyncio.create_task(self._recv_task(ws), name="soniox-recv")
                keepalive = asyncio.create_task(self._keepalive_task(ws), name="soniox-keepalive")
                await asyncio.gather(send, recv)
            finally:
                await utils.aio.gracefully_cancel(send, recv, keepalive)
                with contextlib.suppress(Exception):
                    await ws.close()
        except asyncio.TimeoutError as e:
            raise APITimeoutError("Timeout connecting to Soniox") from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from e
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Soniox connection error: {e}") from e

    @utils.log_exceptions(logger=logger)
    async def _send_task(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        async for data in self._input_ch:
            if isinstance(data, rtc.AudioFrame):
                await ws.send_bytes(data.data.tobytes())
            elif isinstance(data, self._FlushSentinel):
                await ws.send_str(FINALIZE_MSG)
        await ws.send_str("")

    async def _keepalive_task(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        try:
            while True:
                await ws.send_str(KEEPALIVE_MSG)
                await asyncio.sleep(5)
        except Exception:
            return

    @utils.log_exceptions(logger=logger)
    async def _recv_task(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        final: dict[str, _TokenAccumulator] = {
            "original": _TokenAccumulator(),
            "translation": _TokenAccumulator(),
        }
        is_speaking = False

        def flush_endpoint(audio_proc_ms: float) -> None:
            nonlocal is_speaking
            alternatives = [
                final[cat].to_speech_data(self.start_time_offset)
                for cat in ("original", "translation")
                if final[cat].text
            ]
            if alternatives:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=alternatives,
                    )
                )
                self._event_ch.send_nowait(stt.SpeechEvent(type=SpeechEventType.END_OF_SPEECH))
            for acc in final.values():
                acc.reset()
            is_speaking = False
            self._report_processed_audio_duration(audio_proc_ms)

        async for msg in ws:
            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                break

            if msg.type != aiohttp.WSMsgType.TEXT:
                logger.warning("unexpected message type from Soniox: %s", msg.type)
                continue

            try:
                content = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.warning("malformed JSON from Soniox")
                continue

            tokens = content.get("tokens", [])
            total_audio_proc_ms = content.get("total_audio_proc_ms", 0)
            non_final: dict[str, _TokenAccumulator] = {
                "original": _TokenAccumulator(),
                "translation": _TokenAccumulator(),
            }

            for token in tokens:
                cat = (
                    "translation"
                    if token.get("translation_status") == "translation"
                    else "original"
                )
                if token["is_final"]:
                    if token.get("text") in _END_TOKENS:
                        flush_endpoint(total_audio_proc_ms)
                    else:
                        final[cat].update(token)
                else:
                    non_final[cat].update(token)

            has_content = any(
                final[c].text or non_final[c].text for c in ("original", "translation")
            )
            if has_content:
                if not is_speaking:
                    is_speaking = True
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=SpeechEventType.START_OF_SPEECH)
                    )
                alternatives = [
                    final[c].merged_speech_data(non_final[c], self.start_time_offset)
                    for c in ("original", "translation")
                    if final[c].text or non_final[c].text
                ]
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=SpeechEventType.INTERIM_TRANSCRIPT,
                        alternatives=alternatives,
                    )
                )

            if content.get("error_code"):
                code = content["error_code"]
                error_msg = content.get("error_message", "")
                logger.error("Soniox error: %s - %s", code, error_msg)
                flush_endpoint(total_audio_proc_ms)
                if code >= 500:
                    raise APIConnectionError(f"Soniox server error {code}: {error_msg}")
                raise APIStatusError(
                    message=error_msg, status_code=code, request_id=None, body=None
                )

            if content.get("finished"):
                flush_endpoint(total_audio_proc_ms)
                return

        raise APIConnectionError("Soniox connection closed unexpectedly")


class _TokenAccumulator:
    """Accumulates token metadata (text, language, speaker, timing, confidence).

    Tokens are assumed to arrive in chronological order, so start_time is taken
    from the first token and end_time is continuously overwritten by the latest.
    """

    def __init__(self) -> None:
        self.text: str = ""
        self.language: str = ""
        self.speaker_id: str | None = None
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self._confidence_sum: float = 0.0
        self._confidence_count: int = 0
        self._has_start_time: bool = False

    def update(self, token: dict[str, Any]) -> None:
        self.text += token["text"]
        if token.get("language") and not self.language:
            self.language = token["language"]
        if "speaker" in token and self.speaker_id is None:
            self.speaker_id = str(token["speaker"])
        if "start_ms" in token and not self._has_start_time:
            self._has_start_time = True
            self.start_time = float(token["start_ms"])
        if "end_ms" in token:
            self.end_time = float(token["end_ms"])
        if "confidence" in token:
            self._confidence_sum += token["confidence"]
            self._confidence_count += 1

    @property
    def confidence(self) -> float:
        if self._confidence_count == 0:
            return 0.0
        return self._confidence_sum / self._confidence_count

    def reset(self) -> None:
        self.text = ""
        self.language = ""
        self.speaker_id = None
        self.start_time = 0.0
        self.end_time = 0.0
        self._confidence_sum = 0.0
        self._confidence_count = 0
        self._has_start_time = False

    def to_speech_data(self, start_time_offset: float = 0.0) -> stt.SpeechData:
        return stt.SpeechData(
            text=self.text,
            language=LanguageCode(self.language),
            speaker_id=self.speaker_id,
            start_time=self.start_time / 1000 + start_time_offset,
            end_time=self.end_time / 1000 + start_time_offset,
            confidence=self.confidence,
        )

    def merged_speech_data(
        self, other: _TokenAccumulator, start_time_offset: float = 0.0
    ) -> stt.SpeechData:
        """Build a SpeechData combining self (final) with other (non-final)."""
        candidates = [acc.start_time for acc in (self, other) if acc._has_start_time]
        start = min(candidates) if candidates else 0.0
        end = max(self.end_time, other.end_time)
        total_count = self._confidence_count + other._confidence_count
        total_sum = self._confidence_sum + other._confidence_sum
        return stt.SpeechData(
            text=self.text + other.text,
            language=LanguageCode(self.language if self.language else other.language),
            speaker_id=self.speaker_id if self.speaker_id is not None else other.speaker_id,
            start_time=start / 1000 + start_time_offset,
            end_time=end / 1000 + start_time_offset,
            confidence=total_sum / total_count if total_count > 0 else 0.0,
        )
