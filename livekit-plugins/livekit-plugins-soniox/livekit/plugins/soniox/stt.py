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

# Base URL for Soniox Speech-to-Text API.
BASE_URL = "wss://stt-rt.soniox.com/transcribe-websocket"

# WebSocket messages and tokens.
KEEPALIVE_MESSAGE = '{"type": "keepalive"}'
END_TOKEN = "<end>"
FINALIZED_TOKEN = "<fin>"


def is_end_token(token: dict) -> bool:
    """Return True if the given token marks an end or finalized event."""
    return token.get("text") in (END_TOKEN, FINALIZED_TOKEN)


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
    """Translation configuration for the Soniox Speech-to-Text API.

    See: https://soniox.com/docs/stt/api-reference/websocket-api
    """

    type: Literal["one_way", "two_way"]
    target_language: str | None = None
    """Target language for one-way translation."""
    language_a: str | None = None
    """First language for two-way translation."""
    language_b: str | None = None
    """Second language for two-way translation."""

    def __post_init__(self) -> None:
        if self.type == "one_way" and not self.target_language:
            raise ValueError("target_language is required for one_way translation")
        if self.type == "two_way" and not (self.language_a and self.language_b):
            raise ValueError("language_a and language_b are both required for two_way translation")


@dataclass
class STTOptions:
    """Configuration options for Soniox Speech-to-Text service."""

    model: str = "stt-rt-v4"

    language_hints: list[str] | None = None
    language_hints_strict: bool = False
    context: ContextObject | str | None = None

    num_channels: int = 1
    sample_rate: int = 16000

    enable_speaker_diarization: bool = False
    enable_language_identification: bool = True

    max_endpoint_delay_ms: int = 500
    """Maximum delay in milliseconds between speech cessation and endpoint detection.
    Range: 500–3000.
    See: https://soniox.com/docs/stt/rt/endpoint-detection"""

    client_reference_id: str | None = None
    translation: TranslationConfig | None = None

    def __post_init__(self) -> None:
        if not (500 <= self.max_endpoint_delay_ms <= 3000):
            raise ValueError("max_endpoint_delay_ms must be between 500 and 3000")


class STT(stt.STT):
    """Speech-to-Text service using Soniox Speech-to-Text API.

    This service connects to Soniox Speech-to-Text API for real-time transcription
    with support for multiple languages, custom context, speaker diarization,
    and more.

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
        """Set up state and queues for a WebSocket-based transcription stream."""
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._params.sample_rate)
        self._stt: STT = stt
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._reconnect_event = asyncio.Event()

        self.audio_queue: asyncio.Queue[bytes | str] = asyncio.Queue()

        self._reported_duration_ms = 0

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp ClientSession for WebSocket connections."""
        if not self._stt._http_session:
            self._stt._http_session = utils.http_context.http_session()

        return self._stt._http_session

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        """Open a WebSocket connection to the Soniox Speech-to-Text API and send the
        initial configuration."""
        context_raw = self._stt._params.context
        context_value: dict[str, Any] | str | None
        if isinstance(context_raw, ContextObject):
            context_value = asdict(context_raw)
        else:
            context_value = context_raw

        # Create initial config object.
        config: dict[str, Any] = {
            "api_key": self._stt._api_key,
            "model": self._stt._params.model,
            "audio_format": "pcm_s16le",
            "num_channels": self._stt._params.num_channels or 1,
            "enable_endpoint_detection": True,
            "sample_rate": self._stt._params.sample_rate,
            "language_hints": self._stt._params.language_hints,
            "language_hints_strict": self._stt._params.language_hints_strict,
            "context": context_value,
            "enable_speaker_diarization": self._stt._params.enable_speaker_diarization,
            "enable_language_identification": self._stt._params.enable_language_identification,
            "client_reference_id": self._stt._params.client_reference_id,
        }
        config["max_endpoint_delay_ms"] = self._stt._params.max_endpoint_delay_ms
        if self._stt._params.translation is not None:
            tr = self._stt._params.translation
            translation_dict: dict[str, Any] = {"type": tr.type}
            if tr.type == "one_way":
                translation_dict["target_language"] = tr.target_language
            elif tr.type == "two_way":
                translation_dict["language_a"] = tr.language_a
                translation_dict["language_b"] = tr.language_b
            config["translation"] = translation_dict
        # Connect to the Soniox Speech-to-Text API.
        ws = await asyncio.wait_for(
            self._ensure_session().ws_connect(self._stt._base_url),
            timeout=self._conn_options.timeout,
        )
        # Set initial configuration message.
        await ws.send_str(json.dumps(config))
        logger.debug("Soniox Speech-to-Text API connection established!")

        # Reset duration tracking on new connection
        self._reported_duration_ms = 0
        return ws

    def _report_processed_audio_duration(self, total_audio_proc_ms: float) -> None:
        """Report the total audio duration processed by the STT engine."""
        to_report_ms = total_audio_proc_ms - self._reported_duration_ms
        if to_report_ms <= 0:
            return

        usage_event = stt.SpeechEvent(
            type=stt.SpeechEventType.RECOGNITION_USAGE,
            alternatives=[],
            recognition_usage=stt.RecognitionUsage(
                audio_duration=to_report_ms / 1000,
            ),
        )
        self._event_ch.send_nowait(usage_event)
        self._reported_duration_ms = int(total_audio_proc_ms)

    async def _run(self) -> None:
        """Manage connection lifecycle, spawning tasks and handling reconnection."""
        while True:
            try:
                ws = await self._connect_ws()
                self._ws = ws
                # Create task for audio processing, voice turn detection and message handling.
                tasks: list[asyncio.Task[None]] = [
                    asyncio.create_task(self._prepare_audio_task()),
                    asyncio.create_task(self._send_audio_task()),
                    asyncio.create_task(self._recv_messages_task()),
                    asyncio.create_task(self._keepalive_task()),
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
                logger.error(
                    f"Timeout during Soniox Speech-to-Text API connection/initialization: {e}"
                )
                raise APITimeoutError(
                    "Timeout connecting to or initializing Soniox Speech-to-Text API session"
                ) from e

            except aiohttp.ClientResponseError as e:
                logger.error(
                    "Soniox Speech-to-Text API status error during session init:"
                    + f"{e.status} {e.message}"
                )
                raise APIStatusError(
                    message=e.message, status_code=e.status, request_id=None, body=None
                ) from e

            except aiohttp.ClientError as e:
                logger.error(f"Soniox Speech-to-Text API connection error: {e}")
                raise APIConnectionError(f"Soniox Speech-to-Text API connection error: {e}") from e

            except Exception as e:
                logger.exception(f"Unexpected error occurred: {e}")
                raise APIConnectionError(f"An unexpected error occurred: {e}") from e
            # Close the WebSocket connection on finish.
            finally:
                if self._ws is not None:
                    await self._ws.close()
                    self._ws = None

    async def _keepalive_task(self) -> None:
        """Periodically send keepalive messages (while no audio is being sent)
        to maintain the WebSocket connection."""
        try:
            while self._ws:
                await self._ws.send_str(KEEPALIVE_MESSAGE)
                await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error while sending keep alive message: {e}")

    async def _prepare_audio_task(self) -> None:
        """Read audio frames and enqueue PCM data for sending."""
        if not self._ws:
            logger.error("WebSocket connection to Soniox Speech-to-Text API is not established")
            return

        async for data in self._input_ch:
            if isinstance(data, rtc.AudioFrame):
                # Get the raw bytes from the audio frame.
                pcm_data = data.data.tobytes()
                self.audio_queue.put_nowait(pcm_data)

    async def _send_audio_task(self) -> None:
        """Take queued audio data and transmit it over the WebSocket."""
        if not self._ws:
            logger.error("WebSocket connection to Soniox Speech-to-Text API is not established")
            return

        while self._ws:
            try:
                data = await self.audio_queue.get()

                if isinstance(data, bytes):
                    await self._ws.send_bytes(data)
                else:
                    await self._ws.send_str(data)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error while sending audio data: {e}")
                break

    async def _recv_messages_task(self) -> None:
        """Receive transcription messages, handle tokens, errors, and dispatch events."""

        # final tokens are accumulated across messages until an endpoint is detected.
        final = _TokenAccumulator()
        final_original = _TokenAccumulator()
        is_speaking = False

        def send_endpoint_transcript() -> None:
            nonlocal is_speaking
            if final.text:
                src_segs = final_original._lang_segments
                source_languages = [LanguageCode(lang) for lang, _ in src_segs] or None
                source_texts = [t for _, t in src_segs] or None
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[
                            final.to_speech_data(
                                self.start_time_offset,
                                source_languages=source_languages,
                                source_texts=source_texts,
                            )
                        ],
                    )
                )
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=SpeechEventType.END_OF_SPEECH,
                    )
                )

                # Reset buffers.
                final.reset()
                final_original.reset()

                # Reset speaking state, so the next transcript will send START_OF_SPEECH again.
                is_speaking = False
            else:
                final_original.reset()

        is_translation_mode = self._stt._params.translation is not None
        # Method handles receiving messages from the Soniox Speech-to-Text API.
        while self._ws:
            try:
                async for msg in self._ws:
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        break

                    if msg.type != aiohttp.WSMsgType.TEXT:
                        logger.warning(
                            f"Unexpected message type from Soniox Speech-to-Text API: {msg.type}"
                        )
                        continue

                    try:
                        content = json.loads(msg.data)
                        tokens = content["tokens"]

                        non_final = _TokenAccumulator()
                        non_final_original = _TokenAccumulator()
                        total_audio_proc_ms = content.get("total_audio_proc_ms", 0)

                        # 1) process tokens: accumulate final/non-final,
                        #    flush immediately on endpoint tokens.
                        for token in tokens:
                            is_translated = token.get("translation_status") == "translation"
                            if (
                                is_translation_mode
                                and not is_end_token(token)
                                and not is_translated
                            ):
                                # Original-language token: capture text for source_text only.
                                if token["is_final"]:
                                    final_original.update(token)
                                else:
                                    non_final_original.update(token)
                                continue
                            if token["is_final"]:
                                if is_end_token(token):
                                    send_endpoint_transcript()
                                    self._report_processed_audio_duration(
                                        total_audio_proc_ms,
                                    )
                                else:
                                    final.update(token)
                            else:
                                non_final.update(token)

                        # 2) emit START_OF_SPEECH + transcript for remaining content.
                        if final.text or non_final.text:
                            if not is_speaking:
                                is_speaking = True
                                self._event_ch.send_nowait(
                                    stt.SpeechEvent(type=SpeechEventType.START_OF_SPEECH)
                                )
                            interim_segs = _merge_lang_segments(
                                final_original._lang_segments, non_final_original._lang_segments
                            )
                            interim_src_langs = [
                                LanguageCode(lang) for lang, _ in interim_segs
                            ] or None
                            interim_src_texts = [t for _, t in interim_segs] or None

                            # When all tokens in this batch are final (no non-final pending),
                            # speech has reached a stable state — emit PREFLIGHT_TRANSCRIPT to
                            # allow preemptive LLM generation. This mirrors Deepgram v2's
                            # EagerEndOfTurn behavior.
                            event_type = (
                                SpeechEventType.PREFLIGHT_TRANSCRIPT
                                if final.text and not non_final.text
                                else SpeechEventType.INTERIM_TRANSCRIPT
                            )
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(
                                    type=event_type,
                                    alternatives=[
                                        final.merged_speech_data(
                                            non_final,
                                            self.start_time_offset,
                                            source_languages=interim_src_langs,
                                            source_texts=interim_src_texts,
                                        )
                                    ],
                                )
                            )

                        # 3) on error or finish, flush any remaining final tokens.
                        if (
                            content.get("finished")
                            or content.get("error_code")
                            or content.get("error_message")
                        ):
                            send_endpoint_transcript()
                            self._report_processed_audio_duration(total_audio_proc_ms)

                        if content.get("error_code") or content.get("error_message"):
                            logger.error(
                                f"WebSocket error: {content.get('error_code')}"
                                f" - {content.get('error_message')}"
                            )

                        if content.get("finished"):
                            logger.debug("Transcription finished")

                    except Exception as e:
                        logger.exception(f"Error processing message: {e}")

            except aiohttp.ClientError as e:
                logger.error(f"WebSocket error while receiving: {e}")
            except Exception as e:
                logger.error(f"Unexpected error while receiving messages: {e}")


def _merge_lang_segments(
    a: list[tuple[str, str]], b: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Merge two (language, text) segment lists, combining adjacent segments of the same language."""
    result = list(a)
    for lang, text in b:
        if result and result[-1][0] == lang:
            lang, t = result[-1]
            result[-1] = (lang, t + text)
        else:
            result.append((lang, text))
    return result


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
        self._lang_segments: list[tuple[str, str]] = []  # (language, text) pairs

    def update(self, token: dict[str, Any]) -> None:
        text = token["text"]
        lang = token.get("language", "")
        self.text += text
        if lang and not self.language:
            self.language = lang
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
        if text:
            if self._lang_segments and self._lang_segments[-1][0] == lang:
                lang, t = self._lang_segments[-1]
                self._lang_segments[-1] = (lang, t + text)
            else:
                self._lang_segments.append((lang, text))

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
        self._lang_segments = []

    def to_speech_data(
        self,
        start_time_offset: float = 0.0,
        source_languages: list[LanguageCode] | None = None,
        source_texts: list[str] | None = None,
    ) -> stt.SpeechData:
        return stt.SpeechData(
            text=self.text,
            language=LanguageCode(self.language),
            source_languages=source_languages,
            source_texts=source_texts,
            speaker_id=self.speaker_id,
            start_time=self.start_time / 1000 + start_time_offset,
            end_time=self.end_time / 1000 + start_time_offset,
            confidence=self.confidence,
        )

    def merged_speech_data(
        self,
        other: _TokenAccumulator,
        start_time_offset: float = 0.0,
        source_languages: list[LanguageCode] | None = None,
        source_texts: list[str] | None = None,
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
            source_languages=source_languages,
            source_texts=source_texts,
            speaker_id=self.speaker_id if self.speaker_id is not None else other.speaker_id,
            start_time=start / 1000 + start_time_offset,
            end_time=end / 1000 + start_time_offset,
            confidence=total_sum / total_count if total_count > 0 else 0.0,
        )
