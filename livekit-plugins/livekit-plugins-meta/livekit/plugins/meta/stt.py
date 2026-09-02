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

from __future__ import annotations

import asyncio
import json
import math
import os
import time
import weakref
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

DEFAULT_URL = "wss://api.meta.ai/v1/asr/realtime"
DEFAULT_MODEL = "muse-voice-transcribe-1.0"
_SAMPLE_RATE = 24_000
_CHANNELS = 1
_SAMPLE_WIDTH_BYTES = 2
_CHUNK_DURATION = 0.08
_CHUNK_BYTES = int(_SAMPLE_RATE * _CHANNELS * _SAMPLE_WIDTH_BYTES * _CHUNK_DURATION)
_MAX_MESSAGE_BYTES = 1024 * 1024
_MAX_COMPLETED_TURNS = 128
_SUPPORTED_LANGUAGES = (
    "Arabic",
    "Bengali",
    "Dutch",
    "English",
    "French",
    "German",
    "Hebrew",
    "Hindi",
    "Indonesian",
    "Italian",
    "Japanese",
    "Kannada",
    "Korean",
    "Malay",
    "Mandarin Chinese",
    "Marathi",
    "Polish",
    "Portuguese",
    "Spanish",
    "Tagalog",
    "Tamil",
    "Telugu",
    "Thai",
    "Turkish",
    "Vietnamese",
)
_LANGUAGE_NAMES = {language.casefold(): language for language in _SUPPORTED_LANGUAGES}
_LANGUAGE_CODES = {
    "ar": "Arabic",
    "bn": "Bengali",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fil": "Tagalog",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "iw": "Hebrew",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "ms": "Malay",
    "mr": "Marathi",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "zh": "Mandarin Chinese",
}
_RETRYABLE_CLOSE_CODES = frozenset((1011, 1013))
_NON_RETRYABLE_CLOSE_CODES = frozenset((1000, 1008))


@dataclass(slots=True)
class _TurnState:
    provider_started: bool = False
    emitted_start: bool = False
    latest_interim: str | None = None
    emitted_interim: str | None = None
    final_text: str | None = None
    final_emitted: bool = False
    ended: bool = False


class STT(stt.STT[Any]):
    """Streaming speech recognition with Meta Muse Voice Transcribe."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        url: str = DEFAULT_URL,
        keywords: list[str] | None = None,
        language_bias: list[str] | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a Meta Muse streaming STT provider.

        Args:
            api_key: Meta Model API key. Falls back to ``MODEL_API_KEY``.
            model: Muse Voice Transcribe model identifier.
            url: Realtime Muse ASR WebSocket endpoint. Must use ``wss://``.
            keywords: Static recognition keywords sent when each stream starts.
            language_bias: Static supported language names sent when each stream starts.
            http_session: Optional aiohttp session. By default, the LiveKit HTTP
                context session is used.
        """
        resolved_key = (api_key if api_key is not None else os.getenv("MODEL_API_KEY", "")).strip()
        if not resolved_key:
            raise ValueError("Meta Model API key is required. Pass api_key or set MODEL_API_KEY")
        if not model.strip():
            raise ValueError("model must be non-empty")

        parsed_url = urlparse(url)
        if (
            parsed_url.scheme != "wss"
            or not parsed_url.hostname
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.fragment
        ):
            raise ValueError("url must be an absolute wss:// URL without credentials or a fragment")

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=False,
                aligned_transcript=False,
                offline_recognize=False,
                keyterms=False,
            )
        )
        self._api_key = self._normalize_access_token(resolved_key)
        self._model = model.strip()
        self._url = url
        self._keywords = self._normalize_hints(keywords, name="keywords")
        self._language_bias = self._normalize_language_bias(language_bias)
        self._http_session = http_session
        self._streams: weakref.WeakSet[SpeechStream] = weakref.WeakSet()
        self._closed = False

    @staticmethod
    def _normalize_hints(values: list[str] | None, *, name: str) -> list[str]:
        normalized: list[str] = []
        for value in values or ():
            hint = value.strip()
            if not hint:
                raise ValueError(f"{name} entries must be non-empty")
            if hint not in normalized:
                normalized.append(hint)
        return normalized

    @staticmethod
    def _normalize_access_token(api_key: str) -> str:
        parts = api_key.split(None, 1)
        if parts and parts[0].casefold() == "bearer":
            if len(parts) != 2 or not parts[1].strip():
                raise ValueError("Meta Model API key must include a token after Bearer")
            return f"Bearer {parts[1].strip()}"
        return f"Bearer {api_key}"

    @staticmethod
    def _normalize_language_bias(values: list[str] | None) -> list[str]:
        normalized: list[str] = []
        for value in values or ():
            documented_name = _LANGUAGE_NAMES.get(value.strip().casefold())
            if documented_name is None:
                supported = ", ".join(_SUPPORTED_LANGUAGES)
                raise ValueError(
                    f"unsupported language_bias entry {value!r}; supported: {supported}"
                )
            if documented_name not in normalized:
                normalized.append(documented_name)
        return normalized

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "Meta"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            self._http_session = utils.http_context.http_session()
        return self._http_session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise APIError(
            "Meta Muse Voice Transcribe supports streaming recognition only",
            retryable=False,
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        if self._closed:
            raise RuntimeError("Meta STT is closed")

        language_bias = list(self._language_bias)
        if is_given(language):
            language_hint = self._normalize_language_hint(str(language))
            if language_hint not in language_bias:
                language_bias.append(language_hint)

        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            api_key=self._api_key,
            model=self._model,
            url=self._url,
            keywords=list(self._keywords),
            language_bias=language_bias,
            http_session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream

    @staticmethod
    def _normalize_language_hint(language: str) -> str:
        value = language.strip()
        if not value:
            raise ValueError("language must be non-empty")

        documented_name = _LANGUAGE_NAMES.get(value.casefold())
        if documented_name is not None:
            return documented_name

        primary = value.replace("_", "-").split("-", 1)[0].casefold()
        mapped_name = _LANGUAGE_CODES.get(primary)
        if mapped_name is None:
            supported = ", ".join(_SUPPORTED_LANGUAGES)
            raise ValueError(
                f"unsupported Muse Voice language {language!r}; supported: {supported}"
            )
        return mapped_name

    async def aclose(self) -> None:
        self._closed = True
        streams = tuple(self._streams)
        if streams:
            await asyncio.gather(*(stream.aclose() for stream in streams), return_exceptions=True)


class SpeechStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        api_key: str,
        model: str,
        url: str,
        keywords: list[str],
        language_bias: list[str],
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=_SAMPLE_RATE)
        self._api_key = api_key
        self._model = model
        self._url = url
        self._keywords = keywords
        self._language_bias = language_bias
        self._session = http_session
        self._session_id = ""
        self._turns: OrderedDict[str, _TurnState] = OrderedDict()
        self._provider_active_turn_id: str | None = None
        self._completed_turn_ids: set[str] = set()
        self._completed_turn_order: deque[str] = deque()
        self._audio_consumed = False
        self._end_stream_sent = False
        self._last_audio_processed_ms = 0.0

    async def _run(self) -> None:
        self._end_stream_sent = False
        self._last_audio_processed_ms = 0.0
        ws: aiohttp.ClientWebSocketResponse | None = None
        tasks: list[asyncio.Task[None]] = []
        try:
            ws = await self._connect_ws()
            sender = asyncio.create_task(self._send_audio(ws), name="meta-stt-send")
            receiver = asyncio.create_task(self._receive_events(ws), name="meta-stt-receive")
            tasks = [sender, receiver]
            await self._drive_tasks(sender, receiver)
        except asyncio.CancelledError:
            raise
        except APIError as exc:
            if self._audio_consumed and exc.retryable:
                raise APIConnectionError(
                    "Meta Muse realtime ASR failed after audio was consumed",
                    retryable=False,
                ) from None
            raise
        except Exception as exc:
            phase = "audio streaming" if self._audio_consumed else "connection"
            raise APIConnectionError(
                f"Meta Muse realtime ASR {phase} failed ({type(exc).__name__})",
                retryable=not self._audio_consumed,
            ) from None
        finally:
            if tasks:
                await utils.aio.gracefully_cancel(*tasks)
            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        started_at = time.perf_counter()
        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    self._url,
                    max_msg_size=_MAX_MESSAGE_BYTES,
                ),
                timeout=self._conn_options.timeout,
            )
        except asyncio.TimeoutError:
            raise APITimeoutError("Meta Muse realtime ASR connection timed out") from None
        except aiohttp.ClientResponseError as exc:
            raise APIStatusError(
                "Meta Muse realtime ASR connection was rejected",
                status_code=exc.status,
                body=None,
            ) from None
        except Exception as exc:
            raise APIConnectionError(
                f"Meta Muse realtime ASR connection failed ({type(exc).__name__})"
            ) from None

        try:
            await ws.send_str(json.dumps(self._handshake(), separators=(",", ":")))
            raw = await asyncio.wait_for(ws.receive(), timeout=self._conn_options.timeout)
            message = self._parse_ws_message(
                raw,
                phase="handshake",
                close_code=ws.close_code,
            )
            self._accept_handshake(message)
        except asyncio.CancelledError:
            await self._close_quietly(ws)
            raise
        except asyncio.TimeoutError:
            await self._close_quietly(ws)
            raise APITimeoutError("Meta Muse realtime ASR handshake timed out") from None
        except APIError:
            await self._close_quietly(ws)
            raise
        except Exception as exc:
            await self._close_quietly(ws)
            raise APIConnectionError(
                f"Meta Muse realtime ASR handshake failed ({type(exc).__name__})"
            ) from None

        self._report_connection_acquired(time.perf_counter() - started_at, False)
        return ws

    def _handshake(self) -> dict[str, object]:
        handshake: dict[str, object] = {
            "mode": "ENDPOINTING",
            "authorization": {"accessToken": self._api_key},
            "audioEncoding": "PCM_24KHZ",
            "model": self._model,
            "partialMode": "CUMULATIVE",
            "emitAudioProgress": True,
        }
        if self._keywords:
            handshake["keywords"] = self._keywords
        if self._language_bias:
            handshake["languageBias"] = self._language_bias
        return handshake

    def _accept_handshake(self, message: dict[str, Any]) -> None:
        if message.get("type") == "error":
            raise self._server_error(message, phase="handshake")
        session_id = message.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            raise APIConnectionError(
                "Meta Muse realtime ASR sent an invalid handshake response",
                retryable=False,
            )
        self._session_id = session_id

    async def _drive_tasks(self, sender: asyncio.Task[None], receiver: asyncio.Task[None]) -> None:
        done, _ = await asyncio.wait((sender, receiver), return_when=asyncio.FIRST_COMPLETED)
        if receiver in done:
            receiver.result()
            if not self._end_stream_sent:
                raise APIConnectionError(
                    "Meta Muse realtime ASR closed before input ended",
                    retryable=not self._audio_consumed,
                )
            if not sender.done():
                await sender
        else:
            sender.result()
            try:
                await asyncio.wait_for(receiver, timeout=self._conn_options.timeout)
            except asyncio.TimeoutError:
                raise APITimeoutError(
                    "Meta Muse realtime ASR timed out while draining final events",
                    retryable=not self._audio_consumed,
                ) from None
        self._validate_clean_close()

    async def _send_audio(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        pending = bytearray()
        loop = asyncio.get_running_loop()
        pacing_origin: float | None = None
        sent_duration = 0.0

        async def send_packet(packet: bytes) -> None:
            nonlocal pacing_origin, sent_duration
            if not packet:
                return
            if pacing_origin is None:
                pacing_origin = loop.time()
            deadline = pacing_origin + sent_duration
            delay = deadline - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                await ws.send_bytes(packet)
            except Exception as exc:
                raise APIConnectionError(
                    f"Meta Muse realtime ASR audio send failed ({type(exc).__name__})",
                    retryable=False,
                ) from None
            duration = len(packet) / (_SAMPLE_RATE * _CHANNELS * _SAMPLE_WIDTH_BYTES)
            sent_duration += duration

        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                if pending:
                    await send_packet(bytes(pending))
                    pending.clear()
                continue

            self._audio_consumed = True
            if item.num_channels != _CHANNELS:
                raise APIError("Meta Muse realtime ASR requires mono audio", retryable=False)
            pending.extend(item.data.tobytes())
            while len(pending) >= _CHUNK_BYTES:
                await send_packet(bytes(pending[:_CHUNK_BYTES]))
                del pending[:_CHUNK_BYTES]

        if pending:
            await send_packet(bytes(pending))
        if not self._end_stream_sent:
            try:
                await ws.send_str('{"type":"endStream"}')
            except Exception as exc:
                raise APIConnectionError(
                    f"Meta Muse realtime ASR end-of-input send failed ({type(exc).__name__})",
                    retryable=False,
                ) from None
            self._end_stream_sent = True

    async def _receive_events(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while True:
            try:
                raw = await ws.receive()
            except Exception as exc:
                raise APIConnectionError(
                    f"Meta Muse realtime ASR receive failed ({type(exc).__name__})",
                    retryable=not self._audio_consumed,
                ) from None

            if raw.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                close_code = raw.data if isinstance(raw.data, int) else ws.close_code
                if self._end_stream_sent and close_code == 1000:
                    return
                raise self._close_error(close_code, phase="stream")
            if raw.type == aiohttp.WSMsgType.CLOSING:
                continue
            if raw.type == aiohttp.WSMsgType.ERROR:
                raise APIConnectionError(
                    "Meta Muse realtime ASR WebSocket failed",
                    retryable=not self._audio_consumed,
                ) from None

            message = self._parse_ws_message(raw, phase="stream")
            event_type = message.get("type")
            if event_type == "error":
                raise self._server_error(message, phase="stream")
            if event_type == "speechStart":
                self._speech_start(message)
            elif event_type == "transcript":
                self._transcript(message)
            elif event_type == "speechEnd":
                self._speech_end(message)
            elif event_type == "speechComplete":
                self._speech_complete(message)
            elif event_type == "audioProgress":
                self._audio_progress(message)

    def _audio_progress(self, message: dict[str, Any]) -> None:
        processed_ms = message.get("audioProcessedMs")
        if (
            isinstance(processed_ms, bool)
            or not isinstance(processed_ms, (int, float))
            or not math.isfinite(processed_ms)
            or processed_ms < 0
        ):
            raise self._protocol_error("audioProgress event has invalid audioProcessedMs")
        if processed_ms <= self._last_audio_processed_ms:
            return
        delta_seconds = (processed_ms - self._last_audio_processed_ms) / 1000
        self._last_audio_processed_ms = float(processed_ms)
        self._emit_usage(delta_seconds)

    def _speech_start(self, message: dict[str, Any]) -> None:
        turn_id = self._required_turn_id(message, event="speechStart")
        if turn_id in self._completed_turn_ids:
            return
        turn = self._turns.setdefault(turn_id, _TurnState())
        turn.provider_started = True
        self._provider_active_turn_id = turn_id
        self._drain_turns()

    def _transcript(self, message: dict[str, Any]) -> None:
        text = message.get("transcript")
        if not isinstance(text, str):
            raise self._protocol_error("transcript event has invalid text")
        if not text and message.get("turnId") is None and self._provider_active_turn_id is None:
            return
        turn_id = self._transcript_turn_id(message)
        if turn_id in self._completed_turn_ids:
            return
        turn = self._turns.setdefault(turn_id, _TurnState())
        if turn.final_text is not None or turn.latest_interim == text:
            return
        turn.latest_interim = text
        self._drain_turns()

    def _speech_end(self, message: dict[str, Any]) -> None:
        turn_id = self._required_turn_id(message, event="speechEnd")
        if turn_id in self._completed_turn_ids:
            return
        turn = self._turns.setdefault(turn_id, _TurnState())
        turn.ended = True
        if self._provider_active_turn_id == turn_id:
            self._provider_active_turn_id = None
        self._drain_turns()

    def _speech_complete(self, message: dict[str, Any]) -> None:
        turn_id = self._required_turn_id(message, event="speechComplete")
        if turn_id in self._completed_turn_ids:
            return
        text = message.get("transcript")
        if not isinstance(text, str):
            raise self._protocol_error("speechComplete event has invalid transcript")
        turn = self._turns.setdefault(turn_id, _TurnState())
        if turn.final_text is None:
            turn.final_text = text
        self._drain_turns()

    def _drain_turns(self) -> None:
        while self._turns:
            turn_id = next(iter(self._turns))
            turn = self._turns[turn_id]
            has_content = turn.latest_interim is not None or turn.final_text is not None
            if not turn.emitted_start and (turn.provider_started or has_content):
                turn.emitted_start = True
                self._emit(stt.SpeechEventType.START_OF_SPEECH, turn_id)

            if (
                turn.emitted_start
                and not turn.final_emitted
                and turn.latest_interim is not None
                and turn.latest_interim != turn.emitted_interim
            ):
                turn.emitted_interim = turn.latest_interim
                self._emit(
                    stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    turn_id,
                    turn.latest_interim,
                )

            if turn.emitted_start and not turn.final_emitted and turn.final_text is not None:
                turn.final_emitted = True
                self._emit(stt.SpeechEventType.FINAL_TRANSCRIPT, turn_id, turn.final_text)

            if not (turn.final_emitted and turn.ended):
                return

            self._emit(stt.SpeechEventType.END_OF_SPEECH, turn_id)
            del self._turns[turn_id]
            self._remember_completed_turn(turn_id)

    def _remember_completed_turn(self, turn_id: str) -> None:
        if turn_id in self._completed_turn_ids:
            return
        if len(self._completed_turn_order) >= _MAX_COMPLETED_TURNS:
            oldest = self._completed_turn_order.popleft()
            self._completed_turn_ids.discard(oldest)
        self._completed_turn_order.append(turn_id)
        self._completed_turn_ids.add(turn_id)

    def _emit(self, event_type: stt.SpeechEventType, turn_id: str, text: str | None = None) -> None:
        alternatives: list[stt.SpeechData] = []
        if text is not None:
            alternatives.append(stt.SpeechData(language=LanguageCode(""), text=text))
        self._event_ch.send_nowait(
            stt.SpeechEvent(type=event_type, request_id=turn_id, alternatives=alternatives)
        )

    def _emit_usage(self, duration: float) -> None:
        if duration <= 0:
            return
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=self._session_id,
                recognition_usage=stt.RecognitionUsage(audio_duration=duration),
            )
        )

    def _transcript_turn_id(self, message: dict[str, Any]) -> str:
        value = message.get("turnId")
        if value is not None:
            return self._normalize_turn_id(value, event="transcript")

        if self._provider_active_turn_id is not None:
            return self._provider_active_turn_id
        raise self._protocol_error("transcript event is missing turnId outside an active turn")

    def _required_turn_id(self, message: dict[str, Any], *, event: str) -> str:
        value = message.get("turnId")
        if value is None:
            raise self._protocol_error(f"{event} event is missing turnId")
        return self._normalize_turn_id(value, event=event)

    def _normalize_turn_id(self, value: object, *, event: str) -> str:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise self._protocol_error(f"{event} event has an invalid turnId")
        turn_id = str(value).strip()
        if not turn_id:
            raise self._protocol_error(f"{event} event has an invalid turnId")
        return turn_id

    def _validate_clean_close(self) -> None:
        if self._turns:
            raise APIConnectionError(
                "Meta Muse realtime ASR closed with incomplete speech turns",
                retryable=False,
            )

    def _server_error(self, message: dict[str, Any], *, phase: str) -> APIStatusError:
        return APIStatusError(
            f"Meta Muse realtime ASR {phase} error",
            status_code=400,
            request_id=None,
            body=None,
            retryable=False,
        )

    @staticmethod
    def _protocol_error(detail: str) -> APIConnectionError:
        return APIConnectionError(
            f"Meta Muse realtime ASR protocol error: {detail}",
            retryable=False,
        )

    @staticmethod
    def _close_error(close_code: int | None, *, phase: str) -> APIStatusError:
        code = close_code or -1
        if code in _NON_RETRYABLE_CLOSE_CODES:
            retryable = False
        elif code in _RETRYABLE_CLOSE_CODES:
            retryable = True
        else:
            retryable = True
        return APIStatusError(
            f"Meta Muse realtime ASR closed during {phase}",
            status_code=code,
            body=None,
            retryable=retryable,
        )

    @classmethod
    def _parse_ws_message(
        cls,
        raw: aiohttp.WSMessage,
        *,
        phase: str,
        close_code: int | None = None,
    ) -> dict[str, Any]:
        if raw.type != aiohttp.WSMsgType.TEXT or not isinstance(raw.data, str):
            if raw.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                raw_close_code = raw.data if isinstance(raw.data, int) else close_code
                raise cls._close_error(raw_close_code, phase=phase)
            raise cls._protocol_error(f"unexpected message type during {phase}")
        try:
            message = json.loads(raw.data)
        except (json.JSONDecodeError, TypeError):
            raise cls._protocol_error(f"invalid JSON during {phase}") from None
        if not isinstance(message, dict):
            raise cls._protocol_error(f"non-object message during {phase}")
        return message

    @staticmethod
    async def _close_quietly(ws: aiohttp.ClientWebSocketResponse) -> None:
        try:
            await ws.close()
        except Exception:
            pass
