"""Nabrah STT plugin for LiveKit Agents.

The backend relays the recognizer's raw output verbatim no word timings, no
confidence, no end-of-turn interpretation  so everything above the transport
lives here: token stripping, EOT detection, debounce, cross-utterance
accumulation and audio-position tracking.

Wire contract (server -> client):
    {"type": "ready"}
    {"type": "error", "message": ...}
    {"type": "transcript", "text": ..., "is_final": bool,
     "audio_processed": float}

Two facts drive the whole design:

- `text` is cumulative *within one recognizer utterance* and restarts from
  scratch after every `is_final`. `is_final` is utterance-level, not
  turn-level  one spoken sentence routinely produces a dozen  so a final is
  only a diff-cursor reset here, never a turn boundary.
- `audio_processed` is cumulative seconds of audio the recognizer has actually
  consumed. It is the authoritative audio clock; a send-side count runs ahead
  of realtime when input arrives in bursts. The key may be absent on older
  backends, so the local counter stays as a fallback.
"""

import asyncio
import json
import logging
import os
import time
from typing import Literal

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    stt,
    utils,
)
from livekit.agents.language import LanguageCode
from livekit.agents.types import NOT_GIVEN, NotGivenOr, TimedString
from livekit.agents.utils import is_given

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "wss://api.nabrah.ai/api/ext/stt/ws"

NabrahRecognitionModel = Literal["eot_nabrah", ""]

EOT_TOKEN = "<eot>"
EOT_PUNCTUATION = (".", "?", "!", "؟")
_NO_SPACE_BEFORE = frozenset(".,?!:;،؛؟")

_WATCHDOG_POLL_SECONDS = 0.1


def _strip_and_detect_eot(new_text: str) -> tuple[str, bool]:
    """Strip <eot> from a transcript diff and flag an end-of-turn candidate.

    Detection runs on the raw diff so a trailing token isn't lost by its own
    removal. The token becomes a SPACE, never ""  the recognizer doesn't
    reliably pad it, and stripping to empty glues the neighbours together
    ("الله" + "تعالى" -> "اللهتعالى").
    """
    stripped_new_text = new_text.rstrip()
    is_candidate = stripped_new_text.endswith(EOT_TOKEN) or (
        stripped_new_text[-1:] in EOT_PUNCTUATION
    )
    return new_text.replace(EOT_TOKEN, " "), is_candidate


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _word_text(w: dict) -> str:
    return str(w.get("word", "")).replace(EOT_TOKEN, "").strip()


def _timed_word(w: dict, offset: float) -> TimedString:
    """One backend word entry as a TimedString. Timestamps arrive in ms."""
    return TimedString(
        _word_text(w),
        start_time=w.get("start_time", 0) / 1000.0 + offset,
        end_time=w.get("end_time", 0) / 1000.0 + offset,
        confidence=w.get("confidence", NOT_GIVEN),
    )


class STT(stt.STT):
    def __init__(
        self,
        *,
        recognition_model: NabrahRecognitionModel | str = "eot_nabrah",
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        language: str = "ar-SA",
        end_of_utterance_silence_ms: int = -1,
        disable_number_normalization: bool = False,
        priority_words: list[str] | None = None,
        priority_words_strength: float = 0.5,
        max_silence_before_finalize_seconds: float | None = 1.5,
        end_of_turn_confirm_delay_seconds: float | None = 0.4,
        http_session: aiohttp.ClientSession | None = None,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript=False,
                offline_recognize=False,
            ),
        )
        nabrah_api_key = api_key if is_given(api_key) else os.environ.get("NABRAH_API_KEY")
        if not nabrah_api_key:
            raise ValueError(
                "Nabrah API key is required, either as argument or by setting "
                "the NABRAH_API_KEY environment variable",
            )
        self._api_key = nabrah_api_key
        self._base_url = (
            base_url
            if is_given(base_url)
            else os.environ.get(
                "NABRAH_STT_URL",
                DEFAULT_BASE_URL,
            )
        )

        self._recognition_model = recognition_model
        self._language = language
        self._end_of_utterance_silence_ms = end_of_utterance_silence_ms
        self._disable_number_normalization = disable_number_normalization
        self._priority_words = priority_words or []
        self._priority_words_strength = priority_words_strength
        self._max_silence_before_finalize_seconds = max_silence_before_finalize_seconds
        self._end_of_turn_confirm_delay_seconds = end_of_turn_confirm_delay_seconds
        self._session = http_session
        self._label = "nabrah.STT (nabrah-stt-v1)"

    @property
    def model(self) -> str:
        return "nabrah-stt-v1"

    @property
    def provider(self) -> str:
        return "nabrah"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("nabrah STT does not support single-shot recognition")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        lang = language if is_given(language) else self._language
        return SpeechStream(
            stt_instance=self,
            conn_options=conn_options,
            language=lang,
            http_session=self._ensure_session(),
        )


class SpeechStream(stt.SpeechStream):
    _utt_raw: str = ""
    _utt_clean: str = ""
    _utt_closed: bool = False
    _utt_flushed_clean: str = ""
    _input_done: bool = False
    _turn_words: tuple[TimedString, ...] = ()
    _utt_words: tuple[TimedString, ...] = ()
    _utt_flushed_words: int = 0
    _utt_raw_seen: int = 0

    def __init__(
        self,
        *,
        stt_instance: STT,
        conn_options: APIConnectOptions,
        language: str,
        http_session: aiohttp.ClientSession,
    ):
        super().__init__(stt=stt_instance, conn_options=conn_options, sample_rate=16000)
        self._stt: STT = stt_instance
        self._language = LanguageCode(language)
        self._session = http_session

        self._is_speaking = False

        self._turn_text = ""
        self._utt_clean = ""
        self._utt_raw = ""
        self._utt_closed = False
        self._utt_flushed_clean = ""
        self._turn_words = ()
        self._utt_words = ()
        self._utt_flushed_words = 0
        self._utt_raw_seen = 0

        self._segment_start_time: float = 0.0
        self._segment_end_time: float = 0.0
        self._request_id: str = ""

        self._last_progress_at: float = 0.0
        self._pending_eot_at: float | None = None

        self._latest_audio_processed: float | None = None
        self._audio_position: float = 0.0
        self._reported_audio_position: float = 0.0
        self._last_message_position: float = 0.0

    def _config_frame(self) -> dict:
        return {
            "api_key": self._stt._api_key,
            "recognition_model": self._stt._recognition_model,
            "end_of_utterance_silence_ms": self._stt._end_of_utterance_silence_ms,
            "disable_number_normalization": self._stt._disable_number_normalization,
            "priority_words": self._stt._priority_words,
            "priority_words_strength": self._stt._priority_words_strength,
        }

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        try:
            return await asyncio.wait_for(
                self._session.ws_connect(self._stt._base_url),
                self._conn_options.timeout,
            )
        except (
            aiohttp.ClientConnectorError,
            aiohttp.WSServerHandshakeError,
            asyncio.TimeoutError,
        ) as e:
            raise APIConnectionError("failed to connect to nabrah STT") from e

    async def _run(self) -> None:
        self._latest_audio_processed = None
        self._audio_position = 0.0
        self._reported_audio_position = 0.0
        self._last_message_position = 0.0
        self._input_done = False

        ws = await self._connect_ws()
        try:
            await ws.send_str(json.dumps(self._config_frame()))

            async def _await_ready() -> None:
                while True:
                    msg = await ws.receive()
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        raise APIConnectionError("nabrah STT closed before ready")
                    payload = json.loads(msg.data)
                    if payload.get("type") == "error":
                        raise RuntimeError(
                            payload.get("message", "nabrah backend rejected config"),
                        )
                    if payload.get("type") == "ready":
                        return

            try:
                await asyncio.wait_for(_await_ready(), self._conn_options.timeout)
            except asyncio.TimeoutError as e:
                raise APIConnectionError("nabrah STT never sent ready") from e

            watchdog_task = asyncio.create_task(self._eot_watchdog())
            send_task = asyncio.create_task(self._send_task(ws))
            recv_task = asyncio.create_task(self._recv_task(ws))
            try:
                await asyncio.gather(send_task, recv_task)
            finally:
                watchdog_task.cancel()
                await utils.aio.gracefully_cancel(send_task, recv_task, watchdog_task)
                if self._current_text() or self._is_speaking:
                    self._flush_eos()
        finally:
            await ws.close()

    async def _eot_watchdog(self) -> None:
        timeout = self._stt._max_silence_before_finalize_seconds
        confirm_delay = self._stt._end_of_turn_confirm_delay_seconds
        while True:
            await asyncio.sleep(_WATCHDOG_POLL_SECONDS)
            now = time.monotonic()
            if (
                confirm_delay is not None
                and self._pending_eot_at is not None
                and now - self._pending_eot_at >= confirm_delay
            ):
                self._flush_eos()
                continue
            if (
                timeout is not None
                and self._current_text()
                and now - self._last_progress_at >= timeout
            ):
                self._flush_eos()

    async def _send_task(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        try:
            async for data in self._input_ch:
                if isinstance(data, rtc.AudioFrame):
                    audio_bytes = data.data.tobytes()
                    if audio_bytes:
                        self._audio_position += data.samples_per_channel / data.sample_rate
                        await ws.send_bytes(audio_bytes)
            self._input_done = True
            await ws.send_str(json.dumps({"type": "eof"}))
        except (aiohttp.ClientError, ConnectionError) as e:
            raise APIConnectionError("nabrah STT send failed") from e

    async def _recv_task(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while True:
            msg = await ws.receive()
            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                if not self._input_done:
                    raise APIConnectionError("nabrah STT closed unexpectedly")
                return
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                self._process_message(json.loads(msg.data))
            except APIError:
                raise
            except Exception:
                logger.exception("failed to process nabrah STT message")

    def _emit(self, event: stt.SpeechEvent) -> None:
        self._event_ch.send_nowait(event)

    def _audio_clock(self) -> float:
        """Audio-stream position for this connection, in seconds.

        The backend's audio_processed is what the recognizer actually consumed,
        so it cannot run ahead of realtime the way a send-side count does when
        input arrives in bursts. Falls back to audio pushed if absent.
        """
        if self._latest_audio_processed is not None:
            return self._latest_audio_processed
        return self._audio_position

    def _turn_word_list(self) -> tuple[TimedString, ...]:
        """Every word in the turn so far: committed utterances plus the open one."""
        return self._turn_words + self._utt_words

    def _current_text(self) -> str:
        head = self._utt_clean.lstrip()[:1]
        sep = "" if head in _NO_SPACE_BEFORE else " "
        return _normalize_whitespace(self._turn_text + sep + self._utt_clean)

    def _flush_eos(self) -> None:
        text = self._current_text()
        end_time = self._segment_end_time or (self._audio_clock() + self.start_time_offset)
        start_time = min(self._segment_start_time, end_time)

        if text:
            self._emit(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=[
                        stt.SpeechData(
                            language=self._language,
                            text=text,
                            start_time=start_time,
                            end_time=end_time,
                            words=list(self._turn_word_list()) or None,
                        )
                    ],
                )
            )

            audio_clock = self._audio_clock()
            usage_duration = audio_clock - self._reported_audio_position
            if usage_duration > 0:
                self._emit(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.RECOGNITION_USAGE,
                        recognition_usage=stt.RecognitionUsage(
                            audio_duration=usage_duration,
                        ),
                    )
                )
                self._reported_audio_position = audio_clock

        if self._is_speaking:
            self._emit(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.END_OF_SPEECH,
                    alternatives=[
                        stt.SpeechData(
                            language=self._language,
                            text="",
                            start_time=start_time,
                            end_time=end_time,
                        )
                    ],
                )
            )

        self._utt_flushed_clean = _normalize_whitespace(
            self._utt_flushed_clean + " " + self._utt_clean,
        )
        # index into the backend's raw `words` list, so count raw entries consumed,
        # not the filtered survivors, or blank entries shift the cursor backwards
        # and already-emitted words get replayed.
        self._utt_flushed_words = self._utt_raw_seen

        self._turn_text = ""
        self._utt_clean = ""
        self._turn_words = ()
        self._utt_words = ()
        self._is_speaking = False
        self._pending_eot_at = None
        self._segment_start_time = 0.0
        self._segment_end_time = 0.0
        self._request_id = ""

    def _emit_preflight(self) -> None:
        text = self._current_text()
        if not text:
            return

        self._emit(
            stt.SpeechEvent(
                type=stt.SpeechEventType.PREFLIGHT_TRANSCRIPT,
                request_id=self._request_id,
                alternatives=[
                    stt.SpeechData(
                        language=self._language,
                        text=text,
                        start_time=self._segment_start_time,
                        end_time=self._segment_end_time,
                    )
                ],
            )
        )

    def _process_message(self, data: dict) -> None:
        msg_type = data.get("type")

        if msg_type == "error":
            raise APIStatusError(
                message=data.get("message", "nabrah STT error"),
                status_code=-1,
                request_id=None,
                body=None,
                retryable=False,
            )

        if msg_type != "transcript":
            return

        text = data.get("text", "")
        utt_final = bool(data.get("is_final"))

        previous_audio_position = self._last_message_position
        reported = data.get("audio_processed")
        if reported is not None and (
            self._latest_audio_processed is None or reported > self._latest_audio_processed
        ):
            self._latest_audio_processed = reported
        self._last_message_position = self._audio_clock()

        if not text:
            return

        clean_now, _ = _strip_and_detect_eot(text)
        clean_now = _normalize_whitespace(clean_now)
        clean_prev, _ = _strip_and_detect_eot(self._utt_raw)
        clean_prev = _normalize_whitespace(clean_prev)
        if clean_prev:
            continues = clean_now.startswith(clean_prev) or (
                not self._utt_closed and clean_prev.startswith(clean_now)
            )
        else:
            continues = True
        if not continues:
            self._turn_text = _normalize_whitespace(
                self._turn_text + " " + self._utt_clean,
            )
            self._utt_clean = ""
            self._utt_raw = ""
            self._utt_flushed_clean = ""
            self._turn_words = self._turn_words + self._utt_words
            self._utt_words = ()
            self._utt_flushed_words = 0
            self._utt_raw_seen = 0
        self._utt_closed = False

        raw_words = data.get("words") or []
        if raw_words:
            self._utt_words = tuple(
                _timed_word(w, self.start_time_offset)
                for w in raw_words[self._utt_flushed_words :]
                if _word_text(w)
            )
            self._utt_raw_seen = len(raw_words)

        new_text = text[len(self._utt_raw) :] if text.startswith(self._utt_raw) else text
        _, is_eot = _strip_and_detect_eot(new_text)
        self._utt_clean = (
            clean_now[len(self._utt_flushed_clean) :]
            if self._utt_flushed_clean and clean_now.startswith(self._utt_flushed_clean)
            else clean_now
        )
        self._utt_raw = text
        if utt_final:
            self._utt_closed = True

        if not new_text:
            return

        has_real_progress = bool(new_text.replace(EOT_TOKEN, "").strip())

        if has_real_progress:
            self._last_progress_at = time.monotonic()
            self._pending_eot_at = None

            turn_words = self._turn_word_list()

            word_start = turn_words[0].start_time if turn_words else NOT_GIVEN
            word_end = turn_words[-1].end_time if turn_words else NOT_GIVEN

            if not self._is_speaking:
                self._is_speaking = True
                self._segment_start_time = (
                    word_start
                    if is_given(word_start)
                    else min(previous_audio_position, self._audio_clock()) + self.start_time_offset
                )
                self._emit(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.START_OF_SPEECH,
                        speech_start_time=(
                            self.start_time - self.start_time_offset + self._segment_start_time
                        ),
                    )
                )
            self._segment_end_time = (
                word_end if is_given(word_end) else self._audio_clock() + self.start_time_offset
            )

        if not self._request_id:
            self._request_id = utils.shortuuid("nabrah-")

        current = self._current_text()
        if current:
            self._emit(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    alternatives=[stt.SpeechData(language=self._language, text=current)],
                )
            )

        if is_eot:
            if self._stt._end_of_turn_confirm_delay_seconds is None:
                self._flush_eos()
            elif self._pending_eot_at is None:
                self._pending_eot_at = time.monotonic()
                self._emit_preflight()
