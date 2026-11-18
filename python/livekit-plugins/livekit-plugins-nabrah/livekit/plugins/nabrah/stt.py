# Copyright 2026 LiveKit, Inc.
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

"""Nabrah STT plugin for LiveKit Agents.

The backend relays recognizer output with minimal interpretation, so the plugin
handles token stripping, EOT detection, its confirmation, cross-utterance accumulation
and word conversion.

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
  consumed on the connection: the usage reported. The key may be absent on older
  backends.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict, assert_never

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIError,
    LanguageCode,
    stt,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr, TranscriptChunk
from livekit.agents.utils import is_given

DEFAULT_BASE_URL = "wss://api.nabrah.ai/api/ext/stt/ws"

NabrahRecognitionModel = Literal["eot_nabrah", ""]

EOT_TOKEN = "<eot>"
EOT_PUNCTUATION = (".", "?", "!", "؟")
_NO_SPACE_BEFORE = frozenset(".,?!:;،؛؟")


def _strip_and_detect_eot(new_text: str) -> tuple[str, bool]:
    """Strip <eot> from a transcript diff and flag an end-of-turn candidate.

    Detection runs on the raw diff so a trailing token isn't lost by its own
    removal. The token becomes a SPACE, never ""; the recognizer doesn't
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


def _append_text(existing: str, new: str) -> str:
    """Join transcript fragments without inserting spaces before punctuation."""
    head = new.lstrip()[:1]
    separator = "" if not existing or head in _NO_SPACE_BEFORE else " "
    return _normalize_whitespace(existing + separator + new)


class _Word(TypedDict):
    word: NotRequired[str]
    start_time: NotRequired[float]
    end_time: NotRequired[float]
    confidence: NotRequired[float]


class _Ready(TypedDict):
    type: Literal["ready"]


class _Error(TypedDict):
    type: Literal["error"]


class _Transcript(TypedDict):
    type: Literal["transcript"]
    text: NotRequired[str]
    is_final: NotRequired[bool]
    audio_processed: NotRequired[float]
    words: NotRequired[list[_Word]]


class _Config(TypedDict):
    api_key: str
    recognition_model: str
    end_of_utterance_silence_ms: int
    disable_number_normalization: bool
    priority_words: list[str]
    priority_words_strength: float


class _Eof(TypedDict):
    type: Literal["eof"]


def _word_text(word: _Word) -> str:
    return word.get("word", "").replace(EOT_TOKEN, "").strip()


def _timed_word(word: _Word, offset: float) -> TranscriptChunk:
    """One backend word entry as a TranscriptChunk. Timestamps arrive in ms."""
    return TranscriptChunk(
        _word_text(word),
        start_time=word.get("start_time", 0.0) / 1000.0 + offset,
        end_time=word.get("end_time", 0.0) / 1000.0 + offset,
        confidence=word.get("confidence"),
    )


@dataclass
class _STTOptions:
    api_key: str
    base_url: str
    recognition_model: NabrahRecognitionModel | str
    language: str
    end_of_utterance_silence_ms: int
    disable_number_normalization: bool
    priority_words: list[str]
    priority_words_strength: float
    max_transcript_inactivity_seconds: float | None
    end_of_turn_confirm_delay_seconds: float | None


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
        max_transcript_inactivity_seconds: float | None = None,
        end_of_turn_confirm_delay_seconds: float | None = 0.4,
    ) -> None:
        """
        Args:
            max_transcript_inactivity_seconds: Seconds of audio after the last new words that
                end the turn; ``None`` waits for the end-of-turn token.
            end_of_turn_confirm_delay_seconds: Seconds of audio without new words that confirm
                an end-of-turn token; ``None`` ends the turn at the token.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript=False,
                offline_recognize=False,
                # usage events carry the audio the backend reports as processed
                reports_usage=True,
            ),
        )
        nabrah_api_key = api_key if is_given(api_key) else os.environ.get("NABRAH_API_KEY")
        if not nabrah_api_key:
            raise ValueError(
                "Nabrah API key is required, either as argument or by setting "
                "the NABRAH_API_KEY environment variable",
            )
        self._opts = _STTOptions(
            api_key=nabrah_api_key,
            base_url=base_url
            if is_given(base_url)
            else os.environ.get("NABRAH_STT_URL", DEFAULT_BASE_URL),
            recognition_model=recognition_model,
            language=language,
            end_of_utterance_silence_ms=end_of_utterance_silence_ms,
            disable_number_normalization=disable_number_normalization,
            priority_words=priority_words or [],
            priority_words_strength=priority_words_strength,
            max_transcript_inactivity_seconds=max_transcript_inactivity_seconds,
            end_of_turn_confirm_delay_seconds=end_of_turn_confirm_delay_seconds,
        )
        self._label = "nabrah.STT (nabrah-stt-v1)"

    @property
    def model(self) -> str:
        return "nabrah-stt-v1"

    @property
    def provider(self) -> str:
        return "nabrah"

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
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
    ) -> SpeechStream:
        lang = language if is_given(language) else self._opts.language
        return SpeechStream(stt=self, opts=self._opts, conn_options=conn_options, language=lang)


class SpeechStream(stt.WebSocketStream[_Ready | _Error | _Transcript, _Config | _Eof]):
    awaits_ready = True

    def __init__(
        self, *, stt: STT, opts: _STTOptions, conn_options: APIConnectOptions, language: str
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=16000)
        self._opts = opts
        self._language = LanguageCode(language)
        # the audio after the last new words (or an end-of-turn token) that ends the turn; the
        # same silence with no audio at all ends it too
        waits = [
            wait
            for wait in (
                opts.end_of_turn_confirm_delay_seconds,
                opts.max_transcript_inactivity_seconds,
            )
            if wait is not None
        ]
        self.audio_idle = min(waits) if waits else None
        self._reset()

    def _reset(self) -> None:
        """A connection's recognizer starts from nothing: its utterances, words and clock."""
        self._turn_text = ""
        self._turn_words: tuple[TranscriptChunk, ...] = ()
        self._utt_raw = ""
        self._utt_clean = ""
        self._utt_closed = False
        self._utt_flushed_clean = ""
        self._utt_flushed_chars = 0
        self._utt_words: tuple[TranscriptChunk, ...] = ()
        self._utt_flushed_words = 0
        self._utt_raw_seen = 0
        # seconds of audio sent on the connection, at the last new words and at the pending
        # end-of-turn token
        self._audio_sent = 0.0
        self._progress_at = 0.0
        self._eot_at: float | None = None
        self._reported_audio = 0.0

    async def connect(self, conn_options: APIConnectOptions) -> stt.WebSocketTarget:
        self._reset()
        return stt.WebSocketTarget(self._opts.base_url)

    def setup(self) -> _Config:
        return _Config(
            api_key=self._opts.api_key,
            recognition_model=self._opts.recognition_model,
            end_of_utterance_silence_ms=self._opts.end_of_utterance_silence_ms,
            disable_number_normalization=self._opts.disable_number_normalization,
            priority_words=self._opts.priority_words,
            priority_words_strength=self._opts.priority_words_strength,
        )

    def encode(self, item: rtc.AudioFrame | stt.Flush | stt.EndOfAudio) -> bytes | _Eof | None:
        match item:
            case rtc.AudioFrame():
                self._audio_sent += item.duration
                self._end_turn_if_due(self._audio_sent)
                return bytes(item.data)
            case stt.Flush():
                return None
            case stt.EndOfAudio():
                return _Eof(type="eof")
            case _:
                assert_never(item)

    def on_audio_idle(self) -> None:
        # no audio for the shortest wait: the recognizer hears nothing more
        self._end_turn_if_due(float("inf"))

    def _end_turn_if_due(self, audio: float) -> None:
        confirm = self._opts.end_of_turn_confirm_delay_seconds
        inactivity = self._opts.max_transcript_inactivity_seconds
        if (
            confirm is not None and self._eot_at is not None and audio - self._eot_at >= confirm
        ) or (
            inactivity is not None
            and self._current_text()
            and audio - self._progress_at >= inactivity
        ):
            self._end_turn()

    def on_message(self, message: _Ready | _Error | _Transcript) -> None:
        if message["type"] == "ready":
            self.report(stt.Ready())
        elif message["type"] == "error":
            # its message can hold the customer's words: never attached
            raise APIError("nabrah STT reported an error", kind="invalid_request")
        elif message["type"] == "transcript":
            self._transcript(message)
        else:
            self._skip_unknown_message(message)

    def _turn_word_list(self) -> tuple[TranscriptChunk, ...]:
        """Every word in the turn so far: committed utterances plus the open one."""
        return self._turn_words + self._utt_words

    def _current_text(self) -> str:
        return _append_text(self._turn_text, self._utt_clean)

    def _turn(self, *, final: bool = False, preflight: bool = False) -> stt.Transcript:
        words = self._turn_word_list()
        return stt.Transcript(
            self._current_text(),
            self._language,
            final=final,
            preflight=preflight,
            start_time=words[0].start_time or 0.0 if words else 0.0,
            end_time=words[-1].end_time or 0.0 if words else 0.0,
            words=list(words) or None,
        )

    def _end_turn(self) -> None:
        if self._current_text():
            self.report(self._turn(final=True))
        self._utt_flushed_clean = _append_text(self._utt_flushed_clean, self._utt_clean)
        self._utt_flushed_chars = len(self._utt_flushed_clean)
        # index into the backend's raw `words` list, so count raw entries consumed,
        # not the filtered survivors, or blank entries shift the cursor backwards
        # and already-emitted words get replayed.
        self._utt_flushed_words = self._utt_raw_seen
        self._turn_text = ""
        self._utt_clean = ""
        self._turn_words = ()
        self._utt_words = ()
        self._eot_at = None

    def _transcript(self, message: _Transcript) -> None:
        processed = message.get("audio_processed")
        if processed is not None and processed > self._reported_audio:
            self.send(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.RECOGNITION_USAGE,
                    recognition_usage=stt.RecognitionUsage(
                        audio_duration=processed - self._reported_audio
                    ),
                )
            )
            self._reported_audio = processed

        text = message.get("text", "")
        if not text:
            return

        clean_now, _ = _strip_and_detect_eot(text)
        clean_now = _normalize_whitespace(clean_now)
        if self._utt_closed:
            self._turn_text = _append_text(self._turn_text, self._utt_clean)
            self._utt_clean = ""
            self._utt_raw = ""
            self._utt_flushed_clean = ""
            self._utt_flushed_chars = 0
            self._turn_words = self._turn_words + self._utt_words
            self._utt_words = ()
            self._utt_flushed_words = 0
            self._utt_raw_seen = 0
        self._utt_closed = False

        raw_words = message.get("words", [])
        if raw_words:
            self._utt_words = tuple(
                _timed_word(w, self.start_time_offset)
                for w in raw_words[self._utt_flushed_words :]
                if _word_text(w)
            )
            self._utt_raw_seen = len(raw_words)

        new_text = text[len(self._utt_raw) :] if text.startswith(self._utt_raw) else text
        _, is_eot = _strip_and_detect_eot(new_text)
        self._utt_clean = clean_now[self._utt_flushed_chars :]
        self._utt_raw = text
        # is_final closes one recognizer utterance, never the turn
        if message.get("is_final", False):
            self._utt_closed = True

        if not new_text:
            return
        if new_text.replace(EOT_TOKEN, "").strip():
            self._progress_at = self._audio_sent
            self._eot_at = None
        if self._current_text():
            self.report(self._turn())
        if is_eot:
            if self._opts.end_of_turn_confirm_delay_seconds is None:
                self._end_turn()
            elif self._eot_at is None:
                self._eot_at = self._audio_sent
                if self._current_text():
                    self.report(self._turn(preflight=True))
