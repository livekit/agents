from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Sequence, TYPE_CHECKING

from ..log import logger
from ..stt import SpeechEvent
from .audio_recognition import _EndOfTurnInfo, _PreemptiveGenerationInfo, RecognitionHooks
if TYPE_CHECKING:  # avoid circular import at runtime
    from .agent_session import AgentSession


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s']", re.UNICODE)


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def _split_words(text: str) -> list[str]:
    return [w for w in _normalize_text(text).split(" ") if w]


def _parse_csv_env(name: str, default: Sequence[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return list(default)
    return [w.strip().lower() for w in raw.split(",") if w.strip()]


@dataclass
class _Config:
    ignored_words: list[str]
    interruption_keywords: list[str]
    min_confidence: float


def _load_config() -> _Config:
    ignored_words = _parse_csv_env(
        "LIVEKIT_IGNORED_WORDS",
        default=["uh", "umm", "hmm", "haan"],
    )
    interruption_keywords = _parse_csv_env(
        "LIVEKIT_INTERRUPTION_KEYWORDS",
        default=["stop", "wait", "hold on", "one second", "no", "not that"],
    )
    try:
        min_conf = float(os.getenv("LIVEKIT_MIN_ASR_CONFIDENCE", "0.5"))
    except ValueError:
        min_conf = 0.5
    return _Config(
        ignored_words=ignored_words,
        interruption_keywords=interruption_keywords,
        min_confidence=min_conf,
    )


class InterruptionFilter(RecognitionHooks):
    """RecognitionHooks wrapper to suppress filler-only or low-confidence interruptions
    while the agent is currently speaking. Does not modify the underlying VAD logic.

    Configuration via env vars:
      - LIVEKIT_IGNORED_WORDS: comma-separated fillers to ignore during agent TTS.
      - LIVEKIT_INTERRUPTION_KEYWORDS: comma-separated phrases that always interrupt.
      - LIVEKIT_MIN_ASR_CONFIDENCE: float threshold under which transcripts are ignored
        during agent speech (default 0.5).
    """

    def __init__(self, *, hooks: RecognitionHooks, session: "AgentSession") -> None:
        self._hooks = hooks
        self._session = session
        self._cfg = _load_config()
        self._last_interim_text: str = ""
        self._last_interim_confidence: float = 0.0

    # --------------- helper predicates ---------------
    def _agent_currently_speaking(self) -> bool:
        handle = self._session.current_speech
        return bool(handle is not None and not handle.interrupted and self._session.agent_state == "speaking")

    def _contains_interruption_keyword(self, text: str) -> bool:
        nt = _normalize_text(text)
        return any(kw in nt for kw in self._cfg.interruption_keywords)

    def _is_only_fillers(self, text: str) -> bool:
        words = _split_words(text)
        if not words:
            return True
        ig = set(self._cfg.ignored_words)
        return all(w in ig for w in words)

    def _should_ignore(self, text: str, confidence: float | None) -> bool:
        if not self._agent_currently_speaking():
            return False
        if self._contains_interruption_keyword(text):
            return False
        if self._is_only_fillers(text):
            return True
        if confidence is not None and confidence < self._cfg.min_confidence:
            return True
        return False

    # --------------- RecognitionHooks forwarding with filtering ---------------
    def on_start_of_speech(self, ev) -> None:  # type: ignore[no-untyped-def]
        self._hooks.on_start_of_speech(ev)

    def on_vad_inference_done(self, ev) -> None:  # type: ignore[no-untyped-def]
        if self._agent_currently_speaking() and self._should_ignore(self._last_interim_text, self._last_interim_confidence):
            logger.debug(
                "ignored VAD interruption due to filler/low-confidence while agent speaking",
                extra={
                    "last_interim_text": self._last_interim_text,
                    "last_interim_confidence": self._last_interim_confidence,
                    "silence_duration": getattr(ev, "silence_duration", None),
                },
            )
            return
        self._hooks.on_vad_inference_done(ev)

    def on_end_of_speech(self, ev) -> None:  # type: ignore[no-untyped-def]
        self._hooks.on_end_of_speech(ev)

    def on_interim_transcript(self, ev: SpeechEvent, *, speaking: bool | None) -> None:
        text = ev.alternatives[0].text if ev.alternatives else ""
        conf = ev.alternatives[0].confidence if ev.alternatives else 0.0
        self._last_interim_text = text
        self._last_interim_confidence = conf

        if self._should_ignore(text, conf):
            logger.info(
                "ignored interim transcript during agent speech",
                extra={
                    "transcript": text,
                    "confidence": conf,
                    "reason": "filler_only_or_low_confidence",
                },
            )
            return

        self._hooks.on_interim_transcript(ev, speaking=speaking)

    def on_final_transcript(self, ev: SpeechEvent) -> None:
        text = ev.alternatives[0].text if ev.alternatives else ""
        conf = ev.alternatives[0].confidence if ev.alternatives else 0.0

        if self._should_ignore(text, conf):
            logger.info(
                "ignored final transcript during agent speech",
                extra={
                    "transcript": text,
                    "confidence": conf,
                    "reason": "filler_only_or_low_confidence",
                },
            )
            return

        self._hooks.on_final_transcript(ev)

    def on_end_of_turn(self, info: _EndOfTurnInfo) -> bool:
        return self._hooks.on_end_of_turn(info)

    def on_preemptive_generation(self, info: _PreemptiveGenerationInfo) -> None:
        self._hooks.on_preemptive_generation(info)

    def retrieve_chat_ctx(self):  # type: ignore[no-untyped-def]
        return self._hooks.retrieve_chat_ctx()
