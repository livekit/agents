from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import string
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Literal, Mapping, Sequence

from livekit.agents.voice.agent_session import AgentSession
from livekit.agents.voice.events import (
    AgentStateChangedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
)

Classification = Literal["FILLER", "VALID_SPEECH", "UNCERTAIN"]

DEFAULT_IGNORED = ["uh", "umm", "hmm", "haan"]
DEFAULT_COMMANDS = ["stop", "wait", "hold", "no", "not that"]
DEFAULT_CONFIDENCE = 0.6


def _parse_csv_env(var_name: str, fallback: Sequence[str]) -> list[str]:
    raw = os.getenv(var_name)
    if not raw:
        return list(fallback)
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def _parse_float_env(var_name: str, fallback: float | None) -> float | None:
    raw = os.getenv(var_name)
    if raw is None:
        return fallback
    try:
        return float(raw)
    except ValueError:
        return fallback


@dataclass
class InterruptHandlerConfig:
    ignored_words: list[str] = field(default_factory=lambda: DEFAULT_IGNORED.copy())
    confidence_threshold: float = DEFAULT_CONFIDENCE
    command_words: list[str] = field(default_factory=lambda: DEFAULT_COMMANDS.copy())
    uncertain_threshold: float | None = None
    interim_command_threshold: float | None = None
    log_file: str | None = None

    @classmethod
    def from_env(cls) -> "InterruptHandlerConfig":
        return cls(
            ignored_words=_parse_csv_env("INTERRUPT_HANDLER_IGNORED_WORDS", DEFAULT_IGNORED),
            confidence_threshold=_parse_float_env(
                "INTERRUPT_HANDLER_CONFIDENCE_THRESHOLD", DEFAULT_CONFIDENCE
            )
            or DEFAULT_CONFIDENCE,
            command_words=_parse_csv_env(
                "INTERRUPT_HANDLER_COMMAND_WORDS", DEFAULT_COMMANDS
            ),
            uncertain_threshold=_parse_float_env(
                "INTERRUPT_HANDLER_UNCERTAIN_THRESHOLD", None
            ),
            interim_command_threshold=_parse_float_env(
                "INTERRUPT_HANDLER_INTERIM_COMMAND_THRESHOLD", None
            ),
            log_file=os.getenv("INTERRUPT_HANDLER_LOG_FILE"),
        )


WordMeta = Mapping[str, Any]
StopCallable = Callable[[], Awaitable[Any] | Any]


class InterruptHandler:
    """Classifies user ASR segments so filler-only phrases do not interrupt TTS."""

    def __init__(
        self,
        session: AgentSession | None = None,
        *,
        stop_callback: StopCallable | None = None,
        config: InterruptHandlerConfig | None = None,
        logger: logging.Logger | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._session = session
        if stop_callback is None and session is not None:
            stop_callback = session.interrupt

        if stop_callback is None:
            raise ValueError("InterruptHandler requires either a session or a stop_callback.")

        self._config = config or InterruptHandlerConfig.from_env()
        self._ignored_words = {self._normalize_token(w) for w in self._config.ignored_words}
        self._command_terms = self._normalize_command_terms(self._config.command_words)
        self._confidence_threshold = self._config.confidence_threshold
        self._uncertain_threshold = self._config.uncertain_threshold
        self._interim_command_threshold = self._config.interim_command_threshold or max(
            0.9, self._confidence_threshold + 0.2
        )

        self._stop_callback = stop_callback
        self._loop = loop
        self._lock = asyncio.Lock()
        self._tts_is_speaking = False
        self._vad_is_speech = False
        self._attached = False

        self._logger = logger or logging.getLogger("livekit.interrupt_handler")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
        if self._config.log_file:
            file_handler = logging.FileHandler(self._config.log_file)
            file_handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(file_handler)
            self._logger.setLevel(logging.DEBUG)

    # -- Public API -----------------------------------------------------

    def attach(self) -> None:
        if not self._session:
            raise RuntimeError("attach() requires a session instance.")
        if self._attached:
            return

        self._loop = self._loop or asyncio.get_event_loop()
        self._session.on("agent_state_changed", self._handle_agent_state)
        self._session.on("user_state_changed", self._handle_user_state)
        self._session.on("user_input_transcribed", self._handle_transcription_event)
        self._attached = True

    def detach(self) -> None:
        if not self._session or not self._attached:
            return
        self._session.off("agent_state_changed", self._handle_agent_state)
        self._session.off("user_state_changed", self._handle_user_state)
        self._session.off("user_input_transcribed", self._handle_transcription_event)
        self._attached = False

    def update_ignored_words(self, new_list: Iterable[str]) -> None:
        self._ignored_words = {self._normalize_token(w) for w in new_list}
        if self._config:
            self._config.ignored_words = list(new_list)

    async def on_transcription(
        self,
        transcript: str,
        words_meta: Sequence[WordMeta] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        metadata = dict(metadata or {})
        norm_transcript = transcript.strip()
        if not norm_transcript:
            return False

        async with self._lock:
            tokens, confidences = self._tokens_from_words(norm_transcript, words_meta, metadata)
            classification = self.classification_for_segment(
                tokens, confidences, self._tts_is_speaking
            )

            decision = {
                "timestamp": time.time(),
                "transcript": norm_transcript,
                "tokens": tokens,
                "confidences": confidences,
                "classification": classification,
                "tts_speaking": self._tts_is_speaking,
                "vad_speech": self._vad_is_speech,
                "metadata": metadata,
            }

            should_interrupt = False
            if self._tts_is_speaking:
                if classification == "VALID_SPEECH":
                    if metadata.get("is_final", True):
                        should_interrupt = True
                    else:
                        should_interrupt = self._contains_command(
                            tokens, confidences, self._interim_command_threshold
                        )
                else:
                    should_interrupt = False
            else:
                # Agent is quiet, propagate always
                decision["type"] = "passthrough"
                self._log(decision)
                return False

            if should_interrupt:
                decision["type"] = "accepted"
                self._log(decision)
                await self._invoke_stop()
                return True

            decision["type"] = "ignored" if classification == "FILLER" else "uncertain"
            self._log(decision)
            return False

    def on_tts_state(self, is_speaking: bool) -> None:
        self._tts_is_speaking = is_speaking

    def on_vad_state(self, is_speech: bool) -> None:
        self._vad_is_speech = is_speech

    def classification_for_segment(
        self,
        tokens: Sequence[str],
        confidences: Sequence[float],
        tts_is_speaking: bool,
    ) -> Classification:
        if not tokens:
            return "UNCERTAIN"

        if not tts_is_speaking:
            return "VALID_SPEECH"

        normalized = [self._normalize_token(tok) for tok in tokens]
        norm_conf_pairs: list[tuple[str, float]] = []
        for idx, token in enumerate(normalized):
            conf = confidences[idx] if idx < len(confidences) else self._confidence_threshold
            norm_conf_pairs.append((token, conf))

        has_cmd = self._matches_command(normalized, confidences, self._confidence_threshold)
        if has_cmd:
            return "VALID_SPEECH"

        confident_non_filler = any(
            token and token not in self._ignored_words and conf >= self._confidence_threshold
            for token, conf in norm_conf_pairs
        )
        if confident_non_filler:
            return "VALID_SPEECH"

        if all(token in self._ignored_words for token in normalized if token):
            if self._uncertain_threshold is not None and any(
                conf >= self._uncertain_threshold for conf in confidences
            ):
                return "UNCERTAIN"
            return "FILLER"

        return "UNCERTAIN"

    # -- Internal helpers -----------------------------------------------

    def _handle_agent_state(self, ev: AgentStateChangedEvent) -> None:
        self.on_tts_state(ev.new_state == "speaking")

    def _handle_user_state(self, ev: UserStateChangedEvent) -> None:
        self.on_vad_state(ev.new_state == "speaking")

    def _handle_transcription_event(self, ev: UserInputTranscribedEvent) -> None:
        loop = self._loop or asyncio.get_event_loop()
        self._loop = loop
        loop.create_task(
            self.on_transcription(
                ev.transcript,
                words_meta=getattr(ev, "words", None),
                metadata={
                    "is_final": ev.is_final,
                    "speaker_id": ev.speaker_id,
                    "language": ev.language,
                },
            )
        )

    async def _invoke_stop(self) -> None:
        result = self._stop_callback()
        if inspect.isawaitable(result):
            await result

    def _tokens_from_words(
        self,
        transcript: str,
        words_meta: Sequence[WordMeta] | None,
        metadata: Mapping[str, Any],
    ) -> tuple[list[str], list[float]]:
        if words_meta:
            tokens = [word.get("text", "") for word in words_meta]
            confidences = [
                float(word.get("confidence", self._confidence_threshold)) for word in words_meta
            ]
            return tokens, confidences

        tokens = [tok for tok in transcript.split() if tok]
        fallback_conf = float(metadata.get("confidence", self._confidence_threshold))
        confidences = [fallback_conf for _ in tokens]
        return tokens, confidences

    def _contains_command(
        self,
        tokens: Sequence[str],
        confidences: Sequence[float],
        threshold: float,
    ) -> bool:
        normalized = [self._normalize_token(tok) for tok in tokens]
        return self._matches_command(normalized, confidences, threshold)

    def _matches_command(
        self, normalized_tokens: Sequence[str], confidences: Sequence[float], threshold: float
    ) -> bool:
        if not normalized_tokens:
            return False

        for term in self._command_terms:
            if not term:
                continue
            term_len = len(term)
            if term_len == 1:
                token = term[0]
                for idx, current in enumerate(normalized_tokens):
                    if current == token:
                        conf = confidences[idx] if idx < len(confidences) else self._confidence_threshold
                        if conf >= threshold:
                            return True
                continue

            for start in range(0, len(normalized_tokens) - term_len + 1):
                window = normalized_tokens[start : start + term_len]
                if list(window) == list(term):
                    window_conf = min(
                        confidences[start + offset]
                        if start + offset < len(confidences)
                        else self._confidence_threshold
                        for offset in range(term_len)
                    )
                    if window_conf >= threshold:
                        return True

        return False

    def _normalize_token(self, token: str) -> str:
        return token.strip().lower().strip(string.punctuation)

    def _normalize_command_terms(self, commands: Sequence[str]) -> list[tuple[str, ...]]:
        normalized_terms: list[tuple[str, ...]] = []
        for raw in commands:
            pieces = [self._normalize_token(part) for part in raw.split() if part.strip()]
            if pieces:
                normalized_terms.append(tuple(pieces))
        return normalized_terms

    def _log(self, payload: Mapping[str, Any]) -> None:
        try:
            message = json.dumps(payload, ensure_ascii=False)
        except TypeError:
            safe_payload = dict(payload)
            safe_payload["metadata"] = dict(payload.get("metadata", {}))
            message = json.dumps(safe_payload, ensure_ascii=False)

        if payload.get("type") == "accepted":
            self._logger.info(message)
        elif payload.get("type") == "passthrough":
            self._logger.debug(message)
        else:
            self._logger.info(message)

