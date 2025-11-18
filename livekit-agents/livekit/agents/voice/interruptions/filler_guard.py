from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Literal, Sequence, Tuple

_WORD_PATTERN = re.compile(r"\b[\w']+\b", flags=re.UNICODE)

InterruptionReason = Literal[
    "content",
    "filler_agent_speaking",
    "filler_agent_silent",
    "low_confidence",
    "empty",
    "disabled",
]


@dataclass(frozen=True, slots=True)
class InterruptionDecision:
    transcript: str
    tokens: Tuple[str, ...]
    filler_tokens: Tuple[str, ...]
    allow: bool
    reason: InterruptionReason
    confidence: float | None
    agent_speaking: bool
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class LiveAudioTurnState:
    agent_was_speaking: bool = False
    has_semantic_content: bool = False
    filler_only: bool = True
    last_transcript: str = ""

    def reset(self) -> None:
        self.agent_was_speaking = False
        self.has_semantic_content = False
        self.filler_only = True
        self.last_transcript = ""


class FillerAwareInterruptionGuard:
    """
    Lightweight semantic gate that prevents filler-only utterances from
    interrupting the agent while it is speaking. It piggy-backs on STT
    transcripts (interim + final) without modifying the base VAD pipeline.
    """

    def __init__(
        self,
        *,
        ignored_words: Sequence[str],
        min_confidence: float,
        enabled: bool,
        logger: logging.Logger | None = None,
    ) -> None:
        self._ignored_words = self._normalize_words(ignored_words)
        self._min_confidence = min_confidence
        self._enabled = enabled
        self._logger = logger or logging.getLogger(__name__)
        self._turn_state: LiveAudioTurnState | None = None
        self._last_decision: InterruptionDecision | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def update_ignored_words(self, words: Sequence[str]) -> None:
        self._ignored_words = self._normalize_words(words)

    def update_enabled(self, value: bool) -> None:
        self._enabled = value

    def start_turn(self, *, agent_speaking: bool) -> None:
        if not self._enabled:
            return
        self._turn_state = LiveAudioTurnState(agent_was_speaking=agent_speaking)

    def observe_transcript(
        self,
        transcript: str,
        *,
        confidence: float | None,
        agent_speaking: bool,
        stage: Literal["interim", "final"] = "interim",
    ) -> InterruptionDecision:
        decision = self._classify(
            transcript=transcript,
            confidence=confidence,
            agent_speaking=agent_speaking,
        )
        self._last_decision = decision

        if self._enabled and self._turn_state is None:
            self._turn_state = LiveAudioTurnState(agent_was_speaking=agent_speaking)

        if self._enabled and self._turn_state:
            self._turn_state.agent_was_speaking = (
                self._turn_state.agent_was_speaking or agent_speaking
            )
            if decision.reason == "content":
                self._turn_state.has_semantic_content = True
                self._turn_state.filler_only = False
            elif decision.reason == "filler_agent_silent":
                # filler while silent counts as speech, so note that we saw content
                self._turn_state.has_semantic_content = True
                self._turn_state.filler_only = False
            self._turn_state.last_transcript = transcript

        self._log_decision(decision, stage=stage)
        return decision

    def should_interrupt(self, *, transcript: str, agent_speaking: bool) -> bool:
        if not self._enabled or not agent_speaking:
            return True

        transcript = transcript.strip()
        if not transcript:
            # wait for STT text before interrupting
            return False

        cached = self._last_decision
        if cached and cached.transcript == transcript:
            return cached.allow

        decision = self._classify(
            transcript=transcript,
            confidence=cached.confidence if cached else None,
            agent_speaking=agent_speaking,
        )
        self._last_decision = decision
        if not decision.allow:
            self._log_decision(decision, stage="interrupt_gate")
        return decision.allow

    def should_process_turn(self, transcript: str) -> bool:
        if not self._enabled:
            return True

        transcript = transcript.strip()
        if not transcript:
            self._reset_turn()
            return False

        tokens = self._tokenize(transcript)
        if self._has_semantic_tokens(tokens):
            self._reset_turn()
            return True

        state = self._turn_state
        self._reset_turn()
        if state and state.agent_was_speaking:
            self._logger.debug(
                "ignored filler-only turn while agent speaking",
                extra={"transcript": transcript},
            )
            return False

        # Agent was silent the whole time, so treat as normal user speech.
        return True

    def reset(self) -> None:
        self._reset_turn()
        self._last_decision = None

    def _reset_turn(self) -> None:
        if self._turn_state is not None:
            self._turn_state.reset()
        self._turn_state = None

    def _classify(
        self,
        *,
        transcript: str,
        confidence: float | None,
        agent_speaking: bool,
    ) -> InterruptionDecision:
        if not self._enabled:
            return InterruptionDecision(
                transcript=transcript,
                tokens=(),
                filler_tokens=(),
                allow=True,
                reason="disabled",
                confidence=confidence,
                agent_speaking=agent_speaking,
            )

        tokens = self._tokenize(transcript)
        if not tokens:
            return InterruptionDecision(
                transcript=transcript,
                tokens=(),
                filler_tokens=(),
                allow=False,
                reason="empty",
                confidence=confidence,
                agent_speaking=agent_speaking,
            )

        filler_tokens = tuple(tok for tok in tokens if tok in self._ignored_words)
        has_semantic_content = self._has_semantic_tokens(tokens)

        reason: InterruptionReason
        allow = True
        if has_semantic_content:
            reason = "content"
        else:
            # filler-only case
            if confidence is not None and confidence < self._min_confidence:
                allow = False
                reason = "low_confidence"
            elif agent_speaking:
                allow = False
                reason = "filler_agent_speaking"
            else:
                allow = True
                reason = "filler_agent_silent"

        return InterruptionDecision(
            transcript=transcript,
            tokens=tuple(tokens),
            filler_tokens=filler_tokens,
            allow=allow,
            reason=reason,
            confidence=confidence,
            agent_speaking=agent_speaking,
        )

    def _log_decision(self, decision: InterruptionDecision, *, stage: str) -> None:
        if decision.reason in ("content", "filler_agent_silent"):
            self._logger.debug(
                "accepted user speech",
                extra={
                    "stage": stage,
                    "reason": decision.reason,
                    "transcript": decision.transcript,
                },
            )
        elif decision.reason == "disabled":
            return
        else:
            self._logger.debug(
                "ignored filler speech",
                extra={
                    "stage": stage,
                    "reason": decision.reason,
                    "transcript": decision.transcript,
                    "confidence": decision.confidence,
                },
            )

    def _tokenize(self, text: str) -> Tuple[str, ...]:
        return tuple(match.group(0).casefold() for match in _WORD_PATTERN.finditer(text))

    def _has_semantic_tokens(self, tokens: Tuple[str, ...]) -> bool:
        return any(token not in self._ignored_words for token in tokens)

    @staticmethod
    def _normalize_words(words: Sequence[str]) -> Tuple[str, ...]:
        normalized = {word.strip().casefold() for word in words if word.strip()}
        return tuple(sorted(normalized))

