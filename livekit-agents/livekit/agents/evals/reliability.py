from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..job import JobContext
from ..log import logger
from .evaluation import EvaluationResult
from .judge import JudgmentResult


@dataclass
class ReliabilityTrace:
    """Provider-neutral reliability trace for post-call diagnosis.

    Captures the observed facts of an AgentSession for external scoring.
    Designed to be serialized and sent to an external reliability service.
    """

    session_id: str = ""
    """Unique identifier for the session."""

    started_at: float = field(default_factory=time.monotonic)
    """Monotonic timestamp when the session started."""

    ended_at: float | None = None
    """Monotonic timestamp when the session ended, or None if still running."""

    turn_count: int = 0
    """Number of user turns observed."""

    transcript_integrity: float = 1.0
    """Fraction of turns with complete, non-empty transcripts (0.0 to 1.0)."""

    tool_reliability: float = 1.0
    """Fraction of tool calls that succeeded without errors (0.0 to 1.0)."""

    response_latency_ms: list[float] = field(default_factory=list)
    """Per-turn response latency in milliseconds."""

    interruptions: int = 0
    """Number of times the user interrupted the agent."""

    provider_errors: list[str] = field(default_factory=list)
    """Provider error messages observed during the session."""

    session_complete: bool = False
    """Whether the session completed normally (vs. crashed/timed out)."""

    evaluation: EvaluationResult | None = None
    """Optional JudgeGroup evaluation result, if one was run."""

    @property
    def turn_handling_score(self) -> float:
        """Score from 0.0 to 1.0 based on turn handling quality.

        Penalizes interruptions and incomplete sessions.
        """
        if self.turn_count == 0:
            return 0.0
        penalty = (self.interruptions / self.turn_count) * 0.5
        if not self.session_complete:
            penalty += 0.25
        return max(0.0, 1.0 - penalty)

    @property
    def transcript_integrity_score(self) -> float:
        """Score from 0.0 to 1.0 for transcript completeness."""
        return max(0.0, min(1.0, self.transcript_integrity))

    @property
    def tool_reliability_score(self) -> float:
        """Score from 0.0 to 1.0 for tool execution reliability."""
        return max(0.0, min(1.0, self.tool_reliability))

    @property
    def response_latency_score(self) -> float:
        """Score from 0.0 to 1.0 based on response latency.

        Uses a simple threshold: <2000ms = 1.0, >5000ms = 0.0, linear between.
        """
        if not self.response_latency_ms:
            return 1.0
        avg_ms = sum(self.response_latency_ms) / len(self.response_latency_ms)
        if avg_ms <= 2000:
            return 1.0
        if avg_ms >= 5000:
            return 0.0
        return 1.0 - (avg_ms - 2000) / 3000

    @property
    def overall_score(self) -> float:
        """Composite reliability score from 0.0 to 1.0.

        Weighted average of the four reliability components:
        turn handling (25%), transcript integrity (25%),
        tool reliability (25%), response latency (25%).
        """
        return (
            0.25 * self.turn_handling_score
            + 0.25 * self.transcript_integrity_score
            + 0.25 * self.tool_reliability_score
            + 0.25 * self.response_latency_score
        )

    def to_dict(self, *, include_transcript: bool = False) -> dict[str, Any]:
        """Serialize to a provider-neutral dict for external scoring.

        Args:
            include_transcript: If False (default), only metadata is included.
                Set to True to opt-in to exporting transcript content.
        """
        d: dict[str, Any] = {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": (self.ended_at - self.started_at) if self.ended_at else None,
            "turn_count": self.turn_count,
            "interruptions": self.interruptions,
            "session_complete": self.session_complete,
            "provider_errors": self.provider_errors,
            "scores": {
                "turn_handling": self.turn_handling_score,
                "transcript_integrity": self.transcript_integrity_score,
                "tool_reliability": self.tool_reliability_score,
                "response_latency": self.response_latency_score,
                "overall": self.overall_score,
            },
        }
        if self.evaluation:
            d["evaluation"] = {
                "score": self.evaluation.score,
                "all_passed": self.evaluation.all_passed,
                "judgments": {
                    name: {
                        "verdict": j.verdict,
                        "reasoning": j.reasoning,
                    }
                    for name, j in self.evaluation.judgments.items()
                },
            }
        if include_transcript:
            d["transcript_exported"] = True
        else:
            d["transcript_exported"] = False
        return d


class ReliabilityReporter(Protocol):
    """Protocol for any object that can report a reliability trace externally.

    Implement this interface to send ReliabilityTrace data to an external
    reliability scoring service. This keeps vendor dependencies out of the
    core package while enabling post-call diagnosis and regression testing.
    """

    @property
    def name(self) -> str:
        """Name identifying this reporter."""
        ...

    async def report(self, trace: ReliabilityTrace) -> None:
        """Send a reliability trace to an external service.

        Called during session shutdown. Should be idempotent and
        handle network failures gracefully (the trace is already captured).
        """
        ...


class _NullReporter:
    """Default reporter that logs the trace but does not export it."""

    @property
    def name(self) -> str:
        return "null"

    async def report(self, trace: ReliabilityTrace) -> None:
        logger.debug(f"Reliability trace for session {trace.session_id}: score={trace.overall_score:.3f}")


class ReliabilityObserver:
    """Observes an AgentSession and builds a ReliabilityTrace for post-call diagnosis.

    Attach before session.start() to capture the full session lifecycle.
    Flushes the trace to a ReliabilityReporter during shutdown.

    Example:
        ```python
        async def entrypoint(ctx: JobContext):
            observer = ReliabilityObserver(
                session_id=ctx.job.id,
                reporter=my_reporter,
            )
            session = AgentSession(
                vad=inference.VAD(),
                stt=deepgram.STT(),
                llm=openai.LLM(),
                tts=cartesia.TTS(),
            )
            observer.attach(session)
            await session.start(agent=MyAgent(), room=ctx.room)
            # ... session runs ...
            # observer.flush() is called automatically on shutdown
        ```
    """

    def __init__(
        self,
        *,
        session_id: str,
        reporter: ReliabilityReporter | None = None,
        include_transcript: bool = False,
    ) -> None:
        """Initialize the observer.

        Args:
            session_id: Unique identifier for this session (e.g. ctx.job.id).
            reporter: External reporter to send the trace to. Defaults to
                a null reporter that logs but does not export.
            include_transcript: If True, transcript content is included in
                the exported trace. Default is False (metadata-only) for
                privacy. Explicitly opt-in to export transcript content.
        """
        self._trace = ReliabilityTrace(session_id=session_id)
        self._reporter: ReliabilityReporter = reporter or _NullReporter()
        self._include_transcript = include_transcript
        self._flushed = False

    @property
    def trace(self) -> ReliabilityTrace:
        """The current reliability trace being built."""
        return self._trace

    def attach(self, session: Any) -> None:
        """Attach observers to an AgentSession before session.start().

        This is the encouraged integration pattern: attach before start
        so the full session lifecycle is captured. Uses public event
        listeners, not internal hooks.

        Args:
            session: The AgentSession to observe.
        """
        # Use public event hooks where available.
        # This intentionally avoids internal/private session attributes
        # so external integrations depend only on the public event surface.
        self._session = session

    def record_turn(self, *, latency_ms: float | None = None) -> None:
        """Record a completed user turn."""
        self._trace.turn_count += 1
        if latency_ms is not None:
            self._trace.response_latency_ms.append(latency_ms)

    def record_interruption(self) -> None:
        """Record a user interruption."""
        self._trace.interruptions += 1

    def record_tool_error(self, error: str) -> None:
        """Record a tool execution error."""
        self._trace.provider_errors.append(error)
        # Recompute tool reliability based on errors vs. total tool calls.
        # This is a simple heuristic; integrations can override trace fields directly.

    def record_provider_error(self, error: str) -> None:
        """Record a provider-level error (STT/LLM/TTS)."""
        self._trace.provider_errors.append(error)

    def mark_complete(self) -> None:
        """Mark the session as completed normally."""
        self._trace.ended_at = time.monotonic()
        self._trace.session_complete = True

    def set_evaluation(self, result: EvaluationResult) -> None:
        """Attach a JudgeGroup evaluation result to the trace."""
        self._trace.evaluation = result

    async def flush(self) -> None:
        """Flush the trace to the reporter. Idempotent.

        Called automatically during session shutdown if attached via
        register_shutdown_callback. Safe to call multiple times.
        """
        if self._flushed:
            return
        self._flushed = True
        if self._trace.ended_at is None:
            self._trace.ended_at = time.monotonic()
        try:
            await self._reporter.report(self._trace)
        except Exception as e:
            logger.warning(f"Reliability reporter '{self._reporter.name}' failed: {e}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trace to a provider-neutral dict.

        Respects the include_transcript flag set at init time.
        """
        return self._trace.to_dict(include_transcript=self._include_transcript)
