from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import _core

if TYPE_CHECKING:
    from .evals import EvaluationResult


class Tagger:
    """Tag sessions with metadata for observability.

    The Tagger allows adding string tags (key:value format) with optional
    structured metadata to sessions. Tags and metadata are uploaded to
    LiveKit Cloud at session end.

    Example:
        ```python
        # Mark session as successful
        ctx.tagger.success(reason="Task completed successfully")

        # Mark session as failed
        ctx.tagger.fail(reason="User hung up before completing booking")

        # Add custom tags
        ctx.tagger.add("voicemail:true")
        ctx.tagger.add("language:es")

        # Add tags with structured metadata
        ctx.tagger.add(
            "appointment:booked",
            metadata={"slot_id": "abc123", "calendar": "cal.com"},
        )

        # Remove a tag
        ctx.tagger.remove("voicemail:true")
        ```
    """

    def __init__(self) -> None:
        self._core = _core.Tagger()
        self._evaluation_results: list[dict[str, Any]] = []

    def success(self, reason: str | None = None) -> None:
        """Mark the session as successful; ``reason`` is kept apart from the tag."""
        self._core.success(reason)

    def fail(self, reason: str | None = None) -> None:
        """Mark the session as failed; ``reason`` is kept apart from the tag."""
        self._core.fail(reason)

    def add(self, tag: str, *, metadata: dict[str, Any] | None = None) -> None:
        """Add a ``key:value`` tag, with optional structured metadata."""
        self._core.add(tag, metadata)

    def remove(self, tag: str) -> None:
        self._core.remove(tag)

    @property
    def tags(self) -> set[str]:
        return set(self._core.tags())

    @property
    def evaluations(self) -> list[dict[str, Any]]:
        return self._evaluation_results.copy()

    @property
    def outcome(self) -> str | None:
        """``"success"``, ``"fail"`` or ``None``."""
        return self._core.outcome()  # type: ignore[no-any-return]

    @property
    def outcome_reason(self) -> str | None:
        return self._core.outcome_reason()  # type: ignore[no-any-return]

    def _evaluation(self, result: EvaluationResult) -> None:
        """Tag the session with a judge group's results (called by ``JudgeGroup.evaluate``)."""
        for name, judgment in result.judgments.items():
            tag = f"lk.judge.{name}:{judgment.verdict}"
            self._core.add(tag, None)
            self._evaluation_results.append(
                {
                    "name": name,
                    "tag": tag,
                    "verdict": judgment.verdict,
                    "reasoning": judgment.reasoning,
                    "instructions": judgment.instructions,
                }
            )
