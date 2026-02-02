from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .evals import EvaluationResult


class Tagger:
    """Tag sessions with metadata for observability.

    The Tagger allows adding string tags (key:value format) to sessions.
    Tags are uploaded to LiveKit Cloud at session end.

    Example:
        ```python
        # Mark session as successful
        ctx.tagger.success(reason="Task completed successfully")

        # Mark session as failed
        ctx.tagger.fail(reason="User hung up before completing booking")

        # Add custom tags
        ctx.tagger.add("voicemail:true")
        ctx.tagger.add("language:es")

        # Remove a tag
        ctx.tagger.remove("voicemail:true")
        ```
    """

    def __init__(self) -> None:
        self._tags: set[str] = set()
        self._evaluation_results: list[dict[str, Any]] = []
        self._outcome_reason: str | None = None

    def success(self, reason: str | None = None) -> None:
        """Mark the session as successful.

        Args:
            reason: Optional reason for the success (stored separately from the tag).
        """
        # Remove any existing outcome tag
        self._tags.discard("lk.fail")
        self._tags.add("lk.success")
        self._outcome_reason = reason

    def fail(self, reason: str | None = None) -> None:
        """Mark the session as failed.

        Args:
            reason: Optional reason for the failure (stored separately from the tag).
        """
        # Remove any existing outcome tag
        self._tags.discard("lk.success")
        self._tags.add("lk.fail")
        self._outcome_reason = reason

    def add(self, tag: str) -> None:
        """Add a tag to the session.

        Args:
            tag: The tag string in "key:value" format (e.g., "voicemail:true", "language:es").
        """
        self._tags.add(tag)

    def remove(self, tag: str) -> None:
        """Remove a tag from the session.

        Args:
            tag: The tag string to remove.
        """
        self._tags.discard(tag)

    @property
    def tags(self) -> set[str]:
        """All current tags."""
        return self._tags.copy()

    @property
    def evaluations(self) -> list[dict[str, Any]]:
        """All evaluation results."""
        return self._evaluation_results.copy()

    @property
    def outcome_reason(self) -> str | None:
        """Reason for success/failure outcome."""
        return self._outcome_reason

    def _evaluation(self, result: EvaluationResult) -> None:
        """Tag the session with evaluation results (internal use only).

        Called automatically by JudgeGroup.evaluate().
        """
        for name, judgment in result.judgments.items():
            self._tags.add(f"lk.judge.{name}:{judgment.verdict}")
            self._evaluation_results.append(
                {
                    "name": name,
                    "verdict": judgment.verdict,
                    "reasoning": judgment.reasoning,
                }
            )
