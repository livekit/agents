from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .evals import EvaluationResult


@dataclass
class _TagEntry:
    metadata: dict[str, Any] | None = None
    timestamp: float = field(default_factory=time.time)


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
        self._tags: dict[str, _TagEntry] = {}
        self._evaluation_results: list[dict[str, Any]] = []
        self._outcome_reason: str | None = None

    def success(self, reason: str | None = None) -> None:
        """Mark the session as successful.

        Args:
            reason: Optional reason for the success (stored separately from the tag).
        """
        # Remove any existing outcome tag
        self._tags.pop("lk.fail", None)
        self._tags["lk.success"] = _TagEntry()
        self._outcome_reason = reason

    def fail(self, reason: str | None = None) -> None:
        """Mark the session as failed.

        Args:
            reason: Optional reason for the failure (stored separately from the tag).
        """
        # Remove any existing outcome tag
        self._tags.pop("lk.success", None)
        self._tags["lk.fail"] = _TagEntry()
        self._outcome_reason = reason

    def add(self, tag: str, *, metadata: dict[str, Any] | None = None) -> None:
        """Add a tag to the session with optional structured metadata.

        Args:
            tag: The tag string in "key:value" format (e.g., "voicemail:true", "language:es").
            metadata: Optional dict of structured metadata associated with this tag.
        """
        self._tags[tag] = _TagEntry(metadata=metadata)

    def remove(self, tag: str) -> None:
        """Remove a tag from the session.

        Args:
            tag: The tag string to remove.
        """
        self._tags.pop(tag, None)

    @property
    def tags(self) -> set[str]:
        """All current tag strings."""
        return set(self._tags.keys())

    @property
    def evaluations(self) -> list[dict[str, Any]]:
        """All evaluation results."""
        return self._evaluation_results.copy()

    @property
    def outcome(self) -> str | None:
        """The session outcome: 'success', 'fail', or None if not set."""
        if "lk.success" in self._tags:
            return "success"
        elif "lk.fail" in self._tags:
            return "fail"
        return None

    @property
    def outcome_reason(self) -> str | None:
        """Reason for success/failure outcome."""
        return self._outcome_reason

    def _evaluation(self, result: EvaluationResult) -> None:
        """Tag the session with evaluation results (internal use only).

        Called automatically by JudgeGroup.evaluate().
        """
        for name, judgment in result.judgments.items():
            tag = f"lk.judge.{name}:{judgment.verdict}"
            self._tags[tag] = _TagEntry()
            self._evaluation_results.append(
                {
                    "name": name,
                    "tag": tag,
                    "verdict": judgment.verdict,
                    "reasoning": judgment.reasoning,
                    "instructions": judgment.instructions,
                }
            )
