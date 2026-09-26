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


# The only two tags that carry a session outcome. They stay mutually exclusive, and
# `outcome_reason` is only meaningful while one of them is present.
_SUCCESS_TAG = "lk.success"
_FAIL_TAG = "lk.fail"


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
        self.add(_SUCCESS_TAG)
        self._outcome_reason = reason

    def fail(self, reason: str | None = None) -> None:
        """Mark the session as failed.

        Args:
            reason: Optional reason for the failure (stored separately from the tag).
        """
        self.add(_FAIL_TAG)
        self._outcome_reason = reason

    def add(self, tag: str, *, metadata: dict[str, Any] | None = None) -> None:
        """Add a tag to the session with optional structured metadata.

        The two reserved outcome tags go through the same mutual exclusion that
        `success()` and `fail()` maintain, so `outcome` cannot end up disagreeing with
        `tags` whichever public path wrote them.

        Args:
            tag: The tag string in "key:value" format (e.g., "voicemail:true", "language:es").
            metadata: Optional dict of structured metadata associated with this tag.
        """
        if tag == _SUCCESS_TAG:
            self._tags.pop(_FAIL_TAG, None)
            self._outcome_reason = None
        elif tag == _FAIL_TAG:
            self._tags.pop(_SUCCESS_TAG, None)
            self._outcome_reason = None

        self._tags[tag] = _TagEntry(metadata=metadata)

    def remove(self, tag: str) -> None:
        """Remove a tag from the session.

        Removing an outcome tag also clears its reason, so a session left without an
        outcome tag does not keep reporting the reason of the outcome it no longer has.

        Args:
            tag: The tag string to remove.
        """
        removed = self._tags.pop(tag, None)
        if removed is not None and tag in (_SUCCESS_TAG, _FAIL_TAG):
            self._outcome_reason = None

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
        if _SUCCESS_TAG in self._tags:
            return "success"
        elif _FAIL_TAG in self._tags:
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
