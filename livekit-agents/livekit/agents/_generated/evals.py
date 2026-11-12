# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/evals.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import get_annotations
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from builtins import object as _HostJudgeSpec

    from .usage import TokenUsage


@dataclass
class JudgmentResult:
    """One judge's decision."""

    verdict: Literal["pass", "fail", "maybe"]
    reasoning: str
    instructions: str = ""
    """The criteria the judge evaluated against."""

    @property
    def passed(self) -> bool:
        """Whether the evaluation passed. Maybe is treated as not passed."""
        from .. import _core

        return _core._core_judgment_passed(self)

    @property
    def failed(self) -> bool:
        """Whether the evaluation failed. Maybe is treated as not failed."""
        from .. import _core

        return _core._core_judgment_failed(self)

    @property
    def uncertain(self) -> bool:
        """Whether the judge was uncertain about the verdict."""
        from .. import _core

        return _core._core_judgment_uncertain(self)


JudgeSpec: TypeAlias = "_HostJudgeSpec"


@dataclass(kw_only=True, slots=True)
class JudgeOutcomeJudged:
    name: str
    judgment: JudgmentResult


@dataclass(kw_only=True, slots=True)
class JudgeOutcomeFailed:
    """The judge raised or the model gave no usable verdict: no judgment, never a pass."""

    name: str
    error: str


JudgeOutcome: TypeAlias = JudgeOutcomeJudged | JudgeOutcomeFailed
"""What one judge of an evaluation came to."""


@dataclass(kw_only=True, slots=True)
class IntentJudgment:
    """The intent judge's decision on one message (`ChatMessageAssert.judge`)."""

    success: bool
    reason: str
    usage: TokenUsage | None
    """The judge request's usage, when the model reported it."""


@dataclass(kw_only=True, slots=True)
class EvaluationScore:
    """The aggregates of a set of judgments: pass scores 1, maybe 0.5 and fail 0."""

    score: float
    """The mean score; 0 for no judgments."""
    all_passed: bool
    any_passed: bool
    majority_passed: bool
    """More than half passed; true for no judgments."""
    none_failed: bool
    """Nothing failed; maybe is allowed."""


@dataclass
class EvaluationResult:
    """Result of evaluating a conversation with a group of judges."""

    judgments: dict[str, JudgmentResult] = field(default_factory=dict)
    """Individual judgment results keyed by judge name."""

    @property
    def score(self) -> float:
        """Score from 0.0 to 1.0. Pass=1, maybe=0.5, fail=0."""
        return self._score.score

    @property
    def all_passed(self) -> bool:
        """True if all judgments passed. Maybes count as not passed."""
        return self._score.all_passed

    @property
    def any_passed(self) -> bool:
        """True if at least one judgment passed."""
        return self._score.any_passed

    @property
    def majority_passed(self) -> bool:
        """True if more than half of the judgments passed."""
        return self._score.majority_passed

    @property
    def none_failed(self) -> bool:
        """True if no judgments explicitly failed. Maybes are allowed."""
        return self._score.none_failed

    @property
    def _score(self) -> EvaluationScore:
        from .. import _core

        return _core._core_evaluation_score(self)


JudgmentResult.__annotations__ = get_annotations(JudgmentResult, eval_str=True)
JudgmentResult.__module__ = "livekit.agents.evals"
EvaluationResult.__annotations__ = get_annotations(EvaluationResult, eval_str=True)
EvaluationResult.__module__ = "livekit.agents.evals"
