# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/evals.rs

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, kw_only=True, slots=True)
class Judgment:
    """One judge's decision; `verdict` is `"pass"`, `"fail"` or `"maybe"`."""

    verdict: str
    reasoning: str
    instructions: str
    """The criteria the judge evaluated against."""


@dataclass(frozen=True, kw_only=True, slots=True)
class JudgeSpecBuiltIn:
    """A built-in judge, by its name (`"accuracy"`, `"task_completion"`, ...)."""

    name: str


@dataclass(frozen=True, kw_only=True, slots=True)
class JudgeSpecHost:
    """An application evaluator, which the evaluation's host runs."""

    name: str
    evaluator: object


JudgeSpec: TypeAlias = JudgeSpecBuiltIn | JudgeSpecHost
"""One judge of an evaluation."""


@dataclass(frozen=True, kw_only=True, slots=True)
class JudgeOutcomeJudged:
    name: str
    judgment: Judgment


@dataclass(frozen=True, kw_only=True, slots=True)
class JudgeOutcomeFailed:
    """The judge raised or the model gave no usable verdict: no judgment, never a pass."""

    name: str
    error: str


JudgeOutcome: TypeAlias = JudgeOutcomeJudged | JudgeOutcomeFailed
"""What one judge of an evaluation came to."""


@dataclass(frozen=True, kw_only=True, slots=True)
class IntentJudgment:
    """The intent judge's decision on one message (`ChatMessageAssert.judge`)."""

    success: bool
    reason: str
    usage: dict[str, object] | None
    """The judge request's usage, when the model reported it."""


@dataclass(frozen=True, kw_only=True, slots=True)
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
