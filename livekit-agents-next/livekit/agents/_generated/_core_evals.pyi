# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

from livekit.agents.evals import _Evaluation
from livekit.agents.llm import LLM

from .evals import (
    EvaluationResult,
    EvaluationScore,
    IntentJudgment,
    JudgeOutcome,
    JudgeSpec,
    JudgmentResult,
)

class _core_JudgeRunner:
    """Runs judges on one model."""
    def __new__(cls, llm: LLM[Any] | None, host: _Evaluation) -> Self: ...
    async def evaluate(self, judges: Sequence[JudgeSpec]) -> list[JudgeOutcome]:
        """Runs `judges` concurrently: their outcomes in judge order."""
    async def judge_intent(self, intent: str, message: str) -> IntentJudgment:
        """Whether `message` fulfills `intent`. A provider failure raises the provider's error."""

def _core_evaluation_score(result: EvaluationResult) -> EvaluationScore:
    """The aggregates of `result`'s verdicts."""

def _core_judgment_passed(judgment: JudgmentResult) -> bool:
    """`JudgmentResult.passed`: the verdict is pass."""

def _core_judgment_failed(judgment: JudgmentResult) -> bool:
    """`JudgmentResult.failed`: the verdict is fail."""

def _core_judgment_uncertain(judgment: JudgmentResult) -> bool:
    """`JudgmentResult.uncertain`: the verdict is maybe."""
