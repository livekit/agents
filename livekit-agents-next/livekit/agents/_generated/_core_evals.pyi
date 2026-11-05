# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

from livekit.agents.evals import _Evaluation
from livekit.agents.llm import LLM

from .chat_items import ChatItem
from .evals import (
    EvaluationResult,
    EvaluationScore,
    IntentJudgment,
    JudgeOutcome,
    JudgeSpec,
    JudgmentResult,
)

class _core_JudgeRunner:
    """Runs judges on one model. A model the host implements is offered `tools`, the host's live
    judge tools (`judge_tool_schemas`).
    """
    def __new__(cls, llm: LLM[Any] | None, tools: object, host: _Evaluation) -> Self: ...
    async def evaluate(
        self,
        judges: Sequence[JudgeSpec],
        chat_ctx: Sequence[ChatItem],
        reference: Sequence[ChatItem] | None,
    ) -> list[JudgeOutcome]:
        """Runs `judges` concurrently on `chat_ctx`, judged against `reference` when given: their
        outcomes in judge order. An unknown built-in judge is refused before any runs.
        """
    async def judge_intent(self, intent: str, message: str) -> IntentJudgment:
        """Whether `message` fulfills `intent`. A provider failure raises the provider's error."""

def _core_evaluation_score(result: EvaluationResult) -> EvaluationScore:
    """The aggregates of `result`'s verdicts; a verdict other than pass, fail or maybe is
    refused.
    """

def _core_judgment_passed(judgment: JudgmentResult) -> bool:
    """`JudgmentResult.passed`: the verdict is pass."""

def _core_judgment_failed(judgment: JudgmentResult) -> bool:
    """`JudgmentResult.failed`: the verdict is fail."""

def _core_judgment_uncertain(judgment: JudgmentResult) -> bool:
    """`JudgmentResult.uncertain`: the verdict is maybe."""

def _core_judge_tool_schemas() -> object:
    """The judge tools' raw schemas, which the host offers a model it implements."""
