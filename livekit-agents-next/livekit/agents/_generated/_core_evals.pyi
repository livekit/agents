# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from livekit.agents.evals import _Evaluation
from livekit.agents.llm import LLM

from .._core import ChatItem
from .evals import EvaluationScore, IntentJudgment, JudgeOutcome, JudgeSpec

class _core_JudgeRunner:
    """Runs judges on one model. A model the host implements is offered `tools`, the host's live
    judge tools (`judge_tool_schemas`).
    """
    def __init__(self, llm: LLM[Any] | None, tools: object, host: _Evaluation) -> None: ...
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

def _core_evaluation_score(verdicts: Sequence[str]) -> EvaluationScore:
    """The aggregates of `verdicts`; a verdict other than pass, fail or maybe is refused."""

def _core_judge_tool_schemas() -> object:
    """The judge tools' raw schemas, which the host offers a model it implements."""
