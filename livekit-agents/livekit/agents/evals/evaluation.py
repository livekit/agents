from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from ..llm import LLM, ChatContext
from .judge import JudgmentResult

_evals_verbose = int(os.getenv("LIVEKIT_EVALS_VERBOSE", 0))

if TYPE_CHECKING:
    from ..inference import LLMModels


class Evaluator(Protocol):
    """Protocol for any object that can evaluate a conversation."""

    @property
    def name(self) -> str:
        """Name identifying this evaluator."""
        ...

    async def evaluate(
        self,
        *,
        chat_ctx: ChatContext,
        reference: ChatContext | None = None,
        llm: LLM | None = None,
    ) -> JudgmentResult: ...


@dataclass
class EvaluationResult:
    """Result of evaluating a conversation with a group of judges."""

    judgments: dict[str, JudgmentResult] = field(default_factory=dict)
    """Individual judgment results keyed by judge name."""

    @property
    def score(self) -> float:
        """Score from 0.0 to 1.0. Pass=1, maybe=0.5, fail=0."""
        if not self.judgments:
            return 0.0
        total = 0.0
        for j in self.judgments.values():
            if j.passed:
                total += 1.0
            elif j.uncertain:
                total += 0.5
        return total / len(self.judgments)

    @property
    def all_passed(self) -> bool:
        """True if all judgments passed. Maybes count as not passed."""
        return all(j.passed for j in self.judgments.values())

    @property
    def any_passed(self) -> bool:
        """True if at least one judgment passed."""
        return any(j.passed for j in self.judgments.values())

    @property
    def majority_passed(self) -> bool:
        """True if more than half of the judgments passed."""
        if not self.judgments:
            return True
        return self.score > len(self.judgments) / 2

    @property
    def none_failed(self) -> bool:
        """True if no judgments explicitly failed. Maybes are allowed."""
        return not any(j.failed for j in self.judgments.values())

class JudgeGroup:
    """A group of judges that evaluate conversations together.

    Automatically tags the session with judgment results when called within a job context.

    Example:
        ```python
        async def on_session_end(ctx: JobContext) -> None:
            judges = JudgeGroup(
                llm="openai/gpt-4o-mini",
                judges=[
                    task_completion_judge(),
                    accuracy_judge(),
                ],
            )

            report = ctx.make_session_report()
            result = await judges.evaluate(report.chat_history)
            # Results are automatically tagged to the session
        ```
    """

    def __init__(
        self,
        *,
        llm: LLM | LLMModels | str,
        judges: list[Evaluator] | None = None,
    ) -> None:
        """Initialize a JudgeGroup.

        Args:
            llm: The LLM to use for evaluation. Can be an LLM instance or a model
                string like "openai/gpt-4o-mini" (uses LiveKit inference gateway).
            judges: The judges to run during evaluation.
        """
        if isinstance(llm, str):
            from ..inference import LLM as InferenceLLM

            self._llm: LLM = InferenceLLM(llm)
        else:
            self._llm = llm

        self._judges = judges or []

    @property
    def llm(self) -> LLM:
        """The LLM used for evaluation."""
        return self._llm

    @property
    def judges(self) -> list[Evaluator]:
        """The judges to run during evaluation."""
        return self._judges

    async def evaluate(
        self,
        chat_ctx: ChatContext,
        *,
        reference: ChatContext | None = None,
    ) -> EvaluationResult:
        """Evaluate a conversation with all judges.

        Automatically tags the session with results when called within a job context.

        Args:
            chat_ctx: The conversation to evaluate.
            reference: Optional reference conversation for comparison.

        Returns:
            EvaluationResult containing all judgment results.
        """
        from ..job import get_job_context
        from ..log import logger

        # Run all judges concurrently
        async def run_judge(judge: Evaluator) -> tuple[str, JudgmentResult | BaseException]:
            try:
                result = await judge.evaluate(
                    chat_ctx=chat_ctx,
                    reference=reference,
                    llm=self._llm,
                )
                return judge.name, result
            except Exception as e:
                logger.warning(f"Judge '{judge.name}' failed: {e}")
                return judge.name, e

        results = await asyncio.gather(*[run_judge(j) for j in self._judges])

        # Filter out failed judges
        judgments: dict[str, JudgmentResult] = {}
        for name, result in results:
            if isinstance(result, JudgmentResult):
                judgments[name] = result

        evaluation_result = EvaluationResult(judgments=judgments)

        if _evals_verbose:
            print("\n+ JudgeGroup evaluation results:")
            for name, result in results:
                if isinstance(result, JudgmentResult):
                    print(f"  [{name}] verdict={result.verdict}")
                    print(f"    reasoning: {result.reasoning}\n")
                else:
                    print(f"  [{name}] ERROR: {result}\n")

        # Auto-tag if running within a job context
        try:
            ctx = get_job_context()
            ctx.tagger._evaluation(evaluation_result)
        except RuntimeError:
            pass  # Not in a job context, skip tagging

        return evaluation_result
