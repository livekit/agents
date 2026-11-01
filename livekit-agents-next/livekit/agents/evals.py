from __future__ import annotations

import functools
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, cast

from . import _bindings, _core
from ._generated.evals import (
    EvaluationResult as EvaluationResult,
    IntentJudgment,
    JudgeOutcome,
    JudgeOutcomeFailed,
    JudgeSpecBuiltIn,
    JudgeSpecHost,
    JudgmentResult as JudgmentResult,
)
from .llm import LLM, ChatContext, Tool, function_tool
from .log import logger
from .telemetry import capture_content_enabled, trace_types, tracer

if TYPE_CHECKING:
    from pydantic import JsonValue

    from .inference import LLMModels
    from .inference.llm import ChatCompletionOptions

__all__ = [
    "EvaluationResult",
    "Evaluator",
    "JudgeGroup",
    "Judge",
    "JudgmentResult",
    "accuracy_judge",
    "coherence_judge",
    "conciseness_judge",
    "handoff_judge",
    "relevancy_judge",
    "safety_judge",
    "task_completion_judge",
    "tool_use_judge",
]


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


class Judge:
    """Base class for custom evaluation judges.

    Subclass and override :meth:`evaluate` to implement deterministic
    or programmatic checks that don't need an LLM::

        class CitationJudge(Judge):
            def __init__(self):
                super().__init__(name="citation")

            async def evaluate(self, *, chat_ctx, reference=None, llm=None):
                has_citation = any(
                    "[source]" in (m.text_content or "") for m in chat_ctx.messages
                )
                return JudgmentResult(
                    verdict="pass" if has_citation else "fail",
                    reasoning="Found citation markers" if has_citation else "No citations",
                )
    """

    def __init__(self, *, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def evaluate(
        self,
        *,
        chat_ctx: ChatContext,
        reference: ChatContext | None = None,
        llm: LLM | None = None,
    ) -> JudgmentResult:
        """Evaluate a conversation and return a judgment.

        Must be overridden in subclasses.
        """
        raise NotImplementedError(f"Judge '{self._name}' does not implement evaluate().")


class _BuiltInJudge:
    """A built-in judge: the core frames the conversation and asks the model for a verdict."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def evaluate(
        self,
        *,
        chat_ctx: ChatContext,
        reference: ChatContext | None = None,
        llm: LLM | None = None,
    ) -> JudgmentResult:
        (outcome,) = await _evaluate([self], chat_ctx, reference, llm)
        if isinstance(outcome, JudgeOutcomeFailed):
            raise ValueError(outcome.error)
        return outcome.judgment


def task_completion_judge() -> Evaluator:
    """Judge that evaluates if the agent completed its goal based on its instructions.

    Extracts the agent's instructions from AgentConfigUpdate items in the chat context
    and evaluates the whole conversation against them. Considers the overall caller
    experience, including any handoffs between agents.

    Based on First Call Resolution (FCR), the key metric in call centers.
    Useful for: customer service, appointment booking, order management.
    """
    return _BuiltInJudge("task_completion")


def handoff_judge() -> Evaluator:
    """Judge that evaluates context preservation across agent handoffs.

    Handoffs can be silent (seamless) or explicit ("transferring you to...").
    Either is acceptable, but the new agent must preserve context and not
    re-ask for information already provided.
    Automatically passes if no handoffs occurred.

    Useful for: multi-agent systems, transfers to specialists, escalations.
    """
    return _BuiltInJudge("handoff")


def accuracy_judge() -> Evaluator:
    """Judge that evaluates factual accuracy of information provided.

    Focuses on grounding - responses must be supported by function call outputs.
    Catches hallucinations, misquoted data, and contradictions with tool results.

    Useful for: healthcare, insurance, finance - where wrong information has consequences.
    """
    return _BuiltInJudge("accuracy")


def tool_use_judge() -> Evaluator:
    """Judge that evaluates if the agent used tools correctly.

    Checks tool selection, parameter accuracy, output interpretation, and error handling.
    Voice agents rely on function calls for lookups, bookings, transfers, etc.

    Useful for: any agent with tools - appointment systems, order lookups, CRM integrations.
    """
    return _BuiltInJudge("tool_use")


def safety_judge() -> Evaluator:
    """Judge that evaluates if responses are safe, compliant, and appropriate.

    Checks for unauthorized advice, improper disclosure, failure to escalate,
    and toxic or harmful language.

    Useful for: regulated industries, user-facing agents where compliance and tone matter.
    """
    return _BuiltInJudge("safety")


def relevancy_judge() -> Evaluator:
    """Judge that evaluates if responses are relevant and on-topic.

    Checks if the agent directly addresses what the user asked, stays focused
    on the topic, and appropriately redirects off-topic requests.

    Useful for: any conversational agent, scoped agents, customer service.
    """
    return _BuiltInJudge("relevancy")


def coherence_judge() -> Evaluator:
    """Judge that evaluates if responses are coherent and logical.

    Checks if the agent presents ideas in an organized manner without
    contradictions or confusing jumps between topics.

    Useful for: complex explanations, multi-turn conversations, technical support.
    """
    return _BuiltInJudge("coherence")


def conciseness_judge() -> Evaluator:
    """Judge that evaluates if responses are appropriately concise.

    Critical for voice AI where brevity matters. Checks for unnecessary
    verbosity, repetition, and redundant details.

    Useful for: voice agents, chat interfaces, any context where user time matters.
    """
    return _BuiltInJudge("conciseness")


_evals_verbose = int(os.getenv("LIVEKIT_EVALS_VERBOSE", 0))


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
                Model strings default to the lowest reasoning effort the model
                supports and to low inference priority; pass a configured LLM
                instance to override.
            judges: The judges to run during evaluation.
        """
        if isinstance(llm, str):
            from .inference import LLM as InferenceLLM
            from .inference.llm import min_reasoning_effort

            extra_kwargs: ChatCompletionOptions = {}
            if (effort := min_reasoning_effort(llm)) is not None:
                extra_kwargs["reasoning_effort"] = effort

            # Judges run after the conversation, so no caller is waiting on them:
            # they must not compete with live voice traffic for gateway quota.
            self._llm: LLM = InferenceLLM(llm, inference_class="low", extra_kwargs=extra_kwargs)
        else:
            self._llm = llm

        self._judges = judges or []

    async def evaluate(
        self,
        chat_ctx: ChatContext,
        *,
        reference: ChatContext | None = None,
    ) -> EvaluationResult:
        """Evaluate a conversation with all judges, concurrently.

        A judge that fails is logged and absent from the result. Automatically tags the
        session with results when called within a job context.

        Args:
            chat_ctx: The conversation to evaluate.
            reference: Optional reference conversation for comparison.

        Returns:
            EvaluationResult containing all judgment results.
        """
        from .job import get_job_context

        judgments: dict[str, JudgmentResult] = {}
        for outcome in await _evaluate(self._judges, chat_ctx, reference, self._llm):
            if isinstance(outcome, JudgeOutcomeFailed):
                logger.warning(f"Judge '{outcome.name}' failed: {outcome.error}")
                continue
            judgments[outcome.name] = outcome.judgment

        if _evals_verbose:
            print("\n+ JudgeGroup evaluation results:")
            for name, result in judgments.items():
                print(f"  [{name}] verdict={result.verdict}")
                print(f"    reasoning: {result.reasoning}\n")

        try:
            ctx = get_job_context()
        except RuntimeError:
            pass  # not in a job: nothing to tag
        else:
            for name, judgment in judgments.items():
                ctx.tagger._core_record_judgment(
                    name, judgment.verdict, judgment.reasoning, judgment.instructions
                )

        return EvaluationResult(judgments=judgments)


class _Evaluation:
    """The host side of one evaluation: the core calls back into it to run an application
    evaluator on the conversation, with the evaluation's model."""

    def __init__(
        self, chat_ctx: ChatContext, reference: ChatContext | None, llm: LLM | None
    ) -> None:
        self._chat_ctx = chat_ctx
        self._reference = reference
        self._llm = llm

    async def evaluate(self, evaluator: Evaluator) -> JudgmentResult:
        result = await evaluator.evaluate(
            chat_ctx=self._chat_ctx, reference=self._reference, llm=self._llm
        )
        return result


async def _evaluate(
    judges: Sequence[Evaluator],
    chat_ctx: ChatContext,
    reference: ChatContext | None,
    llm: LLM | None,
) -> list[JudgeOutcome]:
    """Every judge's outcome, in judge order, under one span recording each judgment."""
    specs = [
        JudgeSpecBuiltIn(name=j.name)
        if isinstance(j, _BuiltInJudge)
        else JudgeSpecHost(name=j.name, evaluator=j)
        for j in judges
    ]
    with tracer.start_as_current_span("judge_evaluation") as span:
        outcomes = await _runner(llm, _Evaluation(chat_ctx, reference, llm)).evaluate(
            specs,
            chat_ctx.items,
            None if reference is None else reference.items,
        )
        for outcome in outcomes:
            if isinstance(outcome, JudgeOutcomeFailed):
                continue
            attributes = {
                trace_types.ATTR_GEN_AI_EVALUATION_NAME: outcome.name,
                trace_types.ATTR_GEN_AI_EVALUATION_SCORE_LABEL: outcome.judgment.verdict,
            }
            if capture_content_enabled():
                attributes[trace_types.ATTR_GEN_AI_EVALUATION_EXPLANATION] = (
                    outcome.judgment.reasoning
                )
            span.add_event("gen_ai.evaluation.result", attributes)
    return outcomes


async def _judge_intent(llm: LLM, intent: str, message: str) -> IntentJudgment:
    """Whether ``message`` fulfills ``intent``; ``AssertionError`` when it cannot be judged."""
    return await _runner(llm, _Evaluation(ChatContext.empty(), None, llm)).judge_intent(
        intent, message
    )


def _runner(llm: LLM | None, host: _Evaluation) -> _core._core_JudgeRunner:
    return _core._core_JudgeRunner(
        None if llm is None else _bindings.core_leg(llm), _judge_tools(), host
    )


@functools.cache
def _judge_tools() -> list[Tool]:
    """The judge tools a model the host implements is offered; the core reads their calls."""

    async def answer(raw_arguments: dict[str, object]) -> None:
        raise RuntimeError("judge tools are read by the core, never called")

    schemas = cast("list[dict[str, JsonValue]]", _core._core_judge_tool_schemas())
    return [function_tool(answer, raw_schema=s) for s in schemas]
