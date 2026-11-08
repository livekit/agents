from __future__ import annotations

import functools
import json
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, Union, cast

from livekit.protocol import agent_simulation as proto

from . import _bindings, _core
from ._generated import public_evals as _public
from ._generated.evals import (
    IntentJudgment,
    JudgeOutcome,
    JudgeOutcomeFailed,
)
from ._generated.public_evals import *  # noqa: F403
from .llm import LLM, ChatContext, Tool, function_tool
from .log import logger

if TYPE_CHECKING:
    from pydantic import JsonValue

    from .inference import LLMModels
    from .inference.llm import ChatCompletionOptions
    from .job import JobContext


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
                    "[source]" in (m.text_content or "") for m in chat_ctx.messages()
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


_evals_verbose = int(os.getenv("LIVEKIT_EVALS_VERBOSE", 0))


class JudgeGroup:
    """A group of judges that evaluate conversations together.

    Automatically tags the session with judgment results when called within a job context.

    Example:
        ```python
        async def on_session_end(ctx: JobContext) -> None:
            judges = JudgeGroup(
                llm="openai/gpt-4o-mini",
                judges=[CitationJudge()],
            )

            report = ctx.make_session_report()
            result = await judges.evaluate(ChatContext(items=report.chat_history))
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
    """Every judge's outcome, in judge order; the core records one span per judge."""
    return await _runner(llm, _Evaluation(chat_ctx, reference, llm)).evaluate(list(judges))


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


ScenarioUserdata: TypeAlias = dict[str, Union["ScenarioUserdata", Any]]
"""A scenario's ``userdata``, decoded from its JSON string."""


class SimulationContext:
    """Passed to the ``on_simulation_end`` callback while running under a simulation.

    Carries two verdicts, both recorded for the run:
      - :attr:`simulator_verdict`: the simulator's verdict (its LLM judgment of the chat).
      - :attr:`user_verdict`: your own veto, set via :meth:`fail` from richer checks
        (e.g. comparing mock backend state against the benchmark target in
        ``scenario.userdata``). The effective result is the AND of the two: your check
        can fail a run the simulator passed, but it can never rescue one, so there is
        no ``success()``; not calling :meth:`fail` leaves the simulator's verdict to stand.

    Use :attr:`job_context` to reach the running session and the room.
    """

    def __init__(self, dispatch: proto.SimulationDispatch, job_ctx: JobContext) -> None:
        self._dispatch = dispatch
        self._scenario = dispatch.scenario
        self._job_ctx = job_ctx
        self._simulator_verdict: SimulationVerdict | None = None
        self._user_verdict: SimulationVerdict | None = None

    @property
    def scenario(self) -> proto.Scenario:
        return self._scenario

    @property
    def simulation_mode(self) -> Literal["text", "audio"]:
        """How the simulated user interacts with the agent. Unspecified is treated as text,
        since simulations predating the field were all text-only."""
        if self._dispatch.mode == proto.SimulationMode.SIMULATION_MODE_AUDIO:
            return "audio"
        return "text"

    @property
    def simulation_run_id(self) -> str:
        """The run this session belongs to, available as soon as the entrypoint runs."""
        return self._dispatch.simulation_run_id

    @property
    def simulation_job_id(self) -> str:
        """The run job (one scenario) behind this session, likewise available up front."""
        return self._dispatch.job_id

    @property
    def simulator_verdict(self) -> SimulationVerdict:
        """The simulator's verdict (its LLM judgment of the conversation). Read-only;
        recorded alongside your :attr:`user_verdict`.

        Only available once the simulation has ended, i.e. inside ``on_simulation_end``.
        Raises :class:`RuntimeError` if accessed earlier (e.g. from the entrypoint).
        """
        if self._simulator_verdict is None:
            raise RuntimeError(
                "simulator_verdict is only available inside on_simulation_end "
                "(after the simulation completes)"
            )
        return self._simulator_verdict

    @property
    def job_context(self) -> JobContext:
        """The :class:`JobContext` for this run; use it to reach the running session
        (``job_context.primary_session``), the room, and other job state."""
        return self._job_ctx

    def _begin_finalize(
        self,
        *,
        simulator_verdict: SimulationVerdict,
    ) -> None:
        """Internal: populate the simulator verdict / run before on_simulation_end."""
        self._simulator_verdict = simulator_verdict

    def userdata(self) -> ScenarioUserdata:
        """The scenario's ``userdata`` decoded from its JSON string (``{}`` if empty)."""
        if not self._scenario.userdata:
            return {}
        data: ScenarioUserdata = json.loads(self._scenario.userdata)
        return data

    def fail(self, reason: str = "") -> None:
        """Veto this run from your own checks (e.g. final DB state diverged).

        The effective result is the AND of both verdicts, so this can only fail a run
        the simulator passed, never rescue one. The simulator's verdict is still
        reported. The last call wins if you call :meth:`fail` more than once.
        """
        self._user_verdict = SimulationVerdict(success=False, reason=reason)

    @property
    def user_verdict(self) -> SimulationVerdict | None:
        """Your veto set via :meth:`fail`, or None if you didn't veto the run."""
        return self._user_verdict


__all__ = [
    "Evaluator",
    "JudgeGroup",
    "Judge",
    "SimulationContext",
]
__all__ += _public.__all__
