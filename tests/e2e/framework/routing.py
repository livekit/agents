"""Labeled response recognition through ordinary sessions and the production routing boundary."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from livekit.agents import Agent, DelegationOptions, InputRequired, TextSession, llm
from livekit.agents.storage import ContextStore
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from tests.fakes import FakeLLM, tool_call

from .models import write_live_report


@dataclass(frozen=True)
class RoutingCase:
    id: str
    split: Literal["development", "heldout"]
    request: str
    questions: tuple[str, ...]
    presentation: str
    human: str
    matches: tuple[int, ...]
    unclear: bool
    because: str


def routing_cases() -> list[RoutingCase]:
    corpus = json.loads((Path(__file__).parents[1] / "cases/input_routing.json").read_text())
    cases = []
    for item in corpus["cases"]:
        scenario = corpus["scenarios"][item["scenario"]]
        case = RoutingCase(
            **{key: value for key, value in item.items() if key != "scenario" and key != "matches"},
            request=scenario["request"],
            questions=tuple(scenario["questions"]),
            presentation=scenario["presentation"],
            matches=tuple(item["matches"]),
        )
        if len(set(case.questions)) != len(case.questions):
            raise ValueError(f"{case.id}: source questions must be distinguishable")
        if len(set(case.matches)) != len(case.matches) or any(
            index < 0 or index >= len(case.questions) for index in case.matches
        ):
            raise ValueError(f"{case.id}: an expected source is not in the scenario")
        if case.unclear and case.matches:
            raise ValueError(f"{case.id}: unclear must not select a source")
        cases.append(case)
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("routing case IDs must be unique")
    return cases


class RoutingModel(llm.LLM):
    """Observe real provider output; optional prompt experiments keep native tools and validation."""

    def __init__(self, model: llm.LLM, *, prompt: str | None = None) -> None:
        super().__init__()
        self.inner = model
        self.prompt = prompt
        self.requests: list[llm.ChatContext] = []
        self.calls: list[llm.FunctionToolCall] = []
        self.seconds = 0.0

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def provider(self) -> str:
        return self.inner.provider

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> llm.LLMStream:
        history = chat_ctx.copy()
        if self.prompt is not None:
            first = history.items[0]
            assert first.type == "message" and first.role == "system"
            items = history.items
            items[0] = llm.ChatMessage(
                id=first.id, role="system", content=[self.prompt], created_at=first.created_at
            )
            history = llm.ChatContext(items=items)
        self.requests.append(history.copy())
        return _RoutingStream(
            self, chat_ctx=history, tools=tools or [], conn_options=conn_options, extra=kwargs
        )


class _RoutingStream(llm.LLMStream):
    def __init__(self, model: RoutingModel, *, extra: dict[str, Any], **kwargs: Any) -> None:
        self.extra = extra
        super().__init__(model, **kwargs)

    async def _run(self) -> None:
        assert isinstance(self._llm, RoutingModel)
        started = perf_counter()
        try:
            async with self._llm.inner.chat(
                chat_ctx=self._chat_ctx,
                tools=self._tools,
                conn_options=self._conn_options,
                **self.extra,
            ) as stream:
                async for chunk in stream:
                    if chunk.delta is not None:
                        self._llm.calls.extend(chunk.delta.tool_calls)
                    self.send(chunk)
        finally:
            self._llm.seconds += perf_counter() - started


async def evaluate_routing(case: RoutingCase, model: RoutingModel, directory: Path) -> None:
    """Script the conversational model; recognize the person's response with the supplied model."""
    calls = [tool_call("ask", question=question) for question in case.questions]
    presentation = (
        tool_call(
            "present_input_requests", requests=list(range(len(calls))), text=case.presentation
        )
        if len(calls) > 1
        else case.presentation
    )
    agent = Agent(
        name="intake",
        instructions="Collect the details the tools request.",
        llm=FakeLLM([calls, presentation, "Thank you."]),
    )

    @agent.tool
    async def ask(question: str) -> InputRequired:
        """Request a detail from the person."""
        return InputRequired(question)

    store = ContextStore()
    session = TextSession(agent=agent, delegation=DelegationOptions(routing_model=model))
    observed: dict[str, object] = {"case": asdict(case)}
    try:
        opened = await session.run(case.request, context=store.bind())
        assert opened.status == "input_required" and opened.question == case.presentation
        started = perf_counter()
        await session.run(case.human, context=store.bind())
        observed["turn_seconds"] = perf_counter() - started
        assert len(model.requests) == 1, "one eligible human turn should invoke its detector once"
        assert len(model.calls) == 1 and model.calls[0].name == "route_input_response"
        history = model.requests[0]
        prompt = history.items[0].text_content or ""
        offered = json.loads(history.items[-1].text_content or "{}")["candidates"]
        assert sorted(item["question"] for item in offered) == sorted(case.questions), (
            "the native classifier did not offer every presented source exactly once"
        )
        expected_questions = {case.questions[index] for index in case.matches}
        expected = {
            "candidates": sorted(
                item["index"] for item in offered if item["question"] in expected_questions
            ),
            "unclear": case.unclear,
        }
        actual = json.loads(model.calls[0].arguments)
        actual["candidates"] = sorted(actual["candidates"])
        observed.update(
            expected=expected,
            actual=actual,
            detector_seconds=model.seconds,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            prompt=prompt,
            offered=offered,
        )
        state = await store.load_session()
        assert state.current_agent is not None
        inputs = state.current_agent.input_requests.values()
        answered = {
            request.source_message.text_content
            for request in inputs
            if request.response_message is not None
        }
        observed["native_matches"] = sorted(case.questions.index(question) for question in answered)
        observed["native_states"] = {
            request.source_message.text_content: request.state
            for request in state.current_agent.input_requests.values()
        }
        # JSON equality keeps False distinct from a malformed numeric verdict of 0.
        assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True), (
            f"{case.id}: expected input routing {expected}, received {actual}. {case.because}"
        )
        assert answered == expected_questions, "native reservations disagree with the verdict"
        for request in state.current_agent.input_requests.values():
            if request.response_message is not None:
                assert request.response_message.text_content == case.human
            else:
                assert request.state == "open", "an unmatched source did not remain open"
        observed["passed"] = True
    except BaseException as error:
        observed.update(passed=False, error_type=type(error).__name__)
        raise
    finally:
        write_live_report(
            directory,
            models={"frontend": "scripted", "routing": model.model},
            turns=[{"human": case.human}],
            state=await store.load_session(),
            observations=observed,
        )


def routing_summary(directory: Path) -> dict[str, object]:
    """Summarize every case, deduplicating pytest's artifact symlinks; failures stay in the total."""
    paths = {path.resolve() for path in directory.glob("*/live-model-report.json")}
    if not paths:
        raise ValueError(f"no routing reports in {directory}")
    reports = [json.loads(path.read_text()) for path in sorted(paths)]
    cases = [report["observations"] for report in reports]
    judged = [case for case in cases if "actual" in case and "expected" in case]
    durations = sorted(case["detector_seconds"] for case in judged)
    return {
        "cases": len(cases),
        "exact_passes": sum(case.get("passed") is True for case in cases),
        "source_selection_passes": sum(
            json.dumps(case["actual"]["candidates"]) == json.dumps(case["expected"]["candidates"])
            for case in judged
        ),
        "unexpected_selection_cases": sum(
            bool(
                {json.dumps(index) for index in case["actual"]["candidates"]}
                - {json.dumps(index) for index in case["expected"]["candidates"]}
            )
            for case in judged
        ),
        "missed_selection_cases": sum(
            bool(
                {json.dumps(index) for index in case["expected"]["candidates"]}
                - {json.dumps(index) for index in case["actual"]["candidates"]}
            )
            for case in judged
        ),
        "without_verdict": len(cases) - len(judged),
        "detector_seconds": {
            "p50": statistics.median(durations) if durations else None,
            "p95": durations[math.ceil(len(durations) * 0.95) - 1] if durations else None,
        },
        "models": sorted({report["models"]["routing"] for report in reports}),
        "prompt_sha256": sorted({case["prompt_sha256"] for case in judged}),
        "failed_cases": [case["case"]["id"] for case in cases if not case.get("passed")],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize a labeled routing evaluation run.")
    parser.add_argument("directory", type=Path)
    print(json.dumps(routing_summary(parser.parse_args().directory), indent=2))
