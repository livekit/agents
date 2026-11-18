"""The model-quality harness keeps actual provider verdicts and checks native input state."""

import json
from pathlib import Path

import pytest

from tests.e2e.framework.routing import (
    RoutingModel,
    evaluate_routing,
    routing_cases,
    routing_summary,
)
from tests.fakes import FakeLLM, tool_call


@pytest.mark.parametrize(
    ("case_id", "candidates", "unclear"),
    [
        ("date-explanation", [], False),
        ("names-last-only", [1], False),
        ("approvals-ambiguous", [], True),
    ],
)
async def test_a_recorded_verdict_reaches_native_input_state(
    case_id: str, candidates: list[int], unclear: bool, tmp_path: Path
) -> None:
    case = next(case for case in routing_cases() if case.id == case_id)
    routing = RoutingModel(
        FakeLLM([tool_call("route_input_response", candidates=candidates, unclear=unclear)])
    )
    await evaluate_routing(case, routing, tmp_path)
    report = json.loads((tmp_path / "live-model-report.json").read_text())
    assert report["observations"]["passed"] is True
    assert report["models"]["frontend"] == "scripted"


async def test_an_incorrect_model_selection_keeps_the_failed_case_as_evidence(
    tmp_path: Path,
) -> None:
    case = next(case for case in routing_cases() if case.id == "date-explanation")
    routing = RoutingModel(
        FakeLLM([tool_call("route_input_response", candidates=[0], unclear=False)])
    )
    directory = tmp_path / "failed"
    directory.mkdir()
    with pytest.raises(AssertionError, match="expected input routing"):
        await evaluate_routing(case, routing, directory)
    report = json.loads((directory / "live-model-report.json").read_text())
    assert report["observations"]["passed"] is False
    assert report["observations"]["actual"]["candidates"] == [0]
    summary = routing_summary(tmp_path)
    assert summary["cases"] == 1 and summary["exact_passes"] == 0
    assert summary["unexpected_selection_cases"] == 1
