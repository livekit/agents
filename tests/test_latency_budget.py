import pytest

from livekit.agents import AgentSession, LatencyBudgetEvent

pytestmark = pytest.mark.unit


def test_latency_budget_options_validation() -> None:
    with pytest.raises(ValueError, match="budget.*greater than zero"):
        AgentSession(latency_budget={"budget": 0})

    with pytest.raises(ValueError, match="warning.*no greater than budget"):
        AgentSession(latency_budget={"budget": 1.0, "warning": 1.1})


@pytest.mark.parametrize(
    ("latency", "expected_level", "expected_threshold"),
    [(0.8, "warning", 0.5), (1.2, "exceeded", 1.0)],
)
def test_latency_budget_event(
    latency: float, expected_level: str, expected_threshold: float
) -> None:
    session = AgentSession(latency_budget={"budget": 1.0, "warning": 0.5})
    events: list[LatencyBudgetEvent] = []
    session.on("latency_budget", events.append)

    session._evaluate_latency_budget(latency=latency, speech_id="speech-1")

    assert len(events) == 1
    assert events[0].level == expected_level
    assert events[0].threshold == expected_threshold
    assert events[0].budget == 1.0
    assert events[0].latency == latency
    assert events[0].speech_id == "speech-1"


def test_latency_budget_does_not_emit_within_warning_threshold() -> None:
    session = AgentSession(latency_budget={"budget": 1.0, "warning": 0.5})
    events: list[LatencyBudgetEvent] = []
    session.on("latency_budget", events.append)

    session._evaluate_latency_budget(latency=0.49, speech_id="speech-1")

    assert events == []
