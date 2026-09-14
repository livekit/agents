from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from examples.telephony.amd_simulation.__main__ import Trace, monitor_budget
from examples.telephony.amd_simulation.checks import (
    Event,
    check,
    extra_uncertain,
    extra_wait,
    turnaround,
)
from examples.telephony.amd_simulation.scenarios import (
    HUMAN,
    SCENARIOS,
    SCREEN,
    SCREEN_AGAIN,
    Scenario,
)
from livekit.agents.voice.amd._inference import ALLOWED
from livekit.agents.voice.amd.classifier import AMDCategory

pytestmark = pytest.mark.unit


async def test_wait_monitor_aborts_a_call_with_an_expired_uncertainty_interval() -> None:
    scenario = Scenario("expired", (HUMAN,), "human")
    trace = Trace()
    trace.origin -= 6
    trace.events.append(Event(0, "prediction", 0, {"category": "uncertain", "turn_id": 1}))
    with pytest.raises(AssertionError, match="uncertainty/wait budget"):
        await monitor_budget(scenario, trace)
    assert trace.events[-1].kind == "budget_exhausted"


async def test_wait_monitor_aborts_on_the_third_extra_uncertain_turn() -> None:
    scenario = Scenario("three", (SCREEN, SCREEN_AGAIN, HUMAN), "human")
    trace = Trace()
    trace.events.extend(
        Event(0, "classification", i, {"category": "uncertain", "turn_id": i}) for i in range(3)
    )
    with pytest.raises(AssertionError, match="Call-wide uncertainty budget"):
        await monitor_budget(scenario, trace)


async def test_normal_latency_does_not_trip_monitor_and_monitor_can_be_cancelled() -> None:
    trace = Trace()
    trace.origin -= 100
    task = asyncio.create_task(monitor_budget(Scenario("normal", (HUMAN,), "human"), trace))
    try:
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not trace.events


async def test_action_observed_before_waiter_starts_is_not_lost() -> None:
    trace = Trace()
    waiter = asyncio.create_task(trace.until(lambda: bool(trace.current("dtmf")), 1))
    trace.add("dtmf", digit="2")
    await waiter


def turn(index: int, category: str, end: float, delay: float) -> list[Event]:
    ready = end + delay
    return [
        Event(end, "input_end", index),
        Event(ready, "classification", index, {"category": category, "turn_id": index + 1}),
        Event(
            ready,
            "prediction",
            index,
            {"category": category, "turn_id": index + 1, "reason": "prediction"},
        ),
        Event(ready + 0.1, "audio", index),
        Event(ready + 2, "reply", index, {"text": "Alex from Acme Dental.", "interrupted": False}),
        Event(ready + 2, "speech_end", index),
    ]


def completion(time: float, category: str = "human") -> Event:
    return Event(
        time,
        "completed",
        2,
        {"category": category, "reason": "finished", "voicemail_message_played": False},
    )


def test_scenarios_cover_every_allowed_transition() -> None:
    covered = set()
    for scenario in SCENARIOS:
        previous = "uncertain"
        assert len({s.name for s in scenario.steps}) == len(scenario.steps)
        for step in scenario.steps:
            if step.category is not None:
                covered.add((previous, step.category))
                previous = step.category
    required = {
        (current.value, following.value)
        for current, following_set in ALLOWED.items()
        for following in following_set
    }
    assert required <= covered
    assert len({s.name for s in SCENARIOS}) == len(SCENARIOS)


def test_shared_wait_budget_does_not_reset_on_stage_change() -> None:
    scenario = Scenario("budget", (SCREEN, HUMAN), "human")
    events = turn(0, "machine-screening", 10, 5) + turn(1, "human", 30, 4) + [completion(35)]
    events += [
        Event(
            12,
            "prediction",
            0,
            {"category": "uncertain", "turn_id": 8, "reason": "inference_timeout"},
        ),
        Event(
            31,
            "prediction",
            1,
            {"category": "uncertain", "turn_id": 9, "reason": "inference_timeout"},
        ),
    ]
    assert extra_wait(scenario, events) == pytest.approx(6)
    assert any("waiting budget exceeded" in e for e in check(scenario, events))


def test_scripted_pause_and_reply_playback_do_not_consume_budget() -> None:
    scenario = Scenario("pause", (HUMAN,), "human")
    events = [
        Event(0, "prediction", 0, {"category": "uncertain"}),
        Event(1, "pause_start", 0),
        Event(4, "pause_end", 0),
        Event(2, "speech_start", 0),
        Event(5, "speech_end", 0),
        Event(6, "input_start", 0),
        Event(8, "clip_end", 0),
        Event(10, "prediction", 0, {"category": "human"}),
    ]
    assert extra_wait(scenario, events) == pytest.approx(4)


def test_normal_response_latency_is_reported_without_consuming_wait_budget() -> None:
    scenario = Scenario("normal-latency", (SCREEN, HUMAN), "human")
    events = turn(0, "machine-screening", 10, 8) + turn(1, "human", 30, 7) + [completion(38)]
    assert turnaround(scenario, events) == pytest.approx(15.2)
    assert extra_wait(scenario, events) == 0
    assert not any("waiting budget exceeded" in e for e in check(scenario, events))


def test_wait_flag_counts_even_when_category_is_concrete() -> None:
    scenario = Scenario("wait", (SCREEN,), "human")
    events = [
        Event(0, "input_end", 0),
        Event(1, "prediction", 0, {"category": "machine-screening", "should_wait": True}),
        Event(7, "prediction", 0, {"category": "machine-screening", "should_wait": False}),
    ]
    assert extra_wait(scenario, events) == pytest.approx(5.5)


def test_open_wait_interval_uses_live_clock_and_excludes_active_input() -> None:
    scenario = Scenario("live-wait", (HUMAN,), "human")
    events = [Event(1, "prediction", 0, {"category": "uncertain"})]
    assert extra_wait(scenario, events, now=7) == 6
    events.append(Event(3, "input_start", 0))
    assert extra_wait(scenario, events, now=7) == 2


def test_uncertainty_budget_includes_raw_stage_preserving_predictions() -> None:
    scenario = Scenario("uncertain", (SCREEN, SCREEN_AGAIN, HUMAN), "human")
    events = [
        Event(i, "classification", i, {"category": "uncertain", "turn_id": i + 1}) for i in range(3)
    ]
    assert extra_uncertain(scenario, events) == 3
    assert any("uncertainty budget exceeded" in e for e in check(scenario, events))


def test_scripted_uncertain_stage_is_not_extra_uncertainty() -> None:
    scenario = Scenario("expected-uncertainty", (replace(SCREEN, category="uncertain"),), "human")
    events = [Event(1, "classification", 0, {"category": "uncertain", "turn_id": 1})]
    assert extra_uncertain(scenario, events) == 0
    events.append(Event(2, "prediction", 0, {"category": "uncertain"}))
    assert extra_wait(scenario, events, now=30) == 0


def test_timeout_fallback_and_raw_result_do_not_double_count_one_uncertain_turn() -> None:
    scenario = Scenario("fallback", (SCREEN,), "human")
    events = [
        Event(1, "prediction", 0, {"category": "uncertain", "turn_id": 1}),
        Event(2, "classification", 0, {"category": "uncertain", "turn_id": 1}),
    ]
    assert extra_uncertain(scenario, events) == 1


def test_no_action_timeout_without_uncertainty_does_not_consume_wait_budget() -> None:
    scenario = Scenario("no-action", (SCREEN,), "human")
    events = [Event(1, "input_end", 0), Event(7.6, "action_timeout", 0)]
    assert extra_wait(scenario, events) == 0
    assert turnaround(scenario, events) == pytest.approx(6.6)


def test_missing_raw_classification_evidence_cannot_silently_pass() -> None:
    scenario = Scenario("missing-evidence", (HUMAN,), "human")
    events = [e for e in turn(0, "human", 10, 0.3) if e.kind != "classification"]
    events.append(completion(11))
    assert any("missing raw classification evidence" in e for e in check(scenario, events))


def test_wrong_intermediate_stage_fails_even_with_correct_completion() -> None:
    scenario = Scenario("trajectory", (SCREEN, HUMAN), "human")
    events = turn(0, "machine-ivr", 10, 1.5) + turn(1, "human", 20, 0.3) + [completion(21)]
    assert any("unexpected stage machine-ivr" in e for e in check(scenario, events))


def test_stage_preserving_uncertainty_can_precede_correct_transition() -> None:
    scenario = Scenario("preserved", (SCREEN, HUMAN), "human")
    events = turn(0, "machine-screening", 10, 1.5)
    events += [
        Event(20.1, "classification", 1, {"category": "uncertain", "turn_id": 8}),
        Event(
            20.1,
            "prediction",
            1,
            {"category": "machine-screening", "turn_id": 8, "reason": "prediction"},
        ),
    ]
    events += turn(1, "human", 20, 0.3) + [completion(21)]
    assert not any("unexpected stage" in e for e in check(scenario, events))


def test_premature_audio_fails_even_when_reply_text_exists() -> None:
    scenario = Scenario("early", (SCREEN,), "machine-screening")
    errors = check(scenario, turn(0, "machine-screening", 10, 0.5))
    assert "screen-name: premature audio" in errors
    assert "screen-name: premature prediction" in errors


def test_extra_reply_is_not_hidden_by_correct_trajectory() -> None:
    scenario = Scenario("duplicate", (HUMAN,), "human")
    events = turn(0, "human", 10, 0.3) + [completion(11)]
    events.append(Event(14, "reply", 0, {"text": "Please leave a message.", "interrupted": False}))
    assert any("expected 1 reply, got 2" in e for e in check(scenario, events))


def test_dtmf_requires_receipt_of_exact_digits() -> None:
    scenario = Scenario("digits", (replace(SCREEN, reply="", dtmf="2"),), "human")
    events = [Event(1, "input_end", 0), Event(3, "dtmf", 0, {"digit": "1"})]
    assert any("expected DTMF '2', received '1'" in e for e in check(scenario, events))


def test_missing_audio_cannot_pass_on_generated_text() -> None:
    scenario = Scenario("no-audio", (HUMAN,), "human")
    events = [e for e in turn(0, "human", 10, 0.3) if e.kind != "audio"] + [completion(11)]
    assert any("no reply audio reached" in e for e in check(scenario, events))


def test_post_completion_predictions_fail() -> None:
    scenario = Scenario("completed", (HUMAN,), "human")
    events = turn(0, "human", 10, 0.3) + [completion(10.2)]
    assert "AMD predicted after completion" in check(scenario, events)


def test_terminal_states_have_no_outgoing_scenarios() -> None:
    for scenario in SCENARIOS:
        terminal = False
        for step in scenario.steps:
            if terminal:
                assert step.category is None
            terminal = terminal or step.category in (
                AMDCategory.HUMAN,
                AMDCategory.MACHINE_UNAVAILABLE,
            )
