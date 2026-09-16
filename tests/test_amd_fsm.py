from __future__ import annotations

import pytest

from livekit.agents.voice.amd._fsm import _AMDFSM, ReplyDecision, _Transcript
from livekit.agents.voice.amd.events import AMDCategory, AMDCompletedEvent, AMDReason

pytestmark = pytest.mark.unit


def new_fsm(*, silence: float = 0) -> _AMDFSM:
    return _AMDFSM(
        idle_timeout=10,
        voicemail_idle_timeout=60,
        timeout=120,
        inference_timeout=1,
        machine_silence_threshold=silence,
        max_uncertain_turns=3,
    )


@pytest.fixture
def fsm() -> _AMDFSM:
    fsm = new_fsm()
    fsm.enter()
    fsm.start(0)
    return fsm


def request(fsm: _AMDFSM, now: float, text: str = "hello") -> int:
    transcript = _Transcript(text, "session")
    turn_id = fsm.turn_id + 1
    fsm.commit_turn(transcript, now, eot_delay=0, turn_id=turn_id)
    context, _ = fsm.transcript_ready(turn_id, transcript, now)
    assert context is not None
    return turn_id


def test_lifecycle_and_fixed_hard_deadline() -> None:
    fsm = new_fsm()
    assert not fsm.entered and not fsm.enabled and not fsm.started
    fsm.enter()
    assert fsm.enabled and not fsm.started
    fsm.update_idle(3, session_busy=False)
    assert fsm.next_deadline is None
    fsm.start(5)
    assert fsm.next_deadline == 125
    fsm.start(6)
    request(fsm, 7)
    fsm.prediction_received(1, AMDCategory.MACHINE_VM, 8, 1)
    fsm.update_idle(9, session_busy=True)
    assert fsm.next_deadline == 125
    fsm.update_idle(10, session_busy=False)
    assert fsm.next_deadline == 70
    fsm.update_idle(69, session_busy=True)
    assert fsm.next_deadline == 125
    fsm.tick(124)
    assert fsm.started
    fsm.tick(125)
    assert fsm.finished and not fsm.enabled and not fsm.started
    assert fsm.next_deadline is None
    assert fsm.completion().reason == "timeout"
    assert fsm.completion().category == AMDCategory.MACHINE_VM
    with pytest.raises(RuntimeError, match="new AMD instance"):
        fsm.enter()

    delayed = new_fsm()
    delayed.enter()
    delayed.start(0)
    delayed.update_idle(0, session_busy=False)
    delayed.tick(121)
    assert delayed.completion().reason == "idle_timeout"


@pytest.mark.parametrize(
    ("committed_at", "reasons"),
    [
        (118, [AMDReason.INFERENCE_TIMEOUT, AMDReason.TIMEOUT]),
        (119, [AMDReason.TIMEOUT]),
        (119.5, [AMDReason.TIMEOUT]),
    ],
)
def test_delayed_tick_orders_inference_and_hard_timeout(
    fsm: _AMDFSM, committed_at: float, reasons: list[AMDReason]
) -> None:
    request(fsm, committed_at)
    assert [event.reason for event in fsm.tick(121)] == reasons
    assert fsm.finished
    assert fsm.next_deadline is None


@pytest.mark.parametrize(
    ("committed_at", "reasons"),
    [
        (118, [AMDReason.PREDICTION, AMDReason.FINISHED]),
        (118.5, [AMDReason.TIMEOUT]),
        (119, [AMDReason.TIMEOUT]),
    ],
)
def test_delayed_tick_orders_silence_release_and_hard_timeout(
    committed_at: float, reasons: list[AMDReason]
) -> None:
    fsm = new_fsm(silence=1.5)
    fsm.enter()
    fsm.start(0)
    turn_id = request(fsm, committed_at)
    assert (
        fsm.prediction_received(turn_id, AMDCategory.MACHINE_UNAVAILABLE, committed_at + 0.1, 0.1)
        == []
    )
    assert [event.reason for event in fsm.tick(121)] == reasons
    assert fsm.finished
    assert fsm.next_deadline is None


@pytest.mark.parametrize(
    ("stage", "category", "allowed"),
    [
        (AMDCategory.MACHINE_SCREENING, AMDCategory.MACHINE_VM, True),
        (AMDCategory.MACHINE_VM, AMDCategory.MACHINE_IVR, True),
        (AMDCategory.MACHINE_IVR, AMDCategory.MACHINE_VM, True),
        (AMDCategory.MACHINE_SCREENING, AMDCategory.HUMAN, True),
        (AMDCategory.MACHINE_IVR, AMDCategory.MACHINE_UNAVAILABLE, True),
        (AMDCategory.MACHINE_VM, AMDCategory.MACHINE_SCREENING, False),
        (AMDCategory.MACHINE_SCREENING, AMDCategory.MACHINE_IVR, False),
    ],
)
def test_stage_transitions(
    fsm: _AMDFSM, stage: AMDCategory, category: AMDCategory, allowed: bool
) -> None:
    request(fsm, 1)
    fsm.prediction_received(1, stage, 1.1, 0.1)
    request(fsm, 2)
    if not allowed:
        with pytest.raises(ValueError, match="invalid amd stage transition"):
            fsm.prediction_received(2, category, 2.1, 0.1)
        assert fsm.category == stage
        assert fsm.inference_failed(2, 2.1)[0].category == stage
        return
    events = fsm.prediction_received(2, category, 2.1, 0.1)
    event = events[0]
    assert isinstance(events[-1], AMDCompletedEvent) == (
        category in {AMDCategory.HUMAN, AMDCategory.MACHINE_UNAVAILABLE}
    )
    assert event.category == fsm.category == category
    assert event.prev_turn_category == event.prev_stage_category == stage
    assert event.state_changed
    assert fsm.finished == (category in {AMDCategory.HUMAN, AMDCategory.MACHINE_UNAVAILABLE})


@pytest.mark.parametrize("stage", [AMDCategory.UNCERTAIN, AMDCategory.MACHINE_SCREENING])
def test_uncertain_limit_applies_only_before_an_established_stage(
    fsm: _AMDFSM, stage: AMDCategory
) -> None:
    request(fsm, 0)
    fsm.prediction_received(1, stage, 0.1, 0.1)
    for now in (1, 2):
        turn_id = request(fsm, now)
        event = fsm.prediction_received(turn_id, AMDCategory.UNCERTAIN, now + 0.1, 0.1)[0]
        assert event.category == stage
        assert not event.state_changed
    assert fsm.finished == (stage == AMDCategory.UNCERTAIN)
    if fsm.finished:
        assert fsm.completion().reason == "max_uncertain_turns"


def test_timeout_resolves_reply_but_late_result_can_change_stage(fsm: _AMDFSM) -> None:
    request(fsm, 1)
    assert fsm.next_deadline == 2
    fsm.tick(2)
    fallback = fsm.decision(1)
    assert fallback.reason == "inference_timeout"
    assert fsm.authorize_reply(1) == ReplyDecision(True)
    late = fsm.prediction_received(1, AMDCategory.HUMAN, 3, 2)[0]
    assert late.reason == "late_prediction"
    assert late.prev_turn_category == AMDCategory.UNCERTAIN
    assert fsm.decision(1) == fallback
    assert fsm.finished and fsm.authorize_reply(1) == ReplyDecision(True, AMDCategory.HUMAN)
    fsm.tick(4)
    assert fsm.prediction_received(1, AMDCategory.MACHINE_VM, 4, 3) == []


def test_late_inference_failure_preserves_the_timeout_decision(fsm: _AMDFSM) -> None:
    turn_id = request(fsm, 1)
    decision = fsm.tick(2)[0]
    assert decision.reason == AMDReason.INFERENCE_TIMEOUT
    assert fsm.inference_failed(turn_id, 2.1) == []
    assert fsm.decision(turn_id) == decision
    assert fsm.authorize_reply(turn_id) == ReplyDecision(True)
    assert fsm.next_deadline == 120


@pytest.mark.parametrize("pending", ["inferring", "holding", "inference_error"])
def test_new_request_silently_supersedes_pending_and_empty_turns(pending: str) -> None:
    fsm = new_fsm(silence=1.5)
    fsm.enter()
    fsm.start(0)
    first = request(fsm, 0)
    fsm.prediction_received(first, AMDCategory.MACHINE_SCREENING, 0.1, 0.1)
    fsm.tick(1.5)

    previous = request(fsm, 2)
    empty = _Transcript("", None)
    empty_turn = fsm.turn_id + 1
    fsm.commit_turn(empty, 2.2, 0, turn_id=empty_turn)
    assert fsm.transcript_ready(empty_turn, empty, 2.2) == (None, [])
    if pending == "holding":
        assert fsm.prediction_received(previous, AMDCategory.MACHINE_VM, 2.25, 0.25) == []
    elif pending == "inference_error":
        assert fsm.inference_failed(previous, 2.25) == []

    transcript = _Transcript("new turn", "session")
    current = fsm.turn_id + 1
    fsm.commit_turn(transcript, 2.3, 0, turn_id=current)
    context, events = fsm.transcript_ready(current, transcript, 2.3)
    assert context is not None and context.turn_id == current
    assert events == []
    for turn_id in (previous, empty_turn):
        assert fsm.decision(turn_id).reason == AMDReason.SUPERSEDED
        assert not fsm.authorize_reply(turn_id).allow

    assert fsm.prediction_received(current, AMDCategory.MACHINE_SCREENING, 2.4, 0.1) == []
    assert fsm.tick(3.7) == []
    assert [event.turn_id for event in fsm.tick(3.8)] == [current]
    assert fsm.authorize_reply(current) == ReplyDecision(True, AMDCategory.MACHINE_SCREENING)


def test_superseded_request_and_timer_cannot_change_newer_turn(fsm: _AMDFSM) -> None:
    request(fsm, 1)
    second = request(fsm, 1.1)
    fsm.tick(2)
    fsm.prediction_received(1, AMDCategory.HUMAN, 2, 1)
    assert fsm.inference_failed(1, 2) == []
    assert not fsm.authorize_reply(1).allow
    assert not fsm.authorize_reply(second).allow
    fsm.prediction_received(second, AMDCategory.MACHINE_SCREENING, 2, 0.9)
    assert fsm.authorize_reply(second) == ReplyDecision(True, AMDCategory.MACHINE_SCREENING)


def test_older_empty_transcript_cannot_reanchor_newer_silence() -> None:
    fsm = new_fsm(silence=1.5)
    fsm.enter()
    fsm.start(0)
    empty = _Transcript("", None)
    fsm.speech_started(0)
    fsm.speech_ended(0.1, 0)
    old_turn = fsm.turn_id + 1
    fsm.commit_turn(empty, 0.1, 0, turn_id=old_turn)
    fsm.speech_started(0.2)
    fsm.speech_ended(0.3, 0)
    current = request(fsm, 0.3)
    assert fsm.prediction_received(current, AMDCategory.MACHINE_SCREENING, 0.4, 0.1) == []
    assert fsm.next_deadline == 1.8
    assert fsm.transcript_ready(old_turn, empty, 0.6) == (None, [])
    assert fsm.decision(old_turn).reason == "superseded"
    assert fsm.next_deadline == 1.8
    event = fsm.tick(1.8)[0]
    assert event.turn_id == current
    assert event.category == AMDCategory.MACHINE_SCREENING
    assert not fsm.authorize_reply(old_turn).allow
    assert fsm.authorize_reply(current) == ReplyDecision(True, AMDCategory.MACHINE_SCREENING)


def test_new_speech_waits_for_eot_and_empty_turn_reuses_held_prediction() -> None:
    fsm = new_fsm(silence=1.5)
    fsm.enter()
    fsm.start(0)
    fsm.speech_started(0)
    fsm.speech_ended(0.5, 0)
    request(fsm, 0.5, "Please leave a message")
    assert fsm.prediction_received(1, AMDCategory.MACHINE_VM, 0.6, 0.1) == []
    fsm.speech_started(1)
    fsm.speech_ended(1.5, 0)
    assert fsm.tick(2) == []
    assert not fsm.authorize_reply(1).allow
    empty = _Transcript("", None)
    second = fsm.turn_id + 1
    fsm.commit_turn(empty, 2, 0, turn_id=second)
    assert fsm.transcript_ready(second, empty, 2) == (None, [])
    assert fsm.next_deadline == 3
    first, reused = fsm.tick(3)
    assert first.turn_id == 1 and first.transcript == "Please leave a message"
    assert first.inference_duration == 0.1
    assert first.delay == 2.5
    assert reused.turn_id == second and reused.reason == "reused"
    assert fsm.decision(1) == first and fsm.decision(second) == reused
    assert not fsm.authorize_reply(1).allow
    assert fsm.authorize_reply(second) == ReplyDecision(True, AMDCategory.MACHINE_VM)


def test_idle_restarts_after_speech_without_a_transcript(fsm: _AMDFSM) -> None:
    fsm.update_idle(0, session_busy=False)
    assert fsm.next_deadline == 10
    fsm.speech_started(9)
    fsm.update_idle(9, session_busy=False)
    fsm.tick(10)
    assert not fsm.finished
    fsm.speech_ended(11, 0)
    fsm.update_idle(11, session_busy=False)
    assert fsm.next_deadline == 21
    fsm.tick(21)
    assert fsm.completion().reason == "idle_timeout"


def test_voicemail_reply_reservation_is_separate_from_playback(fsm: _AMDFSM) -> None:
    request(fsm, 1)
    fsm.prediction_received(1, AMDCategory.MACHINE_VM, 1.1, 0.1)
    assert fsm.authorize_reply(1) == ReplyDecision(True, AMDCategory.MACHINE_VM)
    assert not fsm.voicemail_message_played
    assert not fsm.authorize_reply(1).allow
    fsm.voicemail_played()
    request(fsm, 2)
    fsm.prediction_received(2, AMDCategory.UNCERTAIN, 2.1, 0.1)
    assert not fsm.authorize_reply(2).allow
    request(fsm, 3)
    fsm.prediction_received(3, AMDCategory.MACHINE_IVR, 3.1, 0.1)
    assert fsm.authorize_reply(3) == ReplyDecision(True, AMDCategory.MACHINE_IVR)
    request(fsm, 4)
    fsm.prediction_received(4, AMDCategory.MACHINE_VM, 4.1, 0.1)
    assert fsm.authorize_reply(4) == ReplyDecision(True, AMDCategory.MACHINE_VM)
    fsm.finish(AMDReason.CANCELLED)
    assert fsm.completion().voicemail_message_played


def test_public_event_mutation_cannot_change_fsm(fsm: _AMDFSM) -> None:
    request(fsm, 1)
    event = fsm.prediction_received(1, AMDCategory.MACHINE_VM, 1.1, 0.1)[0]
    event.category = AMDCategory.HUMAN
    event.transcript = "listener mutation"
    event.turn_id = 99
    request(fsm, 2)
    second = fsm.prediction_received(2, AMDCategory.MACHINE_IVR, 2.1, 0.1)[0]
    assert second.prev_turn_category == second.prev_stage_category == AMDCategory.MACHINE_VM
    fsm.finish(AMDReason.CANCELLED)
    result = fsm.completion()
    assert result.turn_id == 2 and result.category == AMDCategory.MACHINE_IVR
    assert result.transcript == "hello"


def test_history_window_keeps_older_decisions_available(fsm: _AMDFSM) -> None:
    for index in range(1, 24):
        turn_id = request(fsm, index, f"turn {index}")
        fsm.prediction_received(turn_id, AMDCategory.MACHINE_SCREENING, index + 0.1, 0.1)
    assert fsm.decision(1).transcript == "turn 1"
    transcript = _Transcript("current", "session")
    current = fsm.turn_id + 1
    fsm.commit_turn(transcript, 24, 0, turn_id=current)
    context, _ = fsm.transcript_ready(current, transcript, 24)
    assert [turn.turn_id for turn in context.earlier_turns] == list(range(5, 24))


def test_history_uses_session_turn_ids(fsm: _AMDFSM) -> None:
    transcript = _Transcript("hello", "session")
    fsm.commit_turn(transcript, 1, 0, turn_id=40)
    fsm.transcript_ready(40, transcript, 1)
    fsm.prediction_received(40, AMDCategory.MACHINE_SCREENING, 1.1, 0.1)
    fsm.commit_turn(transcript, 2, 0, turn_id=41)
    context, _ = fsm.transcript_ready(41, transcript, 2)
    assert context.turn_id == fsm.turn_id == 41
    assert [turn.turn_id for turn in context.earlier_turns] == [40]
    assert fsm.decision(40).transcript == "hello"


def test_finish_settles_pending_turns_and_ignores_late_work(fsm: _AMDFSM) -> None:
    request(fsm, 1)
    fsm.dtmf_sent("12#")
    fsm.finish(AMDReason.PARTICIPANT_DISCONNECTED)
    assert fsm.decision(1).reason == "participant_disconnected"
    assert fsm.next_deadline is None
    fsm.tick(2)
    assert fsm.prediction_received(1, AMDCategory.HUMAN, 2, 1) == []
    fsm.finish(AMDReason.CANCELLED)
    assert fsm.completion().reason == "participant_disconnected"
