from __future__ import annotations

import pytest

from livekit.agents.voice.amd._fsm import (
    AMDFSM,
    AMDClassifyRequest,
    AMDLifecycle,
    AMDMenuRequest,
    AMDReplyDecision,
    AMDTranscript,
)
from livekit.agents.voice.amd.events import (
    AMDCategory,
    AMDCompletedEvent,
    AMDPredictionEvent,
    AMDReason,
)

pytestmark = pytest.mark.unit


def new_fsm(*, silence: float = 0, max_inference_timeouts: int = 3) -> AMDFSM:
    return AMDFSM(
        idle_timeout=10,
        voicemail_idle_timeout=60,
        timeout=120,
        inference_timeout=1,
        machine_silence_threshold=silence,
        max_uncertain_turns=3,
        max_inference_timeouts=max_inference_timeouts,
    )


@pytest.fixture
def fsm() -> AMDFSM:
    fsm = new_fsm()
    fsm.enter()
    fsm.start(0)
    return fsm


def request(fsm: AMDFSM, now: float, transcript: str = "hello") -> int:
    turn_transcript = AMDTranscript(transcript, "session")
    (effect,) = fsm.commit_turn(turn_transcript, now, eot_delay=0)
    assert isinstance(effect, AMDClassifyRequest)
    assert effect.current_turn.turn_id == fsm.turn_id
    return fsm.turn_id


def test_lifecycle_and_fixed_hard_deadline() -> None:
    fsm = new_fsm()
    assert fsm.lifecycle is AMDLifecycle.INITIALIZED
    fsm.enter()
    assert fsm.lifecycle is AMDLifecycle.PENDING
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
    fsm.deadline_reached(124)
    assert fsm.lifecycle is AMDLifecycle.ACTIVE
    fsm.deadline_reached(125)
    assert fsm.lifecycle is AMDLifecycle.FINISHED
    assert fsm.next_deadline is None
    assert fsm.completion().reason == "timeout"
    assert fsm.completion().category == AMDCategory.MACHINE_VM
    with pytest.raises(RuntimeError, match="new AMD instance"):
        fsm.enter()

    delayed = new_fsm()
    delayed.enter()
    delayed.start(0)
    delayed.update_idle(0, session_busy=False)
    delayed.deadline_reached(121)
    assert delayed.completion().reason == "idle_timeout"


@pytest.mark.parametrize(
    ("committed_at", "reasons"),
    [
        (118, [AMDReason.INFERENCE_TIMEOUT, AMDReason.TIMEOUT]),
        (119, [AMDReason.TIMEOUT]),
        (119.5, [AMDReason.TIMEOUT]),
    ],
)
def test_delayed_deadline_orders_inference_and_hard_timeout(
    fsm: AMDFSM, committed_at: float, reasons: list[AMDReason]
) -> None:
    request(fsm, committed_at)
    assert [event.reason for event in fsm.deadline_reached(121)] == reasons
    assert fsm.lifecycle is AMDLifecycle.FINISHED
    assert fsm.next_deadline is None


@pytest.mark.parametrize(
    ("committed_at", "reasons"),
    [
        (118, [AMDReason.PREDICTION, AMDReason.FINISHED]),
        (118.5, [AMDReason.TIMEOUT]),
        (119, [AMDReason.TIMEOUT]),
    ],
)
def test_delayed_deadline_orders_silence_release_and_hard_timeout(
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
    assert [event.reason for event in fsm.deadline_reached(121)] == reasons
    assert fsm.lifecycle is AMDLifecycle.FINISHED
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
    fsm: AMDFSM, stage: AMDCategory, category: AMDCategory, allowed: bool
) -> None:
    request(fsm, 1)
    fsm.prediction_received(1, stage, 1.1, 0.1)
    request(fsm, 2)
    if not allowed:
        events = fsm.prediction_received(2, category, 2.1, 0.1)
        assert len(events) == 1
        assert events[0].category == stage
        assert events[0].reason == AMDReason.INFERENCE_ERROR
        assert fsm.prediction(2).category == stage
        assert fsm.prediction(2).reason == AMDReason.INFERENCE_ERROR
        assert fsm.category == stage
        assert fsm.prediction_received(2, AMDCategory.HUMAN, 2.2, 0.2) == []
        return
    events = fsm.prediction_received(2, category, 2.1, 0.1)
    if category is AMDCategory.MACHINE_IVR:
        menu, event = events
        assert menu == AMDMenuRequest(2, "hello")
    else:
        event = events[0]
    assert isinstance(event, AMDPredictionEvent)
    assert isinstance(events[-1], AMDCompletedEvent) == (
        category in {AMDCategory.HUMAN, AMDCategory.MACHINE_UNAVAILABLE}
    )
    assert event.category == fsm.category == category
    assert event.prev_turn_category == event.prev_stage_category == stage
    assert event.state_changed
    assert (fsm.lifecycle is AMDLifecycle.FINISHED) == (
        category in {AMDCategory.HUMAN, AMDCategory.MACHINE_UNAVAILABLE}
    )
    if fsm.lifecycle is AMDLifecycle.FINISHED:
        completion = fsm.completion()
        assert completion.turn_id == event.turn_id
        assert completion.transcript == event.transcript
        assert completion.prev_turn_category == event.prev_turn_category
        assert completion.prev_stage_category == event.prev_stage_category


@pytest.mark.parametrize("outcome", ["prediction", "failure"])
def test_repeated_stage_preserves_previous_stage_metadata(fsm: AMDFSM, outcome: str) -> None:
    first = request(fsm, 0)
    fsm.prediction_received(first, AMDCategory.MACHINE_SCREENING, 0.1, 0.1)
    second = request(fsm, 1)
    fsm.prediction_received(second, AMDCategory.MACHINE_VM, 1.1, 0.1)
    current = request(fsm, 2)
    if outcome == "prediction":
        (event,) = fsm.prediction_received(current, AMDCategory.MACHINE_VM, 2.1, 0.1)
    else:
        (event,) = fsm.inference_failed(current, 2.1)
    assert isinstance(event, AMDPredictionEvent)
    assert event.prev_turn_category is AMDCategory.MACHINE_VM
    assert event.prev_stage_category is AMDCategory.MACHINE_SCREENING
    assert not event.state_changed
    completion = fsm.finish(AMDReason.CANCELLED)
    assert completion.turn_id == current
    assert completion.transcript == event.transcript
    assert completion.prev_stage_category is AMDCategory.MACHINE_SCREENING


@pytest.mark.parametrize("late", [False, True])
def test_ivr_menu_request_precedes_prediction_at_release(late: bool) -> None:
    fsm = new_fsm(silence=1.5)
    fsm.enter()
    fsm.start(0)
    transcript = "For sales, press one."
    turn_id = request(fsm, 0, transcript)
    if late:
        (fallback,) = fsm.deadline_reached(1)
        assert isinstance(fallback, AMDPredictionEvent)
        assert fallback.reason is AMDReason.INFERENCE_TIMEOUT
    received_at = 1.1 if late else 0.1
    assert fsm.prediction_received(turn_id, AMDCategory.MACHINE_IVR, received_at, received_at) == []
    assert fsm.deadline_reached(1.4) == []

    menu, prediction = fsm.deadline_reached(1.5)
    assert isinstance(prediction, AMDPredictionEvent)
    assert prediction.reason is (AMDReason.LATE_PREDICTION if late else AMDReason.PREDICTION)
    assert prediction.category is AMDCategory.MACHINE_IVR
    assert menu == AMDMenuRequest(turn_id, transcript)


@pytest.mark.parametrize("outcome", ["failure", "invalid_prediction", "timeout", "reused"])
def test_ivr_fallbacks_do_not_request_a_menu(fsm: AMDFSM, outcome: str) -> None:
    first = request(fsm, 0)
    fsm.prediction_received(first, AMDCategory.MACHINE_IVR, 0.1, 0.1)
    if outcome == "reused":
        assert fsm.commit_turn(AMDTranscript("", None), 1, 0) == []
        assert fsm.prediction(fsm.turn_id).reason is AMDReason.REUSED
        return

    turn_id = request(fsm, 1)
    if outcome == "failure":
        effects = fsm.inference_failed(turn_id, 1.1)
    elif outcome == "invalid_prediction":
        effects = fsm.prediction_received(turn_id, AMDCategory.MACHINE_SCREENING, 1.1, 0.1)
    else:
        effects = fsm.deadline_reached(2)
    (prediction,) = effects
    assert isinstance(prediction, AMDPredictionEvent)
    assert prediction.category is AMDCategory.MACHINE_IVR
    assert prediction.reason in {AMDReason.INFERENCE_ERROR, AMDReason.INFERENCE_TIMEOUT}


@pytest.mark.parametrize("stage", [AMDCategory.UNCERTAIN, AMDCategory.MACHINE_SCREENING])
def test_uncertain_limit_applies_only_before_an_established_stage(
    fsm: AMDFSM, stage: AMDCategory
) -> None:
    request(fsm, 0)
    fsm.prediction_received(1, stage, 0.1, 0.1)
    for now in (1, 2):
        turn_id = request(fsm, now)
        event = fsm.prediction_received(turn_id, AMDCategory.UNCERTAIN, now + 0.1, 0.1)[0]
        assert event.category == stage
        assert not event.state_changed
    assert (fsm.lifecycle is AMDLifecycle.FINISHED) == (stage == AMDCategory.UNCERTAIN)
    if fsm.lifecycle is AMDLifecycle.FINISHED:
        assert fsm.completion().reason == "max_uncertain_turns"


def test_prediction_event_mutation_does_not_change_saved_prediction(fsm: AMDFSM) -> None:
    request(fsm, 1)
    event = fsm.prediction_received(1, AMDCategory.MACHINE_SCREENING, 1.1, 0.1)[0]
    event.category = AMDCategory.HUMAN
    event.transcript = "edited"
    event.turn_id = 99

    prediction = fsm.prediction(1)
    assert prediction is not None
    completed = fsm.finish(AMDReason.TIMEOUT)
    assert prediction.category == completed.category == AMDCategory.MACHINE_SCREENING
    assert prediction.transcript == completed.transcript == "hello"
    assert prediction.turn_id == completed.turn_id == 1


def test_timeout_resolves_reply_but_late_result_can_change_stage(fsm: AMDFSM) -> None:
    request(fsm, 1)
    assert fsm.next_deadline == 2
    fsm.deadline_reached(2)
    fallback = fsm.prediction(1)
    assert fallback.reason == "inference_timeout"
    assert fsm.authorize_reply(1) == AMDReplyDecision(True)
    late = fsm.prediction_received(1, AMDCategory.HUMAN, 3, 2)[0]
    assert late.reason == "late_prediction"
    assert late.prev_turn_category == AMDCategory.UNCERTAIN
    assert fsm.prediction(1) == fallback
    assert fsm.lifecycle is AMDLifecycle.FINISHED
    assert fsm.authorize_reply(1) == AMDReplyDecision(True)
    fsm.deadline_reached(4)
    assert fsm.prediction_received(1, AMDCategory.MACHINE_VM, 4, 3) == []


@pytest.mark.parametrize("late", [False, True])
def test_valid_prediction_resets_the_inference_timeout_count(late: bool) -> None:
    fsm = new_fsm(max_inference_timeouts=2)
    fsm.enter()
    fsm.start(0)
    turn_id = request(fsm, 0)
    fsm.deadline_reached(1)
    assert fsm.lifecycle is not AMDLifecycle.FINISHED
    if not late:
        turn_id = request(fsm, 1.1)
    fsm.prediction_received(turn_id, AMDCategory.MACHINE_SCREENING, 1.2, 0.1)

    request(fsm, 2)
    fsm.deadline_reached(3)
    assert fsm.lifecycle is not AMDLifecycle.FINISHED
    request(fsm, 4)
    fsm.deadline_reached(5)
    assert fsm.lifecycle is AMDLifecycle.FINISHED
    assert fsm.completion().reason == AMDReason.INFERENCE_TIMEOUT


def test_late_inference_failure_preserves_the_timeout_decision(fsm: AMDFSM) -> None:
    turn_id = request(fsm, 1)
    fsm.deadline_reached(2)
    prediction = fsm.prediction(turn_id)
    assert prediction.reason == AMDReason.INFERENCE_TIMEOUT
    assert fsm.inference_failed(turn_id, 2.1) == []
    assert fsm.prediction(turn_id) == prediction
    assert fsm.authorize_reply(turn_id) == AMDReplyDecision(True)
    assert fsm.next_deadline == 120


def test_late_hold_preserves_a_reused_turns_released_fallback() -> None:
    fsm = new_fsm(silence=1.5)
    fsm.enter()
    fsm.start(0)
    previous = request(fsm, 0)
    assert fsm.commit_turn(AMDTranscript("", None), 0.1, 0) == []
    current = fsm.turn_id
    fsm.deadline_reached(1)
    fallback = fsm.prediction(previous)
    assert fallback is not None
    assert fallback.reason is AMDReason.INFERENCE_TIMEOUT
    assert fsm.prediction(current) is fallback
    assert fsm.authorize_reply(current) == AMDReplyDecision(True)

    assert fsm.prediction_received(previous, AMDCategory.MACHINE_VM, 1.1, 1.1) == []
    assert fsm.next_deadline == 1.6
    assert fsm.prediction(previous) is fallback
    assert fsm.prediction(current) is fallback
    assert fsm.authorize_reply(current) == AMDReplyDecision(True)

    (prediction,) = fsm.deadline_reached(1.6)
    assert isinstance(prediction, AMDPredictionEvent)
    assert prediction.reason is AMDReason.LATE_PREDICTION
    assert prediction.category is AMDCategory.MACHINE_VM
    assert fsm.prediction(current) is fallback
    assert fsm.authorize_reply(current) == AMDReplyDecision(
        True, AMDCategory.MACHINE_VM, track_voicemail=True
    )


@pytest.mark.parametrize("limit", [1, 3])
@pytest.mark.parametrize("outcome", ["failure", "invalid_prediction"])
def test_late_inference_failure_preserves_held_timeout_completion(limit: int, outcome: str) -> None:
    fsm = new_fsm(silence=1.5, max_inference_timeouts=limit)
    fsm.enter()
    fsm.start(0)
    first = request(fsm, 0)
    fsm.prediction_received(first, AMDCategory.MACHINE_SCREENING, 0.1, 0.1)
    fsm.deadline_reached(1.5)

    for index in range(limit):
        now = 2 + index * 2
        turn_id = request(fsm, now)
        assert fsm.deadline_reached(now + 1) == []
        assert fsm.prediction(turn_id) is None
        if outcome == "failure":
            assert fsm.inference_failed(turn_id, now + 1.1) == []
        else:
            assert fsm.prediction_received(turn_id, AMDCategory.MACHINE_IVR, now + 1.1, 1.1) == []
        assert fsm.prediction(turn_id) is None
        assert fsm.lifecycle is AMDLifecycle.ACTIVE
        assert fsm.next_deadline == now + 1.5

        events = fsm.deadline_reached(now + 1.5)
        prediction = fsm.prediction(turn_id)
        assert prediction is not None
        assert prediction.reason is AMDReason.INFERENCE_TIMEOUT
        assert prediction.category is AMDCategory.MACHINE_SCREENING
        assert len(events) == (2 if index == limit - 1 else 1)
        assert (fsm.lifecycle is AMDLifecycle.FINISHED) == (index == limit - 1)

    assert fsm.completion().reason is AMDReason.INFERENCE_TIMEOUT
    assert fsm.next_deadline is None


@pytest.mark.parametrize("outcome", ["failure", "invalid_prediction", "human"])
def test_late_inference_does_not_orphan_an_empty_turn(outcome: str) -> None:
    fsm = new_fsm(silence=1.5)
    fsm.enter()
    fsm.start(0)
    first = request(fsm, 0)
    fsm.prediction_received(first, AMDCategory.MACHINE_SCREENING, 0.1, 0.1)
    fsm.deadline_reached(1.5)

    previous = request(fsm, 2)
    fsm.deadline_reached(3.5)
    fallback = fsm.prediction(previous)
    assert fallback is not None
    assert fallback.reason is AMDReason.INFERENCE_TIMEOUT

    assert fsm.commit_turn(AMDTranscript("", None), 4, 0) == []
    current = fsm.turn_id
    assert fsm.prediction(current) is None
    assert not fsm.authorize_reply(current).allow

    if outcome == "human":
        events = fsm.prediction_received(previous, AMDCategory.HUMAN, 4.1, 2.1)
        assert [event.reason for event in events] == [AMDReason.LATE_PREDICTION, AMDReason.FINISHED]
        assert fsm.prediction(previous) is fallback
        assert fsm.completion().category is AMDCategory.HUMAN
        assert fsm.next_deadline is None
        assert fsm.authorize_reply(current) == AMDReplyDecision(True, AMDCategory.HUMAN)
        return

    if outcome == "failure":
        assert fsm.inference_failed(previous, 4.1) == []
    else:
        assert fsm.prediction_received(previous, AMDCategory.MACHINE_IVR, 4.1, 2.1) == []

    assert fsm.prediction(previous) is fallback
    assert fsm.prediction(current) is None
    assert not fsm.authorize_reply(current).allow
    fsm.update_idle(4.1, session_busy=False)
    assert fsm.next_deadline == 5.5
    assert fsm.deadline_reached(5.5) == []
    prediction = fsm.prediction(current)
    assert prediction is not None
    assert prediction.reason is AMDReason.REUSED
    assert prediction.category is AMDCategory.MACHINE_SCREENING
    assert fsm.authorize_reply(current) == AMDReplyDecision(True, AMDCategory.MACHINE_SCREENING)
    fsm.update_idle(5.5, session_busy=False)
    assert fsm.next_deadline == 15.5
    fsm.deadline_reached(15.5)
    assert fsm.completion().reason is AMDReason.IDLE_TIMEOUT


@pytest.mark.parametrize(
    "category",
    [AMDCategory.MACHINE_VM, AMDCategory.MACHINE_IVR, AMDCategory.MACHINE_UNAVAILABLE],
)
@pytest.mark.parametrize("timing", ["waiting", "speaking", "overdue"])
def test_late_machine_prediction_replaces_a_reused_hold(category: AMDCategory, timing: str) -> None:
    fsm = new_fsm(silence=1.5)
    fsm.enter()
    fsm.start(0)
    first = request(fsm, 0)
    fsm.prediction_received(first, AMDCategory.MACHINE_VM, 0.1, 0.1)
    fsm.deadline_reached(1.5)

    transcript = "The number you have dialed is unavailable."
    previous = request(fsm, 2, transcript)
    fsm.deadline_reached(3.5)
    fallback = fsm.prediction(previous)
    assert fallback is not None
    assert fallback.reason is AMDReason.INFERENCE_TIMEOUT

    assert fsm.commit_turn(AMDTranscript("", None), 4, 0) == []
    empty = fsm.turn_id
    assert fsm.commit_turn(AMDTranscript("", None), 4.1, 0) == []
    current = fsm.turn_id
    release_at = 5.6
    assert fsm.next_deadline == release_at
    if timing == "speaking":
        fsm.speech_started(4.2)
        assert fsm.next_deadline == 120

    received_at = 5.7 if timing == "overdue" else 4.3
    effects = fsm.prediction_received(previous, category, received_at, received_at - 2)
    assert fsm.prediction(previous) is fallback
    if timing != "overdue":
        assert effects == []
        for turn_id in (empty, current):
            assert fsm.prediction(turn_id) is None
            assert not fsm.authorize_reply(turn_id).allow
        assert fsm.next_deadline == (120 if timing == "speaking" else release_at)
        if timing == "speaking":
            assert fsm.deadline_reached(5.6) == []
            assert fsm.speech_ended(6, 0) == []
            release_at = 7.5
        fsm.update_idle(release_at - 0.1, session_busy=False)
        assert fsm.next_deadline == release_at
        assert fsm.deadline_reached(release_at - 0.1) == []
        effects = fsm.deadline_reached(release_at)

    if category is AMDCategory.MACHINE_IVR:
        assert effects.pop(0) == AMDMenuRequest(previous, transcript)
    prediction = effects[0]
    assert isinstance(prediction, AMDPredictionEvent)
    assert prediction.turn_id == previous
    assert prediction.transcript == transcript
    assert prediction.category is category
    assert prediction.reason is AMDReason.LATE_PREDICTION
    assert prediction.inference_duration == received_at - 2
    assert prediction.delay == max(received_at, release_at) - 2
    assert fsm.prediction(previous) is fallback
    assert fallback.category is AMDCategory.MACHINE_VM
    assert fallback.reason is AMDReason.INFERENCE_TIMEOUT
    assert fsm.prediction(empty) is fallback
    assert fsm.prediction(current) is fallback
    assert not fsm.authorize_reply(empty).allow
    assert fsm.category is category
    if category is AMDCategory.MACHINE_UNAVAILABLE:
        assert [event.reason for event in effects] == [
            AMDReason.LATE_PREDICTION,
            AMDReason.FINISHED,
        ]
        assert fsm.completion().category is category
        assert not fsm.authorize_reply(current).allow
        assert fsm.next_deadline is None
    else:
        assert len(effects) == 1
        assert fsm.authorize_reply(current) == AMDReplyDecision(
            True, category, track_voicemail=category is AMDCategory.MACHINE_VM
        )
        fsm.update_idle(max(received_at, release_at), session_busy=False)
        assert fsm.next_deadline == max(received_at, release_at) + (
            60 if category is AMDCategory.MACHINE_VM else 10
        )


@pytest.mark.parametrize("pending", ["inferring", "holding", "inference_error"])
def test_new_request_silently_supersedes_pending_and_empty_turns(pending: str) -> None:
    fsm = new_fsm(silence=1.5)
    fsm.enter()
    fsm.start(0)
    first = request(fsm, 0)
    fsm.prediction_received(first, AMDCategory.MACHINE_SCREENING, 0.1, 0.1)
    fsm.deadline_reached(1.5)

    previous = request(fsm, 2)
    empty = AMDTranscript("", None)
    assert fsm.commit_turn(empty, 2.2, 0) == []
    empty_turn = fsm.turn_id
    if pending == "holding":
        assert fsm.prediction_received(previous, AMDCategory.MACHINE_VM, 2.25, 0.25) == []
    elif pending == "inference_error":
        assert fsm.inference_failed(previous, 2.25) == []

    transcript = AMDTranscript("new turn", "session")
    (context,) = fsm.commit_turn(transcript, 2.3, 0)
    current = fsm.turn_id
    assert isinstance(context, AMDClassifyRequest)
    assert context.current_turn.turn_id == current
    for turn_id in (previous, empty_turn):
        assert fsm.prediction(turn_id) is None
        assert not fsm.authorize_reply(turn_id).allow

    assert fsm.prediction_received(current, AMDCategory.MACHINE_SCREENING, 2.4, 0.1) == []
    assert fsm.deadline_reached(3.7) == []
    assert [event.turn_id for event in fsm.deadline_reached(3.8)] == [current]
    assert fsm.authorize_reply(current) == AMDReplyDecision(True, AMDCategory.MACHINE_SCREENING)


def test_superseded_request_and_timer_cannot_change_newer_turn(fsm: AMDFSM) -> None:
    request(fsm, 1)
    second = request(fsm, 1.1)
    fsm.deadline_reached(2)
    fsm.prediction_received(1, AMDCategory.HUMAN, 2, 1)
    assert fsm.inference_failed(1, 2) == []
    assert not fsm.authorize_reply(1).allow
    assert not fsm.authorize_reply(second).allow
    fsm.prediction_received(second, AMDCategory.MACHINE_SCREENING, 2, 0.9)
    assert fsm.authorize_reply(second) == AMDReplyDecision(True, AMDCategory.MACHINE_SCREENING)
    prediction = fsm.prediction(second)
    assert fsm.prediction_received(1, AMDCategory.HUMAN, 2.1, 1.1) == []
    assert fsm.inference_failed(1, 2.1) == []
    assert fsm.prediction(second) is prediction
    assert fsm.category is AMDCategory.MACHINE_SCREENING


def test_empty_turn_settles_at_commit(fsm: AMDFSM) -> None:
    empty = AMDTranscript("", None)
    assert fsm.commit_turn(empty, 1, 0) == []
    assert fsm.prediction(1).transcript == ""
    assert fsm.authorize_reply(1) == AMDReplyDecision(True)
    fsm.update_idle(1, session_busy=False)
    assert fsm.next_deadline == 11


def test_empty_turn_does_not_reuse_a_timed_out_inference(fsm: AMDFSM) -> None:
    previous = request(fsm, 0)
    fsm.deadline_reached(1)
    fallback = fsm.prediction(previous)
    assert fallback is not None
    assert fallback.reason is AMDReason.INFERENCE_TIMEOUT

    assert fsm.commit_turn(AMDTranscript("", None), 1.1, 0) == []
    current = fsm.turn_id
    prediction = fsm.prediction(current)
    assert prediction is not None
    assert prediction.reason is AMDReason.REUSED
    assert prediction.transcript == ""
    assert prediction is not fallback
    assert not fsm.authorize_reply(previous).allow
    assert fsm.authorize_reply(current) == AMDReplyDecision(True)

    (late,) = fsm.prediction_received(previous, AMDCategory.MACHINE_SCREENING, 1.2, 1.2)
    assert isinstance(late, AMDPredictionEvent)
    assert late.reason is AMDReason.LATE_PREDICTION
    assert fsm.prediction(current) is prediction
    assert fsm.authorize_reply(current) == AMDReplyDecision(True, AMDCategory.MACHINE_SCREENING)


@pytest.mark.parametrize("eot_delay", [0, 0.25, -0.5])
def test_empty_eot_without_new_speech_edges_reanchors_a_held_prediction(eot_delay: float) -> None:
    fsm = new_fsm(silence=1.5)
    fsm.enter()
    fsm.start(0)
    fsm.speech_started(0)
    fsm.speech_ended(0.5, 0)
    previous = request(fsm, 0.5)
    assert fsm.prediction_received(previous, AMDCategory.MACHINE_SCREENING, 0.6, 0.1) == []
    assert fsm.next_deadline == 2

    assert fsm.commit_turn(AMDTranscript("", None), 1, eot_delay) == []
    current = fsm.turn_id
    release_at = 2.5 - max(0, eot_delay)
    assert fsm.next_deadline == release_at
    assert fsm.deadline_reached(2) == []
    assert not fsm.authorize_reply(current).allow
    (prediction,) = fsm.deadline_reached(release_at)
    assert isinstance(prediction, AMDPredictionEvent)
    assert prediction.turn_id == previous
    assert prediction.speech_duration == 0.5
    assert fsm.prediction(current) is fsm.prediction(previous)
    assert fsm.authorize_reply(current) == AMDReplyDecision(True, AMDCategory.MACHINE_SCREENING)


@pytest.mark.parametrize("prediction_ready", [False, True])
def test_uncommitted_speech_rearms_a_held_prediction(prediction_ready: bool) -> None:
    fsm = new_fsm(silence=1.5)
    fsm.enter()
    fsm.start(0)
    turn_id = request(fsm, 0)
    if prediction_ready:
        assert fsm.prediction_received(turn_id, AMDCategory.MACHINE_SCREENING, 0.1, 0.1) == []

    fsm.speech_started(0.2)
    assert fsm.speech_ended(0.5, 0) == []
    if not prediction_ready:
        assert fsm.prediction_received(turn_id, AMDCategory.MACHINE_SCREENING, 0.6, 0.6) == []

    assert fsm.next_deadline == 2
    assert fsm.deadline_reached(1.5) == []
    assert not fsm.authorize_reply(turn_id).allow
    (prediction,) = fsm.deadline_reached(2)
    assert prediction.turn_id == fsm.turn_id == turn_id
    assert prediction.category is AMDCategory.MACHINE_SCREENING
    assert fsm.authorize_reply(turn_id) == AMDReplyDecision(True, AMDCategory.MACHINE_SCREENING)
    fsm.update_idle(2, session_busy=False)
    assert fsm.next_deadline == 12


def test_new_speech_rearms_hold_and_empty_turn_reuses_prediction() -> None:
    fsm = new_fsm(silence=1.5)
    fsm.enter()
    fsm.start(0)
    fsm.speech_started(0)
    fsm.speech_ended(0.5, 0)
    request(fsm, 0.5, "Please leave a message")
    assert fsm.prediction_received(1, AMDCategory.MACHINE_VM, 0.6, 0.1) == []
    fsm.speech_started(1)
    fsm.speech_ended(1.5, 0)
    assert fsm.deadline_reached(2) == []
    assert not fsm.authorize_reply(1).allow
    empty = AMDTranscript("", None)
    assert fsm.commit_turn(empty, 2, 0) == []
    second = fsm.turn_id
    assert fsm.next_deadline == 3
    (first,) = fsm.deadline_reached(3)
    assert first.turn_id == 1 and first.transcript == "Please leave a message"
    assert first.inference_duration == 0.1
    assert first.delay == 2.5
    assert fsm.prediction(1).category == AMDCategory.MACHINE_VM
    assert fsm.prediction(second) is fsm.prediction(1)
    assert not fsm.authorize_reply(1).allow
    assert fsm.authorize_reply(second) == AMDReplyDecision(
        True, AMDCategory.MACHINE_VM, track_voicemail=True
    )


def test_idle_restarts_after_speech_without_a_transcript(fsm: AMDFSM) -> None:
    fsm.update_idle(0, session_busy=False)
    assert fsm.next_deadline == 10
    fsm.speech_started(9)
    fsm.update_idle(9, session_busy=False)
    fsm.deadline_reached(10)
    assert fsm.lifecycle is not AMDLifecycle.FINISHED
    fsm.speech_ended(11, 0)
    fsm.update_idle(11, session_busy=False)
    assert fsm.next_deadline == 21
    fsm.deadline_reached(21)
    assert fsm.completion().reason == "idle_timeout"


def test_voicemail_reply_reservation_is_separate_from_playback(fsm: AMDFSM) -> None:
    request(fsm, 1)
    fsm.prediction_received(1, AMDCategory.MACHINE_VM, 1.1, 0.1)
    assert fsm.authorize_reply(1) == AMDReplyDecision(
        True, AMDCategory.MACHINE_VM, track_voicemail=True
    )
    assert not fsm.voicemail_message_played
    assert fsm.authorize_reply(1) == AMDReplyDecision(False)
    assert fsm.commit_voicemail_reply(1)
    fsm.voicemail_played()
    request(fsm, 2)
    fsm.prediction_received(2, AMDCategory.UNCERTAIN, 2.1, 0.1)
    assert not fsm.authorize_reply(2).allow
    request(fsm, 3)
    fsm.prediction_received(3, AMDCategory.MACHINE_IVR, 3.1, 0.1)
    assert fsm.authorize_reply(3) == AMDReplyDecision(True, AMDCategory.MACHINE_IVR)
    request(fsm, 4)
    fsm.prediction_received(4, AMDCategory.MACHINE_VM, 4.1, 0.1)
    assert fsm.authorize_reply(4) == AMDReplyDecision(
        True, AMDCategory.MACHINE_VM, track_voicemail=True
    )
    fsm.finish(AMDReason.CANCELLED)
    assert fsm.completion().voicemail_message_played
    assert fsm.authorize_reply(4) == AMDReplyDecision(True)


@pytest.mark.parametrize("empty", [False, True])
def test_new_turn_releases_an_uncommitted_voicemail_reservation(fsm: AMDFSM, empty: bool) -> None:
    previous = request(fsm, 1)
    fsm.prediction_received(previous, AMDCategory.MACHINE_VM, 1.1, 0.1)
    assert fsm.authorize_reply(previous).track_voicemail
    assert not fsm.authorize_reply(previous).allow

    if empty:
        fsm.commit_turn(AMDTranscript("", None), 2, 0)
    else:
        current = request(fsm, 2)
        fsm.prediction_received(current, AMDCategory.MACHINE_VM, 2.1, 0.1)
    current = fsm.turn_id
    assert fsm.authorize_reply(current) == AMDReplyDecision(
        True, AMDCategory.MACHINE_VM, track_voicemail=True
    )
    assert not fsm.commit_voicemail_reply(previous)
    assert not fsm.authorize_reply(current).allow
    assert fsm.commit_voicemail_reply(current)
    assert not fsm.commit_voicemail_reply(current)

    next_turn = request(fsm, 3)
    fsm.prediction_received(next_turn, AMDCategory.MACHINE_VM, 3.1, 0.1)
    assert not fsm.authorize_reply(next_turn).allow


def test_finished_run_cannot_commit_a_pending_voicemail_reply(fsm: AMDFSM) -> None:
    turn_id = request(fsm, 1)
    fsm.prediction_received(turn_id, AMDCategory.MACHINE_VM, 1.1, 0.1)
    assert fsm.authorize_reply(turn_id).track_voicemail
    fsm.finish(AMDReason.CANCELLED)
    assert not fsm.commit_voicemail_reply(turn_id)


def test_history_window_keeps_older_decisions_available(fsm: AMDFSM) -> None:
    for index in range(1, 24):
        turn_id = request(fsm, index, f"turn {index}")
        fsm.prediction_received(turn_id, AMDCategory.MACHINE_SCREENING, index + 0.1, 0.1)
    assert fsm.prediction(1).transcript == "turn 1"
    transcript = AMDTranscript("current", "session")
    (context,) = fsm.commit_turn(transcript, 24, 0)
    assert isinstance(context, AMDClassifyRequest)
    assert [turn.turn_id for turn in context.earlier_turns] == list(range(5, 24))


def test_turn_ids_are_allocated_for_transcribed_and_empty_turns(fsm: AMDFSM) -> None:
    assert fsm.turn_id == 0
    transcript = AMDTranscript("hello", "session")
    fsm.commit_turn(transcript, 1, 0)
    assert fsm.turn_id == 1
    fsm.prediction_received(1, AMDCategory.MACHINE_SCREENING, 1.1, 0.1)
    assert fsm.commit_turn(AMDTranscript("", None), 2, 0) == []
    assert fsm.turn_id == 2
    (context,) = fsm.commit_turn(transcript, 3, 0)
    assert isinstance(context, AMDClassifyRequest)
    assert context.current_turn.turn_id == fsm.turn_id == 3
    assert [turn.turn_id for turn in context.earlier_turns] == [1, 2]
    assert fsm.prediction(1).transcript == "hello"
    assert fsm.prediction(2).transcript == ""


def test_finish_drops_pending_work_and_ignores_late_work(fsm: AMDFSM) -> None:
    request(fsm, 1)
    fsm.dtmf_sent("12#")
    fsm.finish(AMDReason.PARTICIPANT_DISCONNECTED)
    assert fsm.prediction(1) is None
    assert fsm.authorize_reply(1) == AMDReplyDecision(True)
    assert fsm.next_deadline is None
    fsm.deadline_reached(2)
    assert fsm.prediction_received(1, AMDCategory.HUMAN, 2, 1) == []
    fsm.finish(AMDReason.CANCELLED)
    assert fsm.completion().reason == "participant_disconnected"
