from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from livekit import rtc
from livekit.agents import AMD, Agent, AgentSession, UserStateChangedEvent, llm, stt, vad
from livekit.agents.voice.amd import AMDCategory, AMDLifecycle, _inference
from livekit.agents.voice.speech_handle import SpeechHandle

from .amd_test_utils import detector_clock, next_deadline  # noqa: F401
from .fake_io import FakeAudioOutput
from .fake_llm import FakeLLM
from .fake_stt import DrainingSTT, FakeSTT
from .fake_tts import FakeTTS
from .fake_vad import FakeVAD
from .test_amd_detector import (
    ClassifierLLM,
    CustomerAgent,
    RecordingLLM,
    commit,
    commit_turn,
    end_of_turn,
    eventually,
    push_audio,
    running,
    speech_ended,
    speech_started,
)

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent, pytest.mark.virtual_time]


def test_machine_silence_default_and_validation() -> None:
    assert AMD(AgentSession(), llm=None, stt=None)._options.machine_silence_threshold == 1.5
    with pytest.raises(ValueError, match="machine_silence_threshold"):
        AMD(AgentSession(), machine_silence_threshold=-1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "category",
    [
        AMDCategory.MACHINE_SCREENING,
        AMDCategory.MACHINE_VM,
        AMDCategory.MACHINE_IVR,
        AMDCategory.MACHINE_UNAVAILABLE,
    ],
)
async def test_machine_prediction_updates_state_before_reply_silence(category: AMDCategory) -> None:
    async with running(machine_silence_threshold=1.5, inference_timeout=0.3) as (
        detector,
        session,
        classifier,
        model,
    ):
        events = []
        detector.on("amd_prediction", events.append)
        speech_started(detector)
        speech_ended(detector, 0.5)
        await commit(detector, session, classifier, reply=True)
        await asyncio.sleep(0.2)
        classifier.prediction(1, category)
        await asyncio.sleep(0.7)
        assert len(events) == 1
        assert events[0].category == category
        assert events[0].reason == "prediction"
        assert events[0].delay == pytest.approx(0.2, abs=0.01)
        assert events[0].inference_duration == pytest.approx(0.2, abs=0.01)
        assert detector._state == category
        assert detector._turns[1].prediction is not None
        assert model.calls.empty()
        if category == AMDCategory.MACHINE_UNAVAILABLE:
            assert (await detector.execute()).category == category
            return
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        assert detector._reply_held_at(asyncio.get_running_loop().time())
        assert next_deadline(detector) - asyncio.get_running_loop().time() == pytest.approx(
            0.1, abs=0.01
        )
        await asyncio.sleep(0.11)
        assert len(events) == 1
        assert not model.calls.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("category", [AMDCategory.HUMAN, AMDCategory.UNCERTAIN])
async def test_human_and_initial_uncertain_do_not_wait(category: AMDCategory) -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, _):
        speech_started(detector)
        speech_ended(detector, 0)
        hooks = await commit(detector, session, classifier)
        classifier.prediction(1, category)
        assert await hooks.should_reply(llm.ChatContext())
        assert detector._turns[1].prediction.category == category
        assert detector._turns[1].prediction.delay < 0.01


@pytest.mark.asyncio
async def test_elapsed_silence_and_slow_inference_do_not_add_another_wait() -> None:
    async with running(machine_silence_threshold=1.5, inference_timeout=3) as (
        detector,
        session,
        classifier,
        _,
    ):
        speech_started(detector)
        speech_ended(detector, 0.5)
        hooks = await commit(detector, session, classifier)
        await asyncio.sleep(1.2)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        assert await hooks.should_reply(llm.ChatContext())
        assert detector._turns[1].prediction.delay == pytest.approx(1.2, abs=0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["prediction", "inference_timeout", "inference_error", "reused"])
async def test_machine_fallbacks_and_uncertain_keep_the_silence_wait(reason: str) -> None:
    async with running(machine_silence_threshold=1.5, inference_timeout=0.2) as (
        detector,
        session,
        classifier,
        _,
    ):
        speech_started(detector)
        speech_ended(detector, 1.5)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        assert await first.should_reply(llm.ChatContext())
        events = []
        detector.on("amd_prediction", events.append)
        speech_started(detector)
        speech_ended(detector, 0.5)
        if reason == "reused":
            hooks = commit_turn(detector, end_of_turn(""))
        else:
            hooks = await commit(detector, session, classifier)
            if reason == "prediction":
                classifier.prediction(2, AMDCategory.UNCERTAIN)
            elif reason == "inference_error":
                classifier.respond(2, "invalid JSON")
        ctx = llm.ChatContext()
        reply = asyncio.create_task(hooks.should_reply(ctx))
        await asyncio.sleep(0.9)
        prediction = detector._turns[2].prediction
        assert prediction is not None
        assert len(events) == 1
        assert prediction.reason == reason
        # uncertain keeps the screening stage, so its reply waits for silence too
        assert prediction.category == (
            AMDCategory.UNCERTAIN if reason == "prediction" else AMDCategory.MACHINE_SCREENING
        )
        assert prediction.stage == AMDCategory.MACHINE_SCREENING
        assert not prediction.state_changed
        assert not reply.done()
        await asyncio.sleep(0.11)
        assert await reply
        assert ctx.items[-1].extra["amd_stage"] == "machine-screening"
        assert classifier.requests.empty()


@pytest.mark.asyncio
async def test_resumed_speech_invalidates_release_and_preserves_history() -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, _):
        events = []
        detector.on("amd_prediction", events.append)
        speech_started(detector)
        speech_ended(detector, 0.5)
        hooks = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await asyncio.sleep(0.5)
        speech_started(detector)
        await asyncio.sleep(1)
        speech_ended(detector, 0.5)
        await asyncio.sleep(0.5)
        assert [e.category for e in events] == [AMDCategory.MACHINE_VM]
        assert detector._reply_held_at(asyncio.get_running_loop().time())
        assert detector._turns[1].prediction is not None
        commit_turn(detector, end_of_turn("Hello, can you hear me?"))
        request = await classifier.request()
        assert request.earlier_turns[0].text_content == "hello"
        assert request.stage == AMDCategory.MACHINE_VM
        assert AMDCategory.MACHINE_SCREENING not in request.allowed_next_categories
        assert not await hooks.should_reply(llm.ChatContext())
        classifier.prediction(2, AMDCategory.HUMAN)
        assert (await detector.execute()).category == AMDCategory.HUMAN
        assert [e.category for e in events] == [AMDCategory.MACHINE_VM, AMDCategory.HUMAN]


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["vad", "stt"])
async def test_untranscribed_speech_rearms_the_committed_prediction(source: str) -> None:
    classifier = ClassifierLLM()
    model = RecordingLLM()
    session = AgentSession(
        llm=model,
        stt=FakeSTT(),
        tts=FakeTTS(fake_audio_duration=0.05),
        vad=FakeVAD() if source == "vad" else None,
        turn_handling={"turn_detection": "vad" if source == "vad" else "stt"},
        aec_warmup_duration=None,
    )
    session.output.audio = FakeAudioOutput()
    await session.start(CustomerAgent())
    try:
        async with AMD(
            session,
            llm=classifier,
            stt=None,
            machine_silence_threshold=1.5,
            inference_timeout=3,
        ) as detector:
            await eventually(lambda: detector.lifecycle is AMDLifecycle.ACTIVE)
            recognition = session._activity._audio_recognition
            if source == "vad":
                await asyncio.wait_for(recognition._vad_atask, 2)
            events = []
            speeches = []
            detector.on("amd_prediction", events.append)
            session.on("speech_created", lambda event: speeches.append(event.speech_handle))
            await commit(detector, session, classifier, reply=True)
            classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
            await asyncio.sleep(0.5)
            assert detector._turns[1].prediction is not None
            assert detector._reply_held_at(asyncio.get_running_loop().time())
            assert next_deadline(detector) is not None
            assert next_deadline(detector) - asyncio.get_running_loop().time() == pytest.approx(
                1.0, abs=0.01
            )

            if source == "vad":
                await recognition._on_vad_event(
                    vad.VADEvent(
                        type=vad.VADEventType.START_OF_SPEECH,
                        samples_index=0,
                        timestamp=0,
                        speech_duration=0,
                        silence_duration=0,
                    )
                )
            else:
                recognition._process_stt_event(
                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                )
            await asyncio.sleep(1.1)
            assert len(events) == 1
            assert detector._reply_held_at(asyncio.get_running_loop().time())
            assert model.calls.empty()
            assert not session._activity._user_turn_completed_atask.done()

            if source == "vad":
                await recognition._on_vad_event(
                    vad.VADEvent(
                        type=vad.VADEventType.END_OF_SPEECH,
                        samples_index=0,
                        timestamp=0,
                        speech_duration=0.6,
                        silence_duration=0.5,
                    )
                )
            else:
                recognition._process_stt_event(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.END_OF_SPEECH,
                        speech_end_time=time.time() - 0.5,
                    )
                )
            await asyncio.sleep(0.9)
            assert len(events) == 1
            assert detector._reply_held_at(asyncio.get_running_loop().time())
            assert model.calls.empty()
            await asyncio.sleep(0.11)
            assert [(event.turn_id, event.category, event.transcript) for event in events] == [
                (1, AMDCategory.MACHINE_SCREENING, "hello")
            ]
            assert detector._turn_id == 1
            assert classifier.requests.empty()
            await asyncio.wait_for(model.calls.get(), 2)
            assert len(speeches) == 1
            await asyncio.wait_for(speeches[0], 2)
            assert model.calls.empty()
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_late_human_cannot_replace_a_held_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    original_classify = _inference.classify
    response = asyncio.get_running_loop().create_future()
    cancelled = asyncio.Event()

    async def classify(model, request, **kwargs):
        if request.chat_ctx.messages()[-1].extra["turn_id"] != 2:
            return await original_classify(model, request, **kwargs)
        while True:
            try:
                return await asyncio.shield(response)
            except asyncio.CancelledError:
                cancelled.set()

    monkeypatch.setattr(_inference, "classify", classify)
    async with running(machine_silence_threshold=1.5, inference_timeout=0.2) as (
        detector,
        session,
        classifier,
        _,
    ):
        speech_started(detector)
        speech_ended(detector, 1.5)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await first.should_reply(llm.ChatContext())
        events = []
        detector.on("amd_prediction", events.append)
        speech_started(detector)
        speech_ended(detector, 0)
        commit_turn(detector, end_of_turn())
        task = detector._classifier_atask
        try:
            await asyncio.wait_for(cancelled.wait(), 0.3)
            assert not response.done()
            assert detector._reply_held_at(asyncio.get_running_loop().time())
            assert [event.reason for event in events] == ["inference_timeout"]
            response.set_result(_inference.AMDResponse(category=AMDCategory.HUMAN))
            await task
            await asyncio.sleep(1.31)
            assert detector.lifecycle is AMDLifecycle.ACTIVE
            assert detector._state == AMDCategory.MACHINE_VM
            assert [event.category for event in events] == [AMDCategory.MACHINE_VM]
            assert events[0].reason == "inference_timeout"
        finally:
            if not response.done():
                response.set_result(_inference.AMDResponse(category=AMDCategory.HUMAN))


@pytest.mark.asyncio
async def test_late_machine_prediction_cannot_replace_an_empty_turn_hold() -> None:
    async with running(machine_silence_threshold=1.5, inference_timeout=0.2) as (
        detector,
        session,
        classifier,
        model,
    ):
        events = []
        detector.on("amd_prediction", events.append)
        speech_started(detector)
        speech_ended(detector, 1.5)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert await first.should_reply(llm.ChatContext())
        handle = SpeechHandle.create()
        first.on_agent_turn_committed(handle)
        output = session.output.audio
        await output.capture_frame(rtc.AudioFrame.create(24000, 1, 24))
        output.flush()
        await output.wait_for_playout()
        handle._mark_done()
        await asyncio.wait_for(handle, 2)

        speech_started(detector)
        speech_ended(detector, 1.5)
        second = await commit(detector, session, classifier)
        assert not await second.should_reply(llm.ChatContext())
        fallback = detector._turns[2].prediction
        assert fallback is not None
        assert fallback.reason == "inference_timeout"
        fallback_snapshot = fallback.model_dump()

        speech_started(detector)
        speech_ended(detector, 0)
        activity = session._activity
        assert activity is not None
        assert activity.on_end_of_turn(end_of_turn(""))
        reply_task = activity._user_turn_completed_atask
        assert reply_task is not None
        release_at = next_deadline(detector)
        assert release_at is not None
        assert detector._turn_id == 3
        assert detector._turns[3].prediction.reason == "reused"
        assert detector._reply_held_at(asyncio.get_running_loop().time())
        await asyncio.sleep(0.4)
        classifier.prediction(2, AMDCategory.MACHINE_UNAVAILABLE)
        await asyncio.sleep(0.1)
        assert next_deadline(detector) == release_at
        assert detector._turns[2].prediction is fallback
        assert fallback.model_dump() == fallback_snapshot

        await asyncio.sleep(release_at - asyncio.get_running_loop().time() - 0.1)
        assert not reply_task.done()
        assert model.calls.empty()
        assert [(event.turn_id, event.reason) for event in events] == [
            (1, "prediction"),
            (2, "inference_timeout"),
            (3, "reused"),
        ]
        await asyncio.sleep(0.11)
        assert detector._state == AMDCategory.MACHINE_VM
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        assert detector._turns[3].prediction is not None
        assert detector._turns[2].prediction is fallback
        assert fallback.model_dump() == fallback_snapshot
        assert [(event.turn_id, event.reason) for event in events] == [
            (1, "prediction"),
            (2, "inference_timeout"),
            (3, "reused"),
        ]
        await asyncio.wait_for(reply_task, 2)
        assert model.calls.empty()
        assert classifier.requests.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 3, 4])
async def test_inference_timeout_limit_waits_for_silence_before_completion(limit: int) -> None:
    async with running(
        machine_silence_threshold=1.5, inference_timeout=0.2, max_inference_timeouts=limit
    ) as (
        detector,
        session,
        classifier,
        _,
    ):
        speech_started(detector)
        speech_ended(detector, 1.5)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await first.should_reply(llm.ChatContext())
        for turn_id in range(2, limit + 2):
            speech_started(detector)
            speech_ended(detector, 0.5)
            await commit(detector, session, classifier)
            await asyncio.sleep(0.3)
            assert detector.lifecycle is AMDLifecycle.ACTIVE, (
                session._recorded_events,
                detector._completion(),
            )
            assert detector._turns[turn_id].prediction.reason == "inference_timeout"
            assert detector._reply_held_at(asyncio.get_running_loop().time())
            await asyncio.sleep(0.71)
            assert detector._turns[turn_id].prediction is not None
        assert (await detector.execute()).reason == "inference_timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize("inference_ready", [True, False])
async def test_empty_eot_after_resumed_speech_rearms_a_useful_prediction(
    inference_ready: bool,
) -> None:
    async with running(machine_silence_threshold=1.5, inference_timeout=3) as (
        detector,
        session,
        classifier,
        _,
    ):
        events = []
        detector.on("amd_prediction", events.append)
        speech_started(detector)
        speech_ended(detector, 0.5)
        await commit(detector, session, classifier)
        if inference_ready:
            classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await asyncio.sleep(0.2)
        speech_started(detector)
        speech_ended(detector, 0.5)
        commit_turn(detector, end_of_turn(""))
        if not inference_ready:
            classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        # an empty turn reuses the prediction and emits its own reused event,
        # whether it was committed after the prediction or while it was pending
        expected = [(1, "prediction"), (2, "reused")]
        await asyncio.sleep(0.9)
        assert [(e.turn_id, e.reason) for e in events] == expected
        assert detector._reply_held_at(asyncio.get_running_loop().time())
        await asyncio.sleep(0.11)
        assert [(e.turn_id, e.reason) for e in events] == expected
        assert detector._turns[2].prediction is not None
        assert classifier.requests.empty()


@pytest.mark.asyncio
async def test_empty_turn_does_not_delay_the_next_prediction() -> None:
    stt = DrainingSTT()
    async with running(stt=stt, machine_silence_threshold=1.5) as (detector, _, classifier, _):
        events = []
        detector.on("amd_prediction", events.append)
        speech_started(detector)
        stream = push_audio(detector, stt)
        speech_ended(detector, 0)
        commit_turn(detector, end_of_turn(""))
        assert [(e.turn_id, e.reason) for e in events] == [(1, "reused")]
        assert detector._turns[1].prediction.reason == "reused"

        await asyncio.sleep(0.1)
        speech_started(detector)
        transcript = "Please state your name and why you are calling."
        stream.send_fake_transcript(transcript)
        await eventually(
            lambda: detector._resources.stt._current.snapshot("amd").transcript == transcript
        )
        speech_ended(detector, 0)
        info = end_of_turn(transcript)
        hooks = commit_turn(detector, info)
        assert (await classifier.request()).current_turn.extra["turn_id"] == 2
        classifier.prediction(2, AMDCategory.MACHINE_SCREENING)

        await asyncio.sleep(0.5)
        assert len(events) == 2
        assert detector._reply_held_at(asyncio.get_running_loop().time())
        await asyncio.sleep(1.01)
        assert [(event.turn_id, event.category) for event in events] == [
            (1, AMDCategory.UNCERTAIN),
            (2, AMDCategory.MACHINE_SCREENING),
        ]
        assert not stream.flushed.is_set()
        assert await hooks.should_reply(llm.ChatContext())


@pytest.mark.asyncio
async def test_new_inference_discards_stale_turn_waits() -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, _):
        speech_started(detector)
        speech_ended(detector, 1.5)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await first.should_reply(llm.ChatContext())
        speech_started(detector)
        speech_ended(detector, 0)
        await commit(detector, session, classifier)
        classifier.prediction(2, AMDCategory.MACHINE_SCREENING)
        await asyncio.sleep(0.1)
        commit_turn(detector, end_of_turn(""))
        speech_started(detector)
        speech_ended(detector, 0)
        fourth = await commit(detector, session, classifier)
        await detector._should_reply(2, llm.ChatContext())
        await detector._should_reply(3, llm.ChatContext())
        assert detector._turns[2].prediction.reason == "prediction"
        assert detector._turns[3].prediction.reason == "reused"
        classifier.prediction(4, AMDCategory.MACHINE_SCREENING)
        await fourth.should_reply(llm.ChatContext())
        assert next_deadline(detector) is not None
        assert next_deadline(detector) - asyncio.get_running_loop().time() == pytest.approx(
            10, abs=0.01
        )


@pytest.mark.asyncio
async def test_prediction_listener_failure_cleans_up_a_deferred_release() -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, _):

        def on_prediction(_event: object) -> None:
            raise TypeError("listener failed")

        detector.on("amd_prediction", on_prediction)
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        assert (await detector.execute()).reason == "internal_error"
        assert detector._turns[1].prediction is not None
        assert session.amd is None


@pytest.mark.asyncio
async def test_late_machine_prediction_does_not_postpone_idle_timeout() -> None:
    async with running(machine_silence_threshold=1.5, inference_timeout=0.01, idle_timeout=0.1) as (
        detector,
        session,
        classifier,
        model,
    ):
        events = []
        replies = []
        detector.on("amd_prediction", events.append)
        session.on("speech_created", lambda event: replies.append(event.speech_handle))
        await commit(detector, session, classifier, reply=True)
        await asyncio.wait_for(model.calls.get(), 2)
        assert events[-1].reason == "inference_timeout"
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await asyncio.wait_for(replies[0], 2)
        await asyncio.sleep(0)
        release_at = next_deadline(detector)
        assert release_at is not None
        await asyncio.sleep(0.2)
        assert detector.lifecycle is AMDLifecycle.FINISHED
        assert [(event.category, event.reason) for event in events] == [
            (AMDCategory.UNCERTAIN, "inference_timeout"),
        ]
        assert (await asyncio.wait_for(detector.execute(), 0.2)).reason == "idle_timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize("elapsed", [None, 0.7])
async def test_eot_without_speech_edges_uses_available_timing(elapsed: float | None) -> None:
    async with running(machine_silence_threshold=1.5) as (detector, _, classifier, _):
        info = end_of_turn()
        info.metrics.end_of_turn_delay = elapsed
        commit_turn(detector, info)
        await classifier.request()
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        started = asyncio.get_running_loop().time()
        await detector._should_reply(1, llm.ChatContext())
        waited = asyncio.get_running_loop().time() - started
        assert waited == pytest.approx(1.5 - (elapsed or 0), abs=0.01)
        assert detector._turns[1].prediction.delay < 0.01


@pytest.mark.asyncio
async def test_eot_before_speech_end_waits_for_the_speech_end_anchor() -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, _):
        speech_started(detector)
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await asyncio.sleep(0.6)
        assert detector._turns[1].prediction is not None
        assert detector._reply_held_at(asyncio.get_running_loop().time())
        speech_ended(detector, 0.5)
        started = asyncio.get_running_loop().time()
        await detector._should_reply(1, llm.ChatContext())
        assert asyncio.get_running_loop().time() - started == pytest.approx(1.0, abs=0.01)
        assert detector._turns[1].prediction.delay < 0.01


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("speech_delay", "remaining_silence"), [(0.7, 0.8), (None, 1.5), (-0.5, 1.5)]
)
async def test_speech_timing_is_independent_of_event_creation(
    speech_delay: float | None, remaining_silence: float
) -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, _):
        now = time.time()
        detector._on_user_state_changed(
            UserStateChangedEvent(
                old_state="listening",
                new_state="speaking",
                speech_timestamp=now - speech_delay if speech_delay is not None else None,
                created_at=now - 0.2,
            )
        )
        await asyncio.sleep(1)
        now = time.time()
        detector._on_user_state_changed(
            UserStateChangedEvent(
                old_state="speaking",
                new_state="listening",
                speech_timestamp=now - speech_delay if speech_delay is not None else None,
                created_at=now - 0.2,
            )
        )
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        started = asyncio.get_running_loop().time()
        await detector._should_reply(1, llm.ChatContext())
        assert detector._turns[1].prediction.speech_duration == pytest.approx(1.0, abs=0.01)
        assert asyncio.get_running_loop().time() - started == pytest.approx(
            remaining_silence, abs=0.01
        )
        assert detector._turns[1].prediction.delay < 0.01


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["vad", "stt", "stt_with_vad"])
async def test_audio_recognition_supplies_speech_edges_and_elapsed_silence(
    source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, _, _):
        recognition = session._activity._audio_recognition
        monkeypatch.setattr(recognition, "_run_eou_detection", Mock())
        state_events = []
        session.on("user_state_changed", state_events.append)
        if source == "vad":
            await recognition._on_vad_event(
                vad.VADEvent(
                    type=vad.VADEventType.START_OF_SPEECH,
                    samples_index=0,
                    timestamp=0,
                    speech_duration=0.1,
                    silence_duration=0,
                )
            )
        else:
            monkeypatch.setattr(
                recognition, "_vad", FakeVAD() if source == "stt_with_vad" else None
            )
            recognition._turn_detection_mode = "stt"
            recognition._process_stt_event(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.START_OF_SPEECH, speech_start_time=time.time() - 0.1
                )
            )
        assert detector._speech_started_at is not None
        assert detector._speech_since_commit
        assert state_events[-1].speech_timestamp == pytest.approx(time.time() - 0.1, abs=0.01)
        await asyncio.sleep(1)
        if source == "vad":
            await recognition._on_vad_event(
                vad.VADEvent(
                    type=vad.VADEventType.END_OF_SPEECH,
                    samples_index=0,
                    timestamp=0,
                    speech_duration=0.3,
                    silence_duration=0.5,
                    inference_duration=0.2,
                )
            )
        else:
            recognition._process_stt_event(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.END_OF_SPEECH, speech_end_time=time.time() - 0.7
                )
            )
        assert detector._speech_started_at is None
        assert detector._speech_ended_at == pytest.approx(
            asyncio.get_running_loop().time() - 0.7, abs=0.01
        )
        assert state_events[-1].speech_timestamp == pytest.approx(time.time() - 0.7, abs=0.01)
        assert state_events[-1].created_at == pytest.approx(time.time(), abs=0.01)
        if source == "stt_with_vad":
            speech_ended_at = detector._speech_ended_at
            await recognition._on_vad_event(
                vad.VADEvent(
                    type=vad.VADEventType.END_OF_SPEECH,
                    samples_index=0,
                    timestamp=0,
                    speech_duration=0.3,
                    silence_duration=0.1,
                )
            )
            assert len(state_events) == 2
            assert detector._speech_ended_at == speech_ended_at
            await recognition._on_vad_event(
                vad.VADEvent(
                    type=vad.VADEventType.START_OF_SPEECH,
                    samples_index=0,
                    timestamp=0,
                    speech_duration=0.1,
                    silence_duration=0,
                )
            )
            assert detector._speech_started_at is not None
            assert detector._speech_since_commit


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["vad", "stt"])
async def test_untranscribed_speech_restarts_idle_timer(source: str) -> None:
    session = AgentSession(
        llm=FakeLLM(),
        stt=FakeSTT(),
        vad=FakeVAD() if source == "vad" else None,
        turn_handling={"turn_detection": "vad" if source == "vad" else "stt"},
    )
    await session.start(Agent(instructions="Answer the call."))
    try:
        async with AMD(session, llm=None, stt=None, idle_timeout=0.2, timeout=2) as detector:
            await eventually(lambda: detector.lifecycle is AMDLifecycle.ACTIVE)
            recognition = session._activity._audio_recognition
            if source == "vad":
                # Let the empty fake stream finish before injecting VAD events.
                await recognition._vad_atask
                await recognition._on_vad_event(
                    vad.VADEvent(
                        type=vad.VADEventType.START_OF_SPEECH,
                        samples_index=0,
                        timestamp=0,
                        speech_duration=0,
                        silence_duration=0,
                    )
                )
            else:
                recognition._process_stt_event(
                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                )
            await asyncio.sleep(0.3)
            assert detector.lifecycle is AMDLifecycle.ACTIVE

            if source == "vad":
                await recognition._on_vad_event(
                    vad.VADEvent(
                        type=vad.VADEventType.END_OF_SPEECH,
                        samples_index=0,
                        timestamp=0,
                        speech_duration=0.3,
                        silence_duration=0,
                    )
                )
            else:
                recognition._process_stt_event(
                    stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                )
            await asyncio.sleep(0.1)
            assert detector.lifecycle is AMDLifecycle.ACTIVE
            result = await asyncio.wait_for(detector.execute(), 0.2)
            assert result.reason == "idle_timeout"
            assert result.turn_id == 0
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_speech_end_does_not_start_idle_while_customer_hook_runs() -> None:
    agent = CustomerAgent(stop=True)
    agent.hook_release.clear()
    async with running(agent=agent, idle_timeout=0.1) as (detector, session, classifier, _):
        await commit(detector, session, classifier, reply=True)
        await agent.hook_started.wait()
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await detector._should_reply(1, llm.ChatContext())
        speech_started(detector)
        speech_ended(detector, 0)
        await asyncio.sleep(0.2)
        assert detector.lifecycle is AMDLifecycle.ACTIVE

        agent.hook_release.set()
        result = await asyncio.wait_for(detector.execute(), 0.2)
        assert result.reason == "idle_timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["timeout", "cancelled", "participant_disconnected"])
async def test_completion_cancels_silence_wait_and_releases_waiters(reason: str) -> None:
    async with running(machine_silence_threshold=1.5, inference_timeout=0.2, timeout=0.5) as (
        detector,
        session,
        classifier,
        _,
    ):
        events = []
        detector.on("amd_prediction", events.append)
        await commit(detector, session, classifier)
        committed_at = asyncio.get_running_loop().time()
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await eventually(
            lambda: (
                next_deadline(detector) is not None and next_deadline(detector) > committed_at + 0.4
            )
        )
        waiter = asyncio.create_task(detector._should_reply(1, llm.ChatContext()))
        await asyncio.sleep(0)
        assert not waiter.done()
        if reason == "cancelled":
            await detector.aclose()
        elif reason == "participant_disconnected":
            detector._participant_identity = "callee"
            detector._on_disconnected(SimpleNamespace(identity="callee"))
        assert (await detector.execute()).reason == reason
        await asyncio.wait_for(waiter, 2)
        assert next_deadline(detector) is None
        await asyncio.sleep(1.6)
        assert detector._turns[1].prediction.category == AMDCategory.MACHINE_VM
        assert [e.category for e in events] == [AMDCategory.MACHINE_VM]
        assert not detector._reply_held_at(asyncio.get_running_loop().time())
        assert session.amd is None
        assert not detector._tasks


@pytest.mark.asyncio
async def test_silence_deadline_crossed_while_rearming_still_wakes_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents.voice.amd import detector as detector_module

    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, _):
        hooks = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await eventually(lambda: detector._turns[1].prediction is not None)
        reply = asyncio.create_task(hooks.should_reply(llm.ChatContext()))
        await asyncio.sleep(0)
        assert not reply.done()
        release_at = detector._speech_ended_at + 1.5
        readings = iter((release_at - 0.01, release_at - 0.01))

        def clock() -> float:
            return next(readings, release_at + 0.01)

        with monkeypatch.context() as crossed_deadline:
            crossed_deadline.setattr(detector_module.time, "monotonic", clock)
            detector._update_idle()
            assert detector._timer.when() == pytest.approx(
                asyncio.get_running_loop().time(), abs=1e-6
            )
            assert await asyncio.wait_for(reply, 0.1)
        assert detector.lifecycle is AMDLifecycle.ACTIVE
