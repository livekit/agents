from __future__ import annotations

import asyncio
import json

import pytest

from livekit.agents import llm
from livekit.agents.voice.amd import AMDCategory, AMDLifecycle, AMDReason
from livekit.agents.voice.amd.detector import (
    _DEFAULT_IVR_INSTRUCTIONS,
    _DEFAULT_SCREENING_INSTRUCTIONS,
)

from .amd_test_utils import detector_clock  # noqa: F401
from .test_amd_detector import (
    commit,
    commit_turn,
    dtmf_calls,
    dtmf_executed,
    end_of_turn,
    eventually,
    running,
)

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent, pytest.mark.virtual_time]


@pytest.mark.parametrize(
    ("initial", "corrected", "transcript", "instructions"),
    [
        (
            AMDCategory.MACHINE_VM,
            AMDCategory.MACHINE_SCREENING,
            "Please state your name and why you are calling.",
            _DEFAULT_SCREENING_INSTRUCTIONS,
        ),
        (
            AMDCategory.MACHINE_IVR,
            AMDCategory.MACHINE_SCREENING,
            "Please state your name and why you are calling.",
            _DEFAULT_SCREENING_INSTRUCTIONS,
        ),
        (
            AMDCategory.MACHINE_SCREENING,
            AMDCategory.MACHINE_IVR,
            "Press 1 for appointments.",
            _DEFAULT_IVR_INSTRUCTIONS,
        ),
    ],
)
@pytest.mark.parametrize("bridge", [None, AMDCategory.WAIT, AMDCategory.UNCERTAIN])
async def test_correction_uses_current_turn_and_persists(
    initial: AMDCategory,
    corrected: AMDCategory,
    transcript: str,
    instructions: str,
    bridge: AMDCategory | None,
) -> None:
    async with running() as (detector, _, classifier, _):
        first = commit_turn(detector, end_of_turn("Please record your name."))
        request = await classifier.request()
        assert request.previous_prediction is None
        assert request.allowed_correction_categories == []
        classifier.prediction(1, initial)
        assert await first.should_reply(llm.ChatContext())
        saved_prediction = detector._turns[1].prediction.model_copy(deep=True)
        previous_turn = 1
        if bridge is not None:
            hooks = commit_turn(detector, end_of_turn("One moment."))
            await classifier.request()
            classifier.prediction(2, bridge)
            assert await hooks.should_reply(llm.ChatContext()) == (bridge != AMDCategory.WAIT)
            previous_turn = 2

        current_turn = previous_turn + 1
        hooks = commit_turn(detector, end_of_turn(transcript))
        request = await classifier.request()
        assert request.stage == initial
        assert request.previous_prediction == {
            "turn_id": previous_turn,
            "category": bridge or initial,
            "reason": "prediction",
        }
        assert corrected not in request.allowed_next_categories
        assert request.allowed_correction_categories == [corrected]
        classifier.prediction(
            current_turn, corrected, corrects_stage=True, correction_evidence=transcript
        )
        context = llm.ChatContext()
        assert await hooks.should_reply(context)
        assert context.items[-1].text_content == instructions
        prediction = detector._turns[current_turn].prediction
        assert prediction.reason == AMDReason.PREDICTION
        assert prediction.corrects_stage
        assert prediction.correction_evidence == transcript
        assert prediction.state_changed
        assert prediction.prev_stage_category == initial
        assert prediction.inference_duration is not None
        assert detector._turns[1].prediction == saved_prediction

        human = commit_turn(detector, end_of_turn("Hello, Sam speaking."))
        request = await classifier.request()
        assert request.stage == corrected
        assert request.previous_prediction["category"] == corrected
        assert request.previous_prediction["turn_id"] == current_turn
        assert AMDCategory.HUMAN in request.allowed_next_categories
        assert AMDCategory.HUMAN not in request.allowed_correction_categories
        classifier.prediction(current_turn + 1, AMDCategory.HUMAN)
        assert await human.should_reply(llm.ChatContext())
        assert (await detector.execute()).category == AMDCategory.HUMAN


@pytest.mark.parametrize("evidence", [None, "", "   "])
async def test_correction_without_evidence_uses_the_existing_stage(evidence: str | None) -> None:
    async with running() as (detector, session, classifier, _):
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert await first.should_reply(llm.ChatContext())
        hooks = await commit(detector, session, classifier)
        arguments = {"category": "machine-screening", "corrects_stage": True}
        if evidence is not None:
            arguments["correction_evidence"] = evidence
        classifier.respond(2, json.dumps(arguments))
        assert await hooks.should_reply(llm.ChatContext())
        prediction = detector._turns[2].prediction
        assert prediction.reason == AMDReason.INFERENCE_ERROR
        assert prediction.stage == AMDCategory.MACHINE_VM
        assert not prediction.corrects_stage
        assert prediction.correction_evidence is None
        assert detector.lifecycle == AMDLifecycle.ACTIVE


async def test_reused_turns_do_not_repeat_a_correction() -> None:
    async with running() as (detector, session, classifier, _):
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert await first.should_reply(llm.ChatContext())
        correction = await commit(detector, session, classifier)
        empty = commit_turn(detector, end_of_turn(""))
        classifier.prediction(
            2,
            AMDCategory.MACHINE_SCREENING,
            corrects_stage=True,
            correction_evidence="Please state your name.",
        )
        assert not await correction.should_reply(llm.ChatContext())
        assert await empty.should_reply(llm.ChatContext())
        assert detector._turns[2].prediction.corrects_stage
        empty = commit_turn(detector, end_of_turn(""))
        assert await empty.should_reply(llm.ChatContext())
        for turn_id in (3, 4):
            prediction = detector._turns[turn_id].prediction
            assert prediction.reason == AMDReason.REUSED
            assert prediction.stage == AMDCategory.MACHINE_SCREENING
            assert not prediction.corrects_stage
            assert prediction.correction_evidence is None
        assert classifier.requests.empty()


async def test_correction_preserves_delivered_voicemail_and_dtmf_history() -> None:
    async with running() as (detector, session, classifier, model):
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(model.calls.get(), 2)
        await eventually(lambda: detector._voicemail_message_played)
        dtmf_executed(session, "1")

        activity = session._activity
        assert activity is not None
        transcript = "Please state your name and why you are calling."
        assert activity.on_end_of_turn(end_of_turn(transcript))
        request = await classifier.request()
        assert [json.loads(call.arguments) for call in dtmf_calls(request.chat_ctx)] == [
            {"events": ["1"]}
        ]
        classifier.prediction(
            2, AMDCategory.MACHINE_SCREENING, corrects_stage=True, correction_evidence=transcript
        )
        call = await asyncio.wait_for(model.calls.get(), 2)
        assert call["chat_ctx"].items[-1].text_content == _DEFAULT_SCREENING_INSTRUCTIONS
        await eventually(lambda: activity._no_pending_speech)

        await commit(detector, session, classifier, reply=True)
        classifier.prediction(3, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(activity._user_turn_completed_atask, 2)
        assert detector._voicemail_message_played
        assert model.calls.empty()
        assert detector._turns[3].prediction.stage == AMDCategory.MACHINE_VM
        assert not detector._turns[3].prediction.corrects_stage
