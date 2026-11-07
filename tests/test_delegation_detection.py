from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from livekit.agents import Agent, AgentSession, llm
from livekit.agents.llm import DelegationCheck, DelegationDetector, FunctionToolCall
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.events import AgentStateChangedEvent, UserStateChangedEvent

from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_realtime import FakeRealtimeModel

pytestmark = pytest.mark.unit


@llm.function_tool
async def cancel_appointment() -> None:
    """Cancel the caller's appointment."""


def _check() -> DelegationCheck:
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="Cancel my 3pm.")
    chat_ctx.add_message(role="assistant", content="Done, it's cancelled.")
    return DelegationCheck(chat_ctx=chat_ctx, tools=[cancel_appointment], instructions="Be brief.")


# what the detector sends: the backend tools, the agent's instructions, then the conversation
_INPUT = (
    "## Backend tools\n\n- cancel_appointment: Cancel the caller's appointment.\n\n"
    "## The voice agent's instructions\n\nBe brief.\n\n"
    "## Conversation\n\nuser: Cancel my 3pm.\nassistant: Done, it's cancelled."
)


def _reporting(arguments: str) -> FakeLLM:
    call = FunctionToolCall(name="report", arguments=arguments, call_id="c1")
    return FakeLLM(
        fake_responses=[
            FakeLLMResponse(input=_INPUT, content="", ttft=0, duration=0, tool_calls=[call])
        ]
    )


async def test_an_llm_reports_its_verdict_through_a_tool_call() -> None:
    assert await DelegationDetector(_reporting('{"delegate": true}'))(_check()) is True
    assert await DelegationDetector(_reporting('{"delegate": false}'))(_check()) is False
    # an answer it cannot read leaves the turn alone
    assert await DelegationDetector(_reporting("not json"))(_check()) is False


class _Decided(llm.DecisionModel):
    def __init__(self, probability: float) -> None:
        super().__init__(capabilities=llm.DecisionCapabilities(decision_kinds=["probability"]))
        self._probability = probability

    async def _evaluate_impl(self, *, input, decisions, conn_options) -> llm.DecisionResponse:  # type: ignore[no-untyped-def]
        assert input == _INPUT
        return llm.DecisionResponse(
            results={"delegate": llm.ProbabilityResult(value=self._probability)}
        )


async def test_a_decision_model_delegates_at_even_odds_or_more() -> None:
    assert await DelegationDetector(_Decided(0.5))(_check()) is True
    assert await DelegationDetector(_Decided(0.4))(_check()) is False


async def test_a_turn_said_over_the_agent_is_marked_and_the_backend_told() -> None:
    check = _check()
    # the caller started while the agent's reply was still playing
    check.chat_ctx.items[0].metrics = {"started_speaking_at": 102.0}
    check.chat_ctx.items[1].metrics = {"started_speaking_at": 100.0, "stopped_speaking_at": 105.0}
    marked = _INPUT.replace(
        "user: Cancel", "user (said while the agent was still speaking): Cancel"
    )
    call = FunctionToolCall(name="report", arguments='{"delegate": true}', call_id="c1")
    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(input=marked, content="", ttft=0, duration=0, tool_calls=[call])
        ]
    )
    note = await DelegationDetector(model)(check)
    # the caller's words never reach the backend as a developer message
    assert isinstance(note, str) and "Cancel my 3pm." not in note


# the loop in AgentActivity: a turn the realtime model left undelegated is judged once the agent
# has stopped speaking for the grace period


def _activity(verdict: bool | str) -> tuple[AgentActivity, list[DelegationCheck]]:
    checks: list[DelegationCheck] = []

    async def detector(check: DelegationCheck) -> bool | str:
        checks.append(check)
        return verdict

    activity = AgentActivity(Agent(instructions="test"), AgentSession(llm=FakeRealtimeModel()))
    activity._rt_session = MagicMock()
    activity._rt_session.tools.flatten.return_value = []
    activity._delegation_opts = {"detector": detector, "grace_period": 0}
    return activity, checks


def _turn(activity: AgentActivity, *, delegated: bool = False) -> None:
    # as GPT-Live reports a turn: the final transcript goes out before the speech stops
    activity._on_input_speech_started(llm.InputSpeechStartedEvent())
    activity._on_input_audio_transcription_completed(
        llm.InputTranscriptionCompleted(
            item_id="u1", transcript="Okay.", is_final=True, delegated=delegated
        )
    )
    activity._on_input_speech_stopped(llm.InputSpeechStoppedEvent(user_transcription_enabled=False))
    _user(activity, "speaking", "listening")


def _user(activity: AgentActivity, old: str, new: str) -> None:
    activity._on_speaking_for_delegation(
        UserStateChangedEvent(old_state=old, new_state=new)  # type: ignore[arg-type]
    )


def _agent(activity: AgentActivity, old: str, new: str) -> None:
    activity._on_speaking_for_delegation(
        AgentStateChangedEvent(old_state=old, new_state=new)  # type: ignore[arg-type]
    )


async def test_an_undelegated_turn_is_judged_and_delegated() -> None:
    activity, checks = _activity(True)
    _turn(activity)
    await activity._delegation_check_atask
    assert checks[0].chat_ctx.items[-1].text_content == "Okay."
    activity._rt_session.delegate.assert_called_once_with(None)


async def test_a_note_from_the_detector_goes_to_the_delegated_work() -> None:
    activity, _ = _activity("The caller said it over the agent.")
    _turn(activity)
    await activity._delegation_check_atask
    activity._rt_session.delegate.assert_called_once_with("The caller said it over the agent.")


def test_a_turn_the_model_delegated_is_not_judged() -> None:
    activity, _ = _activity(True)
    _turn(activity, delegated=True)
    assert activity._delegation_check_atask is None


@pytest.mark.parametrize("event", ["delegation", "new turn"])
async def test_a_delegation_or_a_new_turn_drops_the_pending_judgment(event: str) -> None:
    activity, checks = _activity(True)
    activity._delegation_opts["grace_period"] = 10
    _turn(activity)
    pending = activity._delegation_check_atask
    if event == "delegation":
        activity._on_delegation_created(llm.DelegationCreatedEvent(id="d1"))
    else:
        activity._on_input_speech_started(llm.InputSpeechStartedEvent())
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert checks == []


async def test_the_judgment_waits_for_the_agent_to_stop_speaking() -> None:
    activity, checks = _activity(True)
    activity._delegation_opts["grace_period"] = 10
    _turn(activity)
    first = activity._delegation_check_atask
    _agent(activity, "listening", "speaking")
    with pytest.raises(asyncio.CancelledError):
        await first
    activity._delegation_opts["grace_period"] = 0
    _agent(activity, "speaking", "listening")
    await activity._delegation_check_atask
    assert len(checks) == 1


async def test_the_caller_speaking_holds_the_judgment_without_dropping_it() -> None:
    activity, checks = _activity(True)
    activity._delegation_opts["grace_period"] = 10
    _turn(activity)
    first = activity._delegation_check_atask
    _user(activity, "listening", "speaking")  # noise, or the caller starting again
    with pytest.raises(asyncio.CancelledError):
        await first
    activity._delegation_opts["grace_period"] = 0
    _user(activity, "speaking", "listening")
    await activity._delegation_check_atask
    assert len(checks) == 1
