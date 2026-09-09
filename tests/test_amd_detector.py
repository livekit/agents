from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from livekit import rtc
from livekit.agents import (
    AMD,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    StopResponse,
    llm,
    utils,
)
from livekit.agents.types import APIConnectOptions
from livekit.agents.voice.amd import AMDCategory
from livekit.agents.voice.amd.detector import (
    _HUMAN_INSTRUCTIONS,
    DEFAULT_IVR_INSTRUCTIONS,
    DEFAULT_SCREENING_INSTRUCTIONS,
    DEFAULT_VOICEMAIL_INSTRUCTIONS,
)
from livekit.agents.voice.audio_recognition import _EndOfTurnInfo, _EndOfTurnMetrics

from .fake_io import FakeAudioOutput
from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_tts import FakeTTS

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


class ClassifierLLM(FakeLLM):
    """Controllable provider model. Tests use the real SDK inference path."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: asyncio.Queue[Any] = asyncio.Queue()
        self.menu_requests: asyncio.Queue[asyncio.Future[str]] = asyncio.Queue()
        self.responses: dict[int, asyncio.Future[str]] = {}

    def chat(self, **kwargs: Any) -> llm.LLMStream:
        data = json.loads(kwargs["chat_ctx"].items[-1].text_content)
        response = asyncio.get_running_loop().create_future()
        if "stage" in data:
            self.responses[data["turn_id"]] = response
            self.requests.put_nowait(SimpleNamespace(**data))
        else:
            self.menu_requests.put_nowait(response)
        return ControlledStream(self, response=response, **kwargs)

    async def request(self) -> Any:
        return await asyncio.wait_for(self.requests.get(), 2)

    def prediction(self, turn_id: int, category: AMDCategory, **kwargs: Any) -> None:
        self.respond(turn_id, json.dumps({"category": category.value}))

    def respond(self, turn_id: int, text: str) -> None:
        response = self.responses[turn_id]
        if not response.done():
            response.set_result(text)


class ControlledStream(llm.LLMStream):
    def __init__(
        self, model: ClassifierLLM, *, response: asyncio.Future[str], **kwargs: Any
    ) -> None:
        super().__init__(
            model,
            chat_ctx=kwargs["chat_ctx"],
            tools=[],
            conn_options=APIConnectOptions(max_retry=0),
        )
        self.response = response

    async def _run(self) -> None:
        text = await self.response
        self._event_ch.send_nowait(
            llm.ChatChunk(id="amd-test", delta=llm.ChoiceDelta(role="assistant", content=text))
        )


class RecordingLLM(FakeLLM):
    def __init__(self) -> None:
        super().__init__(
            fake_responses=[
                FakeLLMResponse(input=text, content="Please call me back.", ttft=0, duration=0)
                for text in (
                    DEFAULT_SCREENING_INSTRUCTIONS,
                    DEFAULT_VOICEMAIL_INSTRUCTIONS,
                    DEFAULT_IVR_INSTRUCTIONS,
                    _HUMAN_INSTRUCTIONS,
                    "hello",
                )
            ]
        )
        self.calls: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def chat(self, **kwargs: Any) -> llm.LLMStream:
        self.calls.put_nowait({**kwargs, "chat_ctx": kwargs["chat_ctx"].copy()})
        return super().chat(**kwargs)


class CustomerAgent(Agent):
    def __init__(self, *, stop: bool = False) -> None:
        super().__init__(instructions="You are calling about an appointment.")
        self.hooks: list[llm.ChatContext] = []
        self.hook_started = asyncio.Event()
        self.hook_release = asyncio.Event()
        self.hook_release.set()
        self.stop = stop

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        self.hooks.append(turn_ctx.copy())
        self.hook_started.set()
        await self.hook_release.wait()
        if self.stop:
            raise StopResponse()
        turn_ctx.add_message(role="system", content="Customer hook edit")


def end_of_turn(text: str = "hello", *, skip_reply: bool = False) -> _EndOfTurnInfo:
    return _EndOfTurnInfo(
        skip_reply=skip_reply,
        new_transcript=text,
        transcript_confidence=1,
        metrics=_EndOfTurnMetrics(None, None, None, None),
    )


async def eventually(predicate: Any) -> None:
    async def wait() -> None:
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), 2)


@asynccontextmanager
async def running(
    *, classifier: ClassifierLLM | None = None, agent: CustomerAgent | None = None, **options: Any
) -> AsyncIterator[tuple[AMD, AgentSession, ClassifierLLM, RecordingLLM]]:
    classifier = classifier or ClassifierLLM()
    agent = agent or CustomerAgent()
    model = RecordingLLM()
    session = AgentSession(
        llm=model,
        tts=FakeTTS(fake_audio_duration=0.05),
        turn_handling={"turn_detection": "manual"},
        aec_warmup_duration=None,
    )
    session.output.audio = FakeAudioOutput()
    await session.start(agent)
    try:
        detector = AMD(
            session,
            llm=classifier,
            **options,
        )
        async with detector:
            await eventually(lambda: detector.started)
            yield detector, session, classifier, model
    finally:
        await session.aclose()


async def commit(
    detector: AMD, session: AgentSession, classifier: ClassifierLLM, *, reply: bool = False
) -> _EndOfTurnInfo:
    info = end_of_turn()
    if reply:
        assert session._activity is not None
        session._activity.on_end_of_turn(info)
    else:
        detector._on_end_of_turn(info)
    request = await classifier.request()
    assert request.turn_id == info.amd_turn_id
    return info


@pytest.mark.asyncio
async def test_customer_hook_and_prediction_overlap_controls_are_temporary() -> None:
    agent = CustomerAgent()
    agent.hook_release.clear()
    async with running(agent=agent) as (detector, session, classifier, model):
        completed = asyncio.create_task(detector.execute())
        await commit(detector, session, classifier, reply=True)
        await agent.hook_started.wait()
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await eventually(lambda: detector._turns[1].decision.done())
        assert model.calls.empty()
        agent.hook_release.set()
        call = await asyncio.wait_for(model.calls.get(), 2)
        messages = [m for m in call["chat_ctx"].items if m.type == "message"]
        assert messages[-1].text_content == DEFAULT_SCREENING_INSTRUCTIONS
        assert any(m.text_content == "Customer hook edit" for m in messages)
        assert any(m.text_content == "hello" for m in messages)
        assert len(agent.hooks) == 1
        assert not completed.done()
        await eventually(lambda: session._activity._no_pending_speech)
        assert not any(m.id.startswith(detector._control_prefix) for m in agent.chat_ctx.items)
        await detector.aclose()
        assert (await completed).reason == "cancelled"


@pytest.mark.asyncio
async def test_voicemail_sends_one_message_per_stage_and_records_playback() -> None:
    async with running() as (detector, session, classifier, model):
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(model.calls.get(), 2)
        await eventually(lambda: detector._voicemail_message_played)
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(2, AMDCategory.MACHINE_VM)
        await eventually(lambda: detector._turns[2].decision.done())
        await session._activity._user_turn_completed_atask
        assert model.calls.empty()
        assert detector.enabled
        assert [
            m.text_content
            for m in session.current_agent.chat_ctx.items
            if m.type == "message" and m.role == "user"
        ] == ["hello", "hello"]


@pytest.mark.asyncio
async def test_voicemail_idle_defaults_to_one_minute_after_playback() -> None:
    async with running() as (detector, session, classifier, model):
        loop = asyncio.get_running_loop()
        hard_timer = detector._hard_timer
        assert detector._idle_timer is not None
        assert detector._idle_timer.when() - loop.time() == pytest.approx(10, abs=0.1)
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(model.calls.get(), 2)
        await eventually(lambda: detector._voicemail_message_played and detector._idle_timer)
        timer = detector._idle_timer
        assert timer is not None
        assert timer.when() - loop.time() == pytest.approx(60, abs=0.1)
        assert detector._hard_timer is hard_timer
        await detector.aclose()
        assert timer.cancelled()


@pytest.mark.asyncio
async def test_late_post_voicemail_menu_uses_the_normal_ivr_idle_timeout() -> None:
    async with running(idle_timeout=0.03, voicemail_idle_timeout=1.0) as (
        detector,
        session,
        classifier,
        model,
    ):
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(model.calls.get(), 2)
        await eventually(lambda: detector._voicemail_message_played and detector._idle_timer)
        timer = detector._idle_timer
        await asyncio.sleep(0.06)
        assert detector.enabled

        detector._on_user_speech_started()
        assert timer.cancelled()
        assert detector._idle_timer is None
        detector._on_user_speech_ended(0)
        menu = (
            "To replay your message, press 1. To continue recording, press 2. "
            "To delete and re-record your message, press 3. For delivery options, press 4. "
            "To send a fax, press 6. To cancel this message, press star. "
            "To send this message now, press pound or hang up."
        )
        session._activity.on_end_of_turn(end_of_turn(menu))
        await classifier.request()
        classifier.prediction(2, AMDCategory.MACHINE_IVR)
        call = await asyncio.wait_for(model.calls.get(), 2)
        assert call["chat_ctx"].items[-1].text_content.startswith(DEFAULT_IVR_INSTRUCTIONS)
        assert any(tool.id == "send_dtmf_events" for tool in call["tools"])
        result = await asyncio.wait_for(detector.execute(), 0.5)
        assert result.reason == "idle_timeout"
        assert result.category == AMDCategory.MACHINE_IVR
        assert result.voicemail_message_played


@pytest.mark.asyncio
@pytest.mark.parametrize("category", [AMDCategory.MACHINE_VM, AMDCategory.MACHINE_IVR])
async def test_late_stage_change_replaces_the_idle_timer(category: AMDCategory) -> None:
    async with running(idle_timeout=1.0, voicemail_idle_timeout=2.0, inference_timeout=0.01) as (
        detector,
        session,
        classifier,
        _,
    ):
        first = await commit(detector, session, classifier)
        previous = (
            AMDCategory.MACHINE_IVR
            if category == AMDCategory.MACHINE_VM
            else AMDCategory.MACHINE_VM
        )
        classifier.prediction(1, previous)
        assert await detector._prepare_reply(first, llm.ChatContext())
        second = await commit(detector, session, classifier)
        await detector._prepare_reply(second, llm.ChatContext())
        timer = detector._idle_timer
        assert timer is not None
        classifier.prediction(2, category)
        await eventually(
            lambda: detector._latest.turn_id == 2 and detector._latest.reason != "inference_timeout"
        )
        assert timer.cancelled()
        assert detector._idle_timer is not None and detector._idle_timer is not timer
        expected = 2 if category == AMDCategory.MACHINE_VM else 1
        assert detector._idle_timer.when() - asyncio.get_running_loop().time() == pytest.approx(
            expected, abs=0.1
        )


@pytest.mark.asyncio
async def test_voicemail_idle_does_not_extend_the_hard_timeout() -> None:
    async with running(timeout=0.08) as (detector, session, classifier, _):
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.category == AMDCategory.MACHINE_VM
        assert result.reason == "timeout"
        assert detector._idle_timer is None


@pytest.mark.asyncio
@pytest.mark.parametrize("voicemail_idle_timeout", [0, -1])
async def test_nonpositive_voicemail_idle_timeout_is_rejected(
    voicemail_idle_timeout: float,
) -> None:
    with pytest.raises(ValueError, match="timeouts"):
        AMD(AgentSession(), voicemail_idle_timeout=voicemail_idle_timeout)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("category", "prompt"),
    [
        (
            AMDCategory.MACHINE_VM,
            "I'm not available right now. Please leave a message after the beep.",
        ),
        (AMDCategory.MACHINE_SCREENING, "Please state your name and the reason you are calling."),
        (AMDCategory.MACHINE_IVR, "Say who you want to speak to."),
    ],
)
@pytest.mark.parametrize("hook_finishes_after_cleanup", [False, True])
async def test_human_takeover_updates_the_reply_context_during_machine_speech(
    category: AMDCategory, prompt: str, hook_finishes_after_cleanup: bool
) -> None:
    agent = CustomerAgent()
    async with running(agent=agent) as (detector, session, classifier, model):
        activity = session._activity
        assert activity is not None
        assert isinstance(activity.tts, FakeTTS)
        activity.tts.update_options(fake_audio_duration=10)
        activity.on_end_of_turn(end_of_turn(prompt))
        await classifier.request()
        classifier.prediction(1, category)
        await asyncio.wait_for(model.calls.get(), 2)
        await eventually(lambda: session.agent_state == "speaking")
        machine_speech = activity._current_speech
        assert machine_speech is not None and not machine_speech.done()
        activity.tts.update_options(fake_audio_duration=0.05)

        agent.hook_started.clear()
        if hook_finishes_after_cleanup:
            agent.hook_release.clear()
        activity.on_end_of_turn(end_of_turn("Can you hear me?"))
        await classifier.request()
        await asyncio.wait_for(agent.hook_started.wait(), 2)
        classifier.prediction(2, AMDCategory.HUMAN)
        if hook_finishes_after_cleanup:
            await asyncio.wait_for(detector.execute(), 2)
            assert session.amd is None
            agent.hook_release.set()

        call = await asyncio.wait_for(model.calls.get(), 2)
        messages = [m for m in call["chat_ctx"].items if m.type == "message"]
        controls = [m for m in messages if m.id.startswith(detector._control_prefix)]
        assert len(controls) == 1
        assert controls[0].extra["amd_stage"] == "human"
        assert controls[0] is messages[-1]
        assert "a human has answered" in controls[0].text_content
        assert any(m.text_content == "Customer hook edit" for m in messages)
        assert any(m.text_content == prompt for m in messages)
        assert any(m.text_content == "Can you hear me?" for m in messages)
        assert machine_speech.interrupted
        assert len(agent.hooks) == 2
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.category == AMDCategory.HUMAN
        assert not result.voicemail_message_played
        await eventually(lambda: activity._no_pending_speech)
        assert not any(m.id.startswith(detector._control_prefix) for m in agent.chat_ctx.items)

        activity.on_end_of_turn(end_of_turn("Thank you."))
        normal = await asyncio.wait_for(model.calls.get(), 2)
        assert not any(m.id.startswith(detector._control_prefix) for m in normal["chat_ctx"].items)
        assert detector._turn_id == 2


@pytest.mark.asyncio
async def test_menu_is_observability_only_and_dtmf_tool_is_temporary() -> None:
    async with running() as (detector, session, classifier, model):
        menus = []
        detector.on("amd_menu_observed", menus.append)
        info = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_IVR)
        context = llm.ChatContext()
        assert await detector._prepare_reply(info, context)
        before = session.current_agent.tools.copy()
        tools = detector._reply_tools(before)
        assert any(t.id == "send_dtmf_events" for t in tools)
        assert session.current_agent.tools == before
        response = await asyncio.wait_for(classifier.menu_requests.get(), 2)
        response.set_result(
            json.dumps(
                {
                    "menu": "Choose a department",
                    "options": [{"label": "Billing", "dtmf": "1"}],
                }
            )
        )
        await eventually(lambda: len(menus) == 1)
        assert menus[0].options[0].dtmf == "1"
        assert model.calls.empty()
        assert session._activity._no_pending_speech
        assert detector._category == AMDCategory.MACHINE_IVR


@pytest.mark.asyncio
@pytest.mark.parametrize("category", [AMDCategory.HUMAN, AMDCategory.MACHINE_UNAVAILABLE])
async def test_terminal_result_completes_once_and_releases_guard(category: AMDCategory) -> None:
    async with running() as (detector, session, classifier, _):
        events = []
        detector.on("amd_completed", events.append)
        first = asyncio.create_task(detector.execute())
        second = asyncio.create_task(detector.execute())
        info = await commit(detector, session, classifier)
        classifier.prediction(1, category)
        a, b = await asyncio.wait_for(asyncio.gather(first, second), 2)
        assert a is b
        assert a.reason == "finished"
        assert len(events) == 1
        assert session.amd is None
        assert session._activity._authorization_allowed.is_set()
        assert not detector._tasks
        assert await detector._prepare_reply(info, llm.ChatContext()) == (
            category == AMDCategory.HUMAN
        )
        await detector.aclose()
        assert len(events) == 1


@pytest.mark.asyncio
async def test_cancelled_execute_waiter_does_not_cancel_detection() -> None:
    async with running() as (detector, session, classifier, _):
        waiter = asyncio.create_task(detector.execute())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.HUMAN)
        assert (await asyncio.wait_for(detector.execute(), 2)).category == AMDCategory.HUMAN


@pytest.mark.asyncio
async def test_empty_turn_does_not_cancel_a_pending_terminal_prediction() -> None:
    async with running() as (detector, session, classifier, _):
        await commit(detector, session, classifier)
        detector._on_end_of_turn(end_of_turn(""))
        assert not detector._turns[2].decision.done()
        classifier.prediction(1, AMDCategory.HUMAN)
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.turn_id == 1
        assert result.category == AMDCategory.HUMAN


@pytest.mark.asyncio
async def test_timeout_rearms_and_late_result_cannot_change_a_newer_turn() -> None:
    async with running(inference_timeout=0.02) as (detector, session, classifier, _):
        events = []
        detector.on("amd_prediction", events.append)
        info = await commit(detector, session, classifier)
        assert await detector._prepare_reply(info, llm.ChatContext())
        assert events[-1].reason == "inference_timeout"
        assert detector._idle_timer is not None
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert classifier.responses[1].cancelled()
        assert not any(event.reason == "late_prediction" for event in events)
        assert detector._category == AMDCategory.UNCERTAIN
        classifier.prediction(2, AMDCategory.MACHINE_IVR)
        await eventually(
            lambda: detector._latest.turn_id == 2 and detector._latest.reason != "inference_timeout"
        )
        assert detector._category == AMDCategory.MACHINE_IVR


@pytest.mark.asyncio
async def test_stop_response_remains_owned_by_session() -> None:
    agent = CustomerAgent(stop=True)
    async with running(agent=agent, idle_timeout=0.03) as (detector, session, classifier, model):
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        assert (await asyncio.wait_for(detector.execute(), 2)).reason == "idle_timeout"
        assert len(agent.hooks) == 1
        assert model.calls.empty()


@pytest.mark.asyncio
async def test_invalid_model_output_falls_back_and_releases_the_reply() -> None:
    async with running() as (detector, session, classifier, _):
        info = await commit(detector, session, classifier)
        classifier.respond(1, "not JSON")
        assert await detector._prepare_reply(info, llm.ChatContext())
        assert detector._latest.reason == "inference_error"
        assert detector._category == AMDCategory.UNCERTAIN
        assert detector.enabled
        assert detector._idle_timer is not None


@pytest.mark.asyncio
async def test_false_interruption_settlement_rearms_idle_without_a_speech_handle() -> None:
    async with running(idle_timeout=0.03) as (detector, session, _, _):
        activity = session._activity
        activity._false_interruption_pending = True
        detector._rearm_idle()
        assert detector._idle_timer is None
        session.emit("agent_false_interruption", AgentFalseInterruptionEvent(resumed=False))
        activity._false_interruption_pending = False
        assert (await asyncio.wait_for(detector.execute(), 2)).reason == "idle_timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["inference_error", "reused", "prediction"])
async def test_unchanged_voicemail_rearms_idle_without_another_reply(reason: str) -> None:
    async with running(voicemail_idle_timeout=0.03) as (detector, session, classifier, _):
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert await detector._prepare_reply(first, llm.ChatContext())
        detector._on_user_speech_started()
        detector._on_user_speech_ended(0)
        if reason == "reused":
            second = end_of_turn("")
            detector._on_end_of_turn(second)
        else:
            second = await commit(detector, session, classifier)
            if reason == "inference_error":
                classifier.respond(2, "not JSON")
            else:
                classifier.prediction(2, AMDCategory.MACHINE_VM)
        assert not await detector._prepare_reply(second, llm.ChatContext())
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.reason == "idle_timeout"
        assert result.category == AMDCategory.MACHINE_VM


@pytest.mark.asyncio
async def test_three_uncertain_predictions_complete_detection() -> None:
    async with running() as (detector, session, classifier, _):
        for turn_id in range(1, 4):
            await commit(detector, session, classifier)
            classifier.prediction(turn_id, AMDCategory.UNCERTAIN)
            await eventually(lambda expected=turn_id: detector._turns[expected].decision.done())
        assert (await detector.execute()).reason == "max_uncertain_turns"


@pytest.mark.asyncio
async def test_three_timeouts_complete_detection() -> None:
    async with running(inference_timeout=0.01) as (detector, session, classifier, _):
        for turn_id in range(1, 4):
            await commit(detector, session, classifier)
            await eventually(lambda expected=turn_id: detector._turns[expected].decision.done())
        assert (await detector.execute()).reason == "inference_timeout"


@pytest.mark.asyncio
async def test_realtime_model_is_rejected_before_installing_guard() -> None:
    session = SimpleNamespace(_activity=SimpleNamespace(llm=Mock(spec=llm.RealtimeModel)))
    detector = AMD(session)
    with pytest.raises(ValueError, match="pipeline STT/LLM/TTS only"):
        await detector.__aenter__()


@pytest.mark.asyncio
async def test_supplied_model_is_not_closed_by_amd() -> None:
    async with running() as (detector, session, classifier, _):
        classifier.aclose = AsyncMock()
        await detector.aclose()
        classifier.aclose.assert_not_called()
        assert session.amd is None


@pytest.mark.asyncio
@pytest.mark.parametrize("wait_until_answered", [True, False])
async def test_sip_answer_gating_and_early_media(
    monkeypatch: pytest.MonkeyPatch, wait_until_answered: bool
) -> None:
    from livekit.agents.voice.amd import detector as module

    room = utils.EventEmitter()
    participant = SimpleNamespace(
        identity="callee",
        kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
        track_publications={"track": SimpleNamespace(sid="track")},
    )
    room.remote_participants = {"callee": participant}
    room_io = SimpleNamespace(room=room, set_participant=Mock())
    subscribed = asyncio.Event()
    answered = asyncio.Event()

    async def wait_for_track(**kwargs: Any) -> Any:
        assert kwargs["identity"] == "callee"
        assert kwargs["wait_for_subscription"]
        subscribed.set()
        return participant.track_publications["track"]

    async def wait_for_answer(*args: Any, **kwargs: Any) -> None:
        assert kwargs["attribute"] == "sip.callStatus"
        assert kwargs["value"] == "active"
        await answered.wait()

    monkeypatch.setattr(module, "wait_for_track_publication", wait_for_track)
    answer_mock = AsyncMock(side_effect=wait_for_answer)
    monkeypatch.setattr(module, "wait_for_participant_attribute", answer_mock)
    session = AgentSession(llm=FakeLLM(), turn_handling={"turn_detection": "manual"})
    await session.start(Agent(instructions="Hello"))
    session._room_io = room_io
    try:
        async with AMD(
            session,
            participant_identity="callee",
            wait_until_answered=wait_until_answered,
        ) as detector:
            await subscribed.wait()
            room_io.set_participant.assert_called_once_with("callee")
            if wait_until_answered:
                assert not detector.started
                assert detector._discard_pre_answer_audio
                assert detector._hard_timer is None
                answered.set()
            else:
                answer_mock.assert_not_called()
            await eventually(lambda: detector.started)
            assert not detector._discard_pre_answer_audio
            room.emit("participant_disconnected", participant)
            assert (await detector.execute()).reason == "participant_disconnected"
    finally:
        session._room_io = None
        await session.aclose()


@pytest.mark.asyncio
async def test_session_close_completes_amd_and_releases_resources() -> None:
    async with running() as (detector, session, classifier, _):
        await session.aclose()
        assert (await detector.execute()).reason == "cancelled"
        assert session.amd is None
        assert not detector._tasks
        assert not detector._tasks


@pytest.mark.asyncio
async def test_dtmf_digits_are_ordered_and_included_once_with_the_next_eot() -> None:
    async with running() as (detector, _, classifier, _):
        detector.notify_dtmf_sent("1")
        detector.notify_dtmf_sent("2#")
        assert classifier.requests.empty()
        assert detector._category == AMDCategory.UNCERTAIN
        assert detector._turn_id == 0

        detector._on_end_of_turn(end_of_turn())
        detector.notify_dtmf_sent("3")
        first = await classifier.request()
        assert first.dtmf_digits == "12#"
        detector._on_end_of_turn(end_of_turn())
        second = await classifier.request()
        assert second.dtmf_digits == "3"
        detector._on_end_of_turn(end_of_turn())
        assert (await classifier.request()).dtmf_digits == ""


@pytest.mark.asyncio
async def test_dtmf_notification_rejects_invalid_digits_and_ignores_completed_runs() -> None:
    async with running() as (detector, _, classifier, _):
        detector.notify_dtmf_sent("1")
        for digits in ("", "x", "1 2", "1\n"):
            with pytest.raises(ValueError, match="digits must contain only"):
                detector.notify_dtmf_sent(digits)
        detector._on_end_of_turn(end_of_turn())
        assert (await classifier.request()).dtmf_digits == "1"

        detector.notify_dtmf_sent("2")
        classifier.prediction(1, AMDCategory.HUMAN)
        await detector.execute()
        detector.notify_dtmf_sent("3")
        assert detector._pending_dtmf_digits == ""


@pytest.mark.asyncio
async def test_dtmf_tool_works_without_amd() -> None:
    from livekit.agents.beta.tools.send_dtmf import send_dtmf_events
    from livekit.agents.beta.workflows.utils import DtmfEvent

    publisher = AsyncMock()
    tool_session = SimpleNamespace(
        amd=None,
        room_io=SimpleNamespace(
            room=SimpleNamespace(local_participant=SimpleNamespace(publish_dtmf=publisher))
        ),
    )
    result = await send_dtmf_events(SimpleNamespace(session=tool_session), [DtmfEvent.ONE])
    publisher.assert_awaited_once_with(code=1, digit="1")
    assert result == "Successfully sent DTMF events: 1"


@pytest.mark.asyncio
async def test_dtmf_tool_reports_successful_prefix_when_a_later_digit_fails() -> None:
    from livekit.agents.beta.tools.send_dtmf import send_dtmf_events
    from livekit.agents.beta.workflows.utils import DtmfEvent

    async with running() as (detector, session, classifier, _):
        publisher = AsyncMock(side_effect=[None, RuntimeError("publish failed")])
        tool_session = SimpleNamespace(
            amd=detector,
            room_io=SimpleNamespace(
                room=SimpleNamespace(local_participant=SimpleNamespace(publish_dtmf=publisher))
            ),
        )
        result = await send_dtmf_events(
            SimpleNamespace(session=tool_session), [DtmfEvent.ONE, DtmfEvent.TWO]
        )
        assert result.startswith("Failed to send DTMF event: 2.")
        assert publisher.await_count == 2
        detector._on_end_of_turn(end_of_turn())
        assert (await classifier.request()).dtmf_digits == "1"


@pytest.mark.asyncio
async def test_dtmf_tool_does_not_report_an_incomplete_publish() -> None:
    from livekit.agents.beta.tools.send_dtmf import send_dtmf_events
    from livekit.agents.beta.workflows.utils import DtmfEvent

    started = asyncio.Event()

    async def publish(**kwargs: Any) -> None:
        started.set()
        await asyncio.Future()

    detector = SimpleNamespace(notify_dtmf_sent=Mock())
    tool_session = SimpleNamespace(
        amd=detector,
        room_io=SimpleNamespace(
            room=SimpleNamespace(local_participant=SimpleNamespace(publish_dtmf=publish))
        ),
    )
    task = asyncio.create_task(
        send_dtmf_events(SimpleNamespace(session=tool_session), [DtmfEvent.ONE])
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    detector.notify_dtmf_sent.assert_not_called()


@pytest.mark.asyncio
async def test_dtmf_tool_does_not_attach_an_old_send_to_a_new_amd_run() -> None:
    from livekit.agents.beta.tools.send_dtmf import send_dtmf_events
    from livekit.agents.beta.workflows.utils import DtmfEvent

    old = SimpleNamespace(notify_dtmf_sent=Mock())
    new = SimpleNamespace(notify_dtmf_sent=Mock())
    tool_session = SimpleNamespace(amd=old)

    async def publish(**kwargs: Any) -> None:
        tool_session.amd = new

    tool_session.room_io = SimpleNamespace(
        room=SimpleNamespace(local_participant=SimpleNamespace(publish_dtmf=publish))
    )
    await send_dtmf_events(SimpleNamespace(session=tool_session), [DtmfEvent.ONE])
    old.notify_dtmf_sent.assert_not_called()
    new.notify_dtmf_sent.assert_not_called()
