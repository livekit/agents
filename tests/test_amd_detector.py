from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from opentelemetry import trace

from livekit import rtc
from livekit.agents import (
    AMD,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    AMDPredictionEvent,
    StopResponse,
    llm,
    utils,
)
from livekit.agents.types import APIConnectOptions
from livekit.agents.voice._turn_hooks import TurnHooks
from livekit.agents.voice.amd import AMDCategory
from livekit.agents.voice.amd.detector import (
    _HUMAN_INSTRUCTIONS,
    DEFAULT_IVR_INSTRUCTIONS,
    DEFAULT_SCREENING_INSTRUCTIONS,
    DEFAULT_VOICEMAIL_INSTRUCTIONS,
)
from livekit.agents.voice.audio_recognition import (
    _EndOfTurnInfo,
    _EndOfTurnMetrics,
    _PreemptiveGenerationInfo,
)
from livekit.agents.voice.events import UserInputTranscribedEvent
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_io import FakeAudioOutput
from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_tts import FakeTTS

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.mark.parametrize(
    ("category", "previous", "changed"),
    [
        (AMDCategory.UNCERTAIN, None, False),
        (AMDCategory.HUMAN, None, True),
        (AMDCategory.MACHINE_VM, None, True),
        (AMDCategory.MACHINE_VM, AMDCategory.UNCERTAIN, True),
        (AMDCategory.MACHINE_VM, AMDCategory.MACHINE_VM, False),
        (AMDCategory.MACHINE_IVR, AMDCategory.MACHINE_VM, True),
        (AMDCategory.HUMAN, AMDCategory.MACHINE_IVR, True),
    ],
)
def test_prediction_state_changed_is_derived(
    category: AMDCategory, previous: AMDCategory | None, changed: bool
) -> None:
    event = AMDPredictionEvent(
        category=category,
        prev_turn_category=previous,
        prev_stage_category=AMDCategory.UNCERTAIN,
        reason="prediction",
        transcript="hello",
        speech_duration=0,
        delay=0,
    )
    assert event.state_changed is changed
    assert event.model_dump()["state_changed"] is changed
    assert json.loads(event.model_dump_json())["state_changed"] is changed
    assert "state_changed" not in AMDPredictionEvent.model_fields

    event.prev_turn_category = category
    assert not event.state_changed
    event.category = AMDCategory.HUMAN if category != AMDCategory.HUMAN else AMDCategory.MACHINE_IVR
    assert event.state_changed
    assert event.model_dump()["state_changed"] is True


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
            tools=kwargs["tools"],
            conn_options=APIConnectOptions(max_retry=0),
        )
        self.response = response

    async def _run(self) -> None:
        text = await self.response
        self._event_ch.send_nowait(
            llm.ChatChunk(
                id="amd-test",
                delta=llm.ChoiceDelta(
                    role="assistant",
                    tool_calls=[
                        llm.FunctionToolCall(
                            name="record_result", arguments=text, call_id="amd-test"
                        )
                    ],
                ),
            )
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


def commit_turn(detector: AMD, info: _EndOfTurnInfo) -> TurnHooks:
    hooks = detector._session._user_turn_committed(
        info.new_transcript, info.metrics.end_of_turn_delay
    )
    assert hooks is not None
    return hooks


def speech_started(detector: AMD) -> None:
    detector._session._update_user_state("speaking")


def speech_ended(detector: AMD, silence_duration: float) -> None:
    detector._session._update_user_state(
        "listening", last_speaking_time=time.time() - silence_duration
    )


def transcribe(detector: AMD, text: str) -> None:
    detector._session._user_input_transcribed(
        UserInputTranscribedEvent(transcript=text, is_final=True)
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
            **{"machine_silence_threshold": 0, "stt": None, **options},
        )
        async with detector:
            await eventually(lambda: detector.started)
            yield detector, session, classifier, model
    finally:
        await session.aclose()


async def commit(
    detector: AMD, session: AgentSession, classifier: ClassifierLLM, *, reply: bool = False
) -> TurnHooks | None:
    info = end_of_turn()
    hooks = None
    if reply:
        assert session._activity is not None
        session._activity.on_end_of_turn(info)
    else:
        hooks = commit_turn(detector, info)
    request = await classifier.request()
    assert request.turn_id == detector._fsm.turn_id
    return hooks


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_reply", [False, True])
async def test_turn_hooks_control_replies_without_amd(allow_reply: bool) -> None:
    @llm.function_tool
    def reply_tool() -> str:
        """A tool available only to the generated reply."""
        return "done"

    replies: list[SpeechHandle] = []
    calls: list[str] = []

    class Hooks:
        def on_user_turn_committed(
            self, transcript: str, end_of_turn_delay: float | None
        ) -> TurnHooks:
            assert transcript == "hello"
            assert end_of_turn_delay is None
            assert agent.hooks == []
            calls.append("user_turn_committed")
            return self

        def on_reply_generation(
            self, tools: list[llm.Tool | llm.Toolset]
        ) -> list[llm.Tool | llm.Toolset]:
            calls.append("reply_generation")
            return [*tools, reply_tool]

        async def should_reply(self, chat_ctx: llm.ChatContext) -> bool:
            assert len(agent.hooks) == 1
            calls.append("should_reply")
            chat_ctx.add_message(role="system", content="Turn instructions")
            return allow_reply

        def on_agent_turn_committed(self, handle: SpeechHandle) -> None:
            calls.append("agent_turn_committed")
            replies.append(handle)

    model = RecordingLLM()
    agent = CustomerAgent()
    session = AgentSession(llm=model, turn_handling={"turn_detection": "manual"})
    await session.start(agent)
    try:
        assert session.amd is None
        session._turn_hooks = Hooks()
        activity = session._activity
        assert activity is not None
        info = end_of_turn()
        activity.on_end_of_turn(info)
        await asyncio.wait_for(activity._user_turn_completed_atask, 2)

        if allow_reply:
            assert calls == [
                "user_turn_committed",
                "should_reply",
                "reply_generation",
                "agent_turn_committed",
            ]
            call = await asyncio.wait_for(model.calls.get(), 2)
            assert any(
                m.text_content == "Turn instructions"
                for m in call["chat_ctx"].items
                if m.type == "message"
            )
            assert reply_tool in call["tools"]
            assert len(replies) == 1
            await asyncio.wait_for(replies[0], 2)
        else:
            assert calls == ["user_turn_committed", "should_reply"]
            assert model.calls.empty()
            assert replies == []
        assert len(agent.hooks) == 1
        assert not any(
            m.text_content == "Turn instructions"
            for m in agent.chat_ctx.items
            if m.type == "message"
        )
    finally:
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["adopted", "replaced", "suppressed"])
async def test_preemptive_generation_commits_only_the_accepted_reply(outcome: str) -> None:
    hooks = Mock(spec=TurnHooks)
    hooks.on_user_turn_committed.return_value = hooks
    hooks.on_reply_generation.side_effect = lambda tools: tools

    async def should_reply(chat_ctx: llm.ChatContext) -> bool:
        if outcome == "replaced":
            chat_ctx.add_message(role="system", content=_HUMAN_INSTRUCTIONS)
        return outcome != "suppressed"

    hooks.should_reply.side_effect = should_reply
    model = RecordingLLM()
    session = AgentSession(llm=model, turn_handling={"turn_detection": "manual"})
    await session.start(Agent(instructions="Call about an appointment."))
    try:
        session._turn_hooks = hooks
        session.options.preemptive_generation["enabled"] = True
        activity = session._activity
        assert activity is not None
        activity.on_preemptive_generation(
            _PreemptiveGenerationInfo(
                new_transcript="hello", transcript_confidence=1, started_speaking_at=None
            )
        )
        assert activity._preemptive_generation is not None
        preemptive = activity._preemptive_generation.speech_handle
        await asyncio.wait_for(model.calls.get(), 2)

        hooks.on_reply_generation.assert_called_once()
        hooks.on_user_turn_committed.assert_not_called()
        hooks.should_reply.assert_not_awaited()
        hooks.on_agent_turn_committed.assert_not_called()

        activity.on_end_of_turn(end_of_turn())
        await asyncio.wait_for(activity._user_turn_completed_atask, 2)

        hooks.on_user_turn_committed.assert_called_once_with("hello", None)
        hooks.should_reply.assert_awaited_once()
        if outcome == "suppressed":
            hooks.on_agent_turn_committed.assert_not_called()
        else:
            hooks.on_agent_turn_committed.assert_called_once()
            reply = hooks.on_agent_turn_committed.call_args.args[0]
            assert (reply is preemptive) == (outcome == "adopted")
            await asyncio.wait_for(reply, 2)
        assert preemptive.interrupted == (outcome != "adopted")
        if outcome == "replaced":
            await asyncio.wait_for(model.calls.get(), 2)
        assert hooks.on_reply_generation.call_count == (2 if outcome == "replaced" else 1)
        assert model.calls.empty()
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_cancel_pending_speech_includes_preemptive_and_queued_speech() -> None:
    async with running() as (_, session, _, _):
        activity = session._activity
        assert activity is not None
        session.options.preemptive_generation["enabled"] = True
        activity.on_preemptive_generation(
            _PreemptiveGenerationInfo(
                new_transcript="hello", transcript_confidence=1, started_speaking_at=None
            )
        )
        assert activity._preemptive_generation is not None
        preemptive = activity._preemptive_generation.speech_handle
        queued = session.say("Please call me back.")

        activity._cancel_pending_speech()

        await asyncio.wait_for(asyncio.gather(preemptive, queued), 2)
        assert preemptive.interrupted and queued.interrupted
        assert activity._preemptive_generation is None
        assert session.output.audio.captured_playout_segments == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["reused", "inference_error", "inference_timeout"])
async def test_fallback_prediction_has_the_current_previous_category(reason: str) -> None:
    async with running(inference_timeout=0.02) as (detector, session, classifier, _):
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        first = await asyncio.wait_for(detector._wait_for_decision(1), 2)
        assert first.state_changed

        if reason == "reused":
            commit_turn(detector, end_of_turn(""))
        else:
            await commit(detector, session, classifier)
            if reason == "inference_error":
                classifier.respond(2, "invalid JSON")
        fallback = await asyncio.wait_for(detector._wait_for_decision(2), 2)
        assert fallback.reason == reason
        assert fallback.category == fallback.prev_turn_category == AMDCategory.MACHINE_VM
        assert fallback.prev_stage_category == AMDCategory.UNCERTAIN
        assert not fallback.state_changed
        assert fallback.model_dump()["state_changed"] is False


@pytest.mark.asyncio
async def test_customer_hook_and_prediction_overlap_controls_are_temporary() -> None:
    agent = CustomerAgent()
    agent.hook_release.clear()
    async with running(agent=agent) as (detector, session, classifier, model):
        completed = asyncio.create_task(detector.execute())
        info = end_of_turn()
        info.metrics.end_of_turn_delay = 0.5
        session._activity.on_end_of_turn(info)
        assert detector._fsm.turn_id == 1
        assert not agent.hook_started.is_set()
        request = await classifier.request()
        await agent.hook_started.wait()
        assert request.turn_id == 1
        assert request.transcript == "hello"
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await eventually(lambda: detector._fsm.decision(1) is not None)
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
@pytest.mark.parametrize(
    ("category", "instructions"),
    [
        (AMDCategory.MACHINE_SCREENING, DEFAULT_SCREENING_INSTRUCTIONS),
        (AMDCategory.MACHINE_VM, DEFAULT_VOICEMAIL_INSTRUCTIONS),
        (AMDCategory.HUMAN, _HUMAN_INSTRUCTIONS),
    ],
)
async def test_turn_hooks_keep_their_turn_while_an_older_hook_is_blocked(
    category: AMDCategory, instructions: str
) -> None:
    agent = CustomerAgent()
    agent.hook_release.clear()
    async with running(agent=agent) as (detector, session, classifier, model):
        activity = session._activity
        assert activity is not None
        try:
            await commit(detector, session, classifier, reply=True)
            await asyncio.wait_for(agent.hook_started.wait(), 2)
            classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
            await eventually(lambda: detector._fsm.decision(1) is not None)

            activity.on_end_of_turn(end_of_turn("Can you hear me?"))
            assert (await classifier.request()).turn_id == 2
            classifier.prediction(2, category)
            await eventually(lambda: detector._fsm.decision(2) is not None)
            if category == AMDCategory.HUMAN:
                await asyncio.wait_for(detector.execute(), 2)
                assert session.amd is None
            assert len(agent.hooks) == 1
            assert model.calls.empty()
        finally:
            agent.hook_release.set()

        await asyncio.wait_for(activity._user_turn_completed_atask, 2)
        call = await asyncio.wait_for(model.calls.get(), 2)
        messages = [m for m in call["chat_ctx"].items if m.type == "message"]
        assert messages[-1].text_content == instructions
        assert any(m.text_content == "Can you hear me?" for m in messages)
        assert len(agent.hooks) == 2
        if category == AMDCategory.MACHINE_VM:
            await eventually(lambda: detector._fsm.voicemail_message_played)
        await eventually(lambda: activity._no_pending_speech)
        assert model.calls.empty()


@pytest.mark.asyncio
async def test_turn_ids_are_local_to_each_amd_run() -> None:
    session = AgentSession(llm=RecordingLLM(), turn_handling={"turn_detection": "manual"})
    await session.start(CustomerAgent())
    try:
        activity = session._activity
        assert activity is not None
        for _ in range(2):
            activity.on_end_of_turn(end_of_turn(skip_reply=True))
            await asyncio.wait_for(activity._user_turn_completed_atask, 2)
            classifier = ClassifierLLM()
            async with AMD(
                session, llm=classifier, stt=None, machine_silence_threshold=0
            ) as detector:
                await eventually(lambda: detector.started)
                hooks = await commit(detector, session, classifier)
                assert detector._fsm.turn_id == 1
                classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
                assert await hooks.should_reply(llm.ChatContext())
    finally:
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "category", [AMDCategory.MACHINE_SCREENING, AMDCategory.MACHINE_UNAVAILABLE]
)
async def test_amd_turn_hooks_retain_the_adopted_user_turn_span(category: AMDCategory) -> None:
    async with running() as (detector, session, classifier, model):
        span = Mock(spec=trace.Span)
        span.get_span_context.return_value = trace.INVALID_SPAN_CONTEXT
        info = end_of_turn()
        info.user_turn_span = span
        session._activity.on_end_of_turn(info)
        await classifier.request()
        await session.current_agent.hook_started.wait()
        await asyncio.sleep(0)

        assert info.user_turn_span_adopted
        span.end.assert_not_called()
        assert model.calls.empty()

        classifier.prediction(1, category)
        await asyncio.wait_for(session._activity._user_turn_completed_atask, 2)
        span.end.assert_called_once_with()
        assert not info.user_turn_span_adopted
        if category == AMDCategory.MACHINE_UNAVAILABLE:
            assert (await detector.execute()).category == category
            assert model.calls.empty()
        else:
            call = await asyncio.wait_for(model.calls.get(), 2)
            assert call["chat_ctx"].items[-1].text_content == DEFAULT_SCREENING_INSTRUCTIONS


@pytest.mark.asyncio
async def test_voicemail_sends_one_message_per_stage_and_records_playback() -> None:
    async with running() as (detector, session, classifier, model):
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(model.calls.get(), 2)
        await eventually(lambda: detector._fsm.voicemail_message_played)
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(2, AMDCategory.MACHINE_VM)
        await eventually(lambda: detector._fsm.decision(2) is not None)
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
        hard_deadline = detector._fsm._hard_deadline
        assert detector._fsm._idle_deadline is not None
        assert detector._fsm._idle_deadline - loop.time() == pytest.approx(10, abs=0.1)
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(model.calls.get(), 2)
        await eventually(
            lambda: detector._fsm.voicemail_message_played and detector._fsm._idle_deadline
        )
        timer = detector._timer
        assert timer is not None
        assert timer.when() - loop.time() == pytest.approx(60, abs=0.1)
        assert detector._fsm._hard_deadline == hard_deadline
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
        await eventually(
            lambda: detector._fsm.voicemail_message_played and detector._fsm._idle_deadline
        )
        timer = detector._timer
        await asyncio.sleep(0.06)
        assert detector.enabled

        speech_started(detector)
        assert timer.cancelled()
        assert detector._fsm._idle_deadline is None
        speech_ended(detector, 0)
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
        assert await first.should_reply(llm.ChatContext())
        second = await commit(detector, session, classifier)
        await second.should_reply(llm.ChatContext())
        timer = detector._timer
        assert timer is not None
        classifier.prediction(2, category)
        await eventually(
            lambda: (
                detector._fsm._latest.turn_id == 2
                and detector._fsm._latest.reason != "inference_timeout"
            )
        )
        assert timer.cancelled()
        assert detector._fsm._idle_deadline is not None and detector._timer is not timer
        expected = 2 if category == AMDCategory.MACHINE_VM else 1
        assert detector._fsm._idle_deadline - asyncio.get_running_loop().time() == pytest.approx(
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
        assert detector._fsm._idle_deadline is None


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
            assert session._turn_hooks is None
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
        assert detector._fsm.turn_id == 2


@pytest.mark.asyncio
async def test_menu_is_observability_only_and_dtmf_tool_is_temporary() -> None:
    async with running() as (detector, session, classifier, model):
        menus = []
        detector.on("amd_menu_observed", menus.append)
        hooks = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_IVR)
        context = llm.ChatContext()
        assert await hooks.should_reply(context)
        before = session.current_agent.tools.copy()
        tools = hooks.on_reply_generation(before)
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
        assert detector._fsm.category == AMDCategory.MACHINE_IVR


@pytest.mark.asyncio
@pytest.mark.parametrize("category", [AMDCategory.HUMAN, AMDCategory.MACHINE_UNAVAILABLE])
async def test_terminal_result_completes_once_and_releases_turn_hooks(
    category: AMDCategory,
) -> None:
    async with running() as (detector, session, classifier, _):
        assert session._turn_hooks is not None
        events = []
        detector.on("amd_completed", events.append)
        first = asyncio.create_task(detector.execute())
        second = asyncio.create_task(detector.execute())
        hooks = await commit(detector, session, classifier)
        classifier.prediction(1, category)
        a, b = await asyncio.wait_for(asyncio.gather(first, second), 2)
        assert a is b
        assert a.reason == "finished"
        assert len(events) == 1
        assert session.amd is None
        assert session._turn_hooks is None
        assert session._activity._authorization_allowed.is_set()
        assert not detector._tasks
        for event, callback in (
            ("user_state_changed", detector._on_user_state_changed),
            ("user_input_transcribed", detector._on_user_input_transcribed),
        ):
            assert callback not in session._events[event]
        assert await hooks.should_reply(llm.ChatContext()) == (category == AMDCategory.HUMAN)
        await detector.aclose()
        assert len(events) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("playback_started", [False, True])
async def test_late_unavailable_cancels_only_replies_without_playback(
    playback_started: bool,
) -> None:
    async with running(inference_timeout=0.02) as (detector, session, classifier, model):
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await detector._wait_for_decision(1)

        activity = session._activity
        assert activity is not None
        assert isinstance(activity.tts, FakeTTS)
        activity.tts.update_options(
            fake_audio_duration=10,
            fake_timeout=None if playback_started else 2,
        )
        await commit(detector, session, classifier, reply=True)
        await asyncio.wait_for(model.calls.get(), 2)
        await eventually(lambda: activity._current_speech is not None)
        speech = activity._current_speech
        assert speech is not None
        if playback_started:
            await eventually(lambda: session.agent_state == "speaking")
        else:
            assert session.agent_state != "speaking"

        classifier.prediction(2, AMDCategory.MACHINE_UNAVAILABLE)
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.category == AMDCategory.MACHINE_UNAVAILABLE
        assert speech.interrupted is not playback_started
        if playback_started:
            assert session.agent_state == "speaking"
        else:
            await eventually(speech.done)


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
async def test_cancelled_reply_waiter_does_not_block_other_turns() -> None:
    async with running() as (detector, session, classifier, _):
        await commit(detector, session, classifier)
        cancelled = asyncio.create_task(detector._wait_for_decision(1))
        surviving = asyncio.create_task(detector._wait_for_decision(1))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled

        await commit(detector, session, classifier)
        current = asyncio.create_task(detector._wait_for_decision(2))
        assert (await asyncio.wait_for(surviving, 2)).reason == "superseded"
        assert not current.done()
        classifier.prediction(2, AMDCategory.HUMAN)
        assert (await asyncio.wait_for(current, 2)).category == AMDCategory.HUMAN


@pytest.mark.asyncio
async def test_empty_turn_does_not_cancel_a_pending_terminal_prediction() -> None:
    async with running() as (detector, session, classifier, _):
        await commit(detector, session, classifier)
        commit_turn(detector, end_of_turn(""))
        assert detector._fsm.decision(2) is None
        classifier.prediction(1, AMDCategory.HUMAN)
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.turn_id == 1
        assert result.category == AMDCategory.HUMAN


@pytest.mark.asyncio
async def test_timeout_rearms_and_late_result_cannot_change_a_newer_turn() -> None:
    async with running(inference_timeout=0.02) as (detector, session, classifier, _):
        events = []
        detector.on("amd_prediction", events.append)
        hooks = await commit(detector, session, classifier)
        assert await hooks.should_reply(llm.ChatContext())
        assert events[-1].reason == "inference_timeout"
        assert detector._fsm._idle_deadline is not None
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert classifier.responses[1].cancelled()
        assert not any(event.reason == "late_prediction" for event in events)
        assert detector._fsm.category == AMDCategory.UNCERTAIN
        classifier.prediction(2, AMDCategory.MACHINE_IVR)
        await eventually(
            lambda: (
                detector._fsm._latest.turn_id == 2
                and detector._fsm._latest.reason != "inference_timeout"
            )
        )
        assert detector._fsm.category == AMDCategory.MACHINE_IVR


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
        hooks = await commit(detector, session, classifier)
        classifier.respond(1, "not JSON")
        assert await hooks.should_reply(llm.ChatContext())
        assert detector._fsm._latest.reason == "inference_error"
        assert detector._fsm.category == AMDCategory.UNCERTAIN
        assert detector.enabled
        assert detector._fsm._idle_deadline is not None


@pytest.mark.asyncio
async def test_false_interruption_settlement_rearms_idle_without_a_speech_handle() -> None:
    async with running(idle_timeout=0.03) as (detector, session, _, _):
        activity = session._activity
        activity._false_interruption_pending = True
        detector._reschedule_timer()
        assert detector._fsm._idle_deadline is None
        session.emit("agent_false_interruption", AgentFalseInterruptionEvent(resumed=False))
        activity._false_interruption_pending = False
        assert (await asyncio.wait_for(detector.execute(), 2)).reason == "idle_timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["inference_error", "reused", "prediction"])
async def test_unchanged_voicemail_rearms_idle_without_another_reply(reason: str) -> None:
    async with running(voicemail_idle_timeout=0.03) as (detector, session, classifier, _):
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert await first.should_reply(llm.ChatContext())
        speech_started(detector)
        speech_ended(detector, 0)
        if reason == "reused":
            second = commit_turn(detector, end_of_turn(""))
        else:
            second = await commit(detector, session, classifier)
            if reason == "inference_error":
                classifier.respond(2, "not JSON")
            else:
                classifier.prediction(2, AMDCategory.MACHINE_VM)
        assert not await second.should_reply(llm.ChatContext())
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.reason == "idle_timeout"
        assert result.category == AMDCategory.MACHINE_VM


@pytest.mark.asyncio
async def test_three_uncertain_predictions_complete_detection() -> None:
    async with running() as (detector, session, classifier, _):
        for turn_id in range(1, 4):
            await commit(detector, session, classifier)
            classifier.prediction(turn_id, AMDCategory.UNCERTAIN)
            await eventually(lambda expected=turn_id: detector._fsm.decision(expected) is not None)
        assert (await detector.execute()).reason == "max_uncertain_turns"


@pytest.mark.asyncio
async def test_three_timeouts_complete_detection() -> None:
    async with running(inference_timeout=0.01) as (detector, session, classifier, _):
        for turn_id in range(1, 4):
            await commit(detector, session, classifier)
            await eventually(lambda expected=turn_id: detector._fsm.decision(expected) is not None)
        assert (await detector.execute()).reason == "inference_timeout"


@pytest.mark.asyncio
async def test_realtime_model_is_rejected_before_installing_turn_hooks() -> None:
    session = SimpleNamespace(_activity=SimpleNamespace(llm=Mock(spec=llm.RealtimeModel)))
    detector = AMD(session, llm=None, stt=None)
    with pytest.raises(ValueError, match="does not support realtime models"):
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
            llm=None,
            stt=None,
            participant_identity="callee",
            wait_until_answered=wait_until_answered,
        ) as detector:
            await subscribed.wait()
            room_io.set_participant.assert_called_once_with("callee")
            if wait_until_answered:
                assert detector.enabled
                assert not detector.started
                assert detector._fsm._hard_deadline is None
                answered.set()
            else:
                answer_mock.assert_not_called()
            await eventually(lambda: detector.started)
            room.emit("participant_disconnected", participant)
            assert (await detector.execute()).reason == "participant_disconnected"
            assert not detector.enabled
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
        assert detector._fsm.category == AMDCategory.UNCERTAIN
        assert detector._fsm.turn_id == 0

        commit_turn(detector, end_of_turn())
        detector.notify_dtmf_sent("3")
        first = await classifier.request()
        assert first.dtmf_digits == "12#"
        commit_turn(detector, end_of_turn())
        second = await classifier.request()
        assert second.dtmf_digits == "3"
        commit_turn(detector, end_of_turn())
        assert (await classifier.request()).dtmf_digits == ""


@pytest.mark.asyncio
async def test_dtmf_notification_rejects_invalid_digits_and_ignores_completed_runs() -> None:
    async with running() as (detector, _, classifier, _):
        detector.notify_dtmf_sent("1")
        for digits in ("", "x", "1 2", "1\n"):
            with pytest.raises(ValueError, match="digits must contain only"):
                detector.notify_dtmf_sent(digits)
        commit_turn(detector, end_of_turn())
        assert (await classifier.request()).dtmf_digits == "1"

        detector.notify_dtmf_sent("2")
        classifier.prediction(1, AMDCategory.HUMAN)
        await detector.execute()
        detector.notify_dtmf_sent("3")
        assert detector._fsm._pending_dtmf_digits == ""


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
        commit_turn(detector, end_of_turn())
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
