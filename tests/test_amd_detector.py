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
    AMDLifecycle,
    AMDPredictionEvent,
    APIConnectionError,
    StopResponse,
    llm,
    utils,
)
from livekit.agents.types import APIConnectOptions
from livekit.agents.voice._turn_hooks import TurnHooks
from livekit.agents.voice.amd import AMDCategory, _inference
from livekit.agents.voice.amd.detector import (
    _DEFAULT_HUMAN_INSTRUCTIONS,
    _DEFAULT_IVR_INSTRUCTIONS,
    _DEFAULT_SCREENING_INSTRUCTIONS,
    _DEFAULT_VOICEMAIL_INSTRUCTIONS,
)
from livekit.agents.voice.audio_recognition import (
    _EndOfTurnInfo,
    _EndOfTurnMetrics,
    _PreemptiveGenerationInfo,
)
from livekit.agents.voice.events import UserInputTranscribedEvent
from livekit.agents.voice.speech_handle import SpeechHandle

from .amd_test_utils import detector_clock  # noqa: F401
from .fake_io import FakeAudioOutput
from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_stt import DrainingStream, DrainingSTT
from .fake_tts import FakeTTS

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent, pytest.mark.virtual_time]


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
            self.responses[data["current_turn"]["turn_id"]] = response
            self.requests.put_nowait(SimpleNamespace(**data))
        else:
            self.menu_requests.put_nowait(response)
        return ControlledStream(self, response=response, **kwargs)

    async def request(self) -> Any:
        return await asyncio.wait_for(self.requests.get(), 2)

    def prediction(self, turn_id: int, category: AMDCategory, **kwargs: Any) -> None:
        self.respond(turn_id, json.dumps({"category": category.value}))

    def respond(self, turn_id: int, arguments: str) -> None:
        response = self.responses[turn_id]
        if not response.done():
            response.set_result(arguments)


class ControlledStream(llm.LLMStream):
    def __init__(
        self, model: ClassifierLLM, *, response: asyncio.Future[str], **kwargs: Any
    ) -> None:
        super().__init__(
            model,
            chat_ctx=kwargs["chat_ctx"],
            tools=kwargs["tools"],
            conn_options=kwargs.get("conn_options", APIConnectOptions(max_retry=0)),
        )
        self.response = response

    async def _run(self) -> None:
        arguments = await self.response
        self._event_ch.send_nowait(
            llm.ChatChunk(
                id="amd-test",
                delta=llm.ChoiceDelta(
                    role="assistant",
                    tool_calls=[
                        llm.FunctionToolCall(
                            name="record_result", arguments=arguments, call_id="amd-test"
                        )
                    ],
                ),
            )
        )


class RecordingLLM(FakeLLM):
    def __init__(self) -> None:
        super().__init__(
            fake_responses=[
                FakeLLMResponse(input=prompt, content="Please call me back.", ttft=0, duration=0)
                for prompt in (
                    _DEFAULT_SCREENING_INSTRUCTIONS,
                    _DEFAULT_VOICEMAIL_INSTRUCTIONS,
                    _DEFAULT_IVR_INSTRUCTIONS,
                    _DEFAULT_HUMAN_INSTRUCTIONS,
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


def end_of_turn(transcript: str = "hello", *, skip_reply: bool = False) -> _EndOfTurnInfo:
    return _EndOfTurnInfo(
        skip_reply=skip_reply,
        new_transcript=transcript,
        transcript_confidence=1,
        metrics=_EndOfTurnMetrics(None, None, None, None),
    )


def push_audio(detector: AMD, stt: DrainingSTT) -> DrainingStream:
    detector.push_audio(rtc.AudioFrame.create(16000, 1, 320))
    return stt.streams[-1]


def commit_turn(detector: AMD, info: _EndOfTurnInfo) -> TurnHooks:
    hooks = detector._session._turn_hooks
    assert hooks is not None
    return hooks.on_user_turn_committed(info.new_transcript, info.metrics.end_of_turn_delay)


def speech_started(detector: AMD) -> None:
    detector._session._update_user_state("speaking")


def speech_ended(detector: AMD, silence_duration: float) -> None:
    detector._session._update_user_state(
        "listening", last_speaking_time=time.time() - silence_duration
    )


def transcribe(detector: AMD, transcript: str) -> None:
    detector._session._user_input_transcribed(
        UserInputTranscribedEvent(transcript=transcript, is_final=True)
    )


async def eventually(predicate: Any) -> None:
    async def wait() -> None:
        while not predicate():
            await asyncio.sleep(0.001)

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
            await eventually(lambda: detector.lifecycle is AMDLifecycle.ACTIVE)
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
    assert request.current_turn["turn_id"] == detector._fsm.turn_id
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

        def on_user_turn_completed(self) -> None:
            assert activity._user_turn_completed_atask.done()
            calls.append("user_turn_completed")

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
                "user_turn_completed",
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
            assert calls == ["user_turn_committed", "should_reply", "user_turn_completed"]
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
            chat_ctx.add_message(role="system", content=_DEFAULT_HUMAN_INSTRUCTIONS)
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
        hooks.on_user_turn_completed.assert_not_called()
        hooks.should_reply.assert_not_awaited()
        hooks.on_agent_turn_committed.assert_not_called()

        activity.on_end_of_turn(end_of_turn())
        await asyncio.wait_for(activity._user_turn_completed_atask, 2)

        hooks.on_user_turn_committed.assert_called_once_with("hello", None)
        hooks.on_user_turn_completed.assert_called_once_with()
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

        activity._cancel_pending_speeches()

        await asyncio.wait_for(asyncio.gather(preemptive, queued), 2)
        assert preemptive.interrupted and queued.interrupted
        assert activity._preemptive_generation is None
        assert session.output.audio.captured_playout_segments == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["reused", "inference_error", "inference_timeout"])
async def test_fallback_prediction_has_the_current_previous_category(reason: str) -> None:
    async with running(inference_timeout=0.02) as (detector, session, classifier, _):
        events = []
        detector.on("amd_prediction", events.append)
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(detector._wait_for_prediction(1), 2)
        first = events[-1]
        assert first.state_changed

        if reason == "reused":
            commit_turn(detector, end_of_turn(""))
        else:
            await commit(detector, session, classifier)
            if reason == "inference_error":
                classifier.respond(2, "invalid JSON")
        await asyncio.wait_for(detector._wait_for_prediction(2), 2)
        fallback = detector._fsm.prediction(2)
        assert fallback is not None
        assert len(events) == (1 if reason == "reused" else 2)
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
        assert request.current_turn["turn_id"] == 1
        assert request.current_turn["transcript"] == "hello"
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await eventually(lambda: detector._fsm.prediction(1) is not None)
        assert model.calls.empty()
        agent.hook_release.set()
        call = await asyncio.wait_for(model.calls.get(), 2)
        messages = [m for m in call["chat_ctx"].items if m.type == "message"]
        assert messages[-1].text_content == _DEFAULT_SCREENING_INSTRUCTIONS
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
        (AMDCategory.MACHINE_SCREENING, _DEFAULT_SCREENING_INSTRUCTIONS),
        (AMDCategory.MACHINE_VM, _DEFAULT_VOICEMAIL_INSTRUCTIONS),
        (AMDCategory.HUMAN, _DEFAULT_HUMAN_INSTRUCTIONS),
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
            await eventually(lambda: detector._fsm.prediction(1) is not None)

            activity.on_end_of_turn(end_of_turn("Can you hear me?"))
            assert (await classifier.request()).current_turn["turn_id"] == 2
            classifier.prediction(2, category)
            await eventually(lambda: detector._fsm.prediction(2) is not None)
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
                await eventually(lambda: detector.lifecycle is AMDLifecycle.ACTIVE)
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
            assert call["chat_ctx"].items[-1].text_content == _DEFAULT_SCREENING_INSTRUCTIONS


@pytest.mark.asyncio
async def test_voicemail_sends_one_message_per_stage_and_records_playback() -> None:
    async with running() as (detector, session, classifier, model):
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(model.calls.get(), 2)
        await eventually(lambda: detector._fsm.voicemail_message_played)
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(2, AMDCategory.MACHINE_VM)
        await eventually(lambda: detector._fsm.prediction(2) is not None)
        await session._activity._user_turn_completed_atask
        assert model.calls.empty()
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        assert [
            m.text_content
            for m in session.current_agent.chat_ctx.items
            if m.type == "message" and m.role == "user"
        ] == ["hello", "hello"]


@pytest.mark.asyncio
async def test_superseded_voicemail_reply_does_not_consume_the_stage_message() -> None:
    async with running() as (detector, session, classifier, _):
        activity = session._activity
        assert activity is not None
        speeches = []

        def advance_turn(event: Any) -> None:
            speeches.append(event.speech_handle)
            if len(speeches) == 1:
                assert activity.on_end_of_turn(end_of_turn("Please leave your message."))

        session.on("speech_created", advance_turn)
        await commit(detector, session, classifier, reply=True)
        first_turn_task = activity._user_turn_completed_atask
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert (await classifier.request()).current_turn["turn_id"] == 2
        await asyncio.wait_for(first_turn_task, 2)
        assert len(speeches) == 1
        assert speeches[0].interrupted
        assert detector._voicemail_handle is None

        classifier.prediction(2, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(activity._user_turn_completed_atask, 2)
        assert len(speeches) == 2
        assert detector._voicemail_handle is speeches[1]
        await asyncio.wait_for(speeches[1], 2)
        assert not speeches[1].interrupted
        assert detector._fsm.voicemail_message_played
        assert session.output.audio.captured_playout_segments == 1

        await commit(detector, session, classifier, reply=True)
        classifier.prediction(3, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(activity._user_turn_completed_atask, 2)
        assert len(speeches) == 2
        assert session.output.audio.captured_playout_segments == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_commit", ["before_new_handle", "after_new_handle"])
async def test_stale_voicemail_handle_does_not_clear_a_newer_reservation(
    stale_commit: str,
) -> None:
    async with running() as (detector, session, classifier, _):
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert await first.should_reply(llm.ChatContext())
        second = await commit(detector, session, classifier)
        classifier.prediction(2, AMDCategory.MACHINE_VM)
        assert await second.should_reply(llm.ChatContext())

        old_handle = SpeechHandle.create()
        old_handle.interrupt()
        new_handle = SpeechHandle.create()
        if stale_commit == "before_new_handle":
            first.on_agent_turn_committed(old_handle)
        second.on_agent_turn_committed(new_handle)
        if stale_commit == "after_new_handle":
            first.on_agent_turn_committed(old_handle)
        assert detector._voicemail_handle is new_handle

        output = session.output.audio
        assert output is not None
        await output.capture_frame(rtc.AudioFrame.create(24000, 1, 24))
        output.flush()
        await output.wait_for_playout()
        completed = asyncio.Event()
        new_handle.add_done_callback(lambda _: completed.set())
        new_handle._mark_done()
        old_handle._mark_done()
        await asyncio.wait_for(completed.wait(), 2)
        assert detector._fsm.voicemail_message_played

        third = await commit(detector, session, classifier)
        classifier.prediction(3, AMDCategory.MACHINE_VM)
        assert not await third.should_reply(llm.ChatContext())


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["played", "interrupted", "failed", "no_audio", "closed"])
async def test_voicemail_playback_is_recorded_by_its_handle(outcome: str) -> None:
    async with running() as (detector, session, classifier, _):
        hooks = await commit(detector, session, classifier)
        assert hooks is not None
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert await hooks.should_reply(llm.ChatContext())

        handle = SpeechHandle.create()
        hooks.on_agent_turn_committed(handle)
        if outcome != "no_audio":
            output = session.output.audio
            assert output is not None
            await output.capture_frame(rtc.AudioFrame.create(24000, 1, 24))
            output.flush()
            await output.wait_for_playout()

        assert not detector._fsm.voicemail_message_played
        if outcome == "interrupted":
            handle.interrupt()
        elif outcome == "closed":
            await detector.aclose()

        completed = asyncio.Event()
        handle.add_done_callback(lambda _: completed.set())
        handle._mark_done(RuntimeError("speech failed") if outcome == "failed" else None)
        await asyncio.wait_for(completed.wait(), 2)
        assert detector._fsm.voicemail_message_played == (outcome == "played")


@pytest.mark.asyncio
async def test_voicemail_handle_commit_does_not_signal_a_prediction_or_update_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with running() as (detector, session, classifier, _):
        hooks = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert await hooks.should_reply(llm.ChatContext())
        notify_prediction = Mock(wraps=detector._prediction_changed.set)
        update_idle = Mock(wraps=detector._update_idle)
        monkeypatch.setattr(detector._prediction_changed, "set", notify_prediction)
        monkeypatch.setattr(detector, "_update_idle", update_idle)

        handle = SpeechHandle.create()
        hooks.on_agent_turn_committed(handle)

        notify_prediction.assert_not_called()
        update_idle.assert_not_called()
        handle._mark_done()
        await asyncio.wait_for(handle, 2)


@pytest.mark.asyncio
async def test_voicemail_idle_defaults_to_one_minute_after_playback() -> None:
    async with running() as (detector, session, classifier, model):
        loop = asyncio.get_running_loop()
        assert detector._fsm.next_deadline is not None
        assert detector._fsm.next_deadline - loop.time() == pytest.approx(10, abs=0.1)
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(model.calls.get(), 2)
        await eventually(
            lambda: detector._fsm.voicemail_message_played and session._activity._no_pending_speech
        )
        assert detector._fsm.next_deadline is not None
        assert detector._fsm.next_deadline - loop.time() == pytest.approx(60, abs=0.1)
        await detector.aclose()
        assert detector._fsm.next_deadline is None


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
            lambda: detector._fsm.voicemail_message_played and session._activity._no_pending_speech
        )
        voicemail_deadline = detector._fsm.next_deadline
        assert voicemail_deadline is not None
        await asyncio.sleep(0.06)
        assert detector.lifecycle is AMDLifecycle.ACTIVE

        speech_started(detector)
        assert detector._fsm.next_deadline is not None
        assert detector._fsm.next_deadline > voicemail_deadline
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
        assert call["chat_ctx"].items[-1].text_content.startswith(_DEFAULT_IVR_INSTRUCTIONS)
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
        events = []
        detector.on("amd_prediction", events.append)
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
        classifier.prediction(2, category)
        await eventually(lambda: events[-1].turn_id == 2 and events[-1].reason == "late_prediction")
        assert detector._fsm.next_deadline is not None
        expected = 2 if category == AMDCategory.MACHINE_VM else 1
        assert detector._fsm.next_deadline - asyncio.get_running_loop().time() == pytest.approx(
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
        assert detector._fsm.next_deadline is None


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
@pytest.mark.parametrize(
    "human_instructions",
    [None, "Greet the person and ask whether they can discuss the appointment."],
)
async def test_human_takeover_updates_the_reply_context_during_machine_speech(
    category: AMDCategory,
    prompt: str,
    hook_finishes_after_cleanup: bool,
    human_instructions: str | None,
) -> None:
    agent = CustomerAgent()
    options = {"human_instructions": human_instructions} if human_instructions is not None else {}
    async with running(agent=agent, **options) as (detector, session, classifier, model):
        if human_instructions is not None:
            model.fake_response_map[human_instructions] = FakeLLMResponse(
                input=human_instructions,
                content="Hello, can we discuss your appointment?",
                ttft=0,
                duration=0,
            )
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
        assert controls[0].text_content == (human_instructions or _DEFAULT_HUMAN_INSTRUCTIONS)
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
@pytest.mark.parametrize("uncertain_first", [False, True])
async def test_human_instructions_are_skipped_without_a_machine_stage(
    uncertain_first: bool,
) -> None:
    async with running(human_instructions="The automated screening has finished.") as (
        detector,
        session,
        classifier,
        model,
    ):
        if uncertain_first:
            hooks = await commit(detector, session, classifier)
            classifier.prediction(1, AMDCategory.UNCERTAIN)
            assert await hooks.should_reply(llm.ChatContext())

        await commit(detector, session, classifier, reply=True)
        classifier.prediction(detector._fsm.turn_id, AMDCategory.HUMAN)
        call = await asyncio.wait_for(model.calls.get(), 2)
        messages = [m for m in call["chat_ctx"].items if m.type == "message"]
        assert not any(m.id.startswith(detector._control_prefix) for m in messages)
        assert any(m.text_content == "Customer hook edit" for m in messages)
        assert any(m.text_content == "hello" for m in messages)
        assert (await detector.execute()).category == AMDCategory.HUMAN
        await eventually(lambda: session._activity._no_pending_speech)


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
        await detector._wait_for_prediction(1)

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
        cancelled = asyncio.create_task(detector._wait_for_prediction(1))
        surviving = asyncio.create_task(detector._wait_for_prediction(1))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled

        await commit(detector, session, classifier)
        current = asyncio.create_task(detector._wait_for_prediction(2))
        await asyncio.wait_for(surviving, 2)
        assert detector._fsm.prediction(1) is None
        assert not current.done()
        classifier.prediction(2, AMDCategory.HUMAN)
        await asyncio.wait_for(current, 2)
        assert detector._fsm.prediction(2).category == AMDCategory.HUMAN


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["cancelled", "participant_disconnected", "timeout"])
async def test_nonterminal_finish_releases_a_waiter_without_a_prediction(reason: str) -> None:
    async with running(timeout=0.03, inference_timeout=1) as (
        detector,
        session,
        classifier,
        _,
    ):
        events = []
        detector.on("amd_prediction", events.append)
        await commit(detector, session, classifier)
        waiter = asyncio.create_task(detector._wait_for_prediction(1))
        await asyncio.sleep(0)
        assert not waiter.done()
        if reason == "cancelled":
            await detector.aclose()
        elif reason == "participant_disconnected":
            detector._participant_identity = "callee"
            detector._on_disconnected(SimpleNamespace(identity="callee"))
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.reason == reason
        assert result.category == AMDCategory.UNCERTAIN
        await asyncio.wait_for(waiter, 2)
        assert detector._fsm.prediction(1) is None
        assert events == []
        assert detector._fsm.next_deadline is None


@pytest.mark.asyncio
async def test_empty_turn_does_not_cancel_a_pending_terminal_prediction() -> None:
    async with running() as (detector, session, classifier, _):
        await commit(detector, session, classifier)
        commit_turn(detector, end_of_turn(""))
        assert detector._fsm.prediction(2) is None
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
        assert detector._fsm.next_deadline is not None
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert classifier.responses[1].cancelled()
        assert not any(event.reason == "late_prediction" for event in events)
        assert detector._fsm.category == AMDCategory.UNCERTAIN
        classifier.prediction(2, AMDCategory.MACHINE_IVR)
        await eventually(lambda: events[-1].turn_id == 2 and events[-1].reason == "prediction")
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
        assert detector._fsm.prediction(1).reason == "inference_error"
        assert detector._fsm.category == AMDCategory.UNCERTAIN
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        assert detector._fsm.next_deadline is not None


@pytest.mark.asyncio
async def test_false_interruption_settlement_rearms_idle_without_a_speech_handle() -> None:
    async with running(idle_timeout=0.03) as (detector, session, _, _):
        activity = session._activity
        activity._false_interruption_pending = True
        session._update_agent_state("thinking")
        session._update_agent_state("listening")
        await asyncio.sleep(0.04)
        assert detector.lifecycle is AMDLifecycle.ACTIVE
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
        handle = SpeechHandle.create()
        first.on_agent_turn_committed(handle)
        handle._mark_done()
        await asyncio.wait_for(handle, 2)
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
            await eventually(
                lambda expected=turn_id: detector._fsm.prediction(expected) is not None
            )
        assert (await detector.execute()).reason == "max_uncertain_turns"


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [None, 1, 4])
async def test_inference_timeout_limit_completes_detection(limit: int | None) -> None:
    options = {} if limit is None else {"max_inference_timeouts": limit}
    expected_limit = 3 if limit is None else limit
    async with running(inference_timeout=0.01, **options) as (detector, session, classifier, _):
        for turn_id in range(1, expected_limit + 1):
            await commit(detector, session, classifier)
            await eventually(
                lambda expected=turn_id: detector._fsm.prediction(expected) is not None
            )
            assert (detector.lifecycle is AMDLifecycle.ACTIVE) == (turn_id < expected_limit)
        assert (await detector.execute()).reason == "inference_timeout"


@pytest.mark.parametrize("limit", [0, -1])
def test_max_inference_timeouts_must_be_positive(limit: int) -> None:
    with pytest.raises(ValueError, match="max_inference_timeouts must be positive"):
        AMD(AgentSession(), llm=None, stt=None, max_inference_timeouts=limit)


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
                assert detector.lifecycle is AMDLifecycle.PENDING
                assert detector._fsm.next_deadline is None
                answered.set()
            else:
                answer_mock.assert_not_called()
            await eventually(lambda: detector.lifecycle is AMDLifecycle.ACTIVE)
            room.emit("participant_disconnected", participant)
            assert (await detector.execute()).reason == "participant_disconnected"
            assert detector.lifecycle is AMDLifecycle.FINISHED
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
        detector.on_dtmf_event("1")
        detector.on_dtmf_event("2#")
        assert classifier.requests.empty()
        assert detector._fsm.category == AMDCategory.UNCERTAIN
        assert detector._fsm.turn_id == 0

        commit_turn(detector, end_of_turn())
        detector.on_dtmf_event("3")
        first = await classifier.request()
        assert first.current_turn["dtmf_digits"] == "12#"
        commit_turn(detector, end_of_turn())
        second = await classifier.request()
        assert second.current_turn["dtmf_digits"] == "3"
        commit_turn(detector, end_of_turn())
        assert (await classifier.request()).current_turn["dtmf_digits"] == ""


@pytest.mark.asyncio
async def test_dtmf_notification_rejects_invalid_digits_and_ignores_completed_runs() -> None:
    async with running() as (detector, _, classifier, _):
        detector.on_dtmf_event("1")
        for digits in ("", "x", "1 2", "1\n"):
            with pytest.raises(ValueError, match="digits must contain only"):
                detector.on_dtmf_event(digits)
        commit_turn(detector, end_of_turn())
        assert (await classifier.request()).current_turn["dtmf_digits"] == "1"

        detector.on_dtmf_event("2")
        classifier.prediction(1, AMDCategory.HUMAN)
        await detector.execute()
        detector.on_dtmf_event("3")
        assert detector._fsm._turns._pending_dtmf_digits == ""


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
        assert (await classifier.request()).current_turn["dtmf_digits"] == "1"


@pytest.mark.asyncio
async def test_dtmf_tool_does_not_report_an_incomplete_publish() -> None:
    from livekit.agents.beta.tools.send_dtmf import send_dtmf_events
    from livekit.agents.beta.workflows.utils import DtmfEvent

    started = asyncio.Event()

    async def publish(**kwargs: Any) -> None:
        started.set()
        await asyncio.Future()

    detector = SimpleNamespace(on_dtmf_event=Mock())
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
    detector.on_dtmf_event.assert_not_called()


@pytest.mark.asyncio
async def test_dtmf_tool_does_not_attach_an_old_send_to_a_new_amd_run() -> None:
    from livekit.agents.beta.tools.send_dtmf import send_dtmf_events
    from livekit.agents.beta.workflows.utils import DtmfEvent

    old = SimpleNamespace(on_dtmf_event=Mock())
    new = SimpleNamespace(on_dtmf_event=Mock())
    tool_session = SimpleNamespace(amd=old)

    async def publish(**kwargs: Any) -> None:
        tool_session.amd = new

    tool_session.room_io = SimpleNamespace(
        room=SimpleNamespace(local_participant=SimpleNamespace(publish_dtmf=publish))
    )
    await send_dtmf_events(SimpleNamespace(session=tool_session), [DtmfEvent.ONE])
    old.on_dtmf_event.assert_not_called()
    new.on_dtmf_event.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("winner", ["session", "amd"])
async def test_first_final_wins_for_amd_without_changing_agent_transcript(winner: str) -> None:
    stt = DrainingSTT()
    async with running(stt=stt) as (detector, session, classifier, reply_model):
        reply_model.fake_response_map["session transcript"] = FakeLLMResponse(
            input="session transcript", content="Hello.", ttft=0, duration=0
        )
        stream = push_audio(detector, stt)
        if winner == "session":
            transcribe(detector, "session transcript")
        stream.send_fake_transcript("AMD transcript")
        if winner == "amd":
            await eventually(
                lambda: (
                    detector._resources.stt._current.snapshot(
                        detector._resources.stt._source
                    ).transcript
                    == "AMD transcript"
                )
            )
            transcribe(detector, "session transcript")

        info = end_of_turn("session transcript")
        session._activity.on_end_of_turn(info)
        request = await classifier.request()
        assert request.current_turn["transcript"] == (
            f"{winner if winner == 'session' else 'AMD'} transcript"
        )
        assert request.current_turn["transcript_source"] == winner
        assert not stream.flushed.is_set()
        classifier.prediction(1, AMDCategory.UNCERTAIN)
        reply = await asyncio.wait_for(reply_model.calls.get(), 2)
        message_contents = [m.text_content for m in reply["chat_ctx"].items if m.type == "message"]
        assert "session transcript" in message_contents
        assert "AMD transcript" not in message_contents
        assert info.new_transcript == "session transcript"
        await session._activity._user_turn_completed_atask
        await eventually(lambda: session._activity._no_pending_speech)
        assert any(
            m.type == "message" and m.text_content == "session transcript"
            for m in session.current_agent.chat_ctx.items
        )


@pytest.mark.asyncio
async def test_transcript_received_after_eot_is_kept_in_next_request() -> None:
    stt = DrainingSTT()
    async with running(stt=stt) as (detector, _, classifier, _):
        stream = push_audio(detector, stt)
        commit_turn(detector, end_of_turn(""))
        assert detector._fsm.prediction(1).transcript == ""
        stream.send_fake_transcript("Please state your name.")
        await eventually(
            lambda: (
                detector._resources.stt._current.snapshot(
                    detector._resources.stt._source
                ).transcript
                == "Please state your name."
            )
        )
        assert classifier.requests.empty()
        assert detector._fsm.prediction(1).transcript == ""

        commit_turn(detector, end_of_turn(""))
        first = await classifier.request()
        assert first.current_turn["turn_id"] == 2
        classifier.prediction(2, AMDCategory.MACHINE_SCREENING)
        await asyncio.wait_for(detector._wait_for_prediction(2), 2)
        prediction = detector._fsm.prediction(2)
        assert (
            prediction.transcript == first.current_turn["transcript"] == "Please state your name."
        )

        stream.send_fake_transcript("Okay, connecting you.")
        await eventually(
            lambda: (
                detector._resources.stt._current.snapshot(
                    detector._resources.stt._source
                ).transcript
                == "Okay, connecting you."
            )
        )
        commit_turn(detector, end_of_turn("Okay."))
        second = await classifier.request()
        assert second.current_turn["turn_id"] == 3
        assert second.current_turn["transcript"] == "Okay, connecting you."
        assert second.current_turn["transcript_source"] == "amd"
        assert second.earlier_turns[0]["transcript"] == ""
        assert second.earlier_turns[1]["transcript"] == first.current_turn["transcript"]
        assert (
            second.earlier_turns[1]["transcript_source"]
            == first.current_turn["transcript_source"]
            == "amd"
        )


@pytest.mark.asyncio
async def test_late_final_after_many_empty_turns_is_committed_at_the_next_eot() -> None:
    stt = DrainingSTT()
    async with running(stt=stt) as (detector, _, classifier, _):
        stream = push_audio(detector, stt)
        detector.on_dtmf_event("1")
        commit_turn(detector, end_of_turn(""))
        for _ in range(20):
            commit_turn(detector, end_of_turn(""))
        assert detector._fsm.turn_id == 21
        assert all(detector._fsm.prediction(turn_id) is not None for turn_id in range(1, 22))
        stream.send_fake_transcript("Hello, can you hear me?")
        await eventually(
            lambda: (
                detector._resources.stt._current.snapshot(
                    detector._resources.stt._source
                ).transcript
                == "Hello, can you hear me?"
            )
        )
        assert classifier.requests.empty()
        commit_turn(detector, end_of_turn(""))
        request = await classifier.request()
        assert request.current_turn["turn_id"] == 22
        assert request.current_turn["transcript"] == "Hello, can you hear me?"
        assert request.current_turn["dtmf_digits"] == ""
        assert [turn["turn_id"] for turn in request.earlier_turns] == list(range(3, 22))
        classifier.prediction(22, AMDCategory.HUMAN)
        assert (await detector.execute()).category == AMDCategory.HUMAN


@pytest.mark.asyncio
async def test_superseded_inference_keeps_the_classification_context() -> None:
    async with running() as (detector, _, classifier, _):
        detector.on_dtmf_event("1")
        commit_turn(detector, end_of_turn("Hello, can you hear me?"))
        previous = await classifier.request()
        hooks = commit_turn(detector, end_of_turn("Yes, let's schedule that."))
        request = await classifier.request()
        assert request.earlier_turns == [previous.current_turn]
        assert request.earlier_turns[0] == {
            "turn_id": 1,
            "transcript": "Hello, can you hear me?",
            "transcript_source": "session",
            "dtmf_digits": "1",
        }
        assert request.current_turn["transcript"] == "Yes, let's schedule that."
        assert request.stage == previous.stage == "uncertain"
        assert detector._fsm.prediction(1) is None
        assert classifier.responses[1].cancelled()
        classifier.prediction(2, AMDCategory.HUMAN)
        assert await hooks.should_reply(llm.ChatContext())
        assert (await detector.execute()).transcript == request.current_turn["transcript"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_empty_turns_do_not_cancel_or_count_against_pending_inference(failed: bool) -> None:
    async with running(inference_timeout=10, max_uncertain_turns=2) as (
        detector,
        session,
        classifier,
        _,
    ):
        events = []
        detector.on("amd_prediction", events.append)
        first = await commit(detector, session, classifier)
        first_wait = asyncio.create_task(first.should_reply(llm.ChatContext()))
        await asyncio.sleep(0)
        second = commit_turn(detector, end_of_turn(""))
        assert not await asyncio.wait_for(first_wait, 2)
        second_wait = asyncio.create_task(second.should_reply(llm.ChatContext()))
        await asyncio.sleep(0)
        current = commit_turn(detector, end_of_turn(""))
        assert not await asyncio.wait_for(second_wait, 2)
        current_wait = asyncio.create_task(current.should_reply(llm.ChatContext()))
        await asyncio.sleep(0)
        assert not current_wait.done()
        assert detector._fsm.prediction(1) is None
        assert not classifier.responses[1].done()
        assert detector._fsm.prediction(2) is None
        classifier.respond(1, "invalid" if failed else '{"category":"uncertain"}')
        assert await asyncio.wait_for(current_wait, 2)
        assert detector._fsm.prediction(3) is detector._fsm.prediction(1)
        assert detector._fsm.prediction(2) is detector._fsm.prediction(1)
        assert [(e.turn_id, e.reason) for e in events] == [
            (1, "inference_error" if failed else "prediction"),
        ]
        assert classifier.requests.empty()
        final_turn_id = 5 if failed else 4
        for turn_id in range(4, final_turn_id + 1):
            await commit(detector, session, classifier)
            classifier.prediction(turn_id, AMDCategory.UNCERTAIN)
            await asyncio.wait_for(detector._wait_for_prediction(turn_id), 2)
            assert (detector.lifecycle is AMDLifecycle.ACTIVE) == (turn_id < final_turn_id)
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.reason == "max_uncertain_turns"
        assert result.turn_id == final_turn_id


@pytest.mark.asyncio
async def test_dtmf_on_an_empty_turn_is_context_not_an_inference_trigger() -> None:
    async with running() as (detector, _, classifier, _):
        detector.on_dtmf_event("1")
        commit_turn(detector, end_of_turn(""))
        assert classifier.requests.empty()
        commit_turn(detector, end_of_turn("Hello."))
        request = await classifier.request()
        assert request.current_turn["dtmf_digits"] == ""
        assert request.earlier_turns[0]["dtmf_digits"] == "1"


@pytest.mark.asyncio
async def test_invalid_transition_falls_back_and_uncertain_preserves_stage() -> None:
    async with running() as (detector, session, classifier, _):
        events = []
        detector.on("amd_prediction", events.append)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await first.should_reply(llm.ChatContext())
        for turn_id, category, reason in (
            (2, AMDCategory.UNCERTAIN, "prediction"),
            (3, AMDCategory.MACHINE_SCREENING, "inference_error"),
        ):
            hooks = await commit(detector, session, classifier)
            classifier.prediction(turn_id, category)
            await hooks.should_reply(llm.ChatContext())
            assert events[-1].reason == reason
            assert detector._fsm.category == AMDCategory.MACHINE_VM
            assert not events[-1].state_changed


@pytest.mark.asyncio
@pytest.mark.parametrize("menu", [False, True])
@pytest.mark.parametrize(
    ("failures", "max_retry", "retryable", "succeeds", "expected_attempts"),
    [
        (1, 1, True, True, 2),
        (2, 1, True, False, 2),
        (1, 0, True, False, 1),
        (1, 2, False, False, 1),
    ],
)
async def test_llm_calls_use_session_connection_options(
    monkeypatch: pytest.MonkeyPatch,
    menu: bool,
    failures: int,
    max_retry: int,
    retryable: bool,
    succeeds: bool,
    expected_attempts: int,
) -> None:
    original_run = ControlledStream._run
    attempts = 0

    async def run(stream: ControlledStream) -> None:
        nonlocal attempts
        attempts += 1
        if attempts <= failures:
            raise APIConnectionError("provider failure", retryable=retryable)
        await original_run(stream)

    monkeypatch.setattr(ControlledStream, "_run", run)
    async with running() as (detector, session, classifier, _):
        options = APIConnectOptions(max_retry=max_retry, retry_interval=0, timeout=0.25)
        session.conn_options.llm_conn_options = options
        chat = Mock(wraps=classifier.chat)
        monkeypatch.setattr(classifier, "chat", chat)
        if menu:
            menus = []
            detector.on("amd_menu_observed", menus.append)
            task = asyncio.create_task(detector._extract_menu(1, "Press 1 for sales."))
            response = await asyncio.wait_for(classifier.menu_requests.get(), 2)
            response.set_result('{"menu":"Sales","options":[{"label":"Sales","dtmf":"1"}]}')
            await task
            assert len(menus) == int(succeeds)
        else:
            hooks = await commit(detector, session, classifier)
            classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
            assert await hooks.should_reply(llm.ChatContext())
            prediction = detector._fsm.prediction(1)
            assert prediction.reason == ("prediction" if succeeds else "inference_error")
            assert prediction.category == (
                AMDCategory.MACHINE_SCREENING if succeeds else AMDCategory.UNCERTAIN
            )
        assert attempts == expected_attempts
        chat.assert_called_once()
        assert chat.call_args.kwargs["conn_options"] is options


@pytest.mark.asyncio
async def test_prediction_deadline_does_not_wait_for_llm_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_run = ControlledStream._run
    attempts = 0

    async def run(stream: ControlledStream) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise APIConnectionError("provider failure")
        await original_run(stream)

    monkeypatch.setattr(ControlledStream, "_run", run)
    async with running(inference_timeout=0.02) as (detector, session, classifier, _):
        session.conn_options.llm_conn_options = APIConnectOptions(max_retry=1, retry_interval=0)
        predictions = []
        detector.on("amd_prediction", predictions.append)
        hooks = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.HUMAN)
        assert await hooks.should_reply(llm.ChatContext())
        fallback = detector._fsm.prediction(1)
        assert fallback.reason == "inference_timeout"
        assert attempts == 1
        assert (await asyncio.wait_for(detector.execute(), 2)).category == AMDCategory.HUMAN
        assert attempts == 2
        assert [event.reason for event in predictions] == ["inference_timeout", "late_prediction"]
        assert detector._fsm.prediction(1) == fallback


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["classification", "prediction"])
async def test_unexpected_inference_errors_finish_with_internal_error(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    async with running() as (detector, session, classifier, _):
        if source == "classification":
            monkeypatch.setattr(_inference, "classify", AsyncMock(side_effect=RuntimeError("bug")))
            commit_turn(detector, end_of_turn())
        else:
            monkeypatch.setattr(
                detector._fsm, "prediction_received", Mock(side_effect=ValueError("bug"))
            )
            await commit(detector, session, classifier)
            classifier.prediction(1, AMDCategory.MACHINE_IVR)
        assert (await asyncio.wait_for(detector.execute(), 2)).reason == "internal_error"


@pytest.mark.asyncio
async def test_unexpected_menu_error_does_not_end_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extract_menu = AsyncMock(side_effect=RuntimeError("menu bug"))
    monkeypatch.setattr(_inference, "extract_ivr_menu", extract_menu)
    async with running() as (detector, session, classifier, _):
        menus = []
        detector.on("amd_menu_observed", menus.append)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_IVR)
        assert await first.should_reply(llm.ChatContext())
        await eventually(lambda: extract_menu.await_count == 1)
        await asyncio.sleep(0.01)
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        assert menus == []
        await commit(detector, session, classifier)
        classifier.prediction(2, AMDCategory.HUMAN)
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.category == AMDCategory.HUMAN
        assert result.reason == "finished"


@pytest.mark.asyncio
async def test_menu_listener_error_ends_detection_with_internal_error() -> None:
    async with running() as (detector, session, classifier, _):

        def fail_listener(_event: object) -> None:
            raise TypeError("listener bug")

        detector.on("amd_menu_observed", fail_listener)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_IVR)
        assert await first.should_reply(llm.ChatContext())
        menu_response = await asyncio.wait_for(classifier.menu_requests.get(), 2)
        menu_response.set_result(
            '{"menu":"Choose a department","options":[{"label":"Sales","dtmf":"1"}]}'
        )
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.reason == "internal_error"
        assert result.category == AMDCategory.MACHINE_IVR


@pytest.mark.asyncio
async def test_slow_menu_does_not_block_classification_or_next_turn() -> None:
    async with running() as (detector, session, classifier, _):
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_IVR)
        assert await first.should_reply(llm.ChatContext())
        menu_response = await asyncio.wait_for(classifier.menu_requests.get(), 2)
        assert not menu_response.done()
        await commit(detector, session, classifier)
        assert menu_response.cancelled()
        classifier.prediction(2, AMDCategory.HUMAN)
        assert (await detector.execute()).category == AMDCategory.HUMAN


@pytest.mark.asyncio
async def test_prediction_listener_next_turn_cancels_the_previous_menu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extract_menu = AsyncMock()
    monkeypatch.setattr(_inference, "extract_ivr_menu", extract_menu)
    async with running() as (detector, session, classifier, _):

        def advance_turn(event: AMDPredictionEvent) -> None:
            if event.turn_id == 1:
                commit_turn(detector, end_of_turn("Hello, can you hear me?"))

        detector.on("amd_prediction", advance_turn)
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_IVR)
        assert (await classifier.request()).current_turn["turn_id"] == 2
        await asyncio.sleep(0.01)
        extract_menu.assert_not_called()
        classifier.prediction(2, AMDCategory.HUMAN)
        assert (await asyncio.wait_for(detector.execute(), 2)).category == AMDCategory.HUMAN


@pytest.mark.asyncio
async def test_delayed_deadline_does_not_start_menu_after_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents.voice.amd import detector as detector_module

    extract_menu = AsyncMock()
    monkeypatch.setattr(_inference, "extract_ivr_menu", extract_menu)
    async with running(machine_silence_threshold=1.5, inference_timeout=0.2, timeout=2) as (
        detector,
        session,
        classifier,
        _,
    ):
        events = []
        detector.on("amd_prediction", events.append)
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_IVR)
        await asyncio.sleep(0.01)
        assert detector._fsm.prediction(1) is None
        assert detector._fsm.next_deadline is not None
        loop = asyncio.get_running_loop()
        assert detector._fsm.next_deadline - loop.time() == pytest.approx(1.49, abs=0.01)
        with monkeypatch.context() as delayed_clock:
            delayed_clock.setattr(detector_module.time, "monotonic", lambda: loop.time() + 3)
            detector._on_deadline()
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.reason == "timeout"
        assert result.category == AMDCategory.MACHINE_IVR
        assert [(event.turn_id, event.category) for event in events] == [
            (1, AMDCategory.MACHINE_IVR)
        ]
        extract_menu.assert_not_called()


@pytest.mark.asyncio
async def test_screening_prediction_is_forwarded_to_session_observability() -> None:
    async with running() as (detector, session, classifier, _):
        host = Mock()
        session._session_host = host
        try:
            hooks = await commit(detector, session, classifier)
            classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
            await hooks.should_reply(llm.ChatContext())
            assert (
                host._on_amd_prediction.call_args.args[0].category == AMDCategory.MACHINE_SCREENING
            )
        finally:
            session._session_host = None


@pytest.mark.asyncio
async def test_split_screening_rollover_retains_all_turns() -> None:
    async with running() as (detector, _, classifier, _):
        history = [
            ("Please state your name and why you are calling.", AMDCategory.MACHINE_SCREENING),
            ("Okay.", AMDCategory.UNCERTAIN),
            ("They can't take the call.", AMDCategory.UNCERTAIN),
            ("Feel free to leave a message.", AMDCategory.MACHINE_VM),
        ]
        for index, (transcript, category) in enumerate(history, start=1):
            commit_turn(detector, end_of_turn(transcript))
            request = await classifier.request()
            assert [e["transcript"] for e in request.earlier_turns] == [
                prior[0] for prior in history[: index - 1]
            ]
            classifier.prediction(index, category)
            await eventually(lambda turn_id=index: detector._fsm.prediction(turn_id) is not None)
        assert detector._fsm.category == AMDCategory.MACHINE_VM
        assert detector._fsm.prediction(4).prev_stage_category == AMDCategory.MACHINE_SCREENING
