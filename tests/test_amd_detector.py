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
from livekit.agents.voice.amd import AMDCategory, AMDReason, _fsm, _inference
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
from livekit.agents.voice.events import FunctionToolsExecutedEvent, UserInputTranscribedEvent
from livekit.agents.voice.speech_handle import SpeechHandle

from .amd_test_utils import detector_clock, next_deadline  # noqa: F401
from .fake_io import FakeAudioOutput
from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_stt import DrainingStream, DrainingSTT, FakeSTT
from .fake_tts import FakeTTS

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent, pytest.mark.virtual_time]


def test_prediction_event_carries_stage_and_state_changed() -> None:
    event = AMDPredictionEvent(
        category=AMDCategory.WAIT,
        reason="prediction",
        transcript="hello",
        speech_duration=0,
        delay=0,
    )
    assert event.stage is AMDCategory.UNCERTAIN
    assert event.state_changed is False
    dumped = json.loads(event.model_dump_json())
    assert dumped["stage"] == "uncertain"
    assert dumped["state_changed"] is False

    event = AMDPredictionEvent(
        category=AMDCategory.MACHINE_VM,
        stage=AMDCategory.MACHINE_VM,
        state_changed=True,
        prev_turn_category=AMDCategory.UNCERTAIN,
        reason="prediction",
        transcript="hello",
        speech_duration=0,
        delay=0,
    )
    assert event.model_dump()["state_changed"] is True


class ClassifierLLM(FakeLLM):
    """Controllable provider model. Tests use the real SDK inference path."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: asyncio.Queue[Any] = asyncio.Queue()
        self.menu_requests: asyncio.Queue[asyncio.Future[str]] = asyncio.Queue()
        self.responses: dict[int, asyncio.Future[str]] = {}

    def chat(self, **kwargs: Any) -> llm.LLMStream:
        chat_ctx = kwargs["chat_ctx"]
        response = asyncio.get_running_loop().create_future()
        if chat_ctx.items[0].text_content == _inference.CLASSIFY_PROMPT:
            data = json.loads(chat_ctx.items[1].text_content)
            messages = [item for item in chat_ctx.messages() if item.role == "user"]
            self.responses[messages[-1].extra["turn_id"]] = response
            self.requests.put_nowait(
                SimpleNamespace(
                    **data,
                    current_turn=messages[-1],
                    earlier_turns=messages[:-1],
                    chat_ctx=chat_ctx.copy(),
                )
            )
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


def dtmf_executed(
    session: AgentSession,
    digits: str,
    *,
    is_error: bool = False,
    name: str = "send_dtmf_events",
    created_at: float | None = None,
) -> FunctionToolsExecutedEvent:
    call = llm.FunctionCall(
        call_id=utils.shortuuid("dtmf_"),
        name=name,
        arguments=json.dumps({"events": list(digits)}),
        created_at=created_at if created_at is not None else time.time(),
    )
    event = FunctionToolsExecutedEvent(
        function_calls=[call],
        function_call_outputs=[
            llm.FunctionCallOutput(
                call_id=call.call_id,
                name=name,
                output="Send failed" if is_error else f"Successfully sent DTMF events: {digits}",
                is_error=is_error,
            )
        ],
    )
    session.emit("function_tools_executed", event)
    return event


def dtmf_calls(chat_ctx: llm.ChatContext) -> list[llm.FunctionCall]:
    return [item for item in chat_ctx.items if isinstance(item, llm.FunctionCall)]


async def eventually(predicate: Any) -> None:
    async def wait() -> None:
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), 2)


@asynccontextmanager
async def running(
    *,
    classifier: ClassifierLLM | None = None,
    agent: CustomerAgent | None = None,
    session_options: dict[str, Any] | None = None,
    **options: Any,
) -> AsyncIterator[tuple[AMD, AgentSession, ClassifierLLM, RecordingLLM]]:
    classifier = classifier or ClassifierLLM()
    agent = agent or CustomerAgent()
    model = RecordingLLM()
    session = AgentSession(
        **{
            "llm": model,
            "tts": FakeTTS(fake_audio_duration=0.05),
            "turn_handling": {"turn_detection": "manual"},
            "aec_warmup_duration": None,
            **(session_options or {}),
        }
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
    assert request.current_turn.extra["turn_id"] == detector._turn_id
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
        reply_instructions = None

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
        await asyncio.wait_for(detector._should_reply(1, llm.ChatContext()), 2)
        first = events[-1]
        assert first.state_changed

        if reason == "reused":
            commit_turn(detector, end_of_turn(""))
        else:
            await commit(detector, session, classifier)
            if reason == "inference_error":
                classifier.respond(2, "invalid JSON")
        await asyncio.wait_for(detector._should_reply(2, llm.ChatContext()), 2)
        fallback = detector._turns[2].prediction
        assert fallback is not None
        assert len(events) == 2
        assert fallback.reason == reason
        assert fallback.category == fallback.prev_turn_category == AMDCategory.MACHINE_VM
        assert fallback.prev_stage_category == AMDCategory.UNCERTAIN
        assert not fallback.state_changed
        assert fallback.model_dump()["state_changed"] is False

        await detector.aclose()
        completed = await detector.execute()
        latest = first if reason == "reused" else fallback
        assert completed.turn_id == latest.turn_id
        assert completed.prev_turn_category == latest.prev_turn_category


@pytest.mark.asyncio
async def test_customer_hook_and_prediction_overlap_controls_are_temporary() -> None:
    agent = CustomerAgent()
    agent.hook_release.clear()
    async with running(agent=agent) as (detector, session, classifier, model):
        completed = asyncio.create_task(detector.execute())
        info = end_of_turn()
        info.metrics.end_of_turn_delay = 0.5
        session._activity.on_end_of_turn(info)
        assert detector._turn_id == 1
        assert not agent.hook_started.is_set()
        request = await classifier.request()
        await agent.hook_started.wait()
        assert request.current_turn.extra["turn_id"] == 1
        assert request.current_turn.text_content == "hello"
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await eventually(lambda: detector._turns[1].prediction is not None)
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
            await eventually(lambda: detector._turns[1].prediction is not None)

            activity.on_end_of_turn(end_of_turn("Can you hear me?"))
            assert (await classifier.request()).current_turn.extra["turn_id"] == 2
            classifier.prediction(2, category)
            await eventually(lambda: detector._turns[2].prediction is not None)
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
            await eventually(lambda: detector._voicemail_message_played)
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
                assert detector._turn_id == 1
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
async def test_voicemail_sends_one_delivered_message_and_records_playback() -> None:
    async with running() as (detector, session, classifier, model):
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(model.calls.get(), 2)
        await eventually(lambda: detector._voicemail_message_played)
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(2, AMDCategory.MACHINE_VM)
        await eventually(lambda: detector._turns[2].prediction is not None)
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
        assert (await classifier.request()).current_turn.extra["turn_id"] == 2
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
        assert detector._voicemail_message_played
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
        assert detector._voicemail_message_played

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

        assert not detector._voicemail_message_played
        if outcome == "interrupted":
            handle.interrupt()
        elif outcome == "closed":
            await detector.aclose()

        completed = asyncio.Event()
        handle.add_done_callback(lambda _: completed.set())
        handle._mark_done(RuntimeError("speech failed") if outcome == "failed" else None)
        await asyncio.wait_for(completed.wait(), 2)
        assert detector._voicemail_message_played == (outcome == "played")


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
        assert next_deadline(detector) is not None
        assert next_deadline(detector) - loop.time() == pytest.approx(10, abs=0.1)
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(model.calls.get(), 2)
        await eventually(
            lambda: detector._voicemail_message_played and session._activity._no_pending_speech
        )
        assert next_deadline(detector) is not None
        assert next_deadline(detector) - loop.time() == pytest.approx(60, abs=0.1)
        await detector.aclose()
        assert next_deadline(detector) is None


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
            lambda: detector._voicemail_message_played and session._activity._no_pending_speech
        )
        voicemail_deadline = next_deadline(detector)
        assert voicemail_deadline is not None
        await asyncio.sleep(0.06)
        assert detector.lifecycle is AMDLifecycle.ACTIVE

        speech_started(detector)
        assert next_deadline(detector) is not None
        assert next_deadline(detector) > voicemail_deadline
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
async def test_late_prediction_preserves_stage_and_idle_timer(category: AMDCategory) -> None:
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
        deadline = next_deadline(detector)
        classifier.prediction(2, category)
        await asyncio.sleep(0)
        assert classifier.responses[2].cancelled()
        assert events[-1].reason == "inference_timeout"
        assert len(events) == 2
        assert detector._state == previous
        assert next_deadline(detector) == deadline
        assert next_deadline(detector) is not None
        expected = 2 if previous == AMDCategory.MACHINE_VM else 1
        assert next_deadline(detector) - asyncio.get_running_loop().time() == pytest.approx(
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
        assert next_deadline(detector) is None


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
        assert detector._turn_id == 2


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
        classifier.prediction(detector._turn_id, AMDCategory.HUMAN)
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
        assert detector._state == AMDCategory.MACHINE_IVR


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
async def test_late_unavailable_does_not_cancel_reply(
    playback_started: bool,
) -> None:
    async with running(inference_timeout=0.02) as (detector, session, classifier, model):
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await detector._should_reply(1, llm.ChatContext())

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
        await asyncio.sleep(0.1)
        assert classifier.responses[2].cancelled()
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        assert detector._state == AMDCategory.MACHINE_SCREENING
        assert not speech.interrupted
        if playback_started:
            assert session.agent_state == "speaking"
        else:
            assert not speech.done()


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
        cancelled = asyncio.create_task(detector._should_reply(1, llm.ChatContext()))
        surviving = asyncio.create_task(detector._should_reply(1, llm.ChatContext()))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled

        await commit(detector, session, classifier)
        current = asyncio.create_task(detector._should_reply(2, llm.ChatContext()))
        await asyncio.wait_for(surviving, 2)
        assert detector._turns[1].prediction is None
        assert not current.done()
        classifier.prediction(2, AMDCategory.HUMAN)
        await asyncio.wait_for(current, 2)
        assert detector._turns[2].prediction.category == AMDCategory.HUMAN


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
        waiter = asyncio.create_task(detector._should_reply(1, llm.ChatContext()))
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
        assert detector._turns[1].prediction is None
        assert events == []
        assert next_deadline(detector) is None


@pytest.mark.asyncio
async def test_empty_turn_does_not_cancel_a_pending_terminal_prediction() -> None:
    async with running() as (detector, session, classifier, _):
        await commit(detector, session, classifier)
        commit_turn(detector, end_of_turn(""))
        assert detector._turns[2].prediction is None
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
        assert next_deadline(detector) is not None
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert classifier.responses[1].cancelled()
        assert [event.reason for event in events] == ["inference_timeout"]
        assert detector._state == AMDCategory.UNCERTAIN
        classifier.prediction(2, AMDCategory.MACHINE_IVR)
        await eventually(lambda: events[-1].turn_id == 2 and events[-1].reason == "prediction")
        assert detector._state == AMDCategory.MACHINE_IVR


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
        assert detector._turns[1].prediction.reason == "inference_error"
        assert detector._state == AMDCategory.UNCERTAIN
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        assert next_deadline(detector) is not None


@pytest.mark.asyncio
async def test_amd_commits_short_turns_during_agent_speech() -> None:
    session_options = {
        "stt": FakeSTT(),
        "turn_handling": {"turn_detection": "stt", "interruption": {"min_words": 3}},
    }
    async with running(session_options=session_options) as (detector, session, _, _):
        activity = session._activity
        assert activity is not None
        speech = SpeechHandle.create(allow_interruptions=True)
        activity._current_speech = speech
        # AMD classifies every turn, so the min_words filter does not drop a short one
        assert activity.on_end_of_turn(end_of_turn("hi"))
        assert detector._turn_id == 1
        assert detector._turns[1].transcript.transcript == "hi"
        speech._mark_done()
        activity._current_speech = None
        await detector.aclose()
        # without AMD the same short turn is dropped
        assert not activity.on_end_of_turn(end_of_turn("hi"))


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
        output = session.output.audio
        await output.capture_frame(rtc.AudioFrame.create(24000, 1, 24))
        output.flush()
        await output.wait_for_playout()
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
                lambda expected=turn_id: detector._turns[expected].prediction is not None
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
                lambda expected=turn_id: detector._turns[expected].prediction is not None
            )
            assert (detector.lifecycle is AMDLifecycle.ACTIVE) == (turn_id < expected_limit)
        assert (await detector.execute()).reason == "inference_timeout"


@pytest.mark.parametrize("limit", [0, -1])
def test_max_inference_timeouts_must_be_positive(limit: int) -> None:
    with pytest.raises(ValueError, match="max_inference_timeouts must be positive"):
        AMD(AgentSession(), llm=None, stt=None, max_inference_timeouts=limit)


@pytest.mark.asyncio
async def test_supplied_model_is_not_closed_by_amd() -> None:
    async with running() as (detector, session, classifier, _):
        classifier.aclose = AsyncMock()
        await detector.aclose()
        classifier.aclose.assert_not_called()
        assert session.amd is None


@pytest.mark.asyncio
@pytest.mark.parametrize("track_published", [False, True])
async def test_missing_audio_subscription_times_out_and_releases_session(
    monkeypatch: pytest.MonkeyPatch, track_published: bool
) -> None:
    room = utils.EventEmitter()
    room.isconnected = lambda: True
    publication = SimpleNamespace(
        sid="track", kind=rtc.TrackKind.KIND_AUDIO, subscribed=False, track=None
    )
    participant = SimpleNamespace(identity="callee", track_publications={"track": publication})
    room.remote_participants = {"callee": participant} if track_published else {}
    session = AgentSession(llm=FakeLLM(), turn_handling={"turn_detection": "manual"})
    await session.start(Agent(instructions="Hello"))
    activity = session._activity
    assert activity is not None
    session_audio = Mock()
    monkeypatch.setattr(activity._audio_recognition, "_push_audio", session_audio)
    session._room_io = SimpleNamespace(room=room, set_participant=Mock())
    frame = rtc.AudioFrame.create(16000, 1, 320)
    started = asyncio.get_running_loop().time()
    try:
        async with AMD(
            session, llm=None, stt=None, participant_identity="callee", timeout=0.1
        ) as detector:
            await eventually(lambda: bool(room._events.get("track_subscribed")))
            assert detector.lifecycle is AMDLifecycle.PENDING
            assert detector._hard_deadline is None
            assert not activity._authorization_allowed.is_set()
            activity.push_audio(frame)
            session_audio.assert_not_called()

            result = await asyncio.wait_for(detector.execute(), 6)
            assert asyncio.get_running_loop().time() - started == pytest.approx(5, abs=0.01)
            assert result.reason is AMDReason.PARTICIPANT_MISSING
            assert result.category is AMDCategory.UNCERTAIN
            assert detector.lifecycle is AMDLifecycle.FINISHED
            assert session.amd is None
            assert session._turn_hooks is None
            assert activity._authorization_allowed.is_set()
            assert not detector._tasks
            assert not any(room._events.values())
            for emitter, event, handler in detector._subscriptions:
                assert handler not in emitter._events.get(event, ())
            activity.push_audio(frame)
            session_audio.assert_called_once_with(frame, stt_frame=None)
    finally:
        session._room_io = None
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("waiting_for", ["track", "answer"])
@pytest.mark.parametrize("room_disconnected", [False, True])
async def test_disconnect_during_setup_reports_disconnected(
    waiting_for: str, room_disconnected: bool
) -> None:
    room = utils.EventEmitter()
    room.isconnected = Mock(return_value=True)
    publication = SimpleNamespace(
        sid="track",
        kind=rtc.TrackKind.KIND_AUDIO,
        subscribed=waiting_for == "answer",
        track=object() if waiting_for == "answer" else None,
    )
    participant = SimpleNamespace(
        identity="callee",
        kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
        attributes={"sip.callStatus": "ringing"},
        track_publications={"track": publication},
    )
    room.remote_participants = {"callee": participant}
    session = AgentSession(llm=FakeLLM(), turn_handling={"turn_detection": "manual"})
    await session.start(Agent(instructions="Hello"))
    session._room_io = SimpleNamespace(room=room, set_participant=Mock())
    try:
        async with AMD(session, llm=None, stt=None, participant_identity="callee") as detector:
            event = (
                "track_subscribed" if waiting_for == "track" else "participant_attributes_changed"
            )
            await eventually(lambda: bool(room._events.get(event)))
            if room_disconnected:
                room.isconnected.return_value = False
                room.emit("connection_state_changed", rtc.ConnectionState.CONN_DISCONNECTED)
            else:
                room.remote_participants.clear()
                room.emit("participant_disconnected", participant)
            assert (await detector.execute()).reason is AMDReason.PARTICIPANT_DISCONNECTED
            assert session.amd is None
            assert session._activity._authorization_allowed.is_set()
            assert not any(room._events.values())
    finally:
        session._room_io = None
        await session.aclose()


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
                await asyncio.sleep(5.1)
                assert detector.lifecycle is AMDLifecycle.PENDING
                assert next_deadline(detector) is None
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
async def test_dtmf_results_are_collected_once_without_changing_inflight_requests() -> None:
    async with running() as (detector, session, classifier, _):
        first_send = dtmf_executed(session, "12#")
        session.emit("function_tools_executed", first_send)
        assert classifier.requests.empty()
        assert detector._state == AMDCategory.UNCERTAIN
        assert detector._turn_id == 0

        commit_turn(detector, end_of_turn())
        second_send = dtmf_executed(session, "3")
        first = await classifier.request()
        assert dtmf_calls(first.chat_ctx) == first_send.function_calls
        commit_turn(detector, end_of_turn())
        second = await classifier.request()
        assert dtmf_calls(second.chat_ctx) == (
            first_send.function_calls + second_send.function_calls
        )
        assert dtmf_calls(first.chat_ctx) == first_send.function_calls
        assert [item.type for item in second.chat_ctx.items[2:]] == [
            "function_call",
            "function_call_output",
            "message",
            "function_call",
            "function_call_output",
            "message",
        ]


@pytest.mark.asyncio
async def test_amd_collects_only_successful_dtmf_calls_from_its_run() -> None:
    async with running() as (detector, session, classifier, _):
        session.history.add_message(role="user", content="unrelated session transcript")
        session.history.add_message(role="assistant", content="agent response")
        dtmf_executed(session, "1", is_error=True)
        dtmf_executed(session, "2", name="another_tool")
        dtmf_executed(session, "3", created_at=detector._started_at - 1)
        success = dtmf_executed(session, "4")
        commit_turn(detector, end_of_turn())
        request = await classifier.request()
        assert dtmf_calls(request.chat_ctx) == success.function_calls
        assert request.current_turn.text_content == "hello"
        assert request.earlier_turns == []
        classifier.prediction(1, AMDCategory.HUMAN)
        await detector.execute()
        dtmf_executed(session, "5")
        assert dtmf_calls(detector._chat_ctx) == success.function_calls


@pytest.mark.asyncio
async def test_dtmf_tool_works_without_amd() -> None:
    from livekit.agents.beta.tools.send_dtmf import send_dtmf_events
    from livekit.agents.beta.workflows.utils import DtmfEvent

    publisher = AsyncMock()
    tool_session = SimpleNamespace(
        room_io=SimpleNamespace(
            room=SimpleNamespace(local_participant=SimpleNamespace(publish_dtmf=publisher))
        ),
    )
    result = await send_dtmf_events(SimpleNamespace(session=tool_session), [DtmfEvent.ONE])
    publisher.assert_awaited_once_with(code=1, digit="1")
    assert result == "Successfully sent DTMF events: 1"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "error", "cancelled"])
async def test_amd_observes_dtmf_tool_completion_from_session(
    monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    from livekit.agents.beta.tools import send_dtmf

    publisher = AsyncMock(
        side_effect=[None, RuntimeError("publish failed")]
        if failure == "error"
        else [None, asyncio.CancelledError()]
        if failure == "cancelled"
        else None
    )
    room = SimpleNamespace(local_participant=SimpleNamespace(publish_dtmf=publisher))
    monkeypatch.setattr(send_dtmf, "get_job_context", lambda: SimpleNamespace(room=room))
    async with running() as (detector, session, classifier, reply_model):
        executed = []
        session.on("function_tools_executed", executed.append)
        reply_model.fake_response_map[_DEFAULT_IVR_INSTRUCTIONS] = FakeLLMResponse(
            input=_DEFAULT_IVR_INSTRUCTIONS,
            content="",
            ttft=0,
            duration=0,
            tool_calls=[
                llm.FunctionToolCall(
                    name="send_dtmf_events", arguments='{"events":["1","2"]}', call_id="dtmf"
                )
            ],
        )
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_IVR)
        await eventually(lambda: bool(executed))
        assert publisher.await_count == 2
        output = executed[0].function_call_outputs[0]
        assert output.is_error == (failure == "error")
        if failure == "cancelled":
            assert not output.output
        if failure == "error":
            assert "Failed to send DTMF event: 2." in output.output
        commit_turn(detector, end_of_turn("Connecting you."))
        request = await classifier.request()
        calls = dtmf_calls(request.chat_ctx)
        assert calls == ([] if failure else executed[0].function_calls)
        if not failure:
            assert output in request.chat_ctx.items


@pytest.mark.asyncio
async def test_dtmf_tool_propagates_cancelled_publish() -> None:
    from livekit.agents.beta.tools.send_dtmf import send_dtmf_events
    from livekit.agents.beta.workflows.utils import DtmfEvent

    started = asyncio.Event()

    async def publish(**kwargs: Any) -> None:
        started.set()
        await asyncio.Future()

    tool_session = SimpleNamespace(
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


@pytest.mark.asyncio
@pytest.mark.parametrize("winner", ["session", "amd"])
@pytest.mark.parametrize("session_stt", [False, True])
async def test_first_final_wins_for_amd_without_changing_agent_transcript(
    winner: str, session_stt: bool
) -> None:
    stt = DrainingSTT()
    async with running(stt=stt, session_options={"stt": FakeSTT() if session_stt else None}) as (
        detector,
        session,
        classifier,
        reply_model,
    ):
        reply_model.fake_response_map["session transcript"] = FakeLLMResponse(
            input="session transcript", content="Hello.", ttft=0, duration=0
        )
        stream = push_audio(detector, stt)
        if winner == "session":
            transcribe(detector, "session transcript")
        stream.send_fake_transcript("AMD transcript")
        if winner == "amd" or not session_stt:
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
        expected_source = winner if session_stt else "amd"
        assert request.current_turn.text_content == (
            "session transcript" if expected_source == "session" else "AMD transcript"
        )
        assert request.current_turn.extra["transcript_source"] == expected_source
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
        assert detector._turns[1].prediction.transcript == ""
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
        assert detector._turns[1].prediction.transcript == ""

        commit_turn(detector, end_of_turn(""))
        first = await classifier.request()
        assert first.current_turn.extra["turn_id"] == 2
        classifier.prediction(2, AMDCategory.MACHINE_SCREENING)
        await asyncio.wait_for(detector._should_reply(2, llm.ChatContext()), 2)
        prediction = detector._turns[2].prediction
        assert prediction.transcript == first.current_turn.text_content == "Please state your name."

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
        assert second.current_turn.extra["turn_id"] == 3
        assert second.current_turn.text_content == "Okay, connecting you."
        assert second.current_turn.extra["transcript_source"] == "amd"
        assert second.earlier_turns == [first.current_turn]
        assert (
            second.earlier_turns[0].extra["transcript_source"]
            == first.current_turn.extra["transcript_source"]
            == "amd"
        )


@pytest.mark.asyncio
async def test_late_final_after_many_empty_turns_is_committed_at_the_next_eot() -> None:
    stt = DrainingSTT()
    async with running(stt=stt) as (detector, _, classifier, _):
        stream = push_audio(detector, stt)
        dtmf_executed(detector._session, "1")
        commit_turn(detector, end_of_turn(""))
        for _ in range(20):
            commit_turn(detector, end_of_turn(""))
        assert detector._turn_id == 21
        assert all(detector._turns[turn_id].prediction is not None for turn_id in range(1, 22))
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
        assert request.current_turn.extra["turn_id"] == 22
        assert request.current_turn.text_content == "Hello, can you hear me?"
        assert [json.loads(call.arguments) for call in dtmf_calls(request.chat_ctx)] == [
            {"events": ["1"]}
        ]
        assert request.earlier_turns == []
        classifier.prediction(22, AMDCategory.HUMAN)
        assert (await detector.execute()).category == AMDCategory.HUMAN


@pytest.mark.asyncio
async def test_superseded_inference_keeps_the_classification_context() -> None:
    async with running() as (detector, _, classifier, _):
        dtmf_executed(detector._session, "1")
        commit_turn(detector, end_of_turn("Hello, can you hear me?"))
        previous = await classifier.request()
        hooks = commit_turn(detector, end_of_turn("Yes, let's schedule that."))
        request = await classifier.request()
        assert request.earlier_turns == [previous.current_turn]
        assert request.earlier_turns[0].text_content == "Hello, can you hear me?"
        assert request.earlier_turns[0].extra == {"turn_id": 1, "transcript_source": "session"}
        assert [json.loads(call.arguments) for call in dtmf_calls(request.chat_ctx)] == [
            {"events": ["1"]}
        ]
        assert request.current_turn.text_content == "Yes, let's schedule that."
        assert request.stage == previous.stage == "uncertain"
        assert detector._turns[1].prediction is None
        assert classifier.responses[1].cancelled()
        classifier.prediction(2, AMDCategory.HUMAN)
        assert await hooks.should_reply(llm.ChatContext())
        assert (await detector.execute()).transcript == request.current_turn.text_content


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["prediction", "inference_error", "inference_timeout"])
async def test_empty_turns_do_not_cancel_or_count_against_pending_inference(reason: str) -> None:
    async with running(
        inference_timeout=0.1 if reason == "inference_timeout" else 10, max_uncertain_turns=2
    ) as (
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
        assert detector._turns[1].prediction is None
        assert not classifier.responses[1].done()
        assert detector._turns[2].prediction is None
        if reason != "inference_timeout":
            classifier.respond(
                1, "invalid" if reason == "inference_error" else '{"category":"uncertain"}'
            )
        assert await asyncio.wait_for(current_wait, 2)
        source = detector._turns[1].prediction
        for turn_id in (2, 3):
            reused = detector._turns[turn_id].prediction
            assert reused.turn_id == turn_id
            assert reused.reason == "reused"
            assert reused.category == source.category
            assert reused.stage == source.stage
            assert reused.transcript == ""
            assert not reused.state_changed
        assert [(e.turn_id, e.reason) for e in events] == [
            (1, reason),
            (2, "reused"),
            (3, "reused"),
        ]
        assert classifier.requests.empty()
        final_turn_id = 4 if reason == "prediction" else 5
        for turn_id in range(4, final_turn_id + 1):
            await commit(detector, session, classifier)
            classifier.prediction(turn_id, AMDCategory.UNCERTAIN)
            await asyncio.wait_for(detector._should_reply(turn_id, llm.ChatContext()), 2)
            assert (detector.lifecycle is AMDLifecycle.ACTIVE) == (turn_id < final_turn_id)
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.reason == "max_uncertain_turns"
        assert result.turn_id == final_turn_id


@pytest.mark.asyncio
async def test_dtmf_on_an_empty_turn_is_context_not_an_inference_trigger() -> None:
    async with running() as (detector, _, classifier, _):
        dtmf_executed(detector._session, "1")
        commit_turn(detector, end_of_turn(""))
        assert classifier.requests.empty()
        commit_turn(detector, end_of_turn("Hello."))
        request = await classifier.request()
        assert [json.loads(call.arguments) for call in dtmf_calls(request.chat_ctx)] == [
            {"events": ["1"]}
        ]
        assert request.earlier_turns == []


@pytest.mark.asyncio
async def test_invalid_transition_falls_back_and_uncertain_keeps_the_stage() -> None:
    async with running() as (detector, session, classifier, _):
        events = []
        detector.on("amd_prediction", events.append)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await first.should_reply(llm.ChatContext())
        state = detector._state
        for turn_id, category, reason in (
            (2, AMDCategory.MACHINE_SCREENING, "inference_error"),
            (3, AMDCategory.UNCERTAIN, "prediction"),
            (4, AMDCategory.MACHINE_SCREENING, "inference_error"),
        ):
            hooks = await commit(detector, session, classifier)
            classifier.prediction(turn_id, category)
            ctx = llm.ChatContext()
            assert await hooks.should_reply(ctx)
            assert events[-1].reason == reason
            assert detector._state is state
            assert events[-1].stage is state
            assert not events[-1].state_changed
            # the stage instructions still apply while the stage is kept
            assert ctx.items[-1].text_content == _DEFAULT_VOICEMAIL_INSTRUCTIONS
            if turn_id == 2:
                assert events[-1].category is AMDCategory.MACHINE_VM
            elif turn_id == 3:
                assert events[-1].category is AMDCategory.UNCERTAIN
                assert events[-1].prev_turn_category is AMDCategory.MACHINE_VM
                assert classifier.requests.empty()
            else:
                # a fallback keeps the latest accepted prediction
                assert events[-1].category is AMDCategory.UNCERTAIN


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
            task = detector._menu_atask = asyncio.create_task(
                detector._extract_menu(1, "Press 1 for sales.")
            )
            response = await asyncio.wait_for(classifier.menu_requests.get(), 2)
            response.set_result('{"menu":"Sales","options":[{"label":"Sales","dtmf":"1"}]}')
            await task
            assert len(menus) == int(succeeds)
        else:
            hooks = await commit(detector, session, classifier)
            classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
            assert await hooks.should_reply(llm.ChatContext())
            prediction = detector._turns[1].prediction
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
        fallback = detector._turns[1].prediction
        assert fallback.reason == "inference_timeout"
        assert attempts == 1
        await asyncio.sleep(0.2)
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        assert detector._state == AMDCategory.UNCERTAIN
        assert attempts == 1
        assert detector._classifier_atask is None
        assert [event.reason for event in predictions] == ["inference_timeout"]
        assert detector._turns[1].prediction == fallback


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
            monkeypatch.setattr(_fsm, "transition", Mock(side_effect=ValueError("bug")))
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
        assert (await classifier.request()).current_turn.extra["turn_id"] == 2
        await asyncio.sleep(0.01)
        extract_menu.assert_not_called()
        classifier.prediction(2, AMDCategory.HUMAN)
        assert (await asyncio.wait_for(detector.execute(), 2)).category == AMDCategory.HUMAN


@pytest.mark.asyncio
async def test_delayed_deadline_cancels_reply_hold_without_repeating_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents.voice.amd import detector as detector_module

    extract_menu = AsyncMock(return_value=_inference.AMDIVRMenuResponse(menu="", options=[]))
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
        assert detector._turns[1].prediction.category == AMDCategory.MACHINE_IVR
        assert detector._reply_held_at(asyncio.get_running_loop().time())
        extract_menu.assert_awaited_once()
        state = detector._state
        loop = asyncio.get_running_loop()
        timer = detector._timer
        assert timer is not None
        with monkeypatch.context() as delayed_clock:
            # fire the armed timer as if the loop ran it late
            delayed_clock.setattr(detector_module.time, "monotonic", lambda: loop.time() + 3)
            timer._run()
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.reason == "timeout"
        assert result.category == AMDCategory.MACHINE_IVR
        assert detector._state is state
        assert len(events) == 1
        extract_menu.assert_awaited_once()


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
            assert [e.text_content for e in request.earlier_turns] == [
                prior[0] for prior in history[: index - 1]
            ]
            classifier.prediction(index, category)
            await eventually(lambda turn_id=index: detector._turns[turn_id].prediction is not None)
        assert detector._state == AMDCategory.MACHINE_VM
        assert detector._turns[4].prediction.prev_stage_category == AMDCategory.MACHINE_SCREENING
        assert detector._turns[3].prediction.category == AMDCategory.UNCERTAIN
        assert detector._turns[3].prediction.stage == AMDCategory.MACHINE_SCREENING


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled_by", ["timeout", "new_turn", "finish"])
@pytest.mark.parametrize("outcome", ["human", "wait", "provider_error", "unexpected_error"])
async def test_classifier_that_ignores_cancellation_cannot_change_state(
    monkeypatch: pytest.MonkeyPatch, cancelled_by: str, outcome: str
) -> None:
    requests = asyncio.Queue()
    cancelled = asyncio.Event()

    async def classify(model, request, **kwargs):
        response = asyncio.get_running_loop().create_future()
        requests.put_nowait(response)
        while True:
            try:
                result = await asyncio.shield(response)
                break
            except asyncio.CancelledError:
                if request.chat_ctx.messages()[-1].extra["turn_id"] != 1:
                    raise
                cancelled.set()
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(_inference, "classify", classify)
    async with running(inference_timeout=0.02) as (detector, _, _, _):
        events = []
        detector.on("amd_prediction", events.append)
        commit_turn(detector, end_of_turn())
        response = await asyncio.wait_for(requests.get(), 2)
        old_task = detector._classifier_atask
        if cancelled_by == "new_turn":
            hooks = commit_turn(detector, end_of_turn("a new prompt"))
            newer_response = await asyncio.wait_for(requests.get(), 2)
        elif cancelled_by == "finish":
            detector._finish(AMDReason.CANCELLED)
        await asyncio.wait_for(cancelled.wait(), 2)
        state = detector._state
        predictions = list(events)
        if outcome == "human":
            response.set_result(_inference.AMDResponse(category=AMDCategory.HUMAN))
        elif outcome == "wait":
            response.set_result(_inference.AMDResponse(category=AMDCategory.WAIT))
        elif outcome == "provider_error":
            response.set_result(APIConnectionError("stale provider failure"))
        else:
            response.set_result(RuntimeError("stale internal failure"))
        await asyncio.gather(old_task, return_exceptions=True)
        await asyncio.sleep(0)
        assert detector._state == state
        assert events == predictions
        if cancelled_by == "new_turn":
            assert detector._turns[1].prediction is None
            newer_response.set_result(_inference.AMDResponse(category=AMDCategory.MACHINE_VM))
            assert await hooks.should_reply(llm.ChatContext())
            assert detector._state == AMDCategory.MACHINE_VM
        elif cancelled_by == "timeout":
            assert detector._turns[1].prediction.reason == "inference_timeout"
        else:
            assert (await detector.execute()).reason == "cancelled"


@pytest.mark.asyncio
async def test_result_after_deadline_is_ignored_before_timer_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents.voice.amd import detector as detector_module

    async with running(inference_timeout=1) as (detector, session, classifier, _):
        events = []
        detector.on("amd_prediction", events.append)
        hooks = await commit(detector, session, classifier)
        task = detector._classifier_atask
        loop = asyncio.get_running_loop()
        with monkeypatch.context() as delayed_clock:
            delayed_clock.setattr(detector_module.time, "monotonic", lambda: loop.time() + 2)
            classifier.prediction(1, AMDCategory.HUMAN)
            await task
        assert await hooks.should_reply(llm.ChatContext())
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        assert detector._state == AMDCategory.UNCERTAIN
        assert [event.reason for event in events] == ["inference_timeout"]


@pytest.mark.asyncio
async def test_prediction_listener_cannot_mutate_saved_decision() -> None:
    async with running() as (detector, session, classifier, _):

        def mutate(event: AMDPredictionEvent) -> None:
            event.category = AMDCategory.HUMAN
            event.transcript = "changed by listener"

        detector.on("amd_prediction", mutate)
        hooks = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert await hooks.should_reply(llm.ChatContext())
        prediction = detector._turns[1].prediction
        assert prediction.category == AMDCategory.MACHINE_VM
        assert prediction.transcript == "hello"
        assert detector._state == AMDCategory.MACHINE_VM
        await detector.aclose()
        result = await detector.execute()
        assert result.category == AMDCategory.MACHINE_VM
        assert result.transcript == "hello"


@pytest.mark.asyncio
async def test_completed_amd_cannot_be_entered_again() -> None:
    async with running() as (detector, session, _, _):
        await detector.aclose()
        result = await detector.execute()
        with pytest.raises(RuntimeError, match="new AMD instance"):
            await detector.__aenter__()
        assert await detector.execute() is result
        assert session.amd is None
        assert session._turn_hooks is None


@pytest.mark.asyncio
async def test_classifier_history_is_bounded_and_saved_decisions_survive() -> None:
    async with running() as (detector, _, classifier, _):
        commit_turn(detector, end_of_turn("first prompt"))
        await classifier.request()
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await detector._should_reply(1, llm.ChatContext())
        first_prediction = detector._turns[1].prediction
        for turn_id in range(2, 24):
            hooks = commit_turn(detector, end_of_turn(f"prompt {turn_id}"))
            await classifier.request()
            classifier.prediction(turn_id, AMDCategory.MACHINE_SCREENING)
            await hooks.should_reply(llm.ChatContext())
        dtmf_executed(detector._session, "2#")
        commit_turn(detector, end_of_turn("latest prompt"))
        request = await classifier.request()
        assert request.current_turn.extra["turn_id"] == 24
        assert [json.loads(call.arguments) for call in dtmf_calls(request.chat_ctx)] == [
            {"events": ["2", "#"]}
        ]
        assert [turn.extra["turn_id"] for turn in request.earlier_turns] == list(range(7, 24))
        assert len(request.chat_ctx.items[2:]) == 20
        assert all(turn.text_content for turn in request.earlier_turns)
        assert detector._turns[1].prediction is first_prediction
        assert first_prediction.category == AMDCategory.MACHINE_SCREENING


@pytest.mark.asyncio
async def test_empty_turns_preserve_transcript_and_dtmf_history() -> None:
    async with running() as (detector, session, classifier, _):
        hooks = commit_turn(detector, end_of_turn("For sales press 1."))
        first = await classifier.request()
        classifier.prediction(1, AMDCategory.MACHINE_IVR)
        await hooks.should_reply(llm.ChatContext())
        sent = dtmf_executed(session, "1")
        for _ in range(22):
            commit_turn(detector, end_of_turn(""))
        assert classifier.requests.empty()
        commit_turn(detector, end_of_turn("One moment please."))
        request = await classifier.request()
        assert request.earlier_turns == [first.current_turn]
        assert dtmf_calls(request.chat_ctx) == sent.function_calls
        assert request.current_turn.extra["turn_id"] == 24


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", ["idle", "classification", "silence", "speech"])
async def test_amd_enforces_overall_deadline_independently_of_fsm(pending: str) -> None:
    async with running(timeout=0.3, inference_timeout=1, machine_silence_threshold=1.5) as (
        detector,
        session,
        classifier,
        _,
    ):
        deadline = detector._hard_deadline
        assert deadline is not None
        if pending != "idle":
            await commit(detector, session, classifier)
            if pending != "classification":
                classifier.prediction(1, AMDCategory.MACHINE_VM)
                await eventually(lambda: detector._reply_held_at(asyncio.get_running_loop().time()))
                if pending == "speech":
                    speech_started(detector)
        state = detector._state
        assert not hasattr(state, "next_deadline")
        assert next_deadline(detector) == deadline
        assert detector._hard_deadline == deadline
        result = await asyncio.wait_for(detector.execute(), 1)
        assert result.reason == "timeout"
        assert result.category == (
            AMDCategory.MACHINE_VM if pending in {"silence", "speech"} else AMDCategory.UNCERTAIN
        )
        assert detector._hard_deadline is None
        assert detector._state is state
        assert not detector._reply_held_at(asyncio.get_running_loop().time())
        assert next_deadline(detector) is None
        assert detector._timer is None
        assert session.amd is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bridge", [None, AMDCategory.UNCERTAIN, AMDCategory.WAIT, AMDCategory.MACHINE_IVR]
)
@pytest.mark.parametrize("outcome", ["played", "interrupted", "failed", "no_audio"])
async def test_voicemail_delivery_guard_survives_transitions_and_allows_retry(
    bridge: AMDCategory | None,
    outcome: str,
) -> None:
    async with running() as (detector, session, classifier, _):
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert await first.should_reply(llm.ChatContext())
        handle = SpeechHandle.create()
        first.on_agent_turn_committed(handle)
        turn_id = 1
        if bridge is not None:
            turn_id += 1
            await commit(detector, session, classifier)
            classifier.prediction(turn_id, bridge)
            await detector._should_reply(turn_id, llm.ChatContext())
        turn_id += 1
        overlapping = await commit(detector, session, classifier)
        classifier.prediction(turn_id, AMDCategory.MACHINE_VM)
        assert not await overlapping.should_reply(llm.ChatContext())
        state = detector._state
        if outcome != "no_audio":
            output = session.output.audio
            await output.capture_frame(rtc.AudioFrame.create(24000, 1, 24))
            output.flush()
            await output.wait_for_playout()
        if outcome == "interrupted":
            handle.interrupt()
        handle._mark_done(RuntimeError("playback failed") if outcome == "failed" else None)
        await handle
        await eventually(lambda: detector._voicemail_handle is None)
        assert detector._state is state
        assert detector._voicemail_message_played == (outcome == "played")

        if bridge is not None:
            turn_id += 1
            await commit(detector, session, classifier)
            classifier.prediction(turn_id, bridge)
            await detector._should_reply(turn_id, llm.ChatContext())
        turn_id += 1
        retry = await commit(detector, session, classifier)
        classifier.prediction(turn_id, AMDCategory.MACHINE_VM)
        assert await retry.should_reply(llm.ChatContext()) == (outcome != "played")


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["inference_error", "inference_timeout", "reused"])
@pytest.mark.parametrize("category", [AMDCategory.MACHINE_IVR, AMDCategory.WAIT])
async def test_fallback_and_reuse_leave_fsm_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
    category: AMDCategory,
) -> None:
    async with running(inference_timeout=0.1) as (detector, session, classifier, _):
        first = await commit(detector, session, classifier)
        classifier.prediction(1, category)
        assert await first.should_reply(llm.ChatContext()) == (category is not AMDCategory.WAIT)
        state = detector._state
        transition = Mock(wraps=_fsm.transition)
        monkeypatch.setattr(_fsm, "transition", transition)
        if reason == "reused":
            hooks = commit_turn(detector, end_of_turn(""))
        else:
            hooks = await commit(detector, session, classifier)
            if reason == "inference_error":
                classifier.respond(2, "invalid JSON")
        assert await hooks.should_reply(llm.ChatContext()) == (category is not AMDCategory.WAIT)
        assert detector._turns[2].prediction.reason == reason
        assert detector._state is state
        transition.assert_not_called()


@pytest.mark.asyncio
async def test_valid_prediction_resets_timeout_budget_before_reply_is_allowed() -> None:
    async with running(
        inference_timeout=0.1,
        max_inference_timeouts=2,
        machine_silence_threshold=1.5,
    ) as (detector, session, classifier, _):
        first = await commit(detector, session, classifier)
        assert await first.should_reply(llm.ChatContext())
        assert detector._inference_timeouts == 1
        await commit(detector, session, classifier)
        classifier.prediction(2, AMDCategory.MACHINE_SCREENING)
        await eventually(lambda: detector._turns[2].prediction is not None)
        assert detector._reply_held_at(asyncio.get_running_loop().time())
        assert detector._inference_timeouts == 0
        third = await commit(detector, session, classifier)
        assert await third.should_reply(llm.ChatContext())
        assert detector._inference_timeouts == 1
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        fourth = await commit(detector, session, classifier)
        await fourth.should_reply(llm.ChatContext())
        assert (await detector.execute()).reason == "inference_timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize("bridge", [AMDCategory.UNCERTAIN, AMDCategory.WAIT])
async def test_human_instructions_remember_machine_history(bridge: AMDCategory) -> None:
    async with running() as (detector, session, classifier, _):
        for turn_id, category in enumerate(
            (AMDCategory.MACHINE_SCREENING, bridge, AMDCategory.HUMAN),
            start=1,
        ):
            hooks = await commit(detector, session, classifier)
            classifier.prediction(turn_id, category)
            ctx = llm.ChatContext()
            assert await hooks.should_reply(ctx) == (category is not AMDCategory.WAIT)
            if category is bridge:
                # wait blocks the reply; uncertain keeps the screening stage and its instructions
                if bridge is AMDCategory.WAIT:
                    assert not ctx.items
                else:
                    assert ctx.items[-1].extra["amd_stage"] == "machine-screening"
        assert ctx.items[-1].text_content == _DEFAULT_HUMAN_INSTRUCTIONS
        assert (await detector.execute()).prev_stage_category == AMDCategory.MACHINE_SCREENING


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage",
    [None, AMDCategory.MACHINE_SCREENING, AMDCategory.MACHINE_VM, AMDCategory.MACHINE_IVR],
)
async def test_wait_skips_reply_and_allows_a_fresh_prediction(
    monkeypatch: pytest.MonkeyPatch, stage: AMDCategory | None
) -> None:
    async with running(machine_silence_threshold=1.5) as (
        detector,
        session,
        classifier,
        model,
    ):
        events = []
        detector.on("amd_prediction", events.append)
        if stage is not None:
            await commit(detector, session, classifier)
            classifier.prediction(1, stage)
            await detector._should_reply(1, llm.ChatContext())
        previous = detector._state
        previous_stage = detector._previous_stage
        predictions = list(events)
        transition = Mock(wraps=_fsm.transition)
        monkeypatch.setattr(_fsm, "transition", transition)
        activity = session._activity
        activity.on_end_of_turn(end_of_turn("While you wait, hear about our special offers."))
        request = await classifier.request()
        classifier.prediction(request.current_turn.extra["turn_id"], AMDCategory.WAIT)
        await asyncio.wait_for(activity._user_turn_completed_atask, 0.1)

        # wait blocks the reply but keeps the stage
        assert detector._should_wait
        assert detector._state is previous
        assert detector._previous_stage is previous_stage
        assert events[:-1] == predictions
        assert events[-1].category is AMDCategory.WAIT
        assert events[-1].stage is previous
        assert not events[-1].state_changed
        assert not detector._reply_held_at(asyncio.get_running_loop().time())
        assert detector._voicemail_turn_id is None
        assert model.calls.empty()
        transition.assert_called_once_with(previous, AMDCategory.WAIT)
        await asyncio.sleep(1.6)
        assert model.calls.empty()
        assert detector.lifecycle is AMDLifecycle.ACTIVE

        activity.on_end_of_turn(end_of_turn())
        following = await classifier.request()
        assert following.stage == previous
        assert set(following.allowed_next_categories) == set(_fsm.ALLOWED[previous])
        classifier.prediction(following.current_turn.extra["turn_id"], AMDCategory.UNCERTAIN)
        # a kept machine stage still waits for participant silence before replying
        await asyncio.wait_for(activity._user_turn_completed_atask, 2)
        assert not detector._should_wait
        assert detector._state is previous
        await asyncio.wait_for(model.calls.get(), 2)
        assert transition.call_count == 2
        transition.assert_called_with(previous, AMDCategory.UNCERTAIN)


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", [False, True])
async def test_empty_turns_preserve_wait_until_new_classification(pending: bool) -> None:
    async with running() as (detector, session, classifier, _):
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_IVR)
        await detector._should_reply(1, llm.ChatContext())
        hooks = await commit(detector, session, classifier)
        if pending:
            hooks = commit_turn(detector, end_of_turn(""))
        classifier.prediction(2, AMDCategory.WAIT)
        assert not await asyncio.wait_for(hooks.should_reply(llm.ChatContext()), 0.1)

        for _ in range(2):
            hooks = commit_turn(detector, end_of_turn(""))
            assert not await asyncio.wait_for(hooks.should_reply(llm.ChatContext()), 0.1)
            assert detector._turns[detector._turn_id].prediction.category is AMDCategory.WAIT
        assert detector._should_wait
        assert detector._state is AMDCategory.MACHINE_IVR
        assert classifier.requests.empty()
        hooks = await commit(detector, session, classifier)
        classifier.prediction(detector._turn_id, AMDCategory.HUMAN)
        assert await hooks.should_reply(llm.ChatContext())
        assert (await detector.execute()).category is AMDCategory.HUMAN


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [AMDReason.TIMEOUT, AMDReason.CANCELLED])
async def test_wait_does_not_block_completion_or_allow_a_reply_afterward(reason: AMDReason) -> None:
    async with running(
        idle_timeout=0.1,
        timeout=0.1 if reason is AMDReason.TIMEOUT else 10,
    ) as (detector, session, classifier, _):
        hooks = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.WAIT)
        assert not await hooks.should_reply(llm.ChatContext())
        if reason is AMDReason.CANCELLED:
            await detector.aclose()
        assert (await asyncio.wait_for(detector.execute(), 0.2)).reason is reason
        assert not await hooks.should_reply(llm.ChatContext())


@pytest.mark.asyncio
async def test_wait_pauses_the_idle_timer_until_the_next_prediction() -> None:
    async with running(idle_timeout=0.1) as (detector, session, classifier, _):
        hooks = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.WAIT)
        assert not await hooks.should_reply(llm.ChatContext())
        await asyncio.sleep(0.3)
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        assert detector._idle_deadline is None
        hooks = commit_turn(detector, end_of_turn(""))
        assert not await hooks.should_reply(llm.ChatContext())
        await asyncio.sleep(0.3)
        assert detector.lifecycle is AMDLifecycle.ACTIVE

        hooks = await commit(detector, session, classifier)
        classifier.prediction(3, AMDCategory.UNCERTAIN)
        assert await hooks.should_reply(llm.ChatContext())
        assert detector._idle_deadline is not None
        assert (await asyncio.wait_for(detector.execute(), 1)).reason is AMDReason.IDLE_TIMEOUT


@pytest.mark.asyncio
@pytest.mark.parametrize("prior", ["uncertain", "timeout"])
async def test_wait_resets_inference_timeout_and_uncertain_turn_budgets(prior: str) -> None:
    async with running(inference_timeout=0.05, max_inference_timeouts=2, max_uncertain_turns=2) as (
        detector,
        session,
        classifier,
        _,
    ):
        first = await commit(detector, session, classifier)
        if prior == "uncertain":
            classifier.prediction(1, AMDCategory.UNCERTAIN)
        assert await first.should_reply(llm.ChatContext())
        assert detector._uncertain_turns + detector._inference_timeouts == 1

        hooks = await commit(detector, session, classifier)
        classifier.prediction(2, AMDCategory.WAIT)
        assert not await hooks.should_reply(llm.ChatContext())
        assert detector._uncertain_turns == detector._inference_timeouts == 0
        following = await commit(detector, session, classifier)
        if prior == "uncertain":
            classifier.prediction(3, AMDCategory.UNCERTAIN)
        assert await following.should_reply(llm.ChatContext()) == (prior == "uncertain")
        assert detector.lifecycle is AMDLifecycle.ACTIVE
