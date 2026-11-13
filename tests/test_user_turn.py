from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import (
    Agent,
    AgentSession,
    ConversationItemAddedEvent,
    LanguageCode,
    TurnHandlingOptions,
    UserTurnExceededEvent,
    llm,
    vad,
)
from livekit.agents.llm import LLM, ChatChunk, ChatContext, ChoiceDelta, LLMStream
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.telemetry import set_tracer_provider, trace_types, tracer
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.voice.transcription.synchronizer import _SyncedAudioOutput

from .fake_io import FakeAudioInput
from .fake_session import FakeActions, create_session, run_session
from .fake_turn_stt import TurnScriptedSTT
from .trace_schema import assert_trace_well_formed

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

_TRANSCRIPT = "what's the weather tomorrow"


@pytest.fixture
def span_exporter() -> Iterator[InMemorySpanExporter]:
    original_provider = tracer._tracer_provider
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_tracer_provider(provider)
    try:
        yield exporter
    finally:
        set_tracer_provider(original_provider)
        provider.shutdown()


@pytest.fixture
async def running_session(
    request: pytest.FixtureRequest,
) -> AsyncIterator[tuple[AgentSession, Agent]]:
    session = create_session(
        FakeActions(),
        turn_handling={
            "turn_detection": getattr(request, "param", "manual"),
            "endpointing": {"min_delay": 0.5, "max_delay": 3.0},
        },
        can_pause_audio=True,
    )
    agent = Agent(instructions="You are a helpful assistant.")
    audio_output = session.output.audio
    assert isinstance(audio_output, _SyncedAudioOutput)
    try:
        await session.start(agent, record=False)
        assert session._activity is not None
        assert session._activity._on_enter_task is not None
        await session._activity._on_enter_task
        yield session, agent
    finally:
        await session.aclose()
        await audio_output._synchronizer.aclose()


def _final_transcript(text: str, *, confidence: float = 0.9, request_id: str = "") -> SpeechEvent:
    return SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[SpeechData(text=text, language=LanguageCode("en"), confidence=confidence)],
        request_id=request_id,
    )


async def _await_delivery(session: AgentSession) -> None:
    activity = session._activity
    assert activity is not None
    recognition = activity._audio_recognition
    assert recognition is not None
    assert recognition._end_of_turn_task is not None
    await recognition._end_of_turn_task
    assert activity._user_turn_completed_atask is not None
    await activity._user_turn_completed_atask


def _user_transcripts(agent: Agent) -> list[str | None]:
    return [
        item.text_content
        for item in agent.chat_ctx.items
        if item.type == "message" and item.role == "user"
    ]


@pytest.mark.parametrize("phase", ["queued", "pending_delivery", "delivered"])
async def test_clear_preserves_committed_snapshot_and_resets_stt(
    running_session: tuple[AgentSession, Agent], phase: str
) -> None:
    session, agent = running_session
    activity = session._activity
    assert activity is not None
    recognition = activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT))
    speaking_time = recognition.last_speaking_time
    final_time = recognition._last_final_transcript_time
    commit = session.commit_user_turn(skip_reply=True)
    if phase == "queued":
        await asyncio.sleep(0)
        assert commit.done()
        assert commit.result() == _TRANSCRIPT
    else:
        assert await commit == _TRANSCRIPT
    if phase == "delivered":
        await _await_delivery(session)
    else:
        assert recognition._end_of_turn_task is not None
        assert not recognition._end_of_turn_task.done()

    if phase != "delivered":
        assert recognition._current_transcript == _TRANSCRIPT
        assert recognition._final_transcript_confidence == [0.9]
        assert recognition.last_speaking_time == speaking_time
        assert recognition._last_final_transcript_time == final_time

    stt_pipeline = recognition._stt_pipeline
    session.clear_user_turn()

    assert recognition._current_transcript == ""
    assert recognition._stt_pipeline is not stt_pipeline
    await recognition._on_stt_event(
        SpeechEvent(
            type=SpeechEventType.INTERIM_TRANSCRIPT,
            alternatives=[SpeechData(text="next turn", language=LanguageCode("en"))],
        )
    )
    if phase != "delivered":
        await _await_delivery(session)
    assert recognition._audio_interim_transcript == "next turn"
    assert _user_transcripts(agent) == [_TRANSCRIPT]
    [message] = [
        item for item in agent.chat_ctx.items if item.type == "message" and item.role == "user"
    ]
    assert message.transcript_confidence == pytest.approx(0.9)


@pytest.mark.parametrize("clear_order", ["before_final", "after_final", "none"])
async def test_later_final_does_not_change_snapshot_or_get_erased_by_delivery(
    running_session: tuple[AgentSession, Agent], clear_order: str
) -> None:
    session, agent = running_session
    activity = session._activity
    assert activity is not None
    recognition = activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT))
    assert await session.commit_user_turn(skip_reply=True) == _TRANSCRIPT
    task = recognition._end_of_turn_task
    assert task is not None and not task.done()
    if clear_order == "before_final":
        session.clear_user_turn()
    await recognition._on_stt_event(_final_transcript("never mind", confidence=0.3))
    if clear_order == "after_final":
        session.clear_user_turn()
    last_final_time = recognition._last_final_transcript_time

    assert recognition._end_of_turn_task is task
    await _await_delivery(session)

    assert _user_transcripts(agent) == [_TRANSCRIPT]
    assert recognition._current_transcript == ("" if clear_order == "after_final" else "never mind")
    assert recognition._last_final_transcript_time == last_final_time
    if clear_order != "after_final":
        assert recognition._final_transcript_confidence == [0.3]
        assert await session.commit_user_turn(skip_reply=True) == "never mind"
        await _await_delivery(session)
        assert _user_transcripts(agent) == [_TRANSCRIPT, "never mind"]


@pytest.mark.parametrize("clear_between_commits", [False, True])
async def test_overlapping_commits_deliver_both_turns(
    running_session: tuple[AgentSession, Agent], clear_between_commits: bool
) -> None:
    session, agent = running_session
    assert session._activity is not None
    recognition = session._activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT))
    assert await session.commit_user_turn(skip_reply=True) == _TRANSCRIPT
    first_task = recognition._end_of_turn_task
    assert first_task is not None and not first_task.done()
    if clear_between_commits:
        session.clear_user_turn()

    await recognition._on_stt_event(_final_transcript("next turn"))
    assert await session.commit_user_turn(skip_reply=True) == "next turn"
    await _await_delivery(session)

    assert not first_task.cancelled()
    assert _user_transcripts(agent) == [_TRANSCRIPT, "next turn"]


async def test_interim_after_snapshot_is_available_to_next_commit(
    running_session: tuple[AgentSession, Agent],
) -> None:
    session, agent = running_session
    assert session._activity is not None
    recognition = session._activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT))
    assert await session.commit_user_turn(skip_reply=True) == _TRANSCRIPT
    await recognition._on_stt_event(
        SpeechEvent(
            type=SpeechEventType.INTERIM_TRANSCRIPT,
            alternatives=[SpeechData(text="actually tomorrow", language=LanguageCode("en"))],
        )
    )
    await _await_delivery(session)

    assert (
        await session.commit_user_turn(skip_reply=True, transcript_timeout=0) == "actually tomorrow"
    )
    await _await_delivery(session)
    assert _user_transcripts(agent) == [_TRANSCRIPT, "actually tomorrow"]


async def test_queued_commit_waits_for_its_own_final_transcript(
    running_session: tuple[AgentSession, Agent],
) -> None:
    session, agent = running_session
    assert session._activity is not None
    recognition = session._activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT))
    assert await session.commit_user_turn(skip_reply=True) == _TRANSCRIPT
    first_task = recognition._end_of_turn_task
    assert first_task is not None
    await recognition._on_stt_event(
        SpeechEvent(
            type=SpeechEventType.INTERIM_TRANSCRIPT,
            alternatives=[SpeechData(text="next interim", language=LanguageCode("en"))],
        )
    )
    queued = session.commit_user_turn(skip_reply=True, transcript_timeout=1.0)
    await first_task
    await asyncio.sleep(0.1)
    assert not queued.done()
    await recognition._on_stt_event(_final_transcript("next final"))
    assert await queued == "next final"
    await _await_delivery(session)
    assert _user_transcripts(agent) == [_TRANSCRIPT, "next final"]


async def test_empty_commit_does_not_request_reply_or_block_later_transcripts(
    running_session: tuple[AgentSession, Agent], monkeypatch: pytest.MonkeyPatch
) -> None:
    session, agent = running_session
    calls: list[None] = []

    async def llm_node(*args: object, **kwargs: object) -> AsyncIterator[str]:
        calls.append(None)
        yield ""

    monkeypatch.setattr(agent, "llm_node", llm_node)
    assert await session.commit_user_turn(transcript_timeout=0) == ""
    await asyncio.sleep(0.6)
    assert not calls

    assert session._activity is not None
    recognition = session._activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(_final_transcript("next turn"))
    assert await session.commit_user_turn(skip_reply=True) == "next turn"
    await _await_delivery(session)
    assert _user_transcripts(agent) == ["next turn"]


async def test_empty_commit_keeps_stt_mode_ready_for_a_late_final(
    running_session: tuple[AgentSession, Agent], monkeypatch: pytest.MonkeyPatch
) -> None:
    session, agent = running_session
    completed: list[str | None] = []

    async def on_turn(chat_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None:
        completed.append(new_message.text_content)

    monkeypatch.setattr(agent, "on_user_turn_completed", on_turn)
    session.update_options(turn_detection="stt")
    assert session._activity is not None
    recognition = session._activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(SpeechEvent(type=SpeechEventType.END_OF_SPEECH))
    assert await session.commit_user_turn(skip_reply=True, transcript_timeout=0) == ""
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT))
    await asyncio.sleep(0.6)
    assert completed == [_TRANSCRIPT]


@pytest.mark.parametrize("phase", ["scheduled", "waiting_for_stt"])
async def test_clear_before_snapshot_discards_live_transcript(
    running_session: tuple[AgentSession, Agent], phase: str
) -> None:
    session, agent = running_session
    assert session._activity is not None
    recognition = session._activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT))
    recognition._last_final_transcript_time = time.time() - 1.0
    commit = session.commit_user_turn(skip_reply=True)
    if phase == "waiting_for_stt":
        await asyncio.sleep(0)
        assert not recognition._final_transcript_received.is_set()
        assert not commit.done()

    session.clear_user_turn()
    await recognition._on_stt_event(_final_transcript("replacement"))

    assert await commit == "replacement"
    await _await_delivery(session)
    assert _user_transcripts(agent) == ["replacement"]


@pytest.mark.parametrize("clear_before_final", [False, True])
@pytest.mark.parametrize("next_request_id", ["first", "second"])
async def test_snapshot_trace_does_not_consume_next_turn_metadata(
    span_exporter: InMemorySpanExporter,
    running_session: tuple[AgentSession, Agent],
    clear_before_final: bool,
    next_request_id: str,
) -> None:
    session, _ = running_session
    assert session._activity is not None
    recognition = session._activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(
        SpeechEvent(
            type=SpeechEventType.INTERIM_TRANSCRIPT,
            alternatives=[SpeechData(text="what's the weather", language=LanguageCode("en"))],
            request_id="first",
        )
    )
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT, request_id="first"))
    assert await session.commit_user_turn(skip_reply=True) == _TRANSCRIPT
    if clear_before_final:
        session.clear_user_turn()
    await recognition._on_stt_event(
        _final_transcript("next turn", confidence=0.3, request_id=next_request_id)
    )
    await _await_delivery(session)

    [first_turn] = [span for span in span_exporter.get_finished_spans() if span.name == "user_turn"]
    assert first_turn.attributes is not None
    assert first_turn.attributes[trace_types.ATTR_PROVIDER_REQUEST_IDS] == ("first",)
    assert first_turn.attributes[trace_types.ATTR_USER_TRANSCRIPT] == _TRANSCRIPT
    assert first_turn.attributes[trace_types.ATTR_TRANSCRIPT_CONFIDENCE] == pytest.approx(0.9)
    assert len(json.loads(first_turn.attributes[trace_types.ATTR_STT_EVENTS])) == 2
    assert recognition._stt_request_ids == [next_request_id]
    assert len(recognition._stt_events) == 1

    assert await session.commit_user_turn(skip_reply=True, transcript_timeout=0) == "next turn"
    await _await_delivery(session)
    first_turn, second_turn = [
        span for span in span_exporter.get_finished_spans() if span.name == "user_turn"
    ]
    assert second_turn.attributes is not None
    assert second_turn.attributes[trace_types.ATTR_PROVIDER_REQUEST_IDS] == (next_request_id,)
    assert second_turn.attributes[trace_types.ATTR_TRANSCRIPT_CONFIDENCE] == pytest.approx(0.3)
    waits = [span for span in span_exporter.get_finished_spans() if span.name == "eou_wait"]
    assert len(waits) == 2
    assert all(span.attributes[trace_types.ATTR_EOU_OUTCOME] == "committed" for span in waits)
    await session.aclose()
    assert_trace_well_formed(span_exporter.get_finished_spans())


@pytest.mark.parametrize("phase", ["queued", "pending_delivery"])
@pytest.mark.parametrize(
    ("clear_before_cancel", "automatic_delay"), [(False, 0.5), (False, 0.0), (True, 0.5)]
)
async def test_mode_change_preserves_live_span_and_closes_detached_span(
    span_exporter: InMemorySpanExporter,
    running_session: tuple[AgentSession, Agent],
    phase: str,
    clear_before_cancel: bool,
    automatic_delay: float,
) -> None:
    session, agent = running_session
    assert session._activity is not None
    recognition = session._activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT))
    commit = session.commit_user_turn(skip_reply=True)
    if phase == "queued":
        await asyncio.sleep(0)
        assert commit.done()
    assert await commit == _TRANSCRIPT
    manual_task = recognition._end_of_turn_task
    span = recognition._user_turn_span
    assert manual_task is not None and not manual_task.done()
    assert span is not None and span.is_recording()
    if clear_before_cancel:
        session.clear_user_turn()

    session.update_options(turn_detection="vad", endpointing_opts={"min_delay": automatic_delay})
    if not clear_before_cancel:
        recognition._run_eou_detection(agent.chat_ctx, trigger="vad", skip_reply=True)
    with pytest.raises(asyncio.CancelledError):
        await manual_task

    if clear_before_cancel:
        assert not span.is_recording()
    elif automatic_delay > 0:
        assert span.is_recording()
        assert recognition._user_turn_span is span
    if not clear_before_cancel:
        await _await_delivery(session)
        assert _user_transcripts(agent) == [_TRANSCRIPT]
    else:
        assert _user_transcripts(agent) == []
    assert not span.is_recording()
    await session.aclose()
    assert_trace_well_formed(span_exporter.get_finished_spans())


def _start_of_speech() -> vad.VADEvent:
    return vad.VADEvent(
        type=vad.VADEventType.START_OF_SPEECH,
        samples_index=0,
        timestamp=time.time(),
        speech_duration=0.1,
        silence_duration=0.0,
    )


@pytest.mark.parametrize("phase", ["queued", "pending_delivery"])
@pytest.mark.parametrize("clear_before_vad", [False, True])
async def test_manual_snapshot_survives_vad_and_live_clears(
    span_exporter: InMemorySpanExporter,
    running_session: tuple[AgentSession, Agent],
    phase: str,
    clear_before_vad: bool,
) -> None:
    session, agent = running_session
    activity = session._activity
    assert activity is not None
    recognition = activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT))
    commit = session.commit_user_turn(skip_reply=True)
    if phase == "queued":
        await asyncio.sleep(0)
        assert commit.done()
    assert await commit == _TRANSCRIPT
    task = recognition._end_of_turn_task
    assert task is not None and not task.done()
    if clear_before_vad:
        session.clear_user_turn()

    await recognition._on_vad_event(_start_of_speech())
    await recognition._on_stt_event(_final_transcript("never mind"))
    live_span = recognition._user_turn_span
    assert live_span is not None and live_span.is_recording()
    session.clear_user_turn()
    assert live_span.is_recording() is (not clear_before_vad)
    assert recognition._current_transcript == ""
    await task
    assert activity._user_turn_completed_atask is not None
    await activity._user_turn_completed_atask
    assert not live_span.is_recording()
    assert any(
        span.context == live_span.get_span_context() for span in span_exporter.get_finished_spans()
    )

    messages = [
        item for item in agent.chat_ctx.items if item.type == "message" and item.role == "user"
    ]
    assert len(messages) == 1
    assert messages[0].text_content == _TRANSCRIPT
    assert messages[0].transcript_confidence == pytest.approx(0.9)
    assert not recognition._user_turn_committed

    await recognition._on_stt_event(_final_transcript("the next turn"))
    assert recognition._current_transcript == "the next turn"


@pytest.mark.parametrize("running_session", ["vad", "stt"], indirect=True)
async def test_vad_still_cancels_non_manual_delivery(
    running_session: tuple[AgentSession, Agent],
) -> None:
    session, _ = running_session
    assert session._activity is not None
    recognition = session._activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT))
    await session.commit_user_turn(skip_reply=True)
    task = recognition._end_of_turn_task
    assert task is not None and not task.done()

    await recognition._on_vad_event(_start_of_speech())

    with pytest.raises(asyncio.CancelledError):
        await task


async def _aclose(session: AgentSession) -> None:
    audio_output = session.output.audio
    await session.aclose()
    if isinstance(audio_output, _SyncedAudioOutput):
        await audio_output._synchronizer.aclose()


async def test_session_close_commits_trailing_transcript() -> None:
    transcript = "the mailbox is full and cannot accept new messages at this time"
    actions = FakeActions()
    actions.add_user_speech(0.05, 0.2, transcript, stt_delay=0.3)
    session = create_session(
        actions,
        turn_handling={"turn_detection": "stt"},
        extra_kwargs={"session_close_transcript_timeout": 2.0},
    )
    agent = Agent(instructions="You are a helpful assistant.")
    await session.start(agent)
    audio_input = session.input.audio
    assert isinstance(audio_input, FakeAudioInput)
    audio_input.push(0.1)
    await asyncio.sleep(0.1)

    await _aclose(session)

    assert _user_transcripts(agent) == [transcript]


@pytest.mark.parametrize("discard_audio", [False, True])
async def test_user_turn_during_uninterruptible_speech(discard_audio: bool) -> None:
    """A turn heard during uninterruptible speech is recorded without a reply, unless discarded."""
    agent_line = "Your balance is eighty six thousand pesos, forty days overdue. Can you pay today?"
    user_line = "I already paid that yesterday"

    class _UninterruptibleGreetingAgent(Agent):
        async def on_enter(self) -> None:
            self.session.say(agent_line, allow_interruptions=False)

    actions = FakeActions()
    # the user's turn commits around 2s, well before the 10s line ends
    actions.add_tts(10.0, input=agent_line)
    actions.add_user_speech(1.0, 1.5, user_line)
    session: AgentSession = create_session(
        actions,
        turn_handling={"interruption": {"discard_audio_if_uninterruptible": discard_audio}},
    )
    agent = _UninterruptibleGreetingAgent(instructions="You are a collections agent.")
    added_user_items: list[str | None] = []

    def _on_item_added(ev: ConversationItemAddedEvent) -> None:
        if ev.item.type == "message" and ev.item.role == "user":
            added_user_items.append(ev.item.text_content)

    session.on("conversation_item_added", _on_item_added)

    await run_session(session, agent, drain_delay=1.0)

    expected_user = [] if discard_audio else [user_line]
    assert _user_transcripts(agent) == expected_user
    assert added_user_items == expected_user
    [reply] = [i for i in agent.chat_ctx.items if i.type == "message" and i.role == "assistant"]
    assert reply.text_content == agent_line
    assert reply.interrupted is False


class _ExceededAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="test agent")
        self.exceeded_events: list[UserTurnExceededEvent] = []

    async def on_user_turn_exceeded(self, ev: UserTurnExceededEvent) -> None:
        self.exceeded_events.append(ev)


_SPEED = 5.0


@pytest.mark.parametrize(
    ("transcript", "user_turn_limit", "expected_events"),
    [
        ("hello world", {"max_duration": 1.0 / _SPEED}, 1),
        ("one two three four five six seven eight nine ten", None, 0),
    ],
)
async def test_user_turn_limit(
    transcript: str, user_turn_limit: dict | None, expected_events: int
) -> None:
    agent = _ExceededAgent()
    actions = FakeActions()
    actions.add_user_speech(0.5, 3.5, transcript, stt_delay=0.2)
    actions.add_llm("OK", ttft=0.1, duration=0.1)
    actions.add_tts(0.5, ttfb=0.1, duration=0.1)
    turn_handling = {"user_turn_limit": user_turn_limit} if user_turn_limit else {}
    session = create_session(actions, speed_factor=_SPEED, turn_handling=turn_handling)

    await run_session(session, agent)

    assert len(agent.exceeded_events) == expected_events
    if expected_events:
        assert agent.exceeded_events[0].duration > 0


async def test_word_limit_event_carries_the_transcript() -> None:
    agent = _ExceededAgent()
    transcript = "alpha bravo charlie delta echo foxtrot"
    actions = FakeActions()
    actions.add_user_speech(0.5, 3.0, transcript, stt_delay=0.2)
    actions.add_llm("OK", ttft=0.1, duration=0.1)
    actions.add_tts(0.5, ttfb=0.1, duration=0.1)
    session = create_session(
        actions, speed_factor=_SPEED, turn_handling={"user_turn_limit": {"max_words": 3}}
    )

    await run_session(session, agent)

    [ev] = agent.exceeded_events
    assert ev.accumulated_word_count >= 3
    assert ev.transcript == transcript
    assert ev.accumulated_transcript == transcript


async def test_word_limit_resets_once_the_agent_speaks() -> None:
    agent = _ExceededAgent()
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, "one two three", stt_delay=0.2)
    actions.add_llm("reply one", ttft=0.1, duration=0.1)
    actions.add_tts(0.5, ttfb=0.1, duration=0.1)
    actions.add_user_speech(4.0, 5.0, "four five six", stt_delay=0.2)
    actions.add_llm("reply two", input="four five six", ttft=0.1, duration=0.1)
    actions.add_tts(0.5, input="reply two", ttfb=0.1, duration=0.1)
    session = create_session(
        actions, speed_factor=_SPEED, turn_handling={"user_turn_limit": {"max_words": 5}}
    )

    await run_session(session, agent)

    assert agent.exceeded_events == []


async def test_word_limit_accumulates_across_turns_the_agent_never_answered() -> None:
    agent = _ExceededAgent()
    turn1, turn2 = "one two three", "four five six"
    actions = FakeActions()
    # a slow LLM keeps the agent from speaking before the second turn
    actions.add_user_speech(0.5, 1.5, turn1, stt_delay=0.2)
    actions.add_llm("reply one", ttft=5.0, duration=0.1)
    actions.add_tts(0.5, ttfb=0.5, duration=0.5)
    actions.add_user_speech(2.2, 3.2, turn2, stt_delay=0.2)
    actions.add_llm("reply two", input=turn2, ttft=5.0, duration=0.1)
    actions.add_tts(0.5, input="reply two", ttfb=0.5, duration=0.5)
    session = create_session(
        actions, speed_factor=_SPEED, turn_handling={"user_turn_limit": {"max_words": 5}}
    )

    await run_session(session, agent)

    [ev] = agent.exceeded_events
    assert ev.transcript == turn2
    assert turn1 in _user_transcripts(agent)
    assert ev.accumulated_transcript == f"{turn1} {turn2}"


async def test_word_limit_does_not_accumulate_cleared_turns() -> None:
    agent = _ExceededAgent()
    session = create_session(FakeActions(), turn_handling={"user_turn_limit": {"max_words": 5}})
    await session.start(agent)
    try:
        assert session._activity is not None
        recognition = session._activity._audio_recognition
        assert recognition is not None
        recognition._check_user_turn_limit("one two three")
        session.clear_user_turn()
        recognition._check_user_turn_limit("four five six")

        assert recognition._turn_tracker.words == 3
        assert agent.exceeded_events == []
    finally:
        await _aclose(session)


@pytest.mark.parametrize("block", ["before_scheduling", "while_waiting"])
async def test_user_turn_exceeded_is_dropped_during_handoff(block: str) -> None:
    """An exceeded event must not reach the old agent once update_agent() blocks new turns."""
    agent = _ExceededAgent()
    session = create_session(FakeActions(), turn_handling={"user_turn_limit": {"max_words": 3}})
    await session.start(agent)
    try:
        activity = session._activity
        assert activity is not None
        if block == "before_scheduling":
            activity._new_turns_blocked = True
        activity.on_user_turn_exceeded(
            UserTurnExceededEvent(
                transcript="one two three four",
                accumulated_transcript="one two three four",
                accumulated_word_count=4,
                duration=1.0,
            )
        )
        task = activity._user_turn_exceeded_atask
        if block == "before_scheduling":
            assert task is None
        else:
            assert task is not None
            activity._new_turns_blocked = True
            await task
        assert agent.exceeded_events == []
    finally:
        await _aclose(session)


class _OpenStream(LLMStream):
    """Streams one token, then stays open like a provider that is still generating."""

    def __init__(self, llm: _OpenLLM, **kwargs: Any) -> None:
        super().__init__(llm, **kwargs)
        self._owner = llm

    async def _run(self) -> None:
        self._owner.started.set()
        self._event_ch.send_nowait(
            ChatChunk(id="chunk", delta=ChoiceDelta(role="assistant", content="sure, one"))
        )
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self._owner.cancelled.set()
            raise


class _OpenLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    @property
    def model(self) -> str:
        return "open"

    @property
    def provider(self) -> str:
        return "test"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Any] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[Any] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        return _OpenStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


def _stt_event(type: SpeechEventType, text: str) -> SpeechEvent:
    return SpeechEvent(
        type=type,
        alternatives=[SpeechData(text=text, language=LanguageCode("en"), confidence=0.9)],
    )


async def _start_preemptive_session() -> tuple[_OpenLLM, AgentSession[None]]:
    llm = _OpenLLM()
    session: AgentSession[None] = AgentSession(
        stt=TurnScriptedSTT(turns=[]),
        vad=None,
        llm=llm,
        turn_handling=TurnHandlingOptions(turn_detection="stt"),
    )
    await session.start(agent=Agent(instructions="help the caller"))
    return llm, session


async def test_resumed_turn_cancels_the_preemptive_reply() -> None:
    """An interim with new words after a preflight transcript stops the stale preemptive reply."""
    llm, session = await _start_preemptive_session()
    activity = session._activity
    assert activity is not None and activity._audio_recognition is not None
    recognition = activity._audio_recognition

    try:
        await recognition._on_stt_event(
            _stt_event(SpeechEventType.PREFLIGHT_TRANSCRIPT, "I need to cancel my subscription.")
        )
        assert activity._preemptive_generation is not None
        handle = activity._preemptive_generation.speech_handle
        await asyncio.wait_for(llm.started.wait(), 2.0)

        await recognition._on_stt_event(
            _stt_event(
                SpeechEventType.INTERIM_TRANSCRIPT, "I need to cancel my subscription please"
            )
        )

        # dropped on the event itself, not at the next preflight or final transcript
        assert activity._preemptive_generation is None
        assert handle.interrupted
        await asyncio.wait_for(llm.cancelled.wait(), 2.0)
        await asyncio.wait_for(handle.wait_for_playout(), 2.0)

        # later interims find nothing to cancel
        await recognition._on_stt_event(
            _stt_event(
                SpeechEventType.INTERIM_TRANSCRIPT, "I need to cancel my subscription please and"
            )
        )
    finally:
        await asyncio.gather(session.aclose(), return_exceptions=True)


@pytest.mark.parametrize(
    "interim",
    [
        # Flux keeps sending `Update` after `EagerEndOfTurn`, with the same transcript
        pytest.param("Hello, how are you?", id="same-transcript"),
        # AssemblyAI interims join the raw words, its preflight is the formatted utterance
        pytest.param("hello how are you", id="same-words"),
        # carries nothing the user said
        pytest.param("", id="empty"),
    ],
)
async def test_interim_without_new_words_keeps_the_preemptive_reply(interim: str) -> None:
    llm, session = await _start_preemptive_session()
    activity = session._activity
    assert activity is not None and activity._audio_recognition is not None
    recognition = activity._audio_recognition

    try:
        await recognition._on_stt_event(
            _stt_event(SpeechEventType.PREFLIGHT_TRANSCRIPT, "Hello, how are you?")
        )
        assert activity._preemptive_generation is not None
        handle = activity._preemptive_generation.speech_handle
        await asyncio.wait_for(llm.started.wait(), 2.0)

        await recognition._on_stt_event(_stt_event(SpeechEventType.INTERIM_TRANSCRIPT, interim))

        # the reply still matches what the user said, so the final transcript can adopt it
        assert activity._preemptive_generation is not None
        assert activity._preemptive_generation.speech_handle is handle
        assert not handle.interrupted
    finally:
        await asyncio.gather(session.aclose(), return_exceptions=True)
