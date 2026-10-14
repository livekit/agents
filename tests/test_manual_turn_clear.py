from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import Agent, AgentSession, LanguageCode
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.telemetry import set_tracer_provider, trace_types, tracer
from livekit.agents.voice.transcription.synchronizer import _SyncedAudioOutput

from .fake_session import FakeActions, create_session
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
async def running_session() -> AsyncIterator[tuple[AgentSession, Agent]]:
    session = create_session(
        FakeActions(),
        turn_handling={
            "turn_detection": "manual",
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


@pytest.mark.real_time
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


@pytest.mark.real_time
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
