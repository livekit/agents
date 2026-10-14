from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator
from unittest.mock import MagicMock

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

    stt_pipeline = recognition._stt_pipeline
    session.clear_user_turn()

    assert recognition._current_transcript == ""
    assert recognition._stt_pipeline is not stt_pipeline
    if phase != "delivered":
        await _await_delivery(session)
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


async def test_later_interim_stays_live_during_manual_delivery(
    running_session: tuple[AgentSession, Agent],
) -> None:
    session, agent = running_session
    assert session._activity is not None
    recognition = session._activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT))
    assert await session.commit_user_turn(skip_reply=True) == _TRANSCRIPT
    session.clear_user_turn()
    await recognition._on_stt_event(
        SpeechEvent(
            type=SpeechEventType.INTERIM_TRANSCRIPT,
            alternatives=[SpeechData(text="next turn", language=LanguageCode("en"))],
        )
    )
    await _await_delivery(session)

    assert _user_transcripts(agent) == [_TRANSCRIPT]
    assert recognition._audio_interim_transcript == "next turn"


async def test_overlapping_commits_deliver_each_snapshot(
    running_session: tuple[AgentSession, Agent],
) -> None:
    session, agent = running_session
    assert session._activity is not None
    recognition = session._activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT))
    assert await session.commit_user_turn(skip_reply=True) == _TRANSCRIPT
    first_task = recognition._end_of_turn_task
    assert first_task is not None and not first_task.done()
    await recognition._on_stt_event(_final_transcript("next turn", confidence=0.3))
    assert await session.commit_user_turn(skip_reply=True) == "next turn"
    second_task = recognition._end_of_turn_task
    assert second_task is not None and second_task is not first_task
    session.clear_user_turn()

    await _await_delivery(session)

    assert not first_task.cancelled()
    assert _user_transcripts(agent) == [_TRANSCRIPT, "next turn"]
    assert not recognition._user_turn_committed
    assert recognition._current_transcript == ""


async def test_close_delivers_all_pending_manual_snapshots(
    running_session: tuple[AgentSession, Agent],
) -> None:
    session, agent = running_session
    assert session._activity is not None
    recognition = session._activity._audio_recognition
    assert recognition is not None
    for text in (_TRANSCRIPT, "next turn"):
        await recognition._on_stt_event(_final_transcript(text))
        assert await session.commit_user_turn(skip_reply=True) == text
    assert len(recognition._manual_turn_tasks) == 2

    await session.aclose()

    assert _user_transcripts(agent) == [_TRANSCRIPT, "next turn"]
    assert not recognition._manual_turn_tasks


async def test_empty_commit_does_not_replace_pending_snapshot(
    running_session: tuple[AgentSession, Agent],
) -> None:
    session, agent = running_session
    assert session._activity is not None
    recognition = session._activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT))
    assert await session.commit_user_turn(skip_reply=True) == _TRANSCRIPT
    task = recognition._end_of_turn_task
    assert task is not None and not task.done()

    assert await session.commit_user_turn(skip_reply=True, transcript_timeout=0) == ""
    assert recognition._end_of_turn_task is task
    await _await_delivery(session)

    assert _user_transcripts(agent) == [_TRANSCRIPT]


async def test_mode_change_cancels_all_pending_manual_snapshots(
    running_session: tuple[AgentSession, Agent],
) -> None:
    session, _ = running_session
    assert session._activity is not None
    recognition = session._activity._audio_recognition
    assert recognition is not None
    for text in (_TRANSCRIPT, "next turn"):
        await recognition._on_stt_event(_final_transcript(text))
        await session.commit_user_turn(skip_reply=True)
    tasks = list(recognition._manual_turn_tasks)
    assert len(tasks) == 2

    recognition._update_options(turn_detection="vad")
    await asyncio.gather(*tasks, return_exceptions=True)

    assert all(task.cancelled() for task in tasks)
    assert not recognition._manual_turn_tasks
    assert not recognition._user_turn_committed


@pytest.mark.real_time
@pytest.mark.parametrize("phase", ["queued", "pending_delivery"])
async def test_cancelled_snapshot_ends_its_spans_and_preserves_live_data(
    span_exporter: InMemorySpanExporter,
    running_session: tuple[AgentSession, Agent],
    phase: str,
) -> None:
    session, _ = running_session
    assert session._activity is not None
    recognition = session._activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT, request_id="first"))
    commit = session.commit_user_turn(skip_reply=True)
    if phase == "queued":
        await asyncio.sleep(0)
        assert commit.done()
    else:
        await commit
    task = recognition._end_of_turn_task
    assert task is not None and not task.done()
    session.clear_user_turn()
    await recognition._on_stt_event(_final_transcript("next turn", request_id="second"))
    recognition._turn_backchannel_over_agent = True
    recognition._overlap_in_current_turn = True
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not recognition._user_turn_committed
    assert recognition._current_transcript == "next turn"
    assert recognition._stt_request_ids == ["second"]
    assert recognition._turn_backchannel_over_agent
    assert recognition._overlap_in_current_turn
    [wait] = [span for span in span_exporter.get_finished_spans() if span.name == "eou_wait"]
    assert wait.attributes[trace_types.ATTR_EOU_OUTCOME] == "dropped"
    [turn] = [span for span in span_exporter.get_finished_spans() if span.name == "user_turn"]
    assert turn.attributes[trace_types.ATTR_PROVIDER_REQUEST_IDS] == ("first",)
    await session.aclose()
    assert_trace_well_formed(span_exporter.get_finished_spans())


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


async def test_clear_resets_realtime_buffer_during_manual_delivery(
    running_session: tuple[AgentSession, Agent],
) -> None:
    session, _ = running_session
    activity = session._activity
    assert activity is not None
    recognition = activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT))
    await session.commit_user_turn(skip_reply=True)
    realtime_session = MagicMock()
    activity._rt_session = realtime_session
    try:
        session.clear_user_turn()

        realtime_session.clear_audio.assert_called_once_with()
        assert recognition._current_transcript == ""
    finally:
        activity._rt_session = None


@pytest.mark.real_time
async def test_snapshot_trace_does_not_consume_next_turn_metadata(
    span_exporter: InMemorySpanExporter,
    running_session: tuple[AgentSession, Agent],
) -> None:
    session, _ = running_session
    assert session._activity is not None
    recognition = session._activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT, request_id="first"))
    assert await session.commit_user_turn(skip_reply=True) == _TRANSCRIPT
    session.clear_user_turn()
    await recognition._on_stt_event(
        _final_transcript("next turn", confidence=0.3, request_id="second")
    )
    await _await_delivery(session)

    [first_turn] = [span for span in span_exporter.get_finished_spans() if span.name == "user_turn"]
    assert first_turn.attributes is not None
    assert first_turn.attributes[trace_types.ATTR_PROVIDER_REQUEST_IDS] == ("first",)
    assert first_turn.attributes[trace_types.ATTR_USER_TRANSCRIPT] == _TRANSCRIPT
    assert first_turn.attributes[trace_types.ATTR_TRANSCRIPT_CONFIDENCE] == pytest.approx(0.9)
    assert len(json.loads(first_turn.attributes[trace_types.ATTR_STT_EVENTS])) == 1
    assert recognition._stt_request_ids == ["second"]
    assert len(recognition._stt_events) == 1

    assert await session.commit_user_turn(skip_reply=True) == "next turn"
    await _await_delivery(session)
    first_turn, second_turn = [
        span for span in span_exporter.get_finished_spans() if span.name == "user_turn"
    ]
    assert second_turn.attributes is not None
    assert second_turn.attributes[trace_types.ATTR_PROVIDER_REQUEST_IDS] == ("second",)
    assert second_turn.attributes[trace_types.ATTR_TRANSCRIPT_CONFIDENCE] == pytest.approx(0.3)
    waits = [span for span in span_exporter.get_finished_spans() if span.name == "eou_wait"]
    assert len(waits) == 2
    assert all(span.attributes[trace_types.ATTR_EOU_OUTCOME] == "committed" for span in waits)
    await session.aclose()
    assert_trace_well_formed(span_exporter.get_finished_spans())


@pytest.mark.parametrize("running_session", ["vad", "stt"], indirect=True)
async def test_clear_still_discards_non_manual_commit(
    running_session: tuple[AgentSession, Agent],
) -> None:
    session, _ = running_session
    assert session._activity is not None
    recognition = session._activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT))
    await session.commit_user_turn(skip_reply=True)
    assert recognition._end_of_turn_task is not None
    assert not recognition._end_of_turn_task.done()
    stt_pipeline = recognition._stt_pipeline

    session.clear_user_turn()

    assert recognition._current_transcript == ""
    assert recognition._stt_pipeline is not stt_pipeline
