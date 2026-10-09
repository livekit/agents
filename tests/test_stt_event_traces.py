from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Iterator
from unittest.mock import MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import AgentSession, llm, stt
from livekit.agents.telemetry import set_tracer_provider, trace_types, tracer
from livekit.agents.voice.audio_recognition import AudioRecognition
from livekit.agents.voice.endpointing import BaseEndpointing

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
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
async def recognition(exporter: InMemorySpanExporter) -> AsyncIterator[AudioRecognition]:
    session = AgentSession(vad=None, turn_handling={"turn_detection": "manual"})
    root = tracer.start_span("agent_session")
    session._root_span_context = trace.set_span_in_context(root)
    hooks = MagicMock()
    hooks.retrieve_chat_ctx.return_value = llm.ChatContext()
    hooks.on_end_of_turn.return_value = True
    recognition = AudioRecognition(
        session,
        hooks=hooks,
        endpointing=BaseEndpointing(min_delay=0, max_delay=0),
        stt=None,
        vad=None,
        interruption_detection=None,
        turn_detection="manual",
    )
    try:
        yield recognition
    finally:
        await recognition._aclose()
        root.end()


def _turn_events(exporter: InMemorySpanExporter) -> list[list[dict]]:
    return [
        json.loads(span.attributes[trace_types.ATTR_STT_EVENTS])
        for span in exporter.get_finished_spans()
        if span.name == "user_turn"
    ]


async def _commit(recognition: AudioRecognition) -> None:
    recognition._run_eou_detection(llm.ChatContext(), trigger="manual")
    assert recognition._end_of_turn_task is not None
    await recognition._end_of_turn_task


async def test_transcript_arrivals_survive_buffering_and_reset_between_turns(
    recognition: AudioRecognition, exporter: InMemorySpanExporter
) -> None:
    origin = time.time() - 2
    recognition._transcript_gate_active = True
    await recognition._on_stt_event(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
    expected = []
    for i, (kind, text) in enumerate(
        [
            (stt.SpeechEventType.INTERIM_TRANSCRIPT, "你好🙂"),
            (stt.SpeechEventType.PREFLIGHT_TRANSCRIPT, "hello world"),
            (stt.SpeechEventType.FINAL_TRANSCRIPT, "hello world"),
            (stt.SpeechEventType.FINAL_TRANSCRIPT, ""),
        ]
    ):
        received_at = origin + i * 0.1
        await recognition._on_stt_event(
            stt.SpeechEvent(
                type=kind,
                created_at=received_at,
                alternatives=[
                    stt.SpeechData(language="en", text=text),
                    stt.SpeechData(language="en", text="unused alternative"),
                ],
            )
        )
        expected.append(
            {"received_at": received_at, "type": kind.value, "transcript_length": len(text)}
        )

    assert recognition._audio_transcript == ""
    assert recognition._user_turn_span is None
    recognition._ensure_user_turn_span(start_time=origin - 0.1)
    recognition._flush_held_transcripts()
    await _commit(recognition)

    assert _turn_events(exporter) == [expected]
    assert recognition._stt_events == []

    received_at = time.time()
    await recognition._on_stt_event(
        stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            created_at=received_at,
            alternatives=[stt.SpeechData(language="en", text="next")],
        )
    )
    await _commit(recognition)
    assert _turn_events(exporter) == [
        expected,
        [{"received_at": received_at, "type": "final_transcript", "transcript_length": 4}],
    ]


@pytest.mark.parametrize("finish", ["clear", "close"])
async def test_incomplete_turn_keeps_held_transcript_metadata(
    recognition: AudioRecognition, exporter: InMemorySpanExporter, finish: str
) -> None:
    received_at = time.time()
    recognition._transcript_gate_active = True
    await recognition._on_stt_event(
        stt.SpeechEvent(
            type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
            created_at=received_at,
            alternatives=[stt.SpeechData(language="en", text="unfinished")],
        )
    )
    if finish == "clear":
        recognition._clear_user_turn()
    else:
        await recognition._aclose()

    assert _turn_events(exporter) == [
        [{"received_at": received_at, "type": "interim_transcript", "transcript_length": 10}]
    ]
    assert recognition._stt_events == []


async def test_transcript_list_is_not_limited_by_span_event_count(
    recognition: AudioRecognition, exporter: InMemorySpanExporter
) -> None:
    recognition._transcript_gate_active = True
    for _ in range(130):
        await recognition._on_stt_event(
            stt.SpeechEvent(type=stt.SpeechEventType.INTERIM_TRANSCRIPT)
        )
    await recognition._aclose()

    [events] = _turn_events(exporter)
    assert len(events) == 130
    assert all(event["transcript_length"] == 0 for event in events)
