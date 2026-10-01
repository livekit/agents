from __future__ import annotations

import pytest

from livekit.agents.voice import io
from livekit.agents.voice.events import AgentOutputTranscribedEvent
from livekit.agents.voice.generation import (
    _AgentOutputTranscriptionForwarder,
    _finalize_agent_output_transcription,
    _ForwardOutput,
)

pytestmark = pytest.mark.unit


class _RecordingTextOutput(io.TextOutput):
    def __init__(self) -> None:
        super().__init__(label="recording", next_in_chain=None)
        self.chunks: list[str] = []
        self.flush_count = 0

    async def capture_text(self, text: str) -> None:
        self.chunks.append(text)

    def flush(self) -> None:
        self.flush_count += 1


async def test_agent_output_transcription_emits_cumulative_partials_and_final() -> None:
    events: list[AgentOutputTranscribedEvent] = []
    output = _RecordingTextOutput()
    forwarder = _AgentOutputTranscriptionForwarder(emit=events.append, next_in_chain=output)

    await forwarder.capture_text("Hello")
    await forwarder.capture_text(" world")
    forwarder.flush()
    assert [(event.transcript, event.is_final) for event in events] == [
        ("Hello", False),
        ("Hello world", False),
    ]
    forwarder.finalize("Hello")

    assert [(event.transcript, event.is_final) for event in events] == [
        ("Hello", False),
        ("Hello world", False),
        ("Hello", True),
    ]
    assert output.chunks == ["Hello", " world"]
    assert output.flush_count == 1


async def test_agent_output_transcription_strips_markup_across_chunk_boundaries() -> None:
    events: list[AgentOutputTranscribedEvent] = []
    forwarder = _AgentOutputTranscriptionForwarder(emit=events.append, next_in_chain=None)
    markup = '<expr type="expression" label="happy"/>'

    await forwarder.capture_text(markup[:12])
    await forwarder.capture_text(markup[12:] + "Hello")
    forwarder.flush()
    forwarder.finalize(markup + "Hello")

    assert all("<expr" not in event.transcript for event in events)
    assert events[-1].transcript == "Hello"
    assert events[-1].is_final is True


async def test_agent_output_transcription_keeps_unaligned_interrupted_text_speculative() -> None:
    events: list[AgentOutputTranscribedEvent] = []
    forwarder = _AgentOutputTranscriptionForwarder(emit=events.append, next_in_chain=None)

    await forwarder.capture_text("Possibly unheard words")
    forwarder.flush()
    forwarder.finalize(None)

    assert events
    assert all(not event.is_final for event in events)


async def test_agent_output_transcription_final_uses_playback_aligned_text() -> None:
    events: list[AgentOutputTranscribedEvent] = []
    forwarder = _AgentOutputTranscriptionForwarder(emit=events.append, next_in_chain=None)

    await forwarder.capture_text("Hello, welcome back")
    forwarder.flush()
    _finalize_agent_output_transcription(
        forwarder,
        _ForwardOutput(played="partial", synchronized_transcript="Hello"),
    )

    assert events[-1].transcript == "Hello"
    assert events[-1].is_final is True


async def test_agent_output_transcription_empty_flush_only_forwards() -> None:
    events: list[AgentOutputTranscribedEvent] = []
    output = _RecordingTextOutput()
    forwarder = _AgentOutputTranscriptionForwarder(emit=events.append, next_in_chain=output)

    forwarder.flush()
    forwarder.finalize("")

    assert events == []
    assert output.flush_count == 1
