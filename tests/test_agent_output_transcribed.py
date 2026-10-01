from __future__ import annotations

import pytest

from livekit.agents.voice import io
from livekit.agents.voice.events import AgentOutputTranscribedEvent
from livekit.agents.voice.generation import _AgentOutputTranscriptionForwarder

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
        ("Hello world", True),
    ]
    assert output.chunks == ["Hello", " world"]
    assert output.flush_count == 1


async def test_agent_output_transcription_empty_flush_only_forwards() -> None:
    events: list[AgentOutputTranscribedEvent] = []
    output = _RecordingTextOutput()
    forwarder = _AgentOutputTranscriptionForwarder(emit=events.append, next_in_chain=output)

    forwarder.flush()

    assert events == []
    assert output.flush_count == 1
