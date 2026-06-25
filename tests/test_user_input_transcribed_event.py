"""Unit tests for the `item_id` field on ``UserInputTranscribedEvent``.

Pins the contract that the realtime-path transcription event surface exposes
the upstream ``InputTranscriptionCompleted.item_id`` so consumers can dedup
interim transcripts of a single utterance without dropping into
provider-specific raw events.

Regression target: https://github.com/livekit/agents/issues/6109
"""

from __future__ import annotations

import pytest

from livekit.agents.llm.realtime import InputTranscriptionCompleted
from livekit.agents.voice.events import UserInputTranscribedEvent

pytestmark = pytest.mark.unit


def test_user_input_transcribed_event_carries_item_id() -> None:
    """The public event schema accepts ``item_id`` and round-trips it."""
    ev = UserInputTranscribedEvent(
        transcript="hello world",
        is_final=False,
        item_id="item_abc123",
    )
    assert ev.item_id == "item_abc123"


def test_user_input_transcribed_event_item_id_defaults_to_none() -> None:
    """``item_id`` is optional — STT paths that have no upstream item id can
    omit the field and consumers reading ``ev.item_id`` see ``None``."""
    ev = UserInputTranscribedEvent(transcript="hello world", is_final=True)
    assert ev.item_id is None


def test_user_input_transcribed_event_serialises_item_id() -> None:
    """``model_dump`` includes the ``item_id`` field — important because
    downstream consumers (e.g. the host transport in ``test_session_host.py``)
    may serialise this event for cross-process delivery."""
    ev = UserInputTranscribedEvent(
        transcript="hi", is_final=True, item_id="item_xyz"
    )
    dumped = ev.model_dump()
    assert dumped["item_id"] == "item_xyz"


def test_input_transcription_completed_item_id_can_thread_to_event() -> None:
    """Realtime path: every interim/final ``UserInputTranscribedEvent`` for a
    single utterance shares the upstream ``InputTranscriptionCompleted.item_id``.

    Mirrors the data flow inside
    ``AgentActivity._on_input_audio_transcription_completed`` without
    instantiating the full activity — the contract under test is
    "the field exists and can carry the value", not the activity's plumbing.
    """
    upstream = InputTranscriptionCompleted(
        item_id="item_realtime_42",
        transcript="hello world",
        is_final=True,
    )

    emitted = UserInputTranscribedEvent(
        transcript=upstream.transcript,
        is_final=upstream.is_final,
        item_id=upstream.item_id,
    )

    assert emitted.item_id == upstream.item_id == "item_realtime_42"
