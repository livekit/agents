from __future__ import annotations

from types import SimpleNamespace

import pytest

from livekit.agents import APIConnectionError, llm
from livekit.plugins.aws.experimental.realtime.events import (
    Event,
    SonicEventBuilder,
    TextInputContentEvent,
)
from livekit.plugins.aws.experimental.realtime.realtime_model import RealtimeSession
from livekit.plugins.aws.experimental.realtime.turn_tracker import TurnTracker

pytestmark = pytest.mark.unit


async def test_nova_transcript_events_keep_text_and_reject_malformed_content() -> None:
    observed: list[tuple[str, object]] = []

    async def receive(event: object) -> None:
        pass

    session = SimpleNamespace(
        _event_builder=SonicEventBuilder("prompt", "audio"),
        _event_handlers={
            "text_output_content": receive,
            "text_output_content_start": receive,
        },
        _turn_tracker=TurnTracker(lambda name, event: observed.append((name, event)), lambda: None),
    )
    await RealtimeSession._handle_event(
        session,
        {"event": {"textOutput": {"role": "USER", "content": "hello", "contentId": "user"}}},
    )
    await RealtimeSession._handle_event(
        session,
        {
            "event": {
                "contentStart": {
                    "role": "ASSISTANT",
                    "type": "TEXT",
                    "contentId": "assistant",
                    "additionalModelFields": '{"generationStage":"SPECULATIVE"}',
                }
            }
        },
    )
    transcripts = [
        event for name, event in observed if name == "input_audio_transcription_completed"
    ]
    assert [event.transcript for event in transcripts] == ["hello", "hello"]
    assert [event.is_final for event in transcripts] == [False, True]

    with pytest.raises(APIConnectionError, match="invalid Nova Sonic event") as error:
        await RealtimeSession._handle_event(
            session,
            {"event": {"textOutput": {"role": "USER", "content": ["bad"], "contentId": "next"}}},
        )
    assert error.value.__cause__ is not None


def test_nova_history_encodes_plain_instruction_and_dialogue_strings() -> None:
    history = llm.ChatContext()
    history.add_message(role="user", content="Book a table.")
    history.add_message(role="assistant", content="For how many?")
    initial, dialogue = SonicEventBuilder("prompt", "audio").create_prompt_start_block(
        "tiffany", 24_000, "You are the desk.", history
    )

    def text(events: list[str]) -> list[str]:
        parsed = [Event.model_validate_json(event).event for event in events]
        return [
            event.textInput.content for event in parsed if isinstance(event, TextInputContentEvent)
        ]

    assert text(initial) == ["You are the desk."]
    assert text(dialogue) == ["Book a table.", "For how many?"]
