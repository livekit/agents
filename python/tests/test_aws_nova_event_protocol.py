from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from livekit.agents import APIConnectionError, llm
from livekit.plugins.aws.experimental.realtime import realtime_model
from livekit.plugins.aws.experimental.realtime.events import (
    Event,
    SonicEventBuilder,
    TextInputContentEvent,
)
from livekit.plugins.aws.experimental.realtime.realtime_model import (
    RealtimeSession,
    _is_recoverable_validation_error,
)
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


@pytest.mark.parametrize(
    "model, nested",
    [
        ("amazon.nova-sonic-v1:0", False),
        ("us.amazon.nova-sonic-v1:0", False),
        ("amazon.nova-2-sonic-v1:0", True),
        ("us.amazon.nova-2-sonic-v1:0", True),
        ("amazon.nova-2-5-sonic", True),
        ("us.amazon.nova-2-5-sonic", True),
    ],
)
def test_session_start_nests_turn_detection_for_nova_2_and_later(model: str, nested: bool) -> None:
    """Nova Sonic 2 and later reject a sessionStart whose endpointing sensitivity is not under
    ``turnDetectionConfiguration``; Nova Sonic 1 only knows the flat field. Cross-region
    inference-profile ids follow their model."""
    builder = SonicEventBuilder("prompt", "audio", model=model)
    payload = builder.create_session_start_event(endpointing_sensitivity="HIGH")
    start = json.loads(payload)["event"]["sessionStart"]
    if nested:
        assert start["turnDetectionConfiguration"] == {"endpointingSensitivity": "HIGH"}
        assert "endpointingSensitivity" not in start
    else:
        assert start["endpointingSensitivity"] == "HIGH"
        assert "turnDetectionConfiguration" not in start


def test_only_system_instability_is_a_recoverable_validation_error() -> None:
    recoverable = SimpleNamespace(message="System instability detected. Please retry your request.")
    assert _is_recoverable_validation_error(recoverable) is True
    invalid = SimpleNamespace(message="The provided request is invalid.")
    assert _is_recoverable_validation_error(invalid) is False


async def test_a_replaced_response_task_does_not_deactivate_the_restarted_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stream error restarts the session; the old response task, ending after the restart
    installed its replacement, must leave the new session active."""

    class RetryableStreamError(Exception):
        message = "stream dropped"

    class OtherStreamError(Exception):
        pass

    for exception_name in (
        "ValidationException",
        "ThrottlingException",
        "ModelNotReadyException",
        "ModelErrorException",
        "InvalidEventBytes",
        "ModelTimeoutException",
    ):
        monkeypatch.setattr(realtime_model, exception_name, OtherStreamError)
    monkeypatch.setattr(realtime_model, "ModelStreamErrorException", RetryableStreamError)

    class OutputStream:
        async def receive(self) -> None:
            raise RetryableStreamError("stream dropped")

    class StreamResponse:
        async def await_output(self) -> tuple[None, OutputStream]:
            return None, OutputStream()

    session = object.__new__(RealtimeSession)
    session._is_sess_active = asyncio.Event()
    session._is_sess_active.set()
    session._stream_ready = asyncio.Event()
    session._stream_response = StreamResponse()
    session._realtime_model = SimpleNamespace(_label="test")
    session._events = {}
    session._response_task = asyncio.current_task()

    async def restart_session(_exception: Exception) -> None:
        session._response_task = object()

    session._restart_session = restart_session
    await session._process_responses()
    assert session._is_sess_active.is_set()
