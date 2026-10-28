import pytest

from livekit.plugins.ultravox.realtime.events import (
    ClientToolResultEvent,
    UserTextMessageEvent,
    parse_ultravox_event,
)

pytestmark = pytest.mark.unit


def test_deferred_user_message_preserves_interruption_policy() -> None:
    event = UserTextMessageEvent(text="", urgency="immediate", defer_response=True)
    wire = event.model_dump(by_alias=True, exclude_none=True)
    assert wire["deferResponse"] is True
    assert wire["urgency"] == "immediate"
    assert parse_ultravox_event(wire) == event


def test_tool_result_can_set_response_policy_by_python_field_name() -> None:
    event = ClientToolResultEvent(invocation_id="call-1", result="done", agent_reaction="listens")
    wire = event.model_dump(by_alias=True, exclude_none=True)
    assert wire["invocationId"] == "call-1"
    assert wire["agentReaction"] == "listens"
    assert parse_ultravox_event(wire) == event
