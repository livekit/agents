"""Convenience views on the typed messages an endpoint handler receives."""

import pytest

from livekit.agents.a2a import DataPart, FilePart, Message, TextPart

pytestmark = pytest.mark.unit


def test_message_views_preserve_text_order_and_first_object_data():
    first = {"booking": "confirmed", "details": {"room": 42}}
    message = Message(
        message_id="message",
        parts=[
            TextPart(text="Hello"),
            DataPart(data=["non-object data"]),
            FilePart(url="https://example.test/receipt.pdf"),
            DataPart(data=first),
            TextPart(text="Your room is ready"),
            DataPart(data={"booking": "later data"}),
        ],
    )

    assert message.text == "Hello\nYour room is ready"
    assert message.data == first
    assert Message(parts=[]).text == ""
    assert Message(parts=[DataPart(data=42)]).data is None
    assert message.parts[0] == TextPart(text="Hello")
