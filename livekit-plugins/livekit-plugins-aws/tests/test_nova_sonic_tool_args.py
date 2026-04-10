"""
Regression tests for Nova Sonic tool-call argument parsing.

Nova Sonic may deliver toolUse.content as a doubly-encoded JSON string — a
JSON string whose value is itself a JSON object string.  When this reaches
prepare_function_arguments, pydantic_core.from_json() returns a Python str
instead of a dict, causing:

    TypeError: string indices must be integers, not 'str'   (utils.py:404)

The fix peels off one encoding layer *only* when the inner string is itself a
valid JSON object.  Legitimate string-valued schemas (e.g. content="hello")
must be left untouched so that raw tool schemas with primitive top-level types
continue to work correctly.
"""

import json
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub out the optional AWS Smithy/Bedrock SDK not installed in the base venv.
# ---------------------------------------------------------------------------
_AWS_STUBS = [
    "aws_sdk_bedrock_runtime",
    "aws_sdk_bedrock_runtime.client",
    "aws_sdk_bedrock_runtime.models",
    "aws_sdk_bedrock_runtime.config",
    "smithy_aws_core",
    "smithy_aws_core.identity",
    "smithy_aws_event_stream",
    "smithy_aws_event_stream.exceptions",
    "smithy_core",
    "smithy_core.aio",
    "smithy_core.aio.interfaces",
    "smithy_core.aio.interfaces.identity",
]
for _mod in _AWS_STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()


def _make_tool_event(content) -> dict:
    return {
        "event": {
            "toolUse": {
                "toolUseId": "test-id-123",
                "toolName": "check_availability",
                "content": content,
            }
        }
    }


def _make_fake_session(captured: list) -> MagicMock:
    ch = MagicMock()
    ch.send_nowait = lambda call: captured.append(call)

    generation = MagicMock()
    generation.function_ch = ch

    session = MagicMock()
    session._current_generation = generation
    session._pending_tools = set()
    session._close_current_generation = MagicMock()
    return session


class TestHandleToolOutputContentEvent:
    """Unit tests for _handle_tool_output_content_event."""

    async def test_doubly_encoded_string_is_unwrapped(self):
        """Bug case: content is a JSON string wrapping another JSON string.

        Nova Sonic sends:  '"{\\"input\\":{\\"date\\":\\"2026-04-10\\"}}"'
        After fix:         '{"input":{"date":"2026-04-10"}}'  (one layer removed)
        """
        from livekit.plugins.aws.experimental.realtime.realtime_model import (
            RealtimeSession,
        )

        captured = []
        session = _make_fake_session(captured)

        inner_json = json.dumps({"input": {"date": "2026-04-10"}})
        doubly_encoded = json.dumps(inner_json)  # wrap in another JSON string
        event = _make_tool_event(doubly_encoded)

        await RealtimeSession._handle_tool_output_content_event(session, event)

        assert len(captured) == 1
        # arguments must be the inner JSON string (one layer removed), not the
        # doubly-encoded original
        assert captured[0].arguments == inner_json

    async def test_single_encoded_string_passed_through(self):
        """Normal case: content is already a proper JSON object string."""
        from livekit.plugins.aws.experimental.realtime.realtime_model import (
            RealtimeSession,
        )

        captured = []
        session = _make_fake_session(captured)

        json_str = json.dumps({"input": {"date": "2026-04-10"}})
        event = _make_tool_event(json_str)

        await RealtimeSession._handle_tool_output_content_event(session, event)

        assert len(captured) == 1
        assert captured[0].arguments == json_str

    async def test_invalid_json_string_does_not_crash(self):
        """Invalid JSON string → plugin leaves it as-is rather than raising."""
        from livekit.plugins.aws.experimental.realtime.realtime_model import (
            RealtimeSession,
        )

        captured = []
        session = _make_fake_session(captured)

        event = _make_tool_event("not-valid-json")

        await RealtimeSession._handle_tool_output_content_event(session, event)

        assert len(captured) == 1
        assert captured[0].arguments == "not-valid-json"

    async def test_string_primitive_schema_not_unwrapped(self):
        """Regression: content is a JSON string literal (valid primitive schema).

        Bedrock raw tool schemas may legitimately pass a string value such as
        '"hello"'.  This must NOT be unwrapped to 'hello' (which would be invalid
        JSON and cause from_json() to fail downstream).
        """
        from livekit.plugins.aws.experimental.realtime.realtime_model import (
            RealtimeSession,
        )

        captured = []
        session = _make_fake_session(captured)

        string_arg = json.dumps("hello")  # produces '"hello"'
        event = _make_tool_event(string_arg)

        await RealtimeSession._handle_tool_output_content_event(session, event)

        assert len(captured) == 1
        # Must be the original '"hello"', not the bare string 'hello'
        assert captured[0].arguments == string_arg

    async def test_tool_name_and_id_forwarded_correctly(self):
        """call_id and name are passed through regardless of args format."""
        from livekit.plugins.aws.experimental.realtime.realtime_model import (
            RealtimeSession,
        )

        captured = []
        session = _make_fake_session(captured)

        event = _make_tool_event(json.dumps({"date": "2026-04-10"}))

        await RealtimeSession._handle_tool_output_content_event(session, event)

        assert captured[0].call_id == "test-id-123"
        assert captured[0].name == "check_availability"
