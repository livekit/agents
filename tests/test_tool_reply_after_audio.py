from __future__ import annotations

import pytest

from livekit.agents import AgentSession
from livekit.agents.voice.events import FunctionToolsExecutedEvent
from livekit.agents.voice.tool_executor import ToolHandlingOptions

pytestmark = pytest.mark.unit


class TestFunctionToolsExecutedEventResponseHadAudio:
    def test_default_response_had_audio_is_false(self) -> None:
        ev = FunctionToolsExecutedEvent(
            function_calls=[],
            function_call_outputs=[],
        )
        assert ev.response_had_audio is False

    def test_response_had_audio_set_true(self) -> None:
        ev = FunctionToolsExecutedEvent(
            function_calls=[],
            function_call_outputs=[],
        )
        ev._response_had_audio = True
        assert ev.response_had_audio is True

    def test_cancel_tool_reply_still_works(self) -> None:
        ev = FunctionToolsExecutedEvent(
            function_calls=[],
            function_call_outputs=[],
        )
        ev._reply_required = True
        ev._response_had_audio = True
        assert ev.has_tool_reply is True
        ev.cancel_tool_reply()
        assert ev.has_tool_reply is False


class TestToolHandlingOptionsConfig:
    def test_default_tool_reply_after_audio(self) -> None:
        opts: ToolHandlingOptions = {}
        assert opts.get("tool_reply_after_audio", "always") == "always"

    def test_skip_tool_reply_after_audio(self) -> None:
        opts: ToolHandlingOptions = {"tool_reply_after_audio": "skip"}
        assert opts["tool_reply_after_audio"] == "skip"

    def test_always_tool_reply_after_audio(self) -> None:
        opts: ToolHandlingOptions = {"tool_reply_after_audio": "always"}
        assert opts["tool_reply_after_audio"] == "always"


class TestAgentSessionToolReplyConfig:
    def test_session_defaults_to_always(self) -> None:
        session = AgentSession()
        assert session._tool_reply_after_audio == "always"

    def test_session_accepts_skip(self) -> None:
        session = AgentSession(tool_handling={"tool_reply_after_audio": "skip"})
        assert session._tool_reply_after_audio == "skip"

    def test_session_accepts_always_explicit(self) -> None:
        session = AgentSession(tool_handling={"tool_reply_after_audio": "always"})
        assert session._tool_reply_after_audio == "always"

    def test_session_with_empty_tool_handling(self) -> None:
        session = AgentSession(tool_handling={})
        assert session._tool_reply_after_audio == "always"
