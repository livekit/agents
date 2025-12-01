from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from livekit.agents import inference
from livekit.agents.llm import ChatContext
from livekit.agents.voice.guardrail import (
    Guardrail,
    _GuardrailRunner,
    _GuardrailState,
)


class MockAgent:
    def __init__(self):
        self._chat_ctx = ChatContext()

    @property
    def chat_ctx(self):
        return self._chat_ctx

    async def update_chat_ctx(self, ctx):
        self._chat_ctx = ctx


class MockSession:
    def __init__(self):
        self._agent = MockAgent()
        self._handlers: dict[str, list] = {}

    def on(self, event: str, handler=None):
        if handler:
            self._handlers.setdefault(event, []).append(handler)
            return handler

        def decorator(fn):
            self._handlers.setdefault(event, []).append(fn)
            return fn

        return decorator

    def off(self, event: str, handler):
        if event in self._handlers:
            try:
                self._handlers[event].remove(handler)
            except ValueError:
                pass


def test_guardrail_defaults():
    g = Guardrail(instructions="test", llm="openai/gpt-4o-mini")
    assert g.name is None
    assert g.eval_interval == 3
    assert g.max_interventions == 5
    assert g.cooldown == 10.0
    assert g.inject_role == "system"
    assert g.inject_prefix == "[GUARDRAIL ADVISOR]:"


def test_guardrail_custom():
    g = Guardrail(
        name="compliance",
        instructions="watch for X",
        llm="openai/gpt-4o",
        eval_interval=2,
        max_interventions=3,
        cooldown=5.0,
        inject_role="assistant",
        inject_prefix="[ALERT]",
    )
    assert g.name == "compliance"
    assert g.instructions == "watch for X"
    assert g.eval_interval == 2
    assert g.max_interventions == 3
    assert g.cooldown == 5.0
    assert g.inject_role == "assistant"
    assert g.inject_prefix == "[ALERT]"


def test_guardrail_state_initial():
    s = _GuardrailState()
    assert s.turn_count == 0
    assert s.interventions_count == 0
    assert s.last_intervention_time == 0.0


def test_runner_start_stop():
    session = MockSession()
    config = Guardrail(instructions="test", llm="openai/gpt-4o-mini")

    with patch.object(inference.LLM, "from_model_string") as mock:
        mock.return_value = MagicMock()
        runner = _GuardrailRunner(config, session)

    assert not runner._started
    runner.start()
    assert runner._started
    runner.stop()
    assert not runner._started


def test_runner_log_prefix():
    session = MockSession()

    with patch.object(inference.LLM, "from_model_string") as mock:
        mock.return_value = MagicMock()
        runner_no_name = _GuardrailRunner(
            Guardrail(instructions="test", llm="openai/gpt-4o-mini"), session
        )
        runner_with_name = _GuardrailRunner(
            Guardrail(name="compliance", instructions="test", llm="openai/gpt-4o-mini"),
            session,
        )

    assert runner_no_name._log_prefix == "guardrail"
    assert runner_with_name._log_prefix == "guardrail[compliance]"


def test_runner_parse_json():
    session = MockSession()
    config = Guardrail(instructions="test", llm="openai/gpt-4o-mini")

    with patch.object(inference.LLM, "from_model_string") as mock:
        mock.return_value = MagicMock()
        runner = _GuardrailRunner(config, session)

    # Valid JSON
    assert runner._parse_json('{"intervene": true, "advice": "do X"}') == {
        "intervene": True,
        "advice": "do X",
    }
    # Markdown wrapped
    assert runner._parse_json('```json\n{"intervene": false}\n```') == {"intervene": False}
    # Invalid
    assert runner._parse_json("not json") is None


def test_runner_build_transcript():
    """Transcript is built from agent.chat_ctx, not session.history."""
    session = MockSession()
    # Add messages to agent's chat_ctx (this is what _build_transcript uses now)
    session._agent._chat_ctx.add_message(role="user", content="hello")
    session._agent._chat_ctx.add_message(role="assistant", content="hi there")
    session._agent._chat_ctx.add_message(role="user", content="book table")

    config = Guardrail(instructions="test", llm="openai/gpt-4o-mini")

    with patch.object(inference.LLM, "from_model_string") as mock:
        mock.return_value = MagicMock()
        runner = _GuardrailRunner(config, session)

    transcript = runner._build_transcript()
    assert "[USER]: hello" in transcript
    assert "[ASSISTANT]: hi there" in transcript
    assert "[USER]: book table" in transcript


def test_runner_build_transcript_skips_first_system():
    """First system message (main prompt) is skipped, but our advice is included."""
    session = MockSession()
    # Add messages to agent's chat_ctx
    session._agent._chat_ctx.add_message(
        role="system", content="You are a helpful assistant. This is a very long prompt..."
    )
    session._agent._chat_ctx.add_message(role="user", content="hello")
    session._agent._chat_ctx.add_message(role="assistant", content="hi there")
    session._agent._chat_ctx.add_message(
        role="system", content="[GUARDRAIL ADVISOR]: ask about dietary restrictions"
    )
    session._agent._chat_ctx.add_message(role="user", content="I need help")

    config = Guardrail(instructions="test", llm="openai/gpt-4o-mini")

    with patch.object(inference.LLM, "from_model_string") as mock:
        mock.return_value = MagicMock()
        runner = _GuardrailRunner(config, session)

    transcript = runner._build_transcript()

    # User/assistant messages included
    assert "[USER]: hello" in transcript
    assert "[ASSISTANT]: hi there" in transcript
    assert "[USER]: I need help" in transcript

    # First system message (main prompt) excluded
    assert "helpful assistant" not in transcript

    # Our injected advice IS included (marked as PREVIOUS ADVICE)
    assert "[PREVIOUS ADVICE]:" in transcript
    assert "dietary restrictions" in transcript


def test_runner_build_transcript_no_agent():
    """Returns empty string if no agent."""
    session = MockSession()
    session._agent = None

    config = Guardrail(instructions="test", llm="openai/gpt-4o-mini")

    with patch.object(inference.LLM, "from_model_string") as mock:
        mock.return_value = MagicMock()
        runner = _GuardrailRunner(config, session)

    assert runner._build_transcript() == ""


def test_runner_build_transcript_max_history():
    """max_history limits messages included in transcript."""
    session = MockSession()
    # Add many messages
    for i in range(10):
        role = "user" if i % 2 == 0 else "assistant"
        session._agent._chat_ctx.add_message(role=role, content=f"msg{i}")

    # With max_history=4, only last 4 messages
    config = Guardrail(instructions="test", llm="openai/gpt-4o-mini", max_history=4)

    with patch.object(inference.LLM, "from_model_string") as mock:
        mock.return_value = MagicMock()
        runner = _GuardrailRunner(config, session)

    transcript = runner._build_transcript()
    lines = transcript.strip().split("\n")
    assert len(lines) == 4
    assert "msg6" in transcript
    assert "msg9" in transcript
    assert "msg0" not in transcript  # Older messages excluded


def test_runner_build_transcript_no_limit():
    """Without max_history, all messages are included."""
    session = MockSession()
    for i in range(10):
        role = "user" if i % 2 == 0 else "assistant"
        session._agent._chat_ctx.add_message(role=role, content=f"msg{i}")

    config = Guardrail(instructions="test", llm="openai/gpt-4o-mini")  # max_history=None

    with patch.object(inference.LLM, "from_model_string") as mock:
        mock.return_value = MagicMock()
        runner = _GuardrailRunner(config, session)

    transcript = runner._build_transcript()
    lines = transcript.strip().split("\n")
    assert len(lines) == 10
    assert "msg0" in transcript
    assert "msg9" in transcript


@pytest.mark.parametrize("inject_role", ["system", "assistant"])
def test_guardrail_inject_roles(inject_role: str):
    g = Guardrail(instructions="test", llm="openai/gpt-4o-mini", inject_role=inject_role)
    assert g.inject_role == inject_role
