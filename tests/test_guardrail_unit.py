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

# Mock classes


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
        self._history = ChatContext()
        self._handlers: dict[str, list] = {}

    @property
    def history(self):
        return self._history

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


# Tests


def test_guardrail_defaults():
    g = Guardrail(instructions="test", llm="openai/gpt-4o-mini")
    assert g.eval_interval == 3
    assert g.max_interventions == 5
    assert g.cooldown == 10.0
    assert g.inject_role == "system"
    assert g.inject_prefix == "[GUARDRAIL ADVISOR]:"


def test_guardrail_custom():
    g = Guardrail(
        instructions="watch for X",
        llm="openai/gpt-4o",
        eval_interval=2,
        max_interventions=3,
        cooldown=5.0,
        inject_role="assistant",
        inject_prefix="[ALERT]",
    )
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


def test_guardrail_state_mutation():
    s = _GuardrailState()
    s.turn_count = 5
    s.interventions_count = 2
    s.last_intervention_time = 100.0
    assert s.turn_count == 5
    assert s.interventions_count == 2
    assert s.last_intervention_time == 100.0


def test_runner_start_stop():
    session = MockSession()
    config = Guardrail(instructions="test", llm="openai/gpt-4o-mini")

    with patch.object(inference.LLM, "from_model_string") as mock:
        mock.return_value = MagicMock()
        runner = _GuardrailRunner(config, session)

    assert not runner._started
    runner.start()
    assert runner._started
    assert runner._event_handler is not None

    runner.stop()
    assert not runner._started
    assert runner._event_handler is None


def test_runner_double_start():
    session = MockSession()
    config = Guardrail(instructions="test", llm="openai/gpt-4o-mini")

    with patch.object(inference.LLM, "from_model_string") as mock:
        mock.return_value = MagicMock()
        runner = _GuardrailRunner(config, session)

    runner.start()
    runner.start()  # no error
    assert runner._started


def test_runner_parse_json_valid():
    session = MockSession()
    config = Guardrail(instructions="test", llm="openai/gpt-4o-mini")

    with patch.object(inference.LLM, "from_model_string") as mock:
        mock.return_value = MagicMock()
        runner = _GuardrailRunner(config, session)

    result = runner._parse_json('{"intervene": true, "advice": "do X"}')
    assert result == {"intervene": True, "advice": "do X"}


def test_runner_parse_json_markdown():
    session = MockSession()
    config = Guardrail(instructions="test", llm="openai/gpt-4o-mini")

    with patch.object(inference.LLM, "from_model_string") as mock:
        mock.return_value = MagicMock()
        runner = _GuardrailRunner(config, session)

    result = runner._parse_json('```json\n{"intervene": false}\n```')
    assert result == {"intervene": False}


def test_runner_parse_json_invalid():
    session = MockSession()
    config = Guardrail(instructions="test", llm="openai/gpt-4o-mini")

    with patch.object(inference.LLM, "from_model_string") as mock:
        mock.return_value = MagicMock()
        runner = _GuardrailRunner(config, session)

    result = runner._parse_json("not json")
    assert result is None


def test_runner_build_transcript():
    session = MockSession()
    session._history.add_message(role="user", content="hello")
    session._history.add_message(role="assistant", content="hi there")
    session._history.add_message(role="user", content="book table")

    config = Guardrail(instructions="test", llm="openai/gpt-4o-mini")

    with patch.object(inference.LLM, "from_model_string") as mock:
        mock.return_value = MagicMock()
        runner = _GuardrailRunner(config, session)

    transcript = runner._build_transcript()
    assert "[USER]: hello" in transcript
    assert "[ASSISTANT]: hi there" in transcript
    assert "[USER]: book table" in transcript


@pytest.mark.parametrize(
    "inject_role",
    ["system", "assistant", "developer"],
)
def test_guardrail_inject_roles(inject_role: str):
    g = Guardrail(instructions="test", llm="openai/gpt-4o-mini", inject_role=inject_role)
    assert g.inject_role == inject_role
