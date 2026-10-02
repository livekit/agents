from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit.agents import decisions
from livekit.agents.llm import ChatContext

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.fixture
async def receptionist(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    from examples.voice_agents import decision_receptionist as example

    session = MagicMock()
    session.history = ChatContext.empty()
    session.start = AsyncMock()
    handlers = {}

    def on(event):
        def register(callback):
            handlers[event] = callback
            return callback

        return register

    session.on.side_effect = on
    monkeypatch.setattr(example, "AgentSession", lambda **kwargs: session)
    monkeypatch.setattr(example.openai.LLM, "with_openrouter", lambda **kwargs: None)
    monkeypatch.setattr(example.typesafe.Jev, "with_openrouter", lambda: None)
    actions = []
    log_callback = MagicMock(side_effect=lambda **kwargs: actions.append("log"))
    monkeypatch.setattr(example, "log_callback_desire", log_callback)
    session.generate_reply.side_effect = lambda **kwargs: actions.append("confirm")
    await example.entrypoint(SimpleNamespace(room=None))
    return session, handlers["decisions_completed"], log_callback, actions


def callback_event(message, probability=0.95):
    return decisions.DecisionsCompletedEvent(
        results={"wants_callback": decisions.ProbabilityResult(value=probability)},
        source_message_id=message.id,
        agent_id="receptionist",
        activity_id="activity",
    )


async def test_callback_confirmation_follows_logging_and_only_happens_once(receptionist) -> None:
    session, on_decisions, log_callback, actions = receptionist
    old = session.history.add_message(role="user", content="Please call me back.")
    current = session.history.add_message(role="user", content="Yes, a callback please.")
    on_decisions(callback_event(old))
    on_decisions(callback_event(current, probability=0.8))
    assert actions == []
    on_decisions(callback_event(current))
    on_decisions(callback_event(current))
    assert actions == ["log", "confirm"]
    log_callback.assert_called_once_with(source_message_id=current.id, probability=0.95)
    assert session.start.call_args.kwargs["agent"].tools == []


async def test_callback_logging_failure_never_confirms_success(receptionist) -> None:
    session, on_decisions, log_callback, actions = receptionist
    log_callback.side_effect = RuntimeError("callback logging failed")
    message = session.history.add_message(role="user", content="Please call me back.")
    with pytest.raises(RuntimeError, match="callback logging failed"):
        on_decisions(callback_event(message))
    session.generate_reply.assert_not_called()
    assert actions == []

    log_callback.side_effect = lambda **kwargs: actions.append("log")
    repeated = session.history.add_message(role="user", content="Please try that callback again.")
    on_decisions(callback_event(repeated))
    on_decisions(callback_event(repeated))
    assert actions == ["log", "confirm"]
    assert log_callback.call_count == 2
