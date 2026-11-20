from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from livekit.agents import llm
from livekit.plugins.phonic.realtime import (
    PhonicConversationEndedEvent,
    PhonicNoInputTimeoutError,
    RealtimeModel,
    realtime_model,
)

pytestmark = pytest.mark.unit


class _FakeSocket:
    def __init__(self, *, message_type: str | None = None, close_code: int | None = None) -> None:
        self._message_type = message_type
        self._close_code = close_code

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Any]:
        if self._message_type is not None:
            yield SimpleNamespace(type=self._message_type)
        if self._close_code is not None:
            raise ConnectionClosedError(Close(self._close_code, "closed"), None)


class _FakeAgentSession:
    def __init__(self) -> None:
        self.shutdown_drain: list[bool] = []

    def shutdown(self, *, drain: bool = True) -> None:
        self.shutdown_drain.append(drain)


@pytest.fixture
def agent_session(monkeypatch: pytest.MonkeyPatch) -> _FakeAgentSession:
    session = _FakeAgentSession()
    job_ctx = SimpleNamespace(primary_session=session)
    monkeypatch.setattr(realtime_model, "get_job_context", lambda required: job_ctx)
    return session


@pytest.mark.parametrize(
    ("socket", "reason"),
    [
        (_FakeSocket(message_type="assistant_ended_conversation"), "assistant_ended_conversation"),
        (_FakeSocket(close_code=4505), "time_limit_reached"),
    ],
)
async def test_normal_end_shuts_down_the_agent_session_without_an_error(
    agent_session: _FakeAgentSession, socket: _FakeSocket, reason: str
) -> None:
    session = RealtimeModel(api_key="test").session()
    errors: list[llm.RealtimeModelError] = []
    ended: list[PhonicConversationEndedEvent] = []
    session.on("error", errors.append)
    session.on("phonic_conversation_ended", ended.append)
    try:
        await session._recv_task(socket)  # type: ignore[arg-type]

        assert errors == []
        assert [ev.reason for ev in ended] == [reason]
        assert agent_session.shutdown_drain == [True]
    finally:
        await session.aclose()


async def test_no_input_timeout_is_one_typed_unrecoverable_error(
    agent_session: _FakeAgentSession,
) -> None:
    session = RealtimeModel(api_key="test").session()
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    try:
        await session._recv_task(_FakeSocket(close_code=4500))  # type: ignore[arg-type]

        assert len(errors) == 1
        assert not errors[0].recoverable
        assert isinstance(errors[0].error, PhonicNoInputTimeoutError)
        assert agent_session.shutdown_drain == []
    finally:
        await session.aclose()


async def test_other_close_codes_still_raise(agent_session: _FakeAgentSession) -> None:
    session = RealtimeModel(api_key="test").session()
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    try:
        with pytest.raises(ConnectionClosedError):
            await session._recv_task(_FakeSocket(close_code=4501))  # type: ignore[arg-type]

        assert not isinstance(errors[0].error, PhonicNoInputTimeoutError)
        assert agent_session.shutdown_drain == []
    finally:
        await session.aclose()
