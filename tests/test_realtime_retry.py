from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit.agents.llm.realtime import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BASE_DELAY,
    DEFAULT_RETRY_MAX_DELAY,
    GenerationCreatedEvent,
    RealtimeError,
    RealtimeModel,
    RealtimeSession,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.llm.tool_context import Tool, ToolChoice

pytestmark = pytest.mark.unit


class MockRealtimeModel(RealtimeModel):
    def __init__(self) -> None:
        from livekit.agents.llm.realtime import RealtimeCapabilities

        super().__init__(
            capabilities=RealtimeCapabilities(
                message_truncation=False,
                turn_detection=False,
                user_transcription=False,
                auto_tool_reply_generation=False,
                audio_output=False,
                manual_function_calls=False,
            )
        )

    def session(self) -> MockRealtimeSession:
        return MockRealtimeSession(self)

    async def aclose(self) -> None:
        pass


class MockRealtimeSession(RealtimeSession):
    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._call_count = 0
        self._fail_until_attempt = 0
        self._error_type = "recoverable"
        self._generation_event = None

    def set_failure_mode(
        self, fail_until_attempt: int, error_type: str = "recoverable"
    ) -> None:
        self._fail_until_attempt = fail_until_attempt
        self._error_type = error_type

    def _do_generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[GenerationCreatedEvent]:
        self._call_count += 1
        fut: asyncio.Future[GenerationCreatedEvent] = asyncio.Future()

        if self._call_count <= self._fail_until_attempt:
            if self._error_type == "recoverable":
                fut.set_exception(
                    RealtimeError("recoverable error", recoverable=True)
                )
            elif self._error_type == "non_recoverable":
                fut.set_exception(
                    RealtimeError("non-recoverable error", recoverable=False)
                )
            else:
                fut.set_exception(RuntimeError("unexpected error"))
        else:
            mock_event = MagicMock(spec=GenerationCreatedEvent)
            mock_event.user_initiated = True
            fut.set_result(mock_event)

        return fut

    @property
    def chat_ctx(self):
        from livekit.agents.llm.chat_context import ChatContext
        return ChatContext()

    @property
    def tools(self):
        from livekit.agents.llm.tool_context import ToolContext
        return ToolContext.empty()

    async def update_instructions(self, instructions: str) -> None:
        pass

    async def update_chat_ctx(self, chat_ctx) -> None:
        pass

    async def update_tools(self, tools: list[Tool]) -> None:
        pass

    def update_options(self, *, tool_choice=NOT_GIVEN) -> None:
        pass

    def push_audio(self, frame) -> None:
        pass

    def push_video(self, frame) -> None:
        pass

    def commit_audio(self) -> None:
        pass

    def clear_audio(self) -> None:
        pass

    def interrupt(self) -> None:
        pass

    def truncate(self, *, message_id, modalities, audio_end_ms, audio_transcript=NOT_GIVEN) -> None:
        pass

    async def aclose(self) -> None:
        pass


@pytest.fixture
def model():
    return MockRealtimeModel()


@pytest.fixture
def session(model):
    return model.session()


@pytest.mark.asyncio
async def test_realtime_error_has_recoverable_flag():
    err = RealtimeError("test error", recoverable=True)
    assert err.recoverable is True
    assert str(err) == "test error"

    err2 = RealtimeError("test error", recoverable=False)
    assert err2.recoverable is False


@pytest.mark.asyncio
async def test_realtime_error_default_recoverable():
    err = RealtimeError("test error")
    assert err.recoverable is True


@pytest.mark.asyncio
async def test_generate_reply_success_no_retry(session):
    session.set_failure_mode(fail_until_attempt=0)
    fut = session.generate_reply()
    result = await asyncio.wait_for(fut, timeout=5.0)
    assert result is not None
    assert result.user_initiated is True
    assert session._call_count == 1


@pytest.mark.asyncio
async def test_generate_reply_retries_on_recoverable_error(session):
    session.set_failure_mode(fail_until_attempt=2, error_type="recoverable")

    with patch("livekit.agents.llm.realtime.DEFAULT_MAX_RETRIES", 3), \
         patch("livekit.agents.llm.realtime.DEFAULT_RETRY_BASE_DELAY", 0.01), \
         patch("livekit.agents.llm.realtime.DEFAULT_RETRY_MAX_DELAY", 0.1):
        fut = session.generate_reply()
        result = await asyncio.wait_for(fut, timeout=5.0)

    assert result is not None
    assert session._call_count == 3


@pytest.mark.asyncio
async def test_generate_reply_no_retry_on_non_recoverable_error(session):
    session.set_failure_mode(fail_until_attempt=1, error_type="non_recoverable")

    with patch("livekit.agents.llm.realtime.DEFAULT_MAX_RETRIES", 3), \
         patch("livekit.agents.llm.realtime.DEFAULT_RETRY_BASE_DELAY", 0.01), \
         patch("livekit.agents.llm.realtime.DEFAULT_RETRY_MAX_DELAY", 0.1):
        fut = session.generate_reply()
        with pytest.raises(RealtimeError, match="non-recoverable error"):
            await asyncio.wait_for(fut, timeout=5.0)

    assert session._call_count == 1


@pytest.mark.asyncio
async def test_generate_retry_exhaustion(session):
    session.set_failure_mode(fail_until_attempt=10, error_type="recoverable")

    with patch("livekit.agents.llm.realtime.DEFAULT_MAX_RETRIES", 2), \
         patch("livekit.agents.llm.realtime.DEFAULT_RETRY_BASE_DELAY", 0.01), \
         patch("livekit.agents.llm.realtime.DEFAULT_RETRY_MAX_DELAY", 0.1):
        fut = session.generate_reply()
        with pytest.raises(RealtimeError, match="failed after 2 retries"):
            await asyncio.wait_for(fut, timeout=5.0)

    assert session._call_count == 3


@pytest.mark.asyncio
async def test_generate_reply_unexpected_error_no_retry(session):
    session.set_failure_mode(fail_until_attempt=1, error_type="unexpected")

    with patch("livekit.agents.llm.realtime.DEFAULT_MAX_RETRIES", 3), \
         patch("livekit.agents.llm.realtime.DEFAULT_RETRY_BASE_DELAY", 0.01), \
         patch("livekit.agents.llm.realtime.DEFAULT_RETRY_MAX_DELAY", 0.1):
        fut = session.generate_reply()
        with pytest.raises(RuntimeError, match="unexpected error"):
            await asyncio.wait_for(fut, timeout=5.0)

    assert session._call_count == 1


@pytest.mark.asyncio
async def test_generate_reply_env_config():
    import os
    with patch.dict(os.environ, {
        "LIVEKIT_REALTIME_MAX_RETRIES": "5",
        "LIVEKIT_REALTIME_RETRY_BASE_DELAY": "0.5",
        "LIVEKIT_REALTIME_RETRY_MAX_DELAY": "15.0",
    }):
        import importlib
        import livekit.agents.llm.realtime as rt_module
        importlib.reload(rt_module)
        assert rt_module.DEFAULT_MAX_RETRIES == 5
        assert rt_module.DEFAULT_RETRY_BASE_DELAY == 0.5
        assert rt_module.DEFAULT_RETRY_MAX_DELAY == 15.0
