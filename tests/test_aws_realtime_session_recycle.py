from __future__ import annotations

import asyncio

import pytest

from livekit.plugins.aws.experimental.realtime.realtime_model import RealtimeModel, RealtimeSession

pytestmark = pytest.mark.unit


@pytest.fixture
async def aws_realtime_session(monkeypatch: pytest.MonkeyPatch):
    async def fake_initialize_streams(self: RealtimeSession, is_restart: bool = False):
        return self

    monkeypatch.setattr(RealtimeSession, "initialize_streams", fake_initialize_streams)

    session = RealtimeModel().session()
    await session._main_atask
    try:
        yield session
    finally:
        if session._session_recycle_task and not session._session_recycle_task.done():
            session._session_recycle_task.cancel()
            await asyncio.gather(session._session_recycle_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_start_session_recycle_timer_does_not_cancel_current_timer_task(
    aws_realtime_session: RealtimeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aws_realtime_session, "_calculate_session_duration", lambda: 60.0)

    current_task = asyncio.current_task()
    assert current_task is not None

    aws_realtime_session._session_recycle_task = current_task
    aws_realtime_session._start_session_recycle_timer()
    await asyncio.sleep(0)

    assert not current_task.cancelled()
    assert aws_realtime_session._session_recycle_task is not current_task
    assert aws_realtime_session._session_recycle_task is not None


@pytest.mark.asyncio
async def test_start_session_recycle_timer_cancels_previous_timer_task(
    aws_realtime_session: RealtimeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aws_realtime_session, "_calculate_session_duration", lambda: 60.0)

    async def previous_timer() -> None:
        await asyncio.sleep(60)

    previous_task = asyncio.create_task(previous_timer())
    aws_realtime_session._session_recycle_task = previous_task

    aws_realtime_session._start_session_recycle_timer()
    await asyncio.sleep(0)

    assert previous_task.cancelled()
    assert aws_realtime_session._session_recycle_task is not None
    assert aws_realtime_session._session_recycle_task is not previous_task
