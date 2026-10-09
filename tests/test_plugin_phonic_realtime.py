from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from livekit.agents import utils
from livekit.plugins.phonic.realtime.realtime_model import RealtimeModel, RealtimeSession

pytestmark = pytest.mark.unit


@asynccontextmanager
async def _make_session(model: RealtimeModel) -> AsyncIterator[RealtimeSession]:
    """A session whose background connect loop is stopped before it hits the network.

    ``_config_sent`` is set because ``update_options`` deliberately schedules no reset
    until the initial config has gone out, and these tests are about what happens after
    that point.
    """
    session = model.session()
    session._send_ch.close()
    await utils.aio.cancel_and_wait(session._main_atask)
    session._config_sent = True
    try:
        yield session
    finally:
        await session.aclose()


async def test_update_options_resets_every_active_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHONIC_API_KEY", "fake-key")
    model = RealtimeModel(voice="original")

    async with _make_session(model) as first, _make_session(model) as second:
        model.update_options(voice="updated")

        assert first._options_reset_task is not None
        assert second._options_reset_task is not None
        assert first._opts.voice == "updated"
        assert second._opts.voice == "updated"


async def test_update_options_ignores_values_that_did_not_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHONIC_API_KEY", "fake-key")
    model = RealtimeModel(voice="original")

    async with _make_session(model) as session:
        model.update_options(voice="original")

        assert session._options_reset_task is None


async def test_instructions_set_on_one_session_do_not_reach_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHONIC_API_KEY", "fake-key")
    model = RealtimeModel(voice="original")

    async with _make_session(model) as first, _make_session(model) as second:
        # instructions are only accepted before the initial config goes out
        first._config_sent = False
        second._config_sent = False
        untouched = second._opts.instructions

        await first.update_instructions("for the first session only")

        assert first._opts.instructions == "for the first session only"
        assert second._opts.instructions == untouched
