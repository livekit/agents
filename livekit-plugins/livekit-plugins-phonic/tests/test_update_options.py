from __future__ import annotations

import asyncio

import pytest

from livekit.plugins.phonic import realtime
from livekit.plugins.phonic.realtime import RealtimeModel, RealtimeSession

pytestmark = pytest.mark.unit


def _make_ready_session(model: RealtimeModel) -> RealtimeSession:
    """Build a session and put it in the state it would be in after its initial config was sent:
    ready to start and config already flushed, so update_options schedules a reset."""
    sess = model.session()
    sess._config_sent = True
    sess._ready_to_start.set()
    return sess


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    # RealtimeSession.__init__ constructs an AsyncPhonic client and starts a background task that
    # connects a websocket. Stub the client with a connect() that blocks forever so no network is
    # attempted; the background task parks at connect and the test never sends config.
    class _StubSocketCtx:
        async def __aenter__(self) -> object:
            await asyncio.Event().wait()  # never resolves
            raise AssertionError("unreachable")

        async def __aexit__(self, *args: object) -> None:
            pass

    class _StubConversations:
        def connect(self, *args: object, **kwargs: object) -> _StubSocketCtx:
            return _StubSocketCtx()

    class _StubClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.conversations = _StubConversations()

    monkeypatch.setattr(realtime.realtime_model, "AsyncPhonic", _StubClient)


async def test_update_options_resets_every_session() -> None:
    """Regression: with multiple live sessions on one model, model.update_options must schedule a
    reset on EVERY session, not just the first. Each session owns its _opts copy, so the first
    session mutating a shared object no longer makes later sessions see "no change" and skip."""
    model = RealtimeModel(api_key="test", intelligence_level="standard")

    sess_a = _make_ready_session(model)
    sess_b = _make_ready_session(model)

    try:
        model.update_options(intelligence_level="high")

        # Both sessions applied the change to their own options...
        assert sess_a._opts.intelligence_level == "high"
        assert sess_b._opts.intelligence_level == "high"

        # ...and both scheduled an options-reset task (the bug left the second one None).
        assert sess_a._options_reset_task is not None
        assert sess_b._options_reset_task is not None

        # The model template is updated too, so a session created afterwards inherits the new value.
        assert model._opts.intelligence_level == "high"
        sess_c = _make_ready_session(model)
        try:
            assert sess_c._opts.intelligence_level == "high"
        finally:
            await sess_c.aclose()
    finally:
        await sess_a.aclose()
        await sess_b.aclose()


async def test_update_options_independent_change_detection() -> None:
    """A session that already has the target value still leaves other sessions free to change: the
    per-session copies mean one session's state never suppresses another's reset."""
    model = RealtimeModel(api_key="test", audio_speed=1.0)

    sess_a = _make_ready_session(model)
    # sess_a already at the target value; it should not schedule a reset for a no-op change.
    sess_a._opts.audio_speed = 1.5

    sess_b = _make_ready_session(model)

    try:
        model.update_options(audio_speed=1.5)

        # sess_a saw no change (already 1.5) and scheduled nothing.
        assert sess_a._options_reset_task is None
        # sess_b changed from 1.0 to 1.5 and scheduled its reset.
        assert sess_b._opts.audio_speed == 1.5
        assert sess_b._options_reset_task is not None
    finally:
        await sess_a.aclose()
        await sess_b.aclose()
