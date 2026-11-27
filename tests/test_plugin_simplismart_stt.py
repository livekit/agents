from __future__ import annotations

import asyncio
import gc

import pytest

from livekit.plugins.simplismart import stt as simplismart_stt

pytestmark = pytest.mark.unit


async def _idle_run(self: object) -> None:
    del self
    await asyncio.Event().wait()  # cancelled by aclose()


async def test_aclose_closes_every_stream_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(simplismart_stt.SpeechStream, "_run", _idle_run)

    stt = simplismart_stt.STT(api_key="sk_test")
    closed_first = stt.stream()
    sessions = [closed_first._session, stt.stream()._session]
    await closed_first.aclose()

    await stt.aclose()

    assert all(s.closed for s in sessions)
    assert len(stt._streams) == 0


async def test_aclose_closes_session_of_dropped_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """A finished stream the caller dropped must not take its session to the GC."""

    async def _immediate_run(self: object) -> None:
        del self  # return immediately: the stream task finishes on its own

    monkeypatch.setattr(simplismart_stt.SpeechStream, "_run", _immediate_run)

    stt = simplismart_stt.STT(api_key="sk_test")
    stream = stt.stream()
    session = stream._session

    del stream
    gc.collect()

    await stt.aclose()

    assert session.closed
