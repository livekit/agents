from __future__ import annotations

import asyncio
import gc

import pytest

from livekit.plugins.simplismart import stt as simplismart_stt

pytestmark = pytest.mark.unit


async def _idle_run(self: object) -> None:
    del self
    await asyncio.Event().wait()  # cancelled by aclose()


@pytest.mark.asyncio
async def test_simplismart_stt_aclose_closes_tracked_stream_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(simplismart_stt.SpeechStream, "_run", _idle_run)

    stt = simplismart_stt.STT(api_key="sk_test")
    stream_a = stt.stream()
    stream_b = stt.stream()
    sessions = [stream_a._session, stream_b._session]

    assert all(not s.closed for s in sessions)
    assert len(stt._streams) == 2

    await stt.aclose()

    assert all(s.closed for s in sessions), (
        "STT.aclose() must close every per-stream aiohttp session"
    )
    assert len(stt._streams) == 0


@pytest.mark.asyncio
async def test_simplismart_stt_async_context_closes_stream_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(simplismart_stt.SpeechStream, "_run", _idle_run)

    async with simplismart_stt.STT(api_key="sk_test") as stt:
        stream = stt.stream()
        session = stream._session
        assert not session.closed

    assert session.closed, "exiting `async with` must close per-stream sessions"


@pytest.mark.asyncio
async def test_simplismart_stt_aclose_tolerates_already_closed_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(simplismart_stt.SpeechStream, "_run", _idle_run)

    stt = simplismart_stt.STT(api_key="sk_test")
    stream = stt.stream()
    session = stream._session

    await stream.aclose()
    await stt.aclose()  # must not raise on an already-closed stream

    assert session.closed


@pytest.mark.asyncio
async def test_simplismart_stt_aclose_closes_session_of_dropped_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finished stream the caller dropped must not take its session to the GC.

    With weak tracking, a completed SpeechStream can be collected before
    STT.aclose() snapshots the set, leaving its per-stream ClientSession
    unreachable and unclosed. Strong ownership keeps it reachable until
    aclose closes it.
    """

    async def _immediate_run(self: object) -> None:
        del self  # return immediately: the stream task finishes on its own

    monkeypatch.setattr(simplismart_stt.SpeechStream, "_run", _immediate_run)

    stt = simplismart_stt.STT(api_key="sk_test")
    stream = stt.stream()
    session = stream._session

    del stream
    gc.collect()

    assert len(stt._streams) == 1, "a dropped stream must stay tracked until aclose"

    await stt.aclose()

    assert session.closed, "aclose() must close the session of a dropped stream"
    assert len(stt._streams) == 0
