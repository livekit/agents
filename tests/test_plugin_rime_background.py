"""Prism background failures must not silently stop event processing."""

# ruff: noqa: E402 -- load the optional SDK before importing the adapter
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

pytestmark = pytest.mark.unit
pytest.importorskip("rimelabs_sdk.realtime")

from livekit.agents import llm
from livekit.plugins.rime.realtime import PrismError

from .test_plugin_rime_realtime import peer_model


async def test_background_dependency_error_stops_the_session(monkeypatch):
    """Inject an SDK failure while exercising the adapter's public clear method."""
    async with peer_model() as (peer, model, client):
        failure = llm.RealtimeError("background clear failed")
        connect = client.realtime.connect

        async def fail_clear():
            raise failure

        @asynccontextmanager
        async def connect_with_failure(**options):
            async with connect(**options) as sdk:
                monkeypatch.setattr(sdk, "clear_audio", fail_clear)
                yield sdk

        monkeypatch.setattr(client.realtime, "connect", connect_with_failure)
        session = model.session()
        errors = asyncio.Queue()
        session.on("error", errors.put_nowait)
        await session._update_session(instructions="Help")
        session.clear_audio()
        error = await asyncio.wait_for(errors.get(), 3)
        assert error.error is failure
        assert not error.recoverable
        with pytest.raises(llm.RealtimeError, match="background clear failed"):
            await session.set_input_text("new turn")
        assert peer.requests.empty()


async def test_reply_refusal_reaches_caller_without_stopping_session():
    async with peer_model() as (peer, model, _):
        session = model.session()
        errors = asyncio.Queue()
        session.on("error", errors.put_nowait)
        await session._update_session(instructions="Help")
        peer.refusals["response.create"] = ("invalid_request", "event")
        with pytest.raises(PrismError) as raised:
            await asyncio.wait_for(session.generate_reply(), 3)
        assert raised.value.code == "invalid_request"
        assert errors.empty()
        # A named refusal did not create a response. The next explicit reply
        # request can still use this conversation and must not replay the first.
        await peer.emit("prism.typed_input.ready")
        pending = session.generate_reply()
        response_id = await peer.start(await peer.next())
        await peer.end(response_id)
        assert (await asyncio.wait_for(pending, 3)).response_id == response_id
        assert errors.empty()
        assert peer.requests.empty()


async def test_session_fault_reports_one_fatal_error():
    async with peer_model() as (peer, model, _):
        session = model.session()
        errors = asyncio.Queue()
        session.on("error", errors.put_nowait)
        await session._update_session(instructions="Help")
        await peer.emit(
            "error",
            error={
                "code": "unavailable",
                "message": "Session cannot continue",
                "scope": "session",
                "owner": {"kind": "session"},
            },
        )
        error = await asyncio.wait_for(errors.get(), 3)
        assert not error.recoverable
        assert error.retryable is True
        assert error.error.scope == "session"
        await session.aclose()
        assert errors.empty()
