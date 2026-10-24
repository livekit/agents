"""Bounded Prism generation retention must not silently lose playback ownership."""

# ruff: noqa: E402 -- load the optional SDK before importing the adapter
from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.unit
pytest.importorskip("rimelabs_sdk.realtime")

from rimelabs_sdk import RimeResourceLimitError

from .test_plugin_rime_realtime import peer_model


def finish(session, response_id):
    session.playback_finished(
        response_id=response_id,
        message_id=None,
        audio_end_ms=0,
        interrupted=False,
        interrupted_by_user=False,
    )


async def spontaneous_response(peer, generations, index):
    response_id = await peer.start(response_id=f"unannounced-{index}")
    generation = await asyncio.wait_for(generations.get(), 3)
    assert generation.response_id == response_id
    await peer.end(response_id)
    assert [m async for m in generation.message_stream] == []
    return response_id


async def test_generation_limit_fails_instead_of_forgetting_unresolved_playback():
    async with peer_model() as (peer, model, _):
        session = model.session()
        errors = asyncio.Queue()
        generations = asyncio.Queue()
        session.on("error", errors.put_nowait)
        session.on("generation_created", generations.put_nowait)
        await session._update_session()
        oldest = await spontaneous_response(peer, generations, 0)
        for index in range(1, 128):
            response_id = await spontaneous_response(peer, generations, index)
            finish(session, response_id)
            receipt = await asyncio.wait_for(peer.receipts.get(), 3)
            assert receipt["response_id"] == response_id
        await peer.start(response_id="unannounced-128")
        error = await asyncio.wait_for(errors.get(), 3)
        assert isinstance(error.error, RimeResourceLimitError)
        assert "playback" in str(error.error)
        assert not error.recoverable
        assert generations.empty()
        # Cleanup cannot silently send an invalid receipt through an expired SDK
        # reference or continue another turn after playback ownership was lost.
        finish(session, oldest)
        with pytest.raises(RimeResourceLimitError):
            await session.set_input_text("must use a fresh conversation")
        assert peer.receipts.empty()


async def test_generation_limit_retires_completed_records_and_keeps_admission_usable():
    async with peer_model() as (peer, model, _):
        session = model.session()
        errors = asyncio.Queue()
        generations = asyncio.Queue()
        session.on("error", errors.put_nowait)
        session.on("generation_created", generations.put_nowait)
        await session._update_session()
        for index in range(130):
            response_id = await spontaneous_response(peer, generations, index)
            finish(session, response_id)
            receipt = await asyncio.wait_for(peer.receipts.get(), 3)
            assert receipt["response_id"] == response_id
        finish(session, "unannounced-0")  # An already acknowledged, expired callback is harmless.
        await model.get_history()
        assert peer.receipts.empty()
        await session.set_input_text("Still usable")
        reply = session.generate_reply()
        request = await peer.next()
        assert request["response"]["prism_input_text"] == "Still usable"
        response_id = await peer.start(request)
        await peer.end(response_id)
        generation = await asyncio.wait_for(reply, 3)
        assert [m async for m in generation.message_stream] == []
        finish(session, response_id)
        assert (await asyncio.wait_for(peer.receipts.get(), 3))["response_id"] == response_id
        assert errors.empty()
