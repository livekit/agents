"""Typed-turn ownership after refused, unsent, and accepted reply requests."""

# ruff: noqa: E402 -- load the optional SDK before importing the adapter
from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.unit
pytest.importorskip("rimelabs_sdk.realtime")

from rimelabs_sdk import RimeTimeoutError, realtime as r

from livekit.agents import llm
from livekit.plugins.rime.realtime import PrismError

from .test_plugin_rime_realtime import peer_model


async def _finish(peer, pending, request):
    response_id = await peer.start(request)
    await peer.end(response_id)
    generation = await asyncio.wait_for(pending, 3)
    assert [item async for item in generation.message_stream] == []
    return generation


async def test_explicit_reply_reuses_a_typed_turn_after_zero_effect_refusal():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.initialize()
        message = await session.set_input_text("Original question")
        peer.refusals["response.create"] = ("invalid_request", "event")
        with pytest.raises(PrismError):
            await session.generate_reply()
        assert peer.requests.empty()  # Restaging must never schedule a retry.
        await peer.emit("prism.typed_input.ready")
        pending = session.generate_reply()
        request = await peer.next()
        assert request["response"]["prism_input_text"] == "Original question"
        await _finish(peer, pending, request)
        assert [item.id for item in session.chat_ctx.items] == [message.id]


async def test_explicit_reply_reuses_a_typed_turn_after_admission_timeout():
    async with peer_model(timeouts=r.RealtimeTimeouts(ready_s=0.02)) as (peer, model, _):
        session = model.session()
        await session.initialize()
        first = session.generate_reply()
        await _finish(peer, first, await peer.next())
        # No playback receipt or new readiness event has admitted another request.
        await session.set_input_text("Still waiting")
        with pytest.raises(llm.RealtimeError) as raised:
            await session.generate_reply()
        assert raised.value.code == "admission_timeout"
        assert peer.requests.empty()
        await peer.emit("prism.typed_input.ready")
        pending = session.generate_reply()
        request = await peer.next()
        assert request["response"]["prism_input_text"] == "Still waiting"
        await _finish(peer, pending, request)


async def test_new_explicit_input_supersedes_a_restored_unsent_turn():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.initialize()
        old = await session.set_input_text("Old question")
        peer.refusals["response.create"] = ("invalid_request", "event")
        with pytest.raises(PrismError):
            await session.generate_reply()
        new = await session.set_input_text("New question")
        assert session.chat_ctx.get_by_id(old.id) is None
        assert session.chat_ctx.get_by_id(new.id) is not None
        await peer.emit("prism.typed_input.ready")
        pending = session.generate_reply()
        request = await peer.next()
        assert request["response"]["prism_input_text"] == "New question"
        await _finish(peer, pending, request)


async def test_late_refusal_does_not_restore_an_old_turn_over_newer_input():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.initialize()
        old = await session.set_input_text("Old question")
        first = session.generate_reply()
        request = await peer.next()
        new = await session.set_input_text("New question")
        await peer.emit(
            "error",
            error={
                "code": "invalid_request",
                "message": "No response created",
                "scope": "event",
                "owner": {"kind": "event", "event_id": request["event_id"]},
            },
        )
        with pytest.raises(PrismError):
            await first
        assert session.chat_ctx.get_by_id(old.id) is None
        assert session.chat_ctx.get_by_id(new.id) is not None
        await peer.emit("prism.typed_input.ready")
        pending = session.generate_reply()
        request = await peer.next()
        assert request["response"]["prism_input_text"] == "New question"
        await _finish(peer, pending, request)


async def test_accepted_turn_identity_does_not_use_a_newer_staged_item_id():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.initialize()
        old = await session.set_input_text("Accepted first")
        first = session.generate_reply()
        first_request = await peer.next()
        new = await session.set_input_text("Staged second")
        await _finish(peer, first, first_request)
        # Deleting the first local item must delete the first server item, while
        # leaving the second staged turn available for a later explicit request.
        target = llm.ChatContext([i for i in session.chat_ctx.items if i.id != old.id])
        await session.update_chat_ctx(target)
        assert await model.get_history() == ()
        assert [i.id for i in session.chat_ctx.items] == [new.id]
        await peer.emit("prism.typed_input.ready")
        second = session.generate_reply()
        request = await peer.next()
        assert request["response"]["prism_input_text"] == "Staged second"
        await _finish(peer, second, request)


async def test_cancelled_late_accepted_typed_turn_is_never_replayed():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.initialize()
        await session.set_input_text("Accepted once")
        pending = session.generate_reply()
        request = await peer.next()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await peer.start(request)
        await asyncio.wait_for(peer.cancel_received.wait(), 3)
        await asyncio.wait_for(peer.receipts.get(), 3)
        assert peer.requests.empty()
        next_reply = session.generate_reply()
        next_request = await peer.next()
        assert "prism_input_text" not in next_request["response"]
        await _finish(peer, next_reply, next_request)


async def test_unknown_request_outcome_does_not_restore_typed_input():
    async with peer_model(timeouts=r.RealtimeTimeouts(request_s=0.02)) as (peer, model, _):
        session = model.session()
        await session.initialize()
        await session.set_input_text("Unknown outcome")
        pending = session.generate_reply()
        await peer.next()
        with pytest.raises(RimeTimeoutError):
            await pending
        with pytest.raises(RimeTimeoutError):
            await session.generate_reply()
        assert peer.requests.empty()


async def test_superseded_typed_turn_is_not_restored_as_an_unsent_question():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.initialize()
        await session.set_input_text("Superseded")
        peer.refusals["response.create"] = ("typed_turn_superseded", "event")
        with pytest.raises(PrismError):
            await session.generate_reply()
        await peer.emit("prism.typed_input.ready")
        pending = session.generate_reply()
        request = await peer.next()
        assert "prism_input_text" not in request["response"]
        await _finish(peer, pending, request)
