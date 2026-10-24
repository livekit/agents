"""A Prism response drains only after every generated message is accounted for."""

# ruff: noqa: E402 -- load the optional SDK before importing the adapter
from __future__ import annotations

import asyncio
import base64

import pytest

pytestmark = pytest.mark.unit
pytest.importorskip("rimelabs_sdk.realtime")

from livekit.agents import Agent, AgentSession

from .fake_io import FakeAudioOutput
from .test_plugin_rime_realtime import peer_model


async def message(peer, response_id, index, *, seconds=0.02):
    item_id = f"{response_id}-message-{index}"
    await peer.emit(
        "response.output_item.added",
        response_id=response_id,
        output_index=index,
        item={"id": item_id, "type": "message"},
    )
    common = {
        "response_id": response_id,
        "output_index": index,
        "item_id": item_id,
        "content_index": 0,
    }
    await peer.emit("response.text.done", **common, text=f"Part {index}")
    await peer.emit(
        "response.audio.delta",
        **common,
        delta=base64.b64encode(b"\0\0" * int(24000 * seconds)).decode(),
    )
    await peer.emit("response.audio.done", **common)
    return item_id


async def completed_two_messages(peer, session):
    pending = session.generate_reply()
    response_id = await peer.start(await peer.next())
    generation = await pending
    ids = [await message(peer, response_id, index) for index in range(2)]
    await peer.end(response_id)
    assert len([m async for m in generation.message_stream]) == 2
    return response_id, ids


def report(session, response_id, message_id, position, *, interrupted=False, by_user=False):
    session.playback_finished(
        response_id=response_id,
        message_id=message_id,
        audio_end_ms=position,
        interrupted=interrupted,
        interrupted_by_user=by_user,
    )


async def test_agent_session_sends_one_aggregate_receipt_for_multiple_messages():
    async with peer_model() as (peer, model, _):
        async with AgentSession(llm=model) as session:
            session.output.audio = FakeAudioOutput()
            await session.start(Agent(instructions="Help"))
            speech = session.generate_reply()
            response_id = await peer.start(await peer.next())
            await message(peer, response_id, 0, seconds=0.02)
            await message(peer, response_id, 1, seconds=0.04)
            await peer.end(response_id)
            await asyncio.wait_for(speech.wait_for_playout(), 3)
            receipt = await asyncio.wait_for(peer.receipts.get(), 3)
            assert receipt["played_ms"] == 60
            await model.get_history()
            assert peer.receipts.empty()


async def test_first_completed_message_neither_drains_nor_changes_interruption_policy():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.initialize()
        response_id, ids = await completed_two_messages(peer, session)
        report(session, response_id, ids[0], 100)
        await model.get_history()
        assert peer.receipts.empty()
        await model.update_session(interrupt_on_speech=False)
        speech = asyncio.Queue()
        session.on("input_speech_started", speech.put_nowait)
        await peer.emit("input_audio_buffer.speech_started", item_id="caller", audio_start_ms=0)
        assert (await asyncio.wait_for(speech.get(), 3)).interrupt_response
        report(session, response_id, ids[1], 200)
        receipt = await asyncio.wait_for(peer.receipts.get(), 3)
        assert receipt["played_ms"] == 300


async def test_callbacks_before_response_end_wait_for_later_outputs():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.initialize()
        pending = session.generate_reply()
        response_id = await peer.start(await peer.next())
        generation = await pending
        first = await message(peer, response_id, 0)
        assert (await generation.message_stream.__anext__()).message_id == first
        report(session, response_id, first, 100)
        second = await message(peer, response_id, 1)
        await peer.end(response_id)
        assert [m.message_id async for m in generation.message_stream] == [second]
        await model.get_history()
        assert peer.receipts.empty()
        report(session, response_id, second, 200)
        assert (await asyncio.wait_for(peer.receipts.get(), 3))["played_ms"] == 300


@pytest.mark.parametrize("reverse", [False, True])
async def test_duplicate_and_out_of_order_callbacks_drain_once(reverse):
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.initialize()
        response_id, ids = await completed_two_messages(peer, session)
        order = list(reversed(ids)) if reverse else ids
        for item in order:
            report(session, response_id, item, 125)
            report(session, response_id, item, 999)
        assert (await asyncio.wait_for(peer.receipts.get(), 3))["played_ms"] == 250
        for item in ids:
            report(session, response_id, item, 999)
        await model.get_history()
        assert peer.receipts.empty()


async def test_mixed_full_and_caller_interrupted_messages_only_truncate():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.initialize()
        response_id, ids = await completed_two_messages(peer, session)
        report(session, response_id, ids[0], 100)
        report(session, response_id, ids[1], 50, interrupted=True, by_user=True)
        report(session, response_id, ids[1], 75, interrupted=True, by_user=True)
        truncate = await asyncio.wait_for(peer.truncations.get(), 3)
        assert truncate["item_id"] == ids[1]
        assert truncate["audio_end_ms"] == 50
        await model.get_history()
        assert peer.truncations.empty()
        assert peer.receipts.empty()


async def test_empty_response_and_duplicate_none_callbacks_drain_once():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.initialize()
        pending = session.generate_reply()
        response_id = await peer.start(await peer.next())
        generation = await pending
        await peer.end(response_id)
        assert [m async for m in generation.message_stream] == []
        report(session, response_id, None, 0)
        report(session, response_id, None, 0)
        assert (await asyncio.wait_for(peer.receipts.get(), 3))["played_ms"] == 0
        await model.get_history()
        assert peer.receipts.empty()


async def test_response_wide_application_stop_discards_late_outputs_and_drains_once():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.initialize()
        pending = session.generate_reply()
        response_id = await peer.start(await peer.next())
        generation = await pending
        peer.cancel_release.clear()
        try:
            report(session, response_id, None, 0, interrupted=True)
            report(session, response_id, None, 0, interrupted=True)
            await asyncio.wait_for(peer.cancel_received.wait(), 3)
            await message(peer, response_id, 0)
            await message(peer, response_id, 1)
        finally:
            peer.cancel_release.set()
        assert [m async for m in generation.message_stream] == []
        assert (await asyncio.wait_for(peer.receipts.get(), 3))["played_ms"] == 0
        await model.get_history()
        assert peer.receipts.empty()


async def test_all_skipped_messages_send_one_zero_position_receipt():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.initialize()
        response_id, ids = await completed_two_messages(peer, session)
        for item in ids:
            report(session, response_id, item, 0, interrupted=True)
        assert (await asyncio.wait_for(peer.receipts.get(), 3))["played_ms"] == 0
        await model.get_history()
        assert peer.receipts.empty()


async def test_response_wide_stop_discards_late_audio_for_existing_outputs():
    async with peer_model() as (peer, model, _):
        session = model.session()
        errors = asyncio.Queue()
        session.on("error", errors.put_nowait)
        await session.initialize()
        pending = session.generate_reply()
        response_id = await peer.start(await peer.next())
        generation = await pending
        item_id = await message(peer, response_id, 0)
        assert (await generation.message_stream.__anext__()).message_id == item_id
        peer.cancel_release.clear()
        try:
            report(session, response_id, None, 0, interrupted=True)
            await asyncio.wait_for(peer.cancel_received.wait(), 3)
            for _ in range(300):
                await peer.emit(
                    "response.audio.delta",
                    response_id=response_id,
                    output_index=0,
                    item_id=item_id,
                    content_index=0,
                    delta=base64.b64encode(b"\0\0" * 480).decode(),
                )
                await asyncio.sleep(0.001)
        finally:
            peer.cancel_release.set()
        assert [m async for m in generation.message_stream] == []
        assert (await asyncio.wait_for(peer.receipts.get(), 3))["played_ms"] == 0
        assert errors.empty()


@pytest.mark.parametrize("later_already_announced", [False, True])
async def test_caller_stop_accounts_for_unplayed_later_messages(later_already_announced):
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.initialize()
        pending = session.generate_reply()
        response_id = await peer.start(await peer.next())
        generation = await pending
        first = await message(peer, response_id, 0)
        assert (await generation.message_stream.__anext__()).message_id == first
        second = None
        if later_already_announced:
            second = await message(peer, response_id, 1)
            assert (await generation.message_stream.__anext__()).message_id == second
        report(session, response_id, first, 50, interrupted=True, by_user=True)
        if second is None:
            second = await message(peer, response_id, 1)
        await peer.end(response_id)
        assert [m async for m in generation.message_stream] == []
        truncations = [await asyncio.wait_for(peer.truncations.get(), 3) for _ in range(2)]
        assert [(t["item_id"], t["audio_end_ms"]) for t in truncations] == [
            (first, 50),
            (second, 0),
        ]
        await model.update_session(interrupt_on_speech=False)
        speech = asyncio.Queue()
        session.on("input_speech_started", speech.put_nowait)
        await peer.emit(
            "input_audio_buffer.speech_started", item_id="caller-next", audio_start_ms=0
        )
        assert not (await asyncio.wait_for(speech.get(), 3)).interrupt_response
        assert peer.receipts.empty()


@pytest.mark.parametrize("announced", [False, True])
async def test_per_message_application_stop_accounts_remaining_unplayed_outputs(announced):
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.initialize()
        pending = session.generate_reply()
        response_id = await peer.start(await peer.next())
        generation = await pending
        first = await message(peer, response_id, 0)
        assert (await generation.message_stream.__anext__()).message_id == first
        if announced:
            await message(peer, response_id, 1)
            await model.get_history()
        peer.cancel_release.clear()
        try:
            report(session, response_id, first, 15, interrupted=True)
            await asyncio.wait_for(peer.cancel_received.wait(), 3)
            if not announced:
                await message(peer, response_id, 1)
        finally:
            peer.cancel_release.set()
        receipt = await asyncio.wait_for(peer.receipts.get(), 3)
        assert receipt["played_ms"] == 15
        await model.get_history()
        assert peer.receipts.empty()
        assert peer.truncations.empty()
