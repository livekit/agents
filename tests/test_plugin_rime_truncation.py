"""Prism truncation acknowledgments through the SDK and local WebSocket peer."""

# ruff: noqa: E402 -- optional SDK must be checked before importing the plugin
from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.unit
r = pytest.importorskip("rimelabs_sdk.realtime")

from livekit.agents import Agent, AgentSession

from .fake_io import FakeAudioOutput
from .test_plugin_rime_realtime import peer_model
from .test_plugin_rime_settings import wait_until


async def completed_message(peer, session):
    response_id = await peer.start()
    item_id = await peer.message(response_id, text="The full generated answer")
    await peer.end(response_id)
    await wait_until(lambda: session.chat_ctx.get_by_id(item_id) is not None)
    return item_id


async def acknowledge(peer, item_id, kept_text, *, content_index=0):
    await peer.emit(
        "conversation.item.truncated",
        item_id=item_id,
        content_index=content_index,
        audio_end_ms=120,
        **({} if kept_text is None else {"prism_kept_text": kept_text}),
    )


@pytest.mark.parametrize("kept_text", [None, "", "The full", "The full generated answer"])
async def test_retained_text_updates_history_without_changing_existing_snapshots(kept_text):
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.update_instructions("test")
        events = []
        model.on("item_truncated", events.append)
        item_id = await completed_message(peer, session)
        before = session.chat_ctx.get_by_id(item_id)
        await acknowledge(peer, item_id, kept_text)
        await wait_until(lambda: len(events) == 1)
        after = session.chat_ctx.get_by_id(item_id)
        assert after.content == [before.text_content if kept_text is None else kept_text]
        assert after.created_at == before.created_at
        assert before.text_content == "The full generated answer"
        assert events[0] == r.ItemTruncated(
            item_id=item_id, content_index=0, audio_end_ms=120, kept_text=kept_text
        )
        # An acknowledgment from an older server must not erase known retained text.
        await acknowledge(peer, item_id, None)
        await wait_until(lambda: len(events) == 2)
        assert session.chat_ctx.get_by_id(item_id) == after


async def test_unavailable_retained_text_during_generation_keeps_completion_fallback():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.update_instructions("test")
        events = []
        model.on("item_truncated", events.append)
        response_id = await peer.start()
        item_id = await peer.message(response_id, text="Generated words")
        await acknowledge(peer, item_id, None)
        await wait_until(lambda: events)
        await peer.end(response_id)
        await wait_until(lambda: session.chat_ctx.get_by_id(item_id) is not None)
        assert session.chat_ctx.get_by_id(item_id).text_content == "Generated words"


@pytest.mark.parametrize("kept_text", ["The full", ""])
@pytest.mark.parametrize("played_text", [None, "The"])
async def test_early_retained_text_survives_response_completion(kept_text, played_text):
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.update_instructions("test")
        events, ended = [], []
        model.on("item_truncated", events.append)
        model.on("response_ended", ended.append)
        response_id = await peer.start()
        item_id = await peer.message(response_id, text="The full generated answer")
        await wait_until(
            lambda: (
                response_id in session._generations
                and item_id in session._generations[response_id].outputs
            )
        )
        if played_text is not None:
            session.truncate(
                message_id=item_id,
                modalities=["audio"],
                audio_end_ms=120,
                audio_transcript=played_text,
            )
            await wait_until(lambda: events)
            events.clear()
        await acknowledge(peer, item_id, kept_text)
        await wait_until(lambda: events)
        # Missing text in a later acknowledgment must not erase a known value.
        await acknowledge(peer, item_id, None)
        await wait_until(lambda: len(events) == 2)
        await peer.end(response_id)
        await wait_until(lambda: ended)
        assert session.chat_ctx.get_by_id(item_id).content == [kept_text]


@pytest.mark.parametrize("edit", ["delete", "replace"])
async def test_late_acknowledgment_cannot_restore_deleted_or_replaced_history(edit):
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.update_instructions("test")
        events = []
        model.on("item_truncated", events.append)
        item_id = await completed_message(peer, session)
        target = session.chat_ctx
        if edit == "delete":
            target.items.clear()
        else:
            target.get_by_id(item_id).content = ["Replacement"]
        await session.update_chat_ctx(target)
        await acknowledge(peer, item_id, "Stale words")
        await wait_until(lambda: events)
        assert session.chat_ctx.items == target.items


@pytest.mark.parametrize("edit", ["append", "delete", "replace"])
async def test_acknowledgment_during_history_edit_preserves_the_committed_result(edit):
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.update_instructions("test")
        events = []
        model.on("item_truncated", events.append)
        item_id = await completed_message(peer, session)
        target = session.chat_ctx
        if edit == "append":
            target.add_message(role="user", content="Next question")
        elif edit == "delete":
            target.items.clear()
        else:
            target.get_by_id(item_id).content = ["Replacement"]
        peer.history_started.clear()
        peer.history_release.clear()
        update = asyncio.create_task(session.update_chat_ctx(target))
        try:
            await asyncio.wait_for(peer.history_started.wait(), 3)
            await acknowledge(peer, item_id, "Retained words")
            await wait_until(lambda: events)
        finally:
            peer.history_release.set()
            await asyncio.wait_for(update, 3)
        if edit == "append":
            assert session.chat_ctx.get_by_id(item_id).text_content == "Retained words"
            assert session.chat_ctx.items[-1].text_content == "Next question"
        else:
            assert session.chat_ctx.items == target.items


async def test_unknown_item_and_other_content_part_are_diagnostics_only():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.update_instructions("test")
        events = []
        model.on("item_truncated", events.append)
        item_id = await completed_message(peer, session)
        before = session.chat_ctx
        await acknowledge(peer, "unknown", "Unrelated")
        await acknowledge(peer, item_id, "Other part", content_index=1)
        await wait_until(lambda: len(events) == 2)
        assert session.chat_ctx.items == before.items


@pytest.mark.parametrize(
    "summary", [None, {"items_left_out": 3}, {"items_left_out": 3, "first_kept_item_id": "kept"}]
)
@pytest.mark.parametrize("status", ["completed", "cancelled", "failed"])
async def test_response_summary_is_exposed_without_deleting_history(summary, status):
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.update_instructions("test")
        item_id = await completed_message(peer, session)
        before = session.chat_ctx
        events = []
        model.on("response_ended", events.append)
        response_id = await peer.start()
        await peer.emit(
            "response.done",
            response={
                "id": response_id,
                "status": status,
                **({} if summary is None else {"prism_truncation": summary}),
            },
        )
        await wait_until(lambda: events)
        assert events[0].response.response_id == response_id
        assert events[0].status == status
        assert events[0].truncation == (
            None if summary is None else r.ResponseTruncation(**summary)
        )
        assert session.chat_ctx.items == before.items
        # A completed response reference is not required to process a later acknowledgment.
        session._generations.clear()
        truncated = []
        model.on("item_truncated", truncated.append)
        await acknowledge(peer, item_id, "Retained")
        await wait_until(lambda: truncated)
        assert session.chat_ctx.get_by_id(item_id).text_content == "Retained"


async def test_retained_server_text_does_not_replace_agent_playback_transcript():
    async with peer_model() as (peer, model, _):
        output = FakeAudioOutput(can_pause=True)
        async with AgentSession(llm=model) as session:
            session.output.audio = output
            await session.start(Agent(instructions="test"))
            handle = session.generate_reply()
            request = await peer.next()
            response_id = await peer.start(request)
            item_id = await peer.message(response_id, seconds=1)
            await wait_until(lambda: output._started_at is not None)
            await peer.emit("input_audio_buffer.speech_started", item_id="user-1", audio_start_ms=0)
            await peer.end(response_id, status="cancelled")
            await asyncio.wait_for(handle.wait_for_playout(), 5)
            await asyncio.wait_for(peer.truncations.get(), 3)
            before = session.history.to_dict()
            events = []
            model.on("item_truncated", events.append)
            await acknowledge(peer, item_id, "All words retained by the server")
            await wait_until(
                lambda: events and events[-1].kept_text == "All words retained by the server"
            )
            assert model._session.chat_ctx.get_by_id(item_id).text_content == (
                "All words retained by the server"
            )
            assert session.history.to_dict() == before
