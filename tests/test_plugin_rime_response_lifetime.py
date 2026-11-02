"""Tool response ownership through the real SDK and a local protocol peer."""

from __future__ import annotations

import asyncio

import pytest
from rimelabs_sdk import RimeInputError, realtime as r

from livekit.agents import llm
from livekit.plugins.rime.realtime import PrismError

from .test_plugin_rime import peer_model

pytestmark = pytest.mark.unit


async def finish(peer, response):
    await peer.end(response)
    receipt = await asyncio.wait_for(peer.receipts.get(), 3)
    assert receipt["response_id"] == response


async def tool_turn(peer, *calls):
    response = await peer.start()
    for call in calls:
        await peer.emit(
            "response.function_call_arguments.done",
            response_id=response,
            item_id=call,
            call_id=call,
            name="lookup",
            arguments="{}",
        )
    await finish(peer, response)
    return response


async def submit(provider, call, *, reply_required=True):
    await provider.update_chat_ctx(
        llm.ChatContext(
            [
                llm.FunctionCallOutput(
                    call_id=call, output="done", is_error=False, reply_required=reply_required
                )
            ]
        )
    )


async def reply(peer, provider):
    pending = provider.generate_reply()
    request = await peer.next()
    response = await peer.start(request)
    await finish(peer, response)
    await pending
    return request


@pytest.mark.parametrize("superseded_by", ["response", "speech", "report"])
async def test_superseded_tool_replies_do_not_exhaust_response_holds(superseded_by):
    async with peer_model(playback="discard", context_mode="shared") as (peer, model, _):
        provider = model.session()
        await provider.update_instructions("test")
        errors = []
        provider.on("error", errors.append)
        for number in range(140):
            call = f"call-{number}"
            await tool_turn(peer, call)
            if superseded_by == "report":
                await finish(peer, await peer.start())
            await submit(provider, call)
            if superseded_by == "speech":
                started = asyncio.Event()
                provider.once("input_speech_started", lambda _, ready=started: ready.set())
                await peer.emit(
                    "input_audio_buffer.speech_started", item_id=f"input-{number}", audio_start_ms=0
                )
                await asyncio.wait_for(started.wait(), 3)
                await peer.emit("prism.typed_input.ready")
            else:
                await finish(peer, await peer.start())
            request = await reply(peer, provider)
            assert "prism_call_id" not in request["response"]
            assert request["response"]["metadata"]["prism_cause"] == "proactive"
        assert not errors
        assert len(peer.items) == 140
        assert peer.connections == 1


async def test_replaced_tool_reports_release_only_the_old_response():
    async with peer_model(playback="discard", context_mode="shared") as (peer, model, _):
        provider = model.session()
        await provider.update_instructions("test")
        for number in range(140):
            first, second = f"first-{number}", f"second-{number}"
            await tool_turn(peer, first)
            await tool_turn(peer, second)
            await finish(peer, await peer.start())
            await submit(provider, first)
            await submit(provider, second)
            request = await reply(peer, provider)
            assert request["response"]["prism_call_id"] == second
        assert len(peer.items) == 280
        assert peer.connections == 1


async def test_superseded_partial_result_keeps_unresolved_tool_reference():
    async with peer_model(playback="discard", context_mode="shared") as (peer, model, _):
        provider = model.session()
        await provider.update_instructions("test")
        await tool_turn(peer, "first", "pending")
        await submit(provider, "first")
        for _ in range(140):
            await finish(peer, await peer.start())
        await submit(provider, "pending")
        request = await reply(peer, provider)
        assert request["response"]["prism_call_id"] == "pending"
        assert [item["call_id"] for item in peer.items] == ["first", "pending"]


@pytest.mark.parametrize("ending", ["refused", "cancelled_before_start"])
async def test_failed_tool_reply_releases_its_response_hold(ending):
    async with peer_model(playback="discard", context_mode="shared") as (peer, model, _):
        provider = model.session()
        await provider.update_instructions("test")
        for number in range(140):
            call = f"call-{number}"
            await tool_turn(peer, call)
            await submit(provider, call)
            if ending == "refused":
                peer.refusals["response.create"] = ("tool_continuation_not_ready", "event")
                with pytest.raises(PrismError, match="tool_continuation_not_ready"):
                    await provider.generate_reply()
            else:
                pending = provider.generate_reply()
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            assert peer.requests.empty()
        await reply(peer, provider)
        assert len(peer.items) == 140
        assert peer.connections == 1


async def test_in_flight_continuation_keeps_reference_until_fallback_settles():
    async with peer_model(playback="discard", context_mode="shared") as (peer, model, _):
        provider = model.session()
        await provider.update_instructions("test")
        parent = await tool_turn(peer, "call")
        await submit(provider, "call")
        pending = provider.generate_reply()
        request = await peer.next()
        try:
            for _ in range(140):
                await finish(peer, await peer.start())
            await peer.emit(
                "error",
                error={
                    "code": "tool_continuation_unavailable",
                    "message": "Superseded",
                    "scope": "event",
                    "owner": {"kind": "event", "event_id": request["event_id"]},
                },
            )
            report = await peer.next()
            assert report["response"]["prism_call_id"] == "call"
            await finish(peer, await peer.start(report))
            await pending
            for _ in range(140):
                await finish(peer, await peer.start())
            # The public SDK rejects this completed reference once ownership ends.
            assert provider._sdk is not None
            with pytest.raises(RimeInputError):
                provider._sdk.hold_response(
                    r.ResponseRef(session_id=provider._sdk.info.session_id, response_id=parent)
                )
            assert len(peer.items) == 1
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
