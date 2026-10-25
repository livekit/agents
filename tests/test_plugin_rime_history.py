"""Prism history and context policy through the published SDK and a local peer."""

# ruff: noqa: E402 -- optional SDK must be checked before importing the plugin
from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.unit
pytest.importorskip("rimelabs_sdk.realtime")

from rimelabs_sdk import realtime as r

from livekit.agents import Agent, AgentSession, function_tool, llm
from livekit.agents.llm import RealtimeModelFallbackAdapter
from livekit.plugins.rime.realtime import PrismError

from .test_plugin_rime_realtime import peer_model
from .test_plugin_rime_settings import wait_until


def messages(*texts):
    ctx = llm.ChatContext.empty()
    for text in texts:
        ctx.add_message(role="user", content=text)
    return ctx


def tool_history():
    ctx = messages("Earlier question")
    ctx.add_message(role="system", content="Transferred to billing")
    ctx.items.extend(
        [
            llm.FunctionCall(call_id="past-call", name="lookup", arguments='{"order": "1"}'),
            llm.FunctionCallOutput(
                call_id="past-call", name="lookup", output="shipped", is_error=False
            ),
        ]
    )
    ctx.add_message(role="assistant", content="Your order has shipped")
    return ctx


@pytest.mark.parametrize("policy", [None, "auto", "disabled", r.RetentionRatio(0.9)])
async def test_truncation_at_connect_update_and_restart(policy):
    async with peer_model(truncation=policy) as (peer, model, _):
        fallback = RealtimeModelFallbackAdapter([model])
        session = fallback.session()
        try:
            await session._update_session(instructions="Help")
            if policy is None:
                assert "truncation" not in peer.settings
                assert model.session_info.truncation is None
            else:
                assert model.session_info.truncation == policy
            await model.update_session(truncation=r.RetentionRatio(0.7))
            assert model.session_info.truncation == r.RetentionRatio(0.7)
            await fallback.restart_session()
            assert model.session_info.truncation == r.RetentionRatio(0.7)
        finally:
            await session.aclose()


async def test_rejected_truncation_leaves_acknowledged_policy_unchanged():
    async with peer_model(truncation="auto") as (peer, model, _):
        session = model.session()
        await session.update_instructions("test")
        peer.refusals["session.update"] = ("invalid_session", "event")
        with pytest.raises(PrismError):
            await model.update_session(truncation="disabled")
        assert model.session_info.truncation == "auto"
        await model.update_session(truncation="disabled")
        assert model.session_info.truncation == "disabled"


async def test_initial_history_accepts_all_supported_types_without_executing_tools():
    ran = []

    @function_tool
    async def lookup(order: str) -> str:
        """Look up an order."""
        ran.append(order)
        return "unexpected"

    async with peer_model() as (peer, model, _):
        async with AgentSession(llm=model) as session:
            ctx = tool_history()
            await session.start(Agent(instructions="Help", tools=[lookup], chat_ctx=ctx))
            history = await model.get_history()
            assert [type(i) for i in history] == [
                r.HistoryMessage,
                r.HistoryMessage,
                r.HistoryFunctionCall,
                r.HistoryFunctionResult,
                r.HistoryMessage,
            ]
            assert history[1].role == "system"
            assert history[2].arguments == '{"order": "1"}'
            assert history[3].output == "shipped"
            assert [i.id for i in model._session.chat_ctx.items] == [i.id for i in ctx.items]
            assert not ran
            assert peer.requests.empty()


@pytest.mark.parametrize(
    "edit", ["append", "prepend", "middle", "delete", "replace", "reorder", "noop"]
)
async def test_context_edits_preserve_local_identity_and_server_order(edit):
    async with peer_model() as (peer, model, _):
        session = model.session()
        original = messages("one", "two", "three")
        await session._update_session(chat_ctx=original)
        target = original.copy()
        if edit == "append":
            target.add_message(role="system", content="four")
        elif edit in ("prepend", "middle"):
            target.items.insert(
                0 if edit == "prepend" else 1, llm.ChatMessage(role="system", content=["inserted"])
            )
        elif edit == "delete":
            del target.items[1]
        elif edit == "replace":
            target.items[1] = target.items[1].model_copy(update={"content": ["replacement"]})
        elif edit == "reorder":
            target.items.reverse()
        before = len(peer.history_operations)
        await session.update_chat_ctx(target)
        history = await model.get_history()
        assert [i.text for i in history] == [i.text_content for i in target.items]
        assert [i.id for i in session.chat_ctx.items] == [i.id for i in target.items]
        assert peer.requests.empty()
        accepted_operations = len(peer.history_operations)
        if edit == "noop":
            assert accepted_operations == before
        await session.update_chat_ctx(target)
        assert len(peer.history_operations) == accepted_operations
        assert [i.text for i in await model.get_history()] == [i.text_content for i in target.items]


async def test_completed_tool_pair_deletion_and_restart_do_not_execute_tools():
    async with peer_model() as (peer, model, _):
        fallback = RealtimeModelFallbackAdapter([model])
        session = fallback.session()
        try:
            await session._update_session(chat_ctx=tool_history())
            await fallback.restart_session()
            assert len(await model.get_history()) == 5
            target = session.chat_ctx.copy(exclude_function_call=True)
            await session.update_chat_ctx(target)
            history = await model.get_history()
            assert all(isinstance(i, r.HistoryMessage) for i in history)
            assert len(history) == 3
            assert peer.requests.empty()
        finally:
            await session.aclose()


@pytest.mark.parametrize("invalid", ["orphan", "duplicate", "split_pair", "image", "encoding"])
async def test_invalid_batch_sends_nothing_and_keeps_conversation_usable(invalid):
    async with peer_model() as (peer, model, _):
        session = model.session()
        original = messages("keep")
        await session._update_session(chat_ctx=original)
        target = original.copy()
        target.add_message(role="system", content="valid addition")
        if invalid == "orphan":
            target.items.append(llm.FunctionCall(call_id="missing", name="lookup", arguments="{}"))
        elif invalid == "duplicate":
            target.items.append(original.items[0])
        elif invalid == "split_pair":
            target.items.extend(tool_history().items[2:4])
            target.items.insert(
                -1, llm.ChatMessage(role="user", content=["between call and result"])
            )
        elif invalid == "image":
            target.add_message(role="user", content=[llm.ImageContent(image="https://test/image")])
        else:
            target.add_message(role="system", content=chr(0xD800))
        before = len(peer.history_operations)
        with pytest.raises(llm.RealtimeError):
            await session.update_chat_ctx(target)
        assert len(peer.history_operations) == before
        assert session.chat_ctx.is_equivalent(original)
        assert len(await model.get_history()) == 1


async def test_rejected_first_operation_leaves_history_unchanged():
    async with peer_model() as (peer, model, _):
        session = model.session()
        original = messages("keep")
        await session._update_session(chat_ctx=original)
        peer.refusals["conversation.item.create"] = ("invalid_request", "event")
        target = original.copy()
        target.add_message(role="system", content="rejected")
        with pytest.raises(PrismError):
            await session.update_chat_ctx(target)
        assert session.chat_ctx.is_equivalent(original)
        assert len(await model.get_history()) == 1
        await session.update_chat_ctx(target)
        assert len(await model.get_history()) == 2


async def test_partial_history_failure_closes_uncertain_conversation():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session._update_session(chat_ctx=messages("old"))
        peer.fail_history_at = len(peer.history_operations) + 2
        errors = asyncio.Queue()
        session.on("error", errors.put_nowait)
        with pytest.raises(llm.RealtimeError, match="partially applied"):
            await session.update_chat_ctx(messages("new"))
        error = await asyncio.wait_for(errors.get(), 3)
        assert not error.recoverable
        with pytest.raises(llm.RealtimeError):
            await session.update_instructions("must not use uncertain history")


async def test_cancelled_history_update_settles_and_commits_accepted_change():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session._update_session(chat_ctx=messages("old"))
        peer.history_started.clear()
        peer.history_release.clear()
        target = session.chat_ctx
        target.add_message(role="system", content="accepted")
        change = asyncio.create_task(session.update_chat_ctx(target))
        try:
            await asyncio.wait_for(peer.history_started.wait(), 3)
            change.cancel()
            await asyncio.sleep(0)
            change.cancel()
            await asyncio.sleep(0)
            assert not change.done()
        finally:
            peer.history_release.set()
        with pytest.raises(asyncio.CancelledError):
            await change
        assert session.chat_ctx.is_equivalent(target)
        assert len(await model.get_history()) == 2


async def test_running_reply_snapshot_does_not_remove_live_local_items():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.update_instructions("test")
        ctx = session.chat_ctx
        message = ctx.add_message(role="user", content="question")
        await session.update_chat_ctx(ctx)
        pending = session.generate_reply()
        response_id = await peer.start(await peer.next())
        await pending
        await peer.message(response_id)
        assert [item.text for item in await model.get_history()] == ["question"]
        assert session.chat_ctx.get_by_id(message.id) is not None
        with pytest.raises(llm.RealtimeError, match="active replies"):
            await session.update_chat_ctx(messages("replacement"))
        await peer.end(response_id)
        await wait_until(lambda: len(session.chat_ctx.items) == 2)
        await session.update_chat_ctx(llm.ChatContext.empty())
        assert await model.get_history() == ()


@pytest.mark.parametrize("removed_type", [llm.FunctionCall, llm.FunctionCallOutput])
async def test_deleting_half_a_tool_pair_is_rejected_before_mutation(removed_type):
    async with peer_model() as (peer, model, _):
        session = model.session()
        ctx = tool_history()
        await session._update_session(chat_ctx=ctx)
        before = len(peer.history_operations)
        target = llm.ChatContext([i for i in ctx.items if not isinstance(i, removed_type)])
        with pytest.raises(llm.RealtimeError):
            await session.update_chat_ctx(target)
        assert len(peer.history_operations) == before
        assert len(await model.get_history()) == 5


async def test_history_read_refusal_uses_the_adapter_error_and_keeps_session():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.update_instructions("test")
        peer.refusals["prism.conversation.get"] = ("invalid_request", "event")
        with pytest.raises(PrismError) as raised:
            await model.get_history()
        assert raised.value.code == "invalid_request"
        assert await model.get_history() == ()


async def test_reply_waiting_for_readiness_does_not_block_live_tool_results():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.update_instructions("test")
        first = session.generate_reply()
        parent = await peer.start(await peer.next())
        generation = await first
        await peer.emit(
            "response.function_call_arguments.done",
            response_id=parent,
            item_id="live-call",
            call_id="live-call",
            name="lookup",
            arguments="{}",
        )
        call = await generation.function_stream.__anext__()
        await peer.end(parent)
        await wait_until(lambda: session.chat_ctx.get_by_id(call.id) is not None)
        waiting = session.generate_reply()
        try:
            await asyncio.sleep(0)
            target = session.chat_ctx
            target.items.append(
                llm.FunctionCallOutput(call_id=call.call_id, output="done", is_error=False)
            )
            await asyncio.wait_for(session.update_chat_ctx(target), 1)
            assert peer.items[-1]["output"] == "done"
            await peer.emit("prism.typed_input.ready")
            response = await peer.start(await peer.next())
            await peer.end(response)
            await asyncio.wait_for(waiting, 3)
        finally:
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)


@pytest.mark.parametrize("ended", [False, True])
async def test_reply_started_during_final_rebuild_acknowledgment_requires_recovery(ended):
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session._update_session(chat_ctx=messages("old"))
        peer.block_history_at = len(peer.history_operations) + 2
        peer.history_release.clear()
        change = asyncio.create_task(session.update_chat_ctx(messages("replacement")))
        errors = asyncio.Queue()
        session.on("error", errors.put_nowait)
        try:
            await wait_until(lambda: len(peer.history_operations) == peer.block_history_at)
            await peer.emit(
                "response.created",
                response={"id": "racing-reply", "metadata": {"prism_cause": "user_audio"}},
            )
            if ended:
                await peer.end("racing-reply")
                await wait_until(
                    lambda: (
                        session._generations.get("racing-reply")
                        and session._generations["racing-reply"].ended
                    )
                )
            else:
                await wait_until(lambda: "racing-reply" in session._generations)
        finally:
            peer.history_release.set()
        with pytest.raises(llm.RealtimeError, match="partially applied"):
            await change
        event = await asyncio.wait_for(errors.get(), 3)
        assert not event.recoverable


async def test_omitted_new_options_preserve_existing_text_history_requests():
    async with peer_model() as (peer, model, _):
        session = model.session()
        original = messages("seed")
        await session._update_session(chat_ctx=original)
        target = session.chat_ctx
        target.add_message(role="assistant", content="appended history")
        await session.update_chat_ctx(target)
        assert "truncation" not in peer.settings
        assert [event["item"] for event in peer.history_operations] == [
            {"type": "message", "role": "user", "text": "seed"},
            {"type": "message", "role": "assistant", "text": "appended history"},
        ]


async def test_rebuild_validates_completed_live_tool_pair_before_mutation():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.update_instructions("Help")
        pending = session.generate_reply()
        response = await peer.start(await peer.next())
        generation = await pending
        await peer.emit(
            "response.function_call_arguments.done",
            response_id=response,
            item_id="live-tool",
            call_id="live-call",
            name="lookup",
            arguments="{}",
        )
        call = await asyncio.wait_for(anext(generation.function_stream), 3)
        await peer.end(response)
        assert [item async for item in generation.function_stream] == []
        completed = session.chat_ctx
        completed.items.append(
            llm.FunctionCallOutput(
                call_id=call.call_id, name=call.name, output="done", is_error=False
            )
        )
        await session.update_chat_ctx(completed)
        before = len(peer.history_operations)
        invalid = session.chat_ctx
        invalid.items.insert(1, llm.ChatMessage(role="system", content=["between"]))
        with pytest.raises(llm.RealtimeError, match="immediately follow"):
            await session.update_chat_ctx(invalid)
        assert len(peer.history_operations) == before
        assert session.chat_ctx.is_equivalent(completed)
        valid = session.chat_ctx
        valid.items.insert(0, llm.ChatMessage(role="system", content=["before"]))
        await session.update_chat_ctx(valid)
        assert session.chat_ctx.is_equivalent(valid)
        assert [type(item) for item in await model.get_history()] == [
            r.HistoryMessage,
            r.HistoryFunctionCall,
            r.HistoryFunctionResult,
        ]


async def test_chat_snapshot_changes_do_not_mutate_acknowledged_history():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.update_chat_ctx(messages("original"))
        snapshot = session.chat_ctx
        snapshot.items[0].content[:] = ["edited"]
        snapshot.items[0].extra["local"] = True
        assert session.chat_ctx.items[0].text_content == "original"
        assert session.chat_ctx.items[0].extra == {}
        before = len(peer.history_operations)
        await session.update_chat_ctx(snapshot)
        assert len(peer.history_operations) == before + 2
        assert (await model.get_history())[0].text == "edited"
        snapshot.items[0].content[:] = ["changed again"]
        assert session.chat_ctx.items[0].text_content == "edited"


async def test_nonblocking_tool_annotation_is_read_from_execution_context():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.update_instructions("Help")
        pending = session.generate_reply()
        response = await peer.start(await peer.next())
        generation = await pending
        await peer.emit(
            "response.function_call_arguments.done",
            response_id=response,
            item_id="live-tool",
            call_id="live-call",
            name="lookup",
            arguments="{}",
        )
        call = await asyncio.wait_for(anext(generation.function_stream), 3)
        await peer.end(response)
        assert [item async for item in generation.function_stream] == []
        call.extra["__livekit_agents_tool_non_blocking"] = True
        target = llm.ChatContext(
            [
                call,
                llm.FunctionCallOutput(
                    call_id=call.call_id, name=call.name, output="working", is_error=False
                ),
            ]
        )
        before = len(peer.history_operations)
        with pytest.raises(llm.RealtimeError, match="tool progress"):
            await session.update_chat_ctx(target)
        assert len(peer.history_operations) == before
        assert session.chat_ctx.items[0].extra == {}


async def test_partial_history_failure_does_not_submit_a_queued_edit():
    async with peer_model() as (peer, model, _):
        session = model.session()
        original = messages("old")
        await session.update_chat_ctx(original)
        before = len(peer.history_operations)
        peer.block_history_at = peer.fail_history_at = before + 2
        peer.history_release.clear()
        first = asyncio.create_task(session.update_chat_ctx(messages("replacement")))
        queued = None
        try:
            await wait_until(lambda: len(peer.history_operations) == before + 2)
            queued = asyncio.create_task(session.update_chat_ctx(messages("queued")))
            await asyncio.sleep(0)
            peer.history_release.set()
            with pytest.raises(llm.RealtimeError, match="partially applied"):
                await first
            results = await asyncio.gather(queued, return_exceptions=True)
            assert isinstance(results[0], (llm.RealtimeError, asyncio.CancelledError))
            await session.aclose()
            assert len(peer.history_operations) == before + 2
        finally:
            peer.history_release.set()
            first.cancel()
            tasks = [first]
            if queued is not None:
                queued.cancel()
                tasks.append(queued)
            await asyncio.gather(*tasks, return_exceptions=True)
