"""Acknowledged realtime history updates across cancellation and fallback."""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, llm, utils

from .fake_realtime import FakeRealtimeModel

pytestmark = pytest.mark.unit


def context(text):
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="user", content=text)
    return ctx


async def test_history_update_waits_for_swap_and_reaches_the_replacement():
    model = FakeRealtimeModel()
    fallback = llm.RealtimeModelFallbackAdapter([model])
    async with AgentSession(llm=fallback) as session:
        agent = Agent(instructions="Help", chat_ctx=context("old"))
        await session.start(agent)
        old = model.active_session
        old.block_aclose = asyncio.Event()
        restarting = asyncio.create_task(fallback.restart_session())
        change = None
        try:
            await asyncio.wait_for(old.aclose_entered.wait(), 3)
            target = context("new")
            change = asyncio.create_task(agent.update_chat_ctx(target))
            await asyncio.sleep(0)
            assert not change.done()
        finally:
            old.block_aclose.set()
            await asyncio.wait_for(restarting, 3)
        await asyncio.wait_for(change, 3)
        assert model.active_session is not old
        assert model.active_session.chat_ctx.is_equivalent(target)
        assert agent.chat_ctx.is_equivalent(target)


async def test_restart_allows_history_update_from_interruption_cleanup(monkeypatch):
    model = FakeRealtimeModel()
    fallback = llm.RealtimeModelFallbackAdapter([model])
    async with AgentSession(llm=fallback) as session:
        agent = Agent(instructions="Help", chat_ctx=context("old"))
        await session.start(agent)
        original_interrupt = session.interrupt
        target = context("heard before interruption")

        async def interrupt(**kwargs):
            await agent.update_chat_ctx(target)
            await original_interrupt(**kwargs)

        monkeypatch.setattr(session, "interrupt", interrupt)
        await asyncio.wait_for(fallback.restart_session(), 3)
        assert model.active_session.chat_ctx.is_equivalent(target)


async def test_cancelled_update_during_swap_applies_once_after_replacement():
    model = FakeRealtimeModel()
    fallback = llm.RealtimeModelFallbackAdapter([model])
    session = fallback.session()
    try:
        await session.initialize(chat_ctx=context("old"))
        old = model.active_session
        old.block_aclose = asyncio.Event()
        restarting = asyncio.create_task(fallback.restart_session())
        await asyncio.wait_for(old.aclose_entered.wait(), 3)
        target = context("accepted")
        change = asyncio.create_task(session.update_chat_ctx(target))
        await asyncio.sleep(0)
        change.cancel()
        old.block_aclose.set()
        await asyncio.wait_for(restarting, 3)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(change, 3)
        assert model.active_session.chat_ctx.is_equivalent(target)
        await fallback.restart_session()
        assert model.active_session.chat_ctx.is_equivalent(target)
    finally:
        if old.block_aclose is not None:
            old.block_aclose.set()
        await session.aclose()


async def test_new_conversation_item_survives_an_acknowledged_context_edit(monkeypatch):
    model = FakeRealtimeModel()
    async with AgentSession(llm=model) as session:
        agent = Agent(instructions="Help", chat_ctx=context("old"))
        await session.start(agent)
        provider = model.active_session
        update = provider.update_chat_ctx
        entered, release = asyncio.Event(), asyncio.Event()

        async def blocked(target):
            entered.set()
            await release.wait()
            await update(target)

        monkeypatch.setattr(provider, "update_chat_ctx", blocked)
        target = context("replacement")
        change = asyncio.create_task(agent.update_chat_ctx(target))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            speech = session.generate_reply()

            async def wait_for_request():
                while not provider._reply_futs:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_for_request(), 3)
            messages = utils.aio.Chan[llm.MessageGeneration]()
            functions = utils.aio.Chan[llm.FunctionCall]()
            text = utils.aio.Chan[str]()
            audio = utils.aio.Chan()
            modalities = asyncio.get_running_loop().create_future()
            modalities.set_result(["text"])
            text.send_nowait("new reply")
            text.close()
            audio.close()
            functions.close()
            messages.send_nowait(
                llm.MessageGeneration(
                    message_id="arrived",
                    text_stream=text,
                    audio_stream=audio,
                    modalities=modalities,
                )
            )
            messages.close()
            provider._reply_futs[-1].set_result(
                llm.GenerationCreatedEvent(
                    message_stream=messages, function_stream=functions, user_initiated=True
                )
            )
            await asyncio.wait_for(speech.wait_for_playout(), 3)
        finally:
            release.set()
        await asyncio.wait_for(change, 3)
        assert [i.text_content for i in agent.chat_ctx.items] == ["replacement", "new reply"]
