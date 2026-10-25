"""Prism uses the public realtime hooks through the LiveKit fallback adapter."""

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, llm
from livekit.agents.llm import RealtimeModelFallbackAdapter

from .fake_io import FakeAudioOutput
from .test_plugin_rime_realtime import peer_model

pytestmark = pytest.mark.unit


async def test_prism_fallback_typed_input_playback_and_explicit_restart() -> None:
    async with peer_model() as (peer, model, _):
        adapter = RealtimeModelFallbackAdapter([model])
        async with AgentSession(llm=adapter) as session:
            session.output.audio = FakeAudioOutput()
            await session.start(Agent(instructions="test"))
            for text in ("First turn", "After explicit restart"):
                speech = session.generate_reply(user_input=text)
                request = await peer.next()
                assert "prism_input_text" not in request["response"]
                assert peer.items[-1]["text"] == text
                response_id = await peer.start(request)
                await peer.message(response_id)
                await peer.end(response_id)
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                assert peer.receipts.empty()
                if text == "First turn":
                    await adapter.restart_session()
                    assert [item["text"] for item in peer.server_items] == ["First turn", "Hello"]


async def test_prism_startup_uses_backup_after_primary_initialization_failure() -> None:
    async with peer_model() as (primary_peer, primary, _):
        primary_peer.refusals["session.update"] = ("invalid_session", "event")
        async with peer_model() as (backup_peer, backup, _):
            adapter = RealtimeModelFallbackAdapter([primary, backup])
            history = llm.ChatContext.empty()
            history.add_message(role="user", content="Saved history")
            async with AgentSession(llm=adapter) as session:
                await asyncio.wait_for(
                    session.start(Agent(instructions="Saved instructions", chat_ctx=history)), 3
                )
                assert backup_peer.settings["instructions"] == "Saved instructions"
                assert [item["text"] for item in backup_peer.items] == ["Saved history"]
                speech = session.generate_reply(user_input="Hello backup")
                request = await backup_peer.next()
                assert backup_peer.items[-1]["text"] == "Hello backup"
                response_id = await backup_peer.start(request)
                await backup_peer.message(response_id)
                await backup_peer.end(response_id)
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                assert speech.exception() is None


async def test_prism_restart_initialization_failure_opens_backup_once(monkeypatch) -> None:
    async with peer_model() as (primary_peer, primary, _):
        async with peer_model() as (backup_peer, backup, _):
            created = []
            create_session = backup.session

            def counted_session(**kwargs):
                child = create_session(**kwargs)
                created.append(child)
                return child

            monkeypatch.setattr(backup, "session", counted_session)
            adapter = RealtimeModelFallbackAdapter([primary, backup])
            session = adapter.session()
            reconnected = []
            session.on("session_reconnected", reconnected.append)
            try:
                await session._update_session(instructions="test")
                primary_peer.refusals["session.update"] = ("invalid_session", "event")
                await adapter.restart_session()
                if session._swap_task is not None:
                    await asyncio.wait_for(session._swap_task, 3)
                assert len(created) == 1
                assert len(reconnected) == 1
                assert not created[0]._closed

                ctx = session.chat_ctx
                ctx.add_message(role="user", content="Use the backup")
                await session.update_chat_ctx(ctx)
                reply = session.generate_reply()
                request = await backup_peer.next()
                assert backup_peer.items[-1]["text"] == "Use the backup"
                response_id = await backup_peer.start(request)
                await backup_peer.end(response_id)
                assert (await asyncio.wait_for(reply, 3)).response_id == response_id
            finally:
                await session.aclose()
