"""Prism uses the public realtime hooks through the LiveKit fallback adapter."""

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, llm

from .fake_io import FakeAudioOutput
from .test_plugin_rime_realtime import peer_model

pytestmark = pytest.mark.unit


async def test_prism_fallback_typed_input_playback_and_explicit_restart() -> None:
    async with peer_model() as (peer, model, _):
        adapter = llm.RealtimeModelFallbackAdapter([model])
        async with AgentSession(llm=adapter) as session:
            session.output.audio = FakeAudioOutput()
            await session.start(Agent(instructions="test"))
            for text in ("First turn", "After explicit restart"):
                speech = session.generate_reply(user_input=text)
                request = await peer.next()
                assert request["response"]["prism_input_text"] == text
                response_id = await peer.start(request)
                await peer.message(response_id)
                await peer.end(response_id)
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                receipt = await asyncio.wait_for(peer.receipts.get(), 3)
                assert receipt["response_id"] == response_id
                if text == "First turn":
                    await adapter.restart_session()
                    assert [item["text"] for item in peer.items] == ["First turn", "Hello"]


async def test_prism_startup_uses_backup_after_primary_initialization_failure() -> None:
    async with peer_model() as (primary_peer, primary, _):
        primary_peer.refusals["session.update"] = ("invalid_session", "event")
        async with peer_model() as (backup_peer, backup, _):
            adapter = llm.RealtimeModelFallbackAdapter([primary, backup])
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
                assert request["response"]["prism_input_text"] == "Hello backup"
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
            adapter = llm.RealtimeModelFallbackAdapter([primary, backup])
            session = adapter.session()
            reconnected = []
            session.on("session_reconnected", reconnected.append)
            try:
                await session.initialize(instructions="test")
                primary_peer.refusals["session.update"] = ("invalid_session", "event")
                await adapter.restart_session()
                if session._swap_task is not None:
                    await asyncio.wait_for(session._swap_task, 3)
                assert len(created) == 1
                assert len(reconnected) == 1
                assert not created[0]._closed

                await session.set_input_text("Use the backup")
                reply = session.generate_reply()
                request = await backup_peer.next()
                assert request["response"]["prism_input_text"] == "Use the backup"
                response_id = await backup_peer.start(request)
                await backup_peer.end(response_id)
                assert (await asyncio.wait_for(reply, 3)).response_id == response_id
            finally:
                await session.aclose()
