"""Prism settings, diagnostics and playback through the real Rime SDK."""

# ruff: noqa: E402 -- optional SDK must be checked before importing the plugin
from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.unit
pytest.importorskip("rimelabs_sdk.realtime")

from rimelabs_sdk import realtime as r

from livekit.agents import Agent, AgentSession, AgentTask, RunContext, function_tool
from livekit.agents.llm import RealtimeModelFallbackAdapter
from livekit.plugins.rime.realtime import (
    PrismError,
    PrismGenerationCreatedEvent,
    PrismModelError,
    PrismSpeechStoppedEvent,
)

from .test_plugin_rime_realtime import peer_model


async def wait_until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), 3)


@function_tool
async def lookup() -> str:
    """Look up an order."""
    return "shipped"


async def test_connection_settings_fillers_and_session_information():
    async with peer_model(
        voice="test",
        turn_detection=r.TurnDetection(eagerness="low", end_of_turn_timeout_ms=3000),
        request_id="livekit-validation",
        response_metadata={"workflow": "billing"},
        tool_fillers={"lookup": r.PrismFiller(type="text", text="Checking your order.")},
    ) as (peer, model, _):
        session = model.session()
        await session._update_session(instructions="Help", tools=[lookup])
        assert peer.request_id == "livekit-validation"
        assert peer.settings["turn_detection"]["eagerness"] == "low"
        assert peer.settings["turn_detection"]["prism_end_of_turn_timeout_ms"] == 3000
        assert peer.settings["tools"][0]["prism_filler"] == {
            "type": "text",
            "text": "Checking your order.",
        }
        assert model.session_info.request_id == "livekit-validation"
        assert model.session_info.response_metadata == {"workflow": "billing"}


async def test_settings_updates_and_rejection_preserve_session():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session._update_session(instructions="Original")
        await session._update_session(instructions="Billing", tools=[lookup])
        assert peer.settings["instructions"] == "Billing"
        assert peer.updates[-1]["tools"][0]["function"]["name"] == "lookup"
        await model.update_session(voice="new-voice")
        assert model.session_info.voice == "new-voice"
        assert model.session_info.interrupt_on_speech is True
        await model.update_session(response_metadata={"step": "2"})
        assert model.session_info.response_metadata == {"step": "2"}
        old_revision = model.session_info.settings_revision
        peer.refusals["session.update"] = ("invalid_session", "event")
        with pytest.raises(PrismError) as raised:
            await session.update_instructions("Rejected")
        assert raised.value.code == "invalid_session"
        assert raised.value.retryable is False
        assert peer.settings["instructions"] == "Billing"
        assert model.session_info.settings_revision == old_revision
        await session._update_session(instructions="", tools=[])
        assert peer.settings["instructions"] == ""
        assert peer.settings["tools"] == []
        assert peer.connections == 1


async def test_cancelled_update_waits_for_acknowledged_settings():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session._update_session(instructions="Original")
        peer.update_started.clear()
        peer.update_release.clear()
        change = asyncio.create_task(session.update_tools([lookup]))
        await asyncio.wait_for(peer.update_started.wait(), 3)
        change.cancel()
        await asyncio.sleep(0)
        assert not change.done()
        peer.update_release.set()
        with pytest.raises(asyncio.CancelledError):
            await change
        assert "lookup" in session.tools.function_tools
        assert peer.settings["tools"][0]["function"]["name"] == "lookup"


async def test_agent_handoff_reuses_conversation_and_updates_tools():
    async with peer_model() as (peer, model, _):
        async with AgentSession(llm=model) as session:
            first = Agent(instructions="Triage")
            await session.start(first)
            speech = session.generate_reply(user_input="Hello")
            request = await peer.next()
            response_id = await peer.start(request)
            await peer.message(response_id)
            await peer.end(response_id)
            await asyncio.wait_for(speech.wait_for_playout(), 3)
            second = Agent(instructions="Billing", tools=[lookup], chat_ctx=first.chat_ctx)
            session.update_agent(second)
            await wait_until(
                lambda: (
                    peer.settings.get("instructions") == "Billing"
                    and bool(peer.settings.get("tools"))
                )
            )
            assert peer.connections == 1
            assert peer.settings["tools"][0]["function"]["name"] == "lookup"
            assert any(
                item.text_content == "Hello"
                for item in second.chat_ctx.items
                if item.type == "message"
            )


async def test_agent_task_returns_to_parent_on_the_same_connection():
    entered = []
    result = asyncio.Future()

    class Task(AgentTask[str]):
        def __init__(self, chat_ctx):
            super().__init__(instructions="Task", chat_ctx=chat_ctx)

        async def on_enter(self):
            entered.append(self.session.current_agent.instructions)
            self.complete("finished")

    class Parent(Agent):
        def __init__(self):
            super().__init__(instructions="Parent")

        async def on_enter(self):
            if not result.done():
                result.set_result(await Task(self.chat_ctx))

    async with peer_model() as (peer, model, _):
        async with AgentSession(llm=model) as session:
            await session.start(Parent())
            assert await asyncio.wait_for(result, 3) == "finished"
            await wait_until(lambda: peer.settings["instructions"] == "Parent")
            assert entered == ["Task"]
            assert peer.connections == 1


async def test_tool_handoff_preserves_the_pending_tool_result():
    ran = []

    @function_tool
    async def transfer(context: RunContext) -> Agent:
        """Transfer to billing."""
        ran.append("transfer")
        return Agent(instructions="Billing", chat_ctx=context.session.current_agent.chat_ctx)

    async with peer_model() as (peer, model, _):
        async with AgentSession(llm=model) as session:
            await session.start(Agent(instructions="Triage", tools=[transfer]))
            speech = session.generate_reply(user_input="Billing please")
            response_id = await peer.start(await peer.next())
            await peer.emit(
                "response.function_call_arguments.done",
                response_id=response_id,
                item_id="transfer-item",
                call_id="transfer-call",
                name="transfer",
                arguments="{}",
            )
            await peer.end(response_id)
            await wait_until(
                lambda: peer.settings is not None and peer.settings.get("instructions") == "Billing"
            )
            await asyncio.wait_for(speech.wait_for_playout(), 3)
            assert ran == ["transfer"]
            assert peer.connections == 1
            assert any(item.get("call_id") == "transfer-call" for item in peer.items)


async def test_generation_diagnostics_and_transcript_deltas():
    async with peer_model(request_id="diagnostics") as (peer, model, _):
        session = model.session()
        await session._update_session(instructions="Help")
        generations, stopped, transcripts = [], [], []
        model.on("generation_created", generations.append)
        session.on("input_speech_stopped", stopped.append)
        session.on("input_audio_transcription_completed", transcripts.append)
        await peer.emit("input_audio_buffer.speech_started", item_id="input-1", audio_start_ms=0)
        await peer.emit(
            "conversation.item.input_audio_transcription.delta", item_id="input-1", delta="Hel"
        )
        await peer.emit(
            "conversation.item.input_audio_transcription.delta", item_id="input-1", delta="lo"
        )
        await peer.emit(
            "input_audio_buffer.speech_stopped",
            item_id="input-1",
            audio_end_ms=200,
            prism_end_reason="semantic",
        )
        await peer.emit(
            "response.created",
            response={
                "id": "diagnostic-response",
                "metadata": {"prism_cause": "user_audio", "workflow": "billing"},
                "prism_input_item_ids": ["input-1"],
                "prism_settings_revision": peer.revision,
            },
        )
        await wait_until(lambda: generations and stopped and len(transcripts) == 2)
        assert [event.transcript for event in transcripts] == ["Hel", "Hello"]
        assert all(not event.is_final for event in transcripts)
        assert isinstance(stopped[0], PrismSpeechStoppedEvent)
        assert stopped[0].end_reason == "semantic"
        assert isinstance(generations[0], PrismGenerationCreatedEvent)
        assert generations[0].metadata == {"workflow": "billing"}
        assert generations[0].input_item_ids == ("input-1",)
        assert generations[0].provider_request_id == "diagnostics"
        assert generations[0].settings_revision == peer.revision


@pytest.mark.parametrize(
    "code,retryable",
    [("capacity_exceeded", True), ("response_timeout", True), ("context_length_exceeded", False)],
)
@pytest.mark.parametrize("via_fallback", [False, True])
async def test_fault_retryability_does_not_replay_requests(code, retryable, via_fallback):
    async with peer_model() as (peer, model, _):
        realtime_model = RealtimeModelFallbackAdapter([model]) if via_fallback else model
        async with AgentSession(llm=realtime_model) as session:
            errors = []
            session.on("error", errors.append)
            await session.start(Agent(instructions="Help"))
            await peer.emit(
                "error",
                error={
                    "code": code,
                    "message": "Provider refusal",
                    "scope": "response",
                    "owner": {"kind": "response", "response_id": "response-1"},
                },
            )
            await wait_until(lambda: errors)
            model_error = errors[0].error
            assert isinstance(model_error, PrismModelError)
            assert model_error.retryable is retryable
            assert isinstance(model_error.error, PrismError)
            assert model_error.error.retryable is retryable
            assert model_error.error.code == code
            assert model_error.error.scope == "response"
            assert model_error.error.response_id == "response-1"
            assert model_error.error.fault.code == code
            assert errors[0].model_dump()["error"]["retryable"] is retryable
            assert peer.requests.empty()
