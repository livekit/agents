"""Rime adapter and AgentSession tests against a local Prism protocol peer."""

# ruff: noqa: E402 -- optional SDK must be checked before importing the plugin
from __future__ import annotations

import asyncio
import base64
import json
from contextlib import asynccontextmanager

import pytest

pytestmark = pytest.mark.unit

r = pytest.importorskip("rimelabs_sdk.realtime")
from rimelabs_sdk import Rime
from websockets.asyncio.server import serve

from livekit.agents import Agent, AgentSession, function_tool, llm
from livekit.plugins.rime.realtime import PrismUnsupported, RealtimeModel

from .fake_io import FakeAudioOutput


class Peer:
    def __init__(self):
        self.requests = asyncio.Queue()
        self.socket = None
        self.serial = 0
        self.settings = None
        self.receipts = asyncio.Queue()
        self.truncations = asyncio.Queue()
        self.items = []
        self.refusals = {}
        self.cancel_received = asyncio.Event()
        self.cancel_release = asyncio.Event()
        self.cancel_release.set()

    async def emit(self, kind, **body):
        await self.socket.send(json.dumps({"type": kind, "event_id": "server", **body}))

    async def handler(self, socket):
        self.socket = socket
        await self.emit("session.created", session={"id": "session-1"})
        async for raw in socket:
            event = json.loads(raw)
            kind = event["type"]
            if kind in self.refusals:
                code, scope = self.refusals.pop(kind)
                await self.emit(
                    "error",
                    error={
                        "code": code,
                        "message": "Request refused",
                        "scope": scope,
                        "owner": {"kind": scope, "event_id": event["event_id"]},
                    },
                )
                if kind == "response.cancel" and scope == "event":
                    await self.end(event["response_id"])
                continue
            if kind == "session.update":
                self.settings = event["session"]
                await self.emit(
                    "session.updated",
                    prism_request_event_id=event["event_id"],
                    session={"id": "session-1", "voice": "test"},
                )
                await self.emit("prism.typed_input.ready")
            elif kind == "conversation.item.create":
                self.items.append(event["item"])
                await self.emit(
                    "conversation.item.created",
                    prism_request_event_id=event["event_id"],
                    item={"id": f"history-{len(self.items)}", **event["item"]},
                )
            elif kind == "response.cancel":
                self.cancel_received.set()
                await self.cancel_release.wait()
                await self.end(event["response_id"], status="cancelled")
            elif kind == "prism.playback.drained":
                self.receipts.put_nowait(event)
                await self.emit("prism.typed_input.ready")
            elif kind == "conversation.item.truncate":
                self.truncations.put_nowait(event)
                await self.emit(
                    "conversation.item.truncated",
                    item_id=event["item_id"],
                    content_index=event["content_index"],
                    audio_end_ms=event["audio_end_ms"],
                )
            elif kind == "input_audio_buffer.clear":
                await self.emit("input_audio_buffer.cleared")
            else:
                self.requests.put_nowait(event)

    async def next(self, kind="response.create"):
        event = await asyncio.wait_for(self.requests.get(), 3)
        assert event["type"] == kind
        return event

    async def start(self, request=None, response_id=None):
        self.serial += 1
        response_id = response_id or f"response-{self.serial}"
        body = {
            "response": {
                "id": response_id,
                "metadata": request["response"]["metadata"]
                if request
                else {"prism_cause": "user_audio"},
            }
        }
        if request:
            body["prism_request_event_id"] = request["event_id"]
        await self.emit("response.created", **body)
        return response_id

    async def message(self, response_id, *, seconds=0.02, text="Hello"):
        item_id = f"message-{response_id}"
        await self.emit(
            "response.output_item.added",
            response_id=response_id,
            output_index=0,
            item={"id": item_id, "type": "message"},
        )
        common = {
            "response_id": response_id,
            "output_index": 0,
            "item_id": item_id,
            "content_index": 0,
        }
        await self.emit("response.text.delta", **common, delta=text)
        await self.emit("response.text.done", **common, text=text)
        await self.emit(
            "response.audio.delta",
            **common,
            delta=base64.b64encode(b"\0\0" * int(24000 * seconds)).decode(),
        )
        await self.emit("response.audio.done", **common)
        return item_id

    async def end(self, response_id, *, status="completed"):
        await self.emit(
            "response.done",
            response={
                "id": response_id,
                "status": status,
                "metadata": {"prism_drain_token": f"drain-{response_id}"},
            },
        )


@asynccontextmanager
async def peer_model():
    peer = Peer()
    async with serve(peer.handler, "127.0.0.1", 0) as server:
        endpoint = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1/realtime"
        async with Rime(api_key="test") as client:
            model = RealtimeModel(endpoint=endpoint, client=client)
            try:
                yield peer, model, client
            finally:
                await model.aclose()


async def test_history_note_is_not_a_typed_turn():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.initialize(instructions="test", tools=[], chat_ctx=llm.ChatContext.empty())
        ctx = session.chat_ctx
        ctx.add_message(role="user", content="A history note")
        await session.update_chat_ctx(ctx)
        assert peer.items[-1]["text"] == "A history note"
        await session.set_input_text("Actual typed turn")
        future = session.generate_reply()
        request = await peer.next()
        assert request["response"]["prism_input_text"] == "Actual typed turn"
        ref = await peer.start(request)
        await peer.end(ref)
        assert (await future).response_id == ref


async def test_startup_seeds_history_before_first_request():
    async with peer_model() as (peer, model, _):
        session = model.session()
        ctx = llm.ChatContext.empty()
        ctx.items.append(llm.AgentConfigUpdate(instructions="Old configuration"))
        ctx.add_message(role="user", content="Earlier question")
        ctx.add_message(role="assistant", content="Earlier answer")
        await session.initialize(instructions="Be clear", chat_ctx=ctx, tools=[])
        assert peer.settings["instructions"] == "Be clear"
        assert [i["text"] for i in peer.items] == ["Earlier question", "Earlier answer"]
        assert all(isinstance(item, llm.ChatMessage) for item in session.chat_ctx.items)
        await session.update_instructions("Be clear")
        with pytest.raises(PrismUnsupported):
            await session.update_instructions("Different")


async def test_agent_session_typed_reply_and_playback_receipt():
    async with peer_model() as (peer, model, _):
        async with AgentSession(llm=model) as session:
            session.output.audio = FakeAudioOutput()
            await session.start(Agent(instructions="test"))
            handle = session.generate_reply(user_input="Hello Prism")
            request = await peer.next()
            assert request["response"]["prism_input_text"] == "Hello Prism"
            response_id = await peer.start(request)
            await peer.message(response_id)
            await peer.end(response_id)
            await asyncio.wait_for(handle.wait_for_playout(), 5)
            receipt = await asyncio.wait_for(peer.receipts.get(), 3)
            assert receipt["response_id"] == response_id
            assert receipt["drain_token"] == f"drain-{response_id}"
            assert peer.items == []  # no duplicate history turn


@pytest.mark.parametrize("wrapped", [False, True])
async def test_empty_response_allows_next_typed_turn(wrapped):
    async with peer_model() as (peer, model, _):
        realtime = llm.RealtimeModelFallbackAdapter([model]) if wrapped else model
        async with AgentSession(llm=realtime) as session:
            session.output.audio = FakeAudioOutput()
            await session.start(Agent(instructions="test"))
            first = session.generate_reply(user_input="First question")
            response_id = await peer.start(await peer.next())
            await peer.end(response_id)
            await asyncio.wait_for(first.wait_for_playout(), 3)
            assert first.exception() is None
            receipt = await asyncio.wait_for(peer.receipts.get(), 1)
            assert receipt["response_id"] == response_id
            assert receipt["played_ms"] == 0
            assert not peer.cancel_received.is_set()

            second = session.generate_reply(user_input="Next question")
            request = await peer.next()
            assert request["response"]["prism_input_text"] == "Next question"
            next_id = await peer.start(request)
            await peer.message(next_id)
            await peer.end(next_id)
            await asyncio.wait_for(second.wait_for_playout(), 3)
            assert second.exception() is None
            receipt = await asyncio.wait_for(peer.receipts.get(), 1)
            assert receipt["response_id"] == next_id
            assert peer.receipts.empty()


async def test_agent_session_application_stop_reports_drain_not_truncate():
    async with peer_model() as (peer, model, _):
        output = FakeAudioOutput(can_pause=True)
        async with AgentSession(llm=model) as session:
            session.output.audio = output
            await session.start(Agent(instructions="test"))
            handle = session.generate_reply()
            request = await peer.next()
            response_id = await peer.start(request)
            await peer.message(response_id, seconds=1)
            while output._started_at is None:
                await asyncio.sleep(0.001)
            handle.interrupt(force=True)
            await asyncio.wait_for(handle.wait_for_playout(), 5)
            receipt = await asyncio.wait_for(peer.receipts.get(), 3)
            assert receipt["response_id"] == response_id
            assert peer.truncations.empty()


async def test_agent_session_caller_interrupt_reports_truncate():
    async with peer_model() as (peer, model, _):
        output = FakeAudioOutput(can_pause=True)
        async with AgentSession(llm=model) as session:
            session.output.audio = output
            await session.start(Agent(instructions="test"))
            handle = session.generate_reply()
            request = await peer.next()
            response_id = await peer.start(request)
            item_id = await peer.message(response_id, seconds=1)
            while output._started_at is None:
                await asyncio.sleep(0.001)
            await peer.emit("input_audio_buffer.speech_started", item_id="user-1", audio_start_ms=0)
            await peer.end(response_id, status="cancelled")
            await asyncio.wait_for(handle.wait_for_playout(), 5)
            event = await asyncio.wait_for(peer.truncations.get(), 3)
            assert event["item_id"] == item_id
            assert event["content_index"] == 0
            assert peer.receipts.empty()


async def test_late_transcript_keeps_user_before_assistant():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.initialize(instructions="test")
        await peer.emit("input_audio_buffer.speech_started", item_id="u", audio_start_ms=0)
        await peer.emit("input_audio_buffer.committed", item_id="u")
        response_id = await peer.start()
        await peer.message(response_id)
        await peer.end(response_id)
        transcript = asyncio.Future()
        session.on("input_audio_transcription_completed", transcript.set_result)
        await peer.emit(
            "conversation.item.input_audio_transcription.completed",
            item_id="u",
            transcript="Question",
        )
        event = await asyncio.wait_for(transcript, 3)
        assert event.turn_started_at is not None
        assert [i.role for i in session.chat_ctx.items] == ["user", "assistant"]


async def test_agent_session_executes_tool_then_continues():
    ran = []

    @function_tool
    async def lookup() -> str:
        """Look up a value."""
        ran.append(True)
        return "42"

    async with peer_model() as (peer, model, _):
        async with AgentSession(llm=model) as session:
            await session.start(Agent(instructions="test", tools=[lookup]))
            handle = session.generate_reply()
            request = await peer.next()
            parent = await peer.start(request)
            await peer.emit(
                "response.function_call_arguments.done",
                response_id=parent,
                item_id="tool-item",
                call_id="call-1",
                name="lookup",
                arguments="{}",
            )
            await peer.end(parent)
            continuation = await peer.next()
            assert ran == [True]
            assert peer.items[-1]["call_id"] == "call-1"
            assert continuation["response"]["metadata"]["prism_parent_response_id"] == parent
            response_id = await peer.start(continuation)
            await peer.end(response_id)
            await asyncio.wait_for(handle.wait_for_playout(), 5)


async def test_forced_tool_choice_emits_fatal_error():
    async with peer_model() as (_, model, _):
        session = model.session()
        errors = []
        session.on("error", errors.append)
        await session.initialize(instructions="test")
        with pytest.raises(PrismUnsupported):
            session.update_options(tool_choice="none")
        assert errors[-1].recoverable is False


async def test_injected_client_stays_open():
    async with peer_model() as (peer, model, client):
        await model.session().initialize(instructions="test")
        await model.aclose()
        async with client.realtime.connect(endpoint=model._endpoint) as session:
            assert session.info.session_id == "session-1"


async def test_skipped_response_reports_zero_playback_for_its_own_id():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.initialize(instructions="test")
        future = session.generate_reply()
        request = await peer.next()
        response_id = await peer.start(request)
        await future
        session.playback_finished(
            response_id=response_id,
            message_id=None,
            audio_end_ms=0,
            interrupted=True,
            interrupted_by_user=False,
        )
        receipt = await asyncio.wait_for(peer.receipts.get(), 3)
        assert receipt["response_id"] == response_id
        assert receipt["played_ms"] == 0
        assert peer.truncations.empty()


@pytest.mark.parametrize("has_message", [True, False])
async def test_caller_skips_response_before_output_item_arrives(has_message):
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.initialize(instructions="test")
        future = session.generate_reply()
        response_id = await peer.start(await peer.next())
        await future
        session.playback_finished(
            response_id=response_id,
            message_id=None,
            audio_end_ms=0,
            interrupted=True,
            interrupted_by_user=True,
        )
        if has_message:
            item_id = await peer.message(response_id)
            report = await asyncio.wait_for(peer.truncations.get(), 3)
            assert report["item_id"] == item_id
            assert report["audio_end_ms"] == 0
        await peer.end(response_id, status="cancelled")
        await peer.emit("prism.typed_input.ready")
        future = session.generate_reply()
        next_id = await peer.start(await peer.next())
        await peer.end(next_id)
        assert (await future).response_id == next_id
        assert peer.truncations.empty()
        assert peer.receipts.empty()


@pytest.mark.parametrize("max_tool_steps", [0, 1, 2])
async def test_agent_session_tool_step_limit_reports_unsupported_tool_choice(max_tool_steps):
    ran = []

    @function_tool
    async def lookup() -> str:
        """Look up a value."""
        ran.append(True)
        return "42"

    async with peer_model() as (peer, model, _):
        async with AgentSession(llm=model, max_tool_steps=max_tool_steps) as session:
            await session.start(Agent(instructions="test", tools=[lookup]))
            error = asyncio.Future()
            model._session.on(
                "error", lambda event: error.set_result(event) if not error.done() else None
            )
            session.generate_reply()
            for step in range(max_tool_steps + 1):
                request = await peer.next()
                assert ran == [True] * step
                response_id = await peer.start(request)
                await peer.emit(
                    "response.function_call_arguments.done",
                    response_id=response_id,
                    item_id=f"tool-{step}",
                    call_id=f"call-{step}",
                    name="lookup",
                    arguments="{}",
                )
                await peer.end(response_id)
            # LiveKit counts follow-up rounds after the initial tool round. At the
            # limit it requests tool_choice="none", which Prism cannot enforce.
            event = await asyncio.wait_for(error, 3)
            assert isinstance(event.error, PrismUnsupported)
            assert event.recoverable is False
            assert ran == [True] * (max_tool_steps + 1)
            assert peer.requests.empty()


async def test_abandoned_reply_does_not_generate_after_ready():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.initialize(instructions="test")
        # Consume the initial ready, then leave the second request waiting for playback.
        first = session.generate_reply()
        request = await peer.next()
        response_id = await peer.start(request)
        await peer.end(response_id)
        await first
        abandoned = session.generate_reply()
        await asyncio.sleep(0)
        abandoned.cancel()
        await asyncio.gather(abandoned, return_exceptions=True)
        await peer.emit("prism.typed_input.ready")
        await asyncio.sleep(0.02)
        assert peer.requests.empty()


async def test_agent_session_releases_resources_after_connection_failure():
    async with peer_model() as (peer, model, _):
        async with AgentSession(llm=model) as session:
            session.output.audio = FakeAudioOutput()
            await session.start(Agent(instructions="test"))
            closed = asyncio.get_running_loop().create_future()
            session.on("close", closed.set_result)
            session.generate_reply()
            request = await peer.next()
            response_id = await peer.start(request)
            await peer.message(response_id, seconds=1)
            await peer.socket.close()
            event = await asyncio.wait_for(closed, 3)
            assert event.error is not None
            assert session._activity is None
            assert session.output.audio is None
            assert model._session is not None
            # Late cleanup calls must remain safe after the original failure.
            model._session.interrupt()
            model._session.playback_finished(
                response_id=response_id,
                message_id=f"message-{response_id}",
                audio_end_ms=0,
                interrupted=True,
                interrupted_by_user=True,
            )


async def test_cleanup_after_explicit_close_and_fresh_session():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.initialize(instructions="test")
        future = session.generate_reply()
        response_id = await peer.start(await peer.next())
        await future
        await session.aclose()
        session.interrupt()
        session.playback_finished(
            response_id=response_id,
            message_id=None,
            audio_end_ms=0,
            interrupted=True,
            interrupted_by_user=False,
        )
        replacement = model.session()
        await replacement.initialize(instructions="new conversation")
        assert replacement.chat_ctx.items == []
        future = replacement.generate_reply()
        response_id = await peer.start(await peer.next())
        await peer.end(response_id)
        assert (await future).response_id == response_id


@pytest.mark.parametrize(
    ("operation", "code"),
    [
        ("conversation.item.truncate", "truncate_not_current"),
        ("response.cancel", "response_cancel_not_active"),
    ],
)
async def test_playback_refusal_keeps_session_usable(operation, code):
    async with peer_model() as (peer, model, _):
        session = model.session()
        errors = asyncio.Queue()
        session.on("error", errors.put_nowait)
        await session.initialize(instructions="test")
        future = session.generate_reply()
        response_id = await peer.start(await peer.next())
        generation = await future
        await peer.message(response_id)
        message = await anext(aiter(generation.message_stream))
        peer.refusals[operation] = (code, "event")
        session.playback_finished(
            response_id=response_id,
            message_id=message.message_id,
            audio_end_ms=0,
            interrupted=True,
            interrupted_by_user=operation == "conversation.item.truncate",
        )
        error = await asyncio.wait_for(errors.get(), 3)
        assert error.recoverable
        assert error.error.fault.code == code
        if operation == "response.cancel":
            receipt = await asyncio.wait_for(peer.receipts.get(), 3)
            assert receipt["response_id"] == response_id
        else:
            await peer.end(response_id, status="cancelled")
            await peer.emit("prism.typed_input.ready")
        # Verify recovery through a new request on the same connection.
        future = session.generate_reply()
        second_id = await peer.start(await peer.next())
        await peer.end(second_id)
        assert (await future).response_id == second_id
        assert errors.empty()


@pytest.mark.parametrize(
    ("code", "scope"),
    [("invalid_request", "event"), ("truncate_not_current", "session")],
)
async def test_other_playback_failures_remain_fatal(code, scope):
    async with peer_model() as (peer, model, _):
        session = model.session()
        errors = asyncio.Queue()
        session.on("error", errors.put_nowait)
        await session.initialize(instructions="test")
        future = session.generate_reply()
        response_id = await peer.start(await peer.next())
        generation = await future
        await peer.message(response_id)
        message = await anext(aiter(generation.message_stream))
        peer.refusals["conversation.item.truncate"] = (code, scope)
        session.playback_finished(
            response_id=response_id,
            message_id=message.message_id,
            audio_end_ms=0,
            interrupted=True,
            interrupted_by_user=True,
        )
        error = await asyncio.wait_for(errors.get(), 3)
        assert not error.recoverable
        assert error.error.fault.code == code


async def test_admission_timeout_leaves_conversation_usable(monkeypatch):
    async with peer_model() as (peer, model, client):
        connect = client.realtime.connect
        monkeypatch.setattr(
            client.realtime,
            "connect",
            lambda **kwargs: connect(
                **kwargs, timeouts=r.RealtimeTimeouts(ready_s=0.02, request_s=0.5)
            ),
        )
        session = model.session()
        errors = []
        session.on("error", errors.append)
        await session.initialize(instructions="test")
        first = session.generate_reply()
        response_id = await peer.start(await peer.next())
        await first
        with pytest.raises(llm.RealtimeError) as raised:
            await session.generate_reply()
        assert raised.value.code == "admission_timeout"
        assert peer.requests.empty()
        await peer.end(response_id)
        await peer.emit("prism.typed_input.ready")
        next_reply = session.generate_reply()
        next_id = await peer.start(await peer.next())
        await peer.end(next_id)
        assert (await next_reply).response_id == next_id
        assert not errors


async def test_cancel_before_acceptance_reports_zero_playback_after_terminal():
    async with peer_model() as (peer, model, _):
        session = model.session()
        errors = []
        session.on("error", errors.append)
        await session.initialize(instructions="test")
        future = session.generate_reply()
        request = await peer.next()
        future.cancel()
        with pytest.raises(asyncio.CancelledError):
            await future
        peer.cancel_release.clear()
        try:
            response_id = await peer.start(request)
            await asyncio.wait_for(peer.cancel_received.wait(), 1)
            # Output can arrive while cancellation is in flight. It has no player.
            item_id = "unplayed-message"
            await peer.emit(
                "response.output_item.added",
                response_id=response_id,
                output_index=0,
                item={"id": item_id, "type": "message"},
            )
            for _ in range(300):
                await peer.emit(
                    "response.audio.delta",
                    response_id=response_id,
                    item_id=item_id,
                    output_index=0,
                    content_index=0,
                    delta=base64.b64encode(b"\0\0" * 480).decode(),
                )
                await asyncio.sleep(0.001)
            await peer.emit(
                "response.audio.done",
                response_id=response_id,
                item_id=item_id,
                output_index=0,
                content_index=0,
            )
            assert not errors
            assert peer.receipts.empty()
        finally:
            peer.cancel_release.set()
        receipt = await asyncio.wait_for(peer.receipts.get(), 3)
        assert receipt["response_id"] == response_id
        assert receipt["played_ms"] == 0
        assert receipt["drain_token"] == f"drain-{response_id}"
        next_reply = session.generate_reply()
        next_id = await peer.start(await peer.next())
        await peer.end(next_id)
        assert (await next_reply).response_id == next_id
        assert peer.receipts.empty()
        assert not errors


@pytest.mark.parametrize("code", ["tool_continuation_unavailable", "tool_continuation_not_ready"])
async def test_expired_tool_continuation_reports_result_without_reexecution(code):
    ran = []

    @function_tool
    async def lookup() -> str:
        """Look up a value."""
        ran.append(True)
        return "42"

    async with peer_model() as (peer, model, _):
        async with AgentSession(llm=model) as session:
            await session.start(Agent(instructions="test", tools=[lookup]))
            speech = session.generate_reply()
            parent = await peer.start(await peer.next())
            peer.refusals["response.create"] = (code, "event")
            await peer.emit(
                "response.function_call_arguments.done",
                response_id=parent,
                item_id="tool",
                call_id="call",
                name="lookup",
                arguments="{}",
            )
            await peer.end(parent)
            await peer.emit("prism.typed_input.ready")
            if code == "tool_continuation_unavailable":
                report = await peer.next()
                assert report["response"]["metadata"]["prism_cause"] == "proactive"
                assert report["response"]["prism_call_id"] == "call"
                response_id = await peer.start(report)
                await peer.message(response_id, text="The result is 42.")
                await peer.end(response_id)
            await asyncio.wait_for(speech.wait_for_playout(), 3)
            assert ran == [True]
            assert len(peer.items) == 1
            assert peer.items[0]["call_id"] == "call"
            if code == "tool_continuation_unavailable":
                assert speech.exception() is None
                assert any(
                    item.text_content == "The result is 42."
                    for item in session.history.items
                    if isinstance(item, llm.ChatMessage)
                )
            else:
                assert speech.exception().code == code
                assert peer.requests.empty()


@pytest.mark.parametrize("interrupted,by_user", [(True, True), (True, False), (False, False)])
async def test_stale_playback_message_is_ignored(interrupted, by_user):
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.initialize(instructions="test")
        reply = session.generate_reply()
        response_id = await peer.start(await peer.next())
        item_id = await peer.message(response_id)
        await peer.end(response_id)
        await reply
        async with session._generation_ready:
            await asyncio.wait_for(
                session._generation_ready.wait_for(lambda: session._generations[response_id].ended),
                3,
            )
        session.playback_finished(
            response_id=response_id,
            message_id="stale-message",
            audio_end_ms=999,
            audio_transcript="wrong transcript",
            interrupted=interrupted,
            interrupted_by_user=by_user,
        )
        session.playback_finished(
            response_id=response_id,
            message_id=item_id,
            audio_end_ms=20,
            interrupted=False,
            interrupted_by_user=False,
        )
        receipt = await asyncio.wait_for(peer.receipts.get(), 3)
        assert receipt["played_ms"] == 20
        assert peer.receipts.empty()
        assert peer.truncations.empty()
        assert not peer.cancel_received.is_set()
        assert session.chat_ctx.get_by_id(item_id).text_content == "Hello"
