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
from rimelabs_sdk import Rime, RimeInputError
from websockets.asyncio.server import serve

from livekit.agents import Agent, AgentSession, function_tool, llm
from livekit.agents.llm import RealtimeModelFallbackAdapter
from livekit.plugins import rime
from livekit.plugins.rime.realtime import (
    PrismUnsupported,
)

from .fake_io import FakeAudioOutput


class Peer:
    def __init__(self):
        self.requests = asyncio.Queue()
        self.socket = None
        self.serial = 0
        self.settings = None
        self.updates = []
        self.revision = 0
        self.request_id = None
        self.update_started = asyncio.Event()
        self.update_release = asyncio.Event()
        self.update_release.set()
        self.connections = 0
        self.receipts = asyncio.Queue()
        self.truncations = asyncio.Queue()
        self.items = []
        self.server_items = []
        self.history_operations = []
        self.history_started = asyncio.Event()
        self.history_release = asyncio.Event()
        self.history_release.set()
        self.fail_history_at = None
        self.block_history_at = None
        self.running = {}
        self.history_serial = 0
        self.refusals = {}
        self.cancel_received = asyncio.Event()
        self.cancel_release = asyncio.Event()
        self.cancel_release.set()

    async def emit(self, kind, **body):
        if kind == "response.created":
            response = body["response"]
            self.running[response["id"]] = set(response.get("prism_input_item_ids", []))
        elif kind == "response.function_call_arguments.done":
            self.server_items.append(
                {
                    "id": body["item_id"],
                    "type": "function_call",
                    "call_id": body["call_id"],
                    "name": body["name"],
                    "arguments": body["arguments"],
                }
            )
            self.running.setdefault(body["response_id"], set()).add(body["item_id"])
        elif kind == "response.text.done":
            self.server_items.append(
                {
                    "id": body["item_id"],
                    "type": "message",
                    "role": "assistant",
                    "text": body["text"],
                }
            )
            self.running.setdefault(body["response_id"], set()).add(body["item_id"])
        elif kind == "response.done":
            self.running.pop(body["response"]["id"], None)
        await self.socket.send(json.dumps({"type": kind, "event_id": "server", **body}))

    async def handler(self, socket):
        self.socket = socket
        self.connections += 1
        self.settings = None
        self.server_items = []
        self.running = {}
        self.request_id = socket.request.headers.get("x-request-id", "peer-request")
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
                self.update_started.set()
                await self.update_release.wait()
                self.updates.append(event["session"])
                self.settings = {**(self.settings or {}), **event["session"]}
                self.revision += 1
                await self.emit(
                    "session.updated",
                    prism_request_event_id=event["event_id"],
                    session={
                        **self.settings,
                        "id": "session-1",
                        "voice": self.settings.get("voice", "test"),
                        "request_id": self.request_id,
                        "prism_settings_revision": self.revision,
                        "prism_effective_interrupt_response": self.settings.get(
                            "turn_detection", {}
                        ).get("interrupt_response", True),
                    },
                )
                await self.emit("prism.typed_input.ready")
            elif kind in ("conversation.item.create", "conversation.item.delete"):
                self.history_operations.append(event)
                self.history_started.set()
                if self.block_history_at is None or self.block_history_at == len(
                    self.history_operations
                ):
                    await self.history_release.wait()
                if self.fail_history_at == len(self.history_operations):
                    await self.emit(
                        "error",
                        error={
                            "code": "invalid_request",
                            "message": "History refused",
                            "scope": "event",
                            "owner": {"kind": "event", "event_id": event["event_id"]},
                        },
                    )
                    continue
                if kind == "conversation.item.create":
                    self.history_serial += 1
                    item = {"id": f"history-{self.history_serial}", **event["item"]}
                    self.items.append(event["item"])
                    self.server_items.append(item)
                    await self.emit(
                        "conversation.item.created",
                        prism_request_event_id=event["event_id"],
                        item=item,
                    )
                else:
                    item = next((i for i in self.server_items if i["id"] == event["item_id"]), None)
                    if item is not None:
                        call_id = item.get("call_id")
                        self.server_items[:] = [
                            i
                            for i in self.server_items
                            if i["id"] != item["id"]
                            and (not call_id or i.get("call_id") != call_id)
                        ]
                    await self.emit(
                        "conversation.item.deleted",
                        prism_request_event_id=event["event_id"],
                        item_id=event["item_id"],
                    )
            elif kind == "prism.conversation.get":
                active_ids = {i for ids in self.running.values() for i in ids}
                await self.emit(
                    "prism.conversation",
                    prism_request_event_id=event["event_id"],
                    items=[i for i in self.server_items if i["id"] not in active_ids],
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
                "prism_settings_revision": self.revision,
                "metadata": request["response"]["metadata"]
                if request
                else {"prism_cause": "user_audio"},
            }
        }
        if request and "prism_input_text" in request["response"]:
            input_id = f"typed-{self.serial}"
            body["response"]["prism_input_item_ids"] = [input_id]
            self.server_items.append(
                {
                    "id": input_id,
                    "type": "message",
                    "role": "user",
                    "text": request["response"]["prism_input_text"],
                }
            )
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

        await self.emit("prism.typed_input.ready")


@asynccontextmanager
async def peer_model(**options):
    peer = Peer()
    async with serve(peer.handler, "127.0.0.1", 0) as server:
        endpoint = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1/realtime"
        async with Rime(api_key="test") as client:
            model = rime.realtime.RealtimeModel(endpoint=endpoint, client=client, **options)
            try:
                yield peer, model, client
            finally:
                await model.aclose()


async def test_startup_seeds_history_before_first_request():
    async with peer_model() as (peer, model, _):
        session = model.session()
        ctx = llm.ChatContext.empty()
        ctx.items.append(llm.AgentConfigUpdate(instructions="Old configuration"))
        ctx.add_message(role="user", content="Earlier question")
        ctx.add_message(role="assistant", content="Earlier answer")
        await session._update_session(instructions="Be clear", chat_ctx=ctx, tools=[])
        assert peer.settings["instructions"] == "Be clear"
        assert [i["text"] for i in peer.items] == ["Earlier question", "Earlier answer"]
        assert all(isinstance(item, llm.ChatMessage) for item in session.chat_ctx.items)
        await session.update_instructions("Be clear")
        await session.update_instructions("Different")
        assert peer.settings["instructions"] == "Different"


@pytest.mark.parametrize("invalid_item", ["developer", "image", "tool", "long"])
async def test_history_update_validates_all_new_items_before_sending(invalid_item):
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session._update_session(instructions="test")
        ctx = session.chat_ctx
        ctx.add_message(role="user", content="Valid new history")
        if invalid_item == "developer":
            ctx.add_message(role="developer", content="Unsupported role")
        elif invalid_item == "image":
            ctx.add_message(role="user", content=[llm.ImageContent(image="https://test/image")])
        elif invalid_item == "tool":
            ctx.items.append(
                llm.FunctionCallOutput(call_id="unknown", output="Result", is_error=False)
            )
        else:
            ctx.add_message(role="user", content=" \t" if invalid_item == "blank" else "x" * 4001)
        with pytest.raises((llm.RealtimeError, RimeInputError)):
            await session.update_chat_ctx(ctx)
        assert peer.items == []
        assert session.chat_ctx.items == []


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
        await session._update_session(instructions="test")
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
        await session._update_session(instructions="test")
        with pytest.raises(PrismUnsupported):
            session.update_options(tool_choice="required")
        assert errors[-1].recoverable is False


async def test_injected_client_stays_open():
    async with peer_model() as (peer, model, client):
        await model.session()._update_session(instructions="test")
        await model.aclose()
        async with client.realtime.connect(endpoint=model._endpoint) as session:
            assert session.info.session_id == "session-1"


async def test_abandoned_reply_does_not_generate_after_ready():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session._update_session(instructions="test")
        # Keep the first response active so the second request waits for readiness.
        first = session.generate_reply()
        request = await peer.next()
        response_id = await peer.start(request)
        await first
        abandoned = session.generate_reply()
        await asyncio.sleep(0)
        abandoned.cancel()
        await asyncio.gather(abandoned, return_exceptions=True)
        await peer.end(response_id)
        await asyncio.sleep(0.02)
        assert peer.requests.empty()


async def test_admission_timeout_leaves_conversation_usable(monkeypatch):
    async with peer_model() as (peer, model, client):
        connect = client.realtime.connect
        monkeypatch.setattr(
            client.realtime,
            "connect",
            lambda **kwargs: connect(
                **{**kwargs, "timeouts": r.RealtimeTimeouts(ready_s=0.02, request_s=0.5)}
            ),
        )
        session = model.session()
        errors = []
        session.on("error", errors.append)
        await session._update_session(instructions="test")
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


@pytest.mark.parametrize("tool_choice", ["auto", "none"])
async def test_connection_failure_cancels_reply_waiting_for_adapter_observation(
    monkeypatch, tool_choice
):
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session._update_session(instructions="test")
        observed = session._on_event

        def delay_response_started(event):
            # The SDK has accepted the response, but the adapter has not yet
            # registered its generation when the connection fails.
            if not isinstance(event.payload, r.ResponseStarted):
                observed(event)

        monkeypatch.setattr(session, "_on_event", delay_response_started)
        waiting = asyncio.Event()
        wait = session._generation_ready.wait

        async def mark_waiting():
            waiting.set()
            return await wait()

        monkeypatch.setattr(session._generation_ready, "wait", mark_waiting)
        reply = session.generate_reply(tool_choice=tool_choice)
        await peer.start(await peer.next())
        await asyncio.wait_for(waiting.wait(), 3)
        assert not reply.done()
        await peer.socket.close()
        done, _ = await asyncio.wait([reply], timeout=3)
        assert reply in done
        assert reply.cancelled()
        assert session._failure is not None
        await asyncio.wait_for(session.aclose(), 3)


@pytest.mark.parametrize("wrapped", [False, True])
async def test_standard_session_text_and_empty_replies_need_no_playback_receipts(wrapped):
    async with peer_model() as (peer, model, _):
        provider = RealtimeModelFallbackAdapter([model]) if wrapped else model
        async with AgentSession(llm=provider) as session:
            session.output.audio = FakeAudioOutput()
            await session.start(Agent(instructions="Help"))
            for text in ("First question", "Next question"):
                speech = session.generate_reply(user_input=text)
                request = await peer.next()
                assert "prism_input_text" not in request["response"]
                assert request["response"]["metadata"]["prism_cause"] == "proactive"
                assert [i["text"] for i in peer.items if i.get("text") == text] == [text]
                response = await peer.start(request)
                if text == "Next question":
                    await peer.message(response)
                await peer.end(response)
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                assert speech.exception() is None
            assert peer.receipts.empty()


@pytest.mark.parametrize("max_tool_steps", [0, 1, 2])
async def test_standard_session_tool_limit_produces_final_reply_and_restores_tools(max_tool_steps):
    ran = []

    @function_tool
    async def lookup() -> str:
        """Look up a value."""
        ran.append(True)
        return "42"

    async with peer_model() as (peer, model, _):
        async with AgentSession(llm=model, max_tool_steps=max_tool_steps) as session:
            await session.start(Agent(instructions="Help", tools=[lookup]))
            speech = session.generate_reply()
            for step in range(max_tool_steps + 1):
                request = await peer.next()
                assert peer.settings["tools"]
                parent = await peer.start(request)
                await peer.emit(
                    "response.function_call_arguments.done",
                    response_id=parent,
                    item_id=f"tool-{step}",
                    call_id=f"call-{step}",
                    name="lookup",
                    arguments="{}",
                )
                await peer.end(parent)
            final = await peer.next()
            assert peer.settings["tools"] == []
            assert final["response"]["metadata"]["prism_cause"] == "tool_continuation"
            response = await peer.start(final)
            await peer.message(response, text="The result is 42.")
            await peer.end(response)
            await asyncio.wait_for(speech.wait_for_playout(), 3)
            assert speech.exception() is None
            assert ran == [True] * (max_tool_steps + 1)
            assert peer.settings["tools"][0]["function"]["name"] == "lookup"
            assert model._session._tool_choice == "auto"
            assert peer.receipts.empty()


async def test_tool_disabled_response_never_executes_unexpected_call():
    ran = []

    @function_tool
    async def lookup() -> str:
        """Look up a value."""
        ran.append(True)
        return "42"

    async with peer_model() as (peer, model, _):
        async with AgentSession(llm=model) as session:
            await session.start(Agent(instructions="Help", tools=[lookup]))
            errors = asyncio.Queue()
            model._session.on("error", errors.put_nowait)
            session.generate_reply(tool_choice="none")
            request = await peer.next()
            assert peer.settings["tools"] == []
            response = await peer.start(request)
            # Even if the model calls a tool from history, it must not reach execution.
            await peer.emit(
                "response.function_call_arguments.done",
                response_id=response,
                item_id="unexpected",
                call_id="unexpected",
                name="lookup",
                arguments="{}",
            )
            error = await asyncio.wait_for(errors.get(), 3)
            assert isinstance(error.error, PrismUnsupported)
            assert not error.recoverable
            assert ran == []


async def test_cancelled_tool_disabled_reply_restores_after_late_acceptance():
    @function_tool
    async def lookup() -> str:
        """Look up a value."""
        return "42"

    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.update_tools([lookup])
        pending = session.generate_reply(tool_choice="none")
        request = await peer.next()
        pending.cancel()
        await asyncio.sleep(0)
        assert peer.settings["tools"] == []
        response = await peer.start(request)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, 3)
        assert peer.cancel_received.is_set()
        assert peer.settings["tools"][0]["function"]["name"] == "lookup"
        assert peer.requests.empty()
        assert session._generations[response].abandoned


async def test_refused_tool_disabled_reply_restores_tools_and_keeps_session():
    @function_tool
    async def lookup() -> str:
        """Look up a value."""
        return "42"

    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.update_tools([lookup])
        errors = []
        session.on("error", errors.append)
        peer.refusals["response.create"] = ("invalid_request", "event")
        with pytest.raises(llm.RealtimeError, match="invalid_request"):
            await session.generate_reply(tool_choice="none")
        assert peer.settings["tools"][0]["function"]["name"] == "lookup"
        assert not errors
        pending = session.generate_reply(tool_choice="auto")
        response = await peer.start(await peer.next())
        await peer.end(response)
        assert (await pending).response_id == response


async def test_failed_tool_restoration_fails_session():
    @function_tool
    async def lookup() -> str:
        """Look up a value."""
        return "42"

    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.update_tools([lookup])
        errors = asyncio.Queue()
        session.on("error", errors.put_nowait)
        pending = session.generate_reply(tool_choice="none")
        request = await peer.next()
        peer.refusals["session.update"] = ("invalid_session", "event")
        await peer.start(request)
        await asyncio.gather(pending, return_exceptions=True)
        error = await asyncio.wait_for(errors.get(), 3)
        assert not error.recoverable
        assert "restoration" in str(error.error)
        assert session._failure is not None


async def test_ordinary_session_application_stop_truncates_without_completion_receipt():
    async with peer_model() as (peer, model, _):
        output = FakeAudioOutput(can_pause=True)
        async with AgentSession(llm=model) as session:
            session.output.audio = output
            await session.start(Agent(instructions="Help"))
            speech = session.generate_reply()
            response = await peer.start(await peer.next())
            item = await peer.message(response, seconds=1)
            async with asyncio.timeout(3):
                while output._started_at is None:
                    await asyncio.sleep(0)
            speech.interrupt(force=True)
            await asyncio.wait_for(speech.wait_for_playout(), 3)
            truncate = await asyncio.wait_for(peer.truncations.get(), 3)
            assert truncate["item_id"] == item
            assert peer.cancel_received.is_set()
            assert peer.receipts.empty()


@pytest.mark.parametrize("revision", ["older", "suppressed", "missing"])
async def test_automatic_reply_uses_its_own_tool_settings_during_suppression(revision):
    @function_tool
    async def lookup() -> str:
        """Look up a value."""
        return "42"

    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.update_tools([lookup])
        old_revision = peer.revision
        generations = asyncio.Queue()
        errors = asyncio.Queue()
        session.on("generation_created", generations.put_nowait)
        session.on("error", errors.put_nowait)
        pending = session.generate_reply(tool_choice="none")
        try:
            request = await peer.next()
            response = {"id": "early-speech", "metadata": {"prism_cause": "user_audio"}}
            if revision != "missing":
                response["prism_settings_revision"] = (
                    old_revision if revision == "older" else peer.revision
                )
            await peer.emit("response.created", response=response)
            generation = await asyncio.wait_for(generations.get(), 3)
            await peer.emit(
                "response.function_call_arguments.done",
                response_id="early-speech",
                item_id="early-tool",
                call_id="early-call",
                name="lookup",
                arguments="{}",
            )
            if revision == "older":
                call = await asyncio.wait_for(anext(generation.function_stream), 3)
                assert call.name == "lookup"
                assert errors.empty()
                await peer.end("early-speech")
                response_id = await peer.start(request)
                await peer.end(response_id)
                await asyncio.wait_for(pending, 3)
                assert peer.settings["tools"][0]["function"]["name"] == "lookup"
            else:
                error = await asyncio.wait_for(errors.get(), 3)
                assert isinstance(error.error, PrismUnsupported)
                assert not error.recoverable
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
