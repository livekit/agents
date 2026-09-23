# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from openai.types.beta.realtime.session import TurnDetection
from openai.types.realtime import (
    AudioTranscription,
    RealtimeError,
    RealtimeErrorEvent,
    RealtimeResponse,
    ResponseCreatedEvent,
)
from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

from livekit import rtc
from livekit.agents import llm
from livekit.plugins.alibaba.models import (
    DEFAULT_MODEL,
    DEFAULT_REGION,
    DEFAULT_VOICE,
    INPUT_SAMPLE_RATE,
    get_realtime_url,
)
from livekit.plugins.alibaba.realtime.realtime_model import (
    RealtimeModel,
    RealtimeSession,
    _DashScopeWSAdapter,
)

pytestmark = pytest.mark.unit


class DashScopeServer:
    """A local protocol peer; tests control acknowledgements and server events."""

    def __init__(self) -> None:
        self.messages: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.connections: asyncio.Queue[web.WebSocketResponse] = asyncio.Queue()
        self.requests: list[web.Request] = []
        self.session_updates: list[dict[str, Any]] = []
        self.ack_updates = True

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        self.requests.append(request)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.connections.put_nowait(ws)
        await ws.send_json({"type": "session.created", "session": {"id": "test_session"}})
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                event = json.loads(msg.data)
                self.messages.put_nowait(event)
                if event["type"] == "session.update" and self.ack_updates:
                    self.session_updates.append(event["session"])
                    await ws.send_json({"type": "session.updated", "session": event["session"]})
        return ws

    async def receive(self, event_type: str) -> dict[str, Any]:
        event = await asyncio.wait_for(self.messages.get(), timeout=2)
        assert event["type"] == event_type
        return event


@asynccontextmanager
async def connected_session(
    turn_detection_disabled: bool = False,
    **options: Any,
) -> AsyncIterator[tuple[RealtimeSession, DashScopeServer, web.WebSocketResponse]]:
    peer = DashScopeServer()
    app = web.Application()
    app.router.add_get("/realtime", peer.handle)
    async with TestServer(app) as server, aiohttp.ClientSession() as http:
        model = RealtimeModel(
            api_key="test-api-key",
            base_url=str(server.make_url("/realtime")),
            http_session=http,
            **options,
        )
        session = model.session(turn_detection_disabled=turn_detection_disabled)
        try:
            ws = await asyncio.wait_for(peer.connections.get(), timeout=2)
            await peer.receive("session.update")
            yield session, peer, ws
        finally:
            await session.aclose()
            await model.aclose()


async def test_region_environment_selects_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.plugins.alibaba.models import REALTIME_BASE_URLS

    peer = DashScopeServer()
    app = web.Application()
    app.router.add_get("/realtime", peer.handle)
    async with TestServer(app) as server, aiohttp.ClientSession() as http:
        monkeypatch.setenv("DASHSCOPE_REGION", "intl")
        monkeypatch.delenv("DASHSCOPE_WORKSPACE_ID", raising=False)
        monkeypatch.setitem(REALTIME_BASE_URLS, "intl", str(server.make_url("/realtime")))
        monkeypatch.setitem(REALTIME_BASE_URLS, "cn", str(server.make_url("/wrong-region")))
        model = RealtimeModel(api_key="test-api-key", http_session=http)
        session = model.session()
        try:
            await peer.receive("session.update")
            assert peer.requests[0].query["model"] == "qwen-audio-3.1-realtime-plus"
            assert peer.requests[0].headers["Authorization"] == "Bearer test-api-key"
        finally:
            await session.aclose()
            await model.aclose()


async def test_manual_audio_commit_preserves_200ms_of_16khz_audio() -> None:
    async with connected_session(turn_detection=None) as (session, peer, _):
        pcm = b"\x01\x00" * 3200
        session.push_audio(rtc.AudioFrame(pcm, 16000, 1, 3200))
        session.commit_audio()
        chunks = [await peer.receive("input_audio_buffer.append") for _ in range(2)]
        await peer.receive("input_audio_buffer.commit")
        assert b"".join(base64.b64decode(chunk["audio"]) for chunk in chunks) == pcm


@pytest.mark.parametrize("query", ["?tenant=test", "?tenant=test&model=endpoint-model"])
async def test_custom_endpoint_preserves_query_without_duplicating_model(query: str) -> None:
    peer = DashScopeServer()
    app = web.Application()
    app.router.add_get("/realtime", peer.handle)
    async with TestServer(app) as server, aiohttp.ClientSession() as http:
        model = RealtimeModel(
            api_key="test",
            model="model/with special+characters",
            base_url=str(server.make_url("/realtime")) + query,
            http_session=http,
        )
        session = model.session()
        try:
            await peer.receive("session.update")
            assert peer.requests[0].query["tenant"] == "test"
            expected = "endpoint-model" if "model=" in query else model.model
            assert peer.requests[0].query.getall("model") == [expected]
        finally:
            await session.aclose()
            await model.aclose()


async def test_voice_update_does_not_change_manual_turn_detection() -> None:
    async with connected_session(turn_detection=None) as (session, peer, _):
        session.update_options(voice="Cherry")
        event = await peer.receive("session.update")
        assert event["session"] == {"voice": "Cherry"}
        assert session.capabilities.turn_detection is False


async def test_tool_choice_update_reaches_provider() -> None:
    async with connected_session() as (session, peer, _):
        session.update_options(tool_choice="none")
        event = await peer.receive("session.update")
        assert event["session"] == {"tool_choice": "none"}


async def test_client_turn_taking_tool_policy_update_does_not_resend_vad() -> None:
    async with connected_session(turn_detection_disabled=True) as (session, peer, _):
        session.update_options(tool_choice="none")
        assert (await peer.receive("session.update"))["session"] == {"tool_choice": "none"}


async def test_rejected_chat_update_completes_without_waiting_for_timeout() -> None:
    async with connected_session() as (session, peer, ws):
        ctx = llm.ChatContext()
        ctx.add_message(role="user", content="Hello")
        update = asyncio.create_task(session.update_chat_ctx(ctx))
        try:
            event = await peer.receive("conversation.item.create")
            await ws.send_json(
                {
                    "type": "error",
                    "event_id": "error_1",
                    "error": {
                        "type": "invalid_request_error",
                        "code": "invalid_value",
                        "message": "invalid item",
                        "event_id": event["event_id"],
                    },
                }
            )
            await asyncio.wait_for(update, timeout=1)
            assert session.chat_ctx.items == []
        finally:
            if not update.done():
                update.cancel()
            await asyncio.gather(update, return_exceptions=True)


async def test_transcription_update_reaches_provider() -> None:
    async with connected_session() as (session, peer, _):
        session.update_options(input_audio_transcription=None)
        assert (await peer.receive("session.update"))["session"] == {
            "input_audio_transcription": None,
        }
        session.update_options(
            input_audio_transcription=AudioTranscription(model="gummy-realtime-v1")
        )
        assert (await peer.receive("session.update"))["session"] == {
            "input_audio_transcription": {"model": "gummy-realtime-v1"},
        }


@pytest.mark.parametrize("option,value", [("speed", 1.2), ("reasoning", {"effort": "low"})])
def test_unsupported_constructor_options_are_rejected(option: str, value: Any) -> None:
    with pytest.raises(ValueError, match=f"{option} is not supported"):
        RealtimeModel(api_key="test", **{option: value})


async def test_unsupported_session_options_are_rejected() -> None:
    async with connected_session() as (session, _, _):
        with pytest.raises(ValueError, match="speed is not supported"):
            session.update_options(speed=1.2)


@pytest.mark.parametrize(
    "turn_detection",
    [
        TurnDetection(type="semantic_vad"),
        TurnDetection(type="server_vad", create_response=False),
        TurnDetection(type="server_vad", interrupt_response=False),
        TurnDetection(type="server_vad", eagerness="high"),
        ServerVad(type="server_vad", idle_timeout_ms=5000),
    ],
)
def test_unsupported_vad_configuration_is_rejected(
    turn_detection: TurnDetection | ServerVad,
) -> None:
    with pytest.raises(ValueError, match="turn_detection"):
        RealtimeModel(api_key="test", turn_detection=turn_detection)


async def test_rejected_interrupted_create_does_not_reset_next_response() -> None:
    async with connected_session() as (session, peer, ws):
        reply = session.generate_reply()
        request = await peer.receive("response.create")
        session.interrupt()
        await peer.receive("response.cancel")
        assert reply.cancelled()
        await ws.send_json(
            {
                "type": "error",
                "event_id": "rejected",
                "error": {
                    "type": "invalid_request_error",
                    "code": "invalid_value",
                    "message": "invalid request",
                    "event_id": request["event_id"],
                },
            }
        )
        # A following server event is an ordered barrier for rejection handling.
        rejected = asyncio.Event()
        session.on("input_speech_started", lambda _: rejected.set())
        await ws.send_json(
            {
                "type": "input_audio_buffer.speech_started",
                "event_id": "barrier",
                "item_id": "user",
                "audio_start_ms": 0,
            }
        )
        await asyncio.wait_for(rejected.wait(), 1)
        next_reply = session.generate_reply()
        await peer.receive("response.create")
        await ws.send_json(
            {
                "type": "response.created",
                "event_id": "created",
                "response": {"id": "next", "status": "in_progress"},
            }
        )
        assert (await asyncio.wait_for(next_reply, 1)).response_id == "next"
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(peer.connections.get(), 10.5)
        assert not ws.closed


async def test_model_updates_active_session_voice() -> None:
    peer = DashScopeServer()
    app = web.Application()
    app.router.add_get("/realtime", peer.handle)
    async with TestServer(app) as server, aiohttp.ClientSession() as http:
        model = RealtimeModel(
            api_key="test", base_url=str(server.make_url("/realtime")), http_session=http
        )
        session = model.session()
        try:
            await peer.receive("session.update")
            model.update_options(voice="Cherry")
            assert (await peer.receive("session.update"))["session"] == {"voice": "Cherry"}
        finally:
            await session.aclose()
            await model.aclose()


async def test_voice_update_after_audio_is_rejected_without_breaking_session() -> None:
    async with connected_session() as (session, peer, _):
        session.push_audio(rtc.AudioFrame(b"\x00" * 6400, 16000, 1, 3200))
        await peer.receive("input_audio_buffer.append")
        await peer.receive("input_audio_buffer.append")
        for target in (session, session.realtime_model):
            with pytest.raises(ValueError, match="voice.*audio"):
                target.update_options(voice="longanlingxin")
        session.update_options(tool_choice="none")
        assert (await peer.receive("session.update"))["session"] == {"tool_choice": "none"}


async def test_runtime_tool_update_does_not_resend_immutable_audio_config() -> None:
    async with connected_session() as (session, peer, _):
        await session.update_tools([])
        assert (await peer.receive("session.update"))["session"] == {"tools": []}


async def test_client_turn_taking_cannot_be_reenabled_by_option_update() -> None:
    async with connected_session(turn_detection_disabled=True) as (session, _, _):
        with pytest.raises(ValueError, match="turn_detection is fixed"):
            session.update_options(turn_detection=ServerVad(type="server_vad"))
        assert session.capabilities.turn_detection is False


async def test_timed_out_reply_reconnects_before_accepting_another_request() -> None:
    async with connected_session(turn_detection=None) as (session, peer, ws):
        generations: list[llm.GenerationCreatedEvent] = []
        session.on("generation_created", generations.append)
        reply = session.generate_reply()
        await peer.receive("response.create")
        with pytest.raises(llm.RealtimeError, match="timed out"):
            await asyncio.wait_for(asyncio.shield(reply), timeout=12)
        # No response from the old connection can be assigned to a new request.
        if not ws.closed:
            with pytest.raises(llm.RealtimeError, match="reconnect"):
                session.generate_reply()
        new_ws = await asyncio.wait_for(peer.connections.get(), timeout=2)
        assert new_ws is not ws
        await peer.receive("session.update")
        next_reply = session.generate_reply()
        await peer.receive("response.create")
        await new_ws.send_json(
            {
                "type": "response.created",
                "event_id": "new_event",
                "response": {"id": "new_response", "status": "in_progress"},
            }
        )
        assert (await asyncio.wait_for(next_reply, 1)).response_id == "new_response"
        assert len(generations) == 1


@pytest.mark.parametrize("echo_metadata", [False, True])
async def test_reply_correlation_consumes_acknowledged_request(echo_metadata: bool) -> None:
    async with connected_session(turn_detection=None) as (session, peer, ws):
        done = asyncio.Event()
        session.on("metrics_collected", lambda _: done.set())
        for index in range(2):
            done.clear()
            reply = session.generate_reply()
            request = await peer.receive("response.create")
            response: dict[str, Any] = {"id": f"reply_{index}", "status": "in_progress"}
            if echo_metadata and index == 0:
                response["metadata"] = request["response"]["metadata"]
            await ws.send_json(
                {"type": "response.created", "event_id": f"event_{index}", "response": response}
            )
            generation = await asyncio.wait_for(reply, 1)
            assert generation.response_id == f"reply_{index}"
            assert generation.user_initiated
            await ws.send_json(
                {
                    "type": "response.done",
                    "event_id": f"done_{index}",
                    "response": {"id": f"reply_{index}", "status": "completed", "output": []},
                }
            )
            await asyncio.wait_for(done.wait(), 1)


async def test_rejected_reply_does_not_steal_next_acknowledgement() -> None:
    async with connected_session(turn_detection=None) as (session, peer, ws):
        rejected = session.generate_reply()
        request = await peer.receive("response.create")
        await ws.send_json(
            {
                "type": "error",
                "event_id": "err_1",
                "error": {
                    "type": "invalid_request_error",
                    "code": "invalid_value",
                    "message": "invalid request",
                    "event_id": request["event_id"],
                },
            }
        )
        with pytest.raises(llm.RealtimeError, match="invalid request"):
            await asyncio.wait_for(rejected, 1)
        reply = session.generate_reply()
        await peer.receive("response.create")
        await ws.send_json(
            {
                "type": "response.created",
                "event_id": "ev_2",
                "response": {"id": "accepted", "status": "in_progress"},
            }
        )
        assert (await asyncio.wait_for(reply, 1)).response_id == "accepted"


async def test_native_response_error_without_request_id_settles_pending() -> None:
    async with connected_session() as (session, peer, ws):
        reply = session.generate_reply()
        await peer.receive("response.create")
        await ws.send_json(
            {
                "type": "error",
                "event_id": "server_generated_id",
                "error": {
                    "type": "invalid_request_error",
                    "code": "invalid_value",
                    "param": "response.create",
                    "message": "Cannot create response: conversation has no messages or no user message.",
                },
            }
        )
        with pytest.raises(llm.RealtimeError, match="uncorrelated.*reconnecting"):
            await asyncio.wait_for(reply, 1)


async def test_first_chat_item_omits_openai_root_sentinel() -> None:
    async with connected_session() as (session, peer, ws):
        ctx = llm.ChatContext()
        ctx.add_message(role="user", content="Hello", id="greeting")
        update = asyncio.create_task(session.update_chat_ctx(ctx))
        try:
            event = await peer.receive("conversation.item.create")
            assert event.get("previous_item_id") in (None, "")
            assert event["item"]["id"] == "greeting"
            await ws.send_json(
                {
                    "type": "conversation.item.created",
                    "event_id": "created",
                    "previous_item_id": None,
                    "item": event["item"],
                }
            )
            await asyncio.wait_for(update, 1)
        finally:
            update.cancel()
            await asyncio.gather(update, return_exceptions=True)


async def test_cancelled_reply_discards_its_late_acknowledgement() -> None:
    async with connected_session(turn_detection=None) as (session, peer, ws):
        generations: list[llm.GenerationCreatedEvent] = []
        session.on("generation_created", generations.append)
        cancelled = session.generate_reply()
        await peer.receive("response.create")
        cancelled.cancel()
        await peer.receive("response.cancel")
        await ws.send_json(
            {
                "type": "response.created",
                "event_id": "ev_cancelled",
                "response": {"id": "cancelled", "status": "in_progress"},
            }
        )
        assert (await peer.receive("response.cancel"))["response_id"] == "cancelled"
        assert generations == []


async def test_reconnect_fails_pending_reply_and_resets_correlation() -> None:
    async with connected_session(turn_detection=None) as (session, peer, ws):
        disconnected = session.generate_reply()
        await peer.receive("response.create")
        await ws.close()
        new_ws = await asyncio.wait_for(peer.connections.get(), 2)
        await peer.receive("session.update")
        with pytest.raises(llm.RealtimeError, match="reconnection"):
            await disconnected
        reply = session.generate_reply()
        await peer.receive("response.create")
        await new_ws.send_json(
            {
                "type": "response.created",
                "event_id": "ev_reconnected",
                "response": {"id": "reconnected", "status": "in_progress"},
            }
        )
        assert (await asyncio.wait_for(reply, 1)).response_id == "reconnected"


async def test_reconnected_listener_can_immediately_request_reply() -> None:
    async with connected_session() as (session, peer, ws):
        replies: list[asyncio.Future[llm.GenerationCreatedEvent]] = []
        session.on("session_reconnected", lambda _: replies.append(session.generate_reply()))
        await ws.close()
        new_ws = await asyncio.wait_for(peer.connections.get(), 2)
        await peer.receive("session.update")
        await peer.receive("response.create")
        await new_ws.send_json(
            {
                "type": "response.created",
                "event_id": "created",
                "response": {"id": "new", "status": "in_progress"},
            }
        )
        assert (await asyncio.wait_for(replies[0], 1)).response_id == "new"


@pytest.mark.parametrize("server_vad", [True, False])
async def test_tool_call_and_result_round_trip_preserves_item_ids(server_vad: bool) -> None:
    @llm.function_tool
    def weather(city: str) -> str:
        """Look up the weather for a city."""
        return f"Sunny in {city}"

    options = {} if server_vad else {"turn_detection": None}
    async with connected_session(**options) as (session, peer, ws):
        await session.update_tools([weather])
        tools = (await peer.receive("session.update"))["session"]["tools"]
        assert tools[0]["name"] == "weather"
        assert tools[0]["parameters"]["required"] == ["city"]
        reply = session.generate_reply()
        await peer.receive("response.create")
        await ws.send_json(
            {
                "type": "response.created",
                "event_id": "created",
                "response": {"id": "tool_response", "status": "in_progress"},
            }
        )
        generation = await asyncio.wait_for(reply, 1)
        item = {
            "id": "tool_item",
            "type": "function_call",
            "call_id": "call_weather",
            "name": "weather",
            "arguments": '{"city":"Paris"}',
            "status": "completed",
        }
        await ws.send_json(
            {
                "type": "conversation.item.created",
                "event_id": "item_created",
                "previous_item_id": None,
                "item": item,
            }
        )
        await ws.send_json(
            {
                "type": "response.output_item.done",
                "event_id": "item_done",
                "response_id": "tool_response",
                "output_index": 0,
                "item": item,
            }
        )
        await ws.send_json(
            {
                "type": "response.done",
                "event_id": "done",
                "response": {"id": "tool_response", "status": "completed", "output": [item]},
            }
        )
        calls = [call async for call in generation.function_stream]
        assert len(calls) == 1
        assert calls[0].call_id == "call_weather"
        assert json.loads(calls[0].arguments) == {"city": "Paris"}

        ctx = session.chat_ctx.copy()
        ctx.items.append(
            llm.FunctionCallOutput(
                id="result_item",
                call_id=calls[0].call_id,
                name=calls[0].name,
                output=weather("Paris"),
                is_error=False,
            )
        )
        update = asyncio.create_task(session.update_chat_ctx(ctx))
        try:
            request = await peer.receive("conversation.item.create")
            assert request["previous_item_id"] == "tool_item"
            assert request["item"]["id"] == "result_item"
            assert request["item"]["output"] == "Sunny in Paris"
            await ws.send_json(
                {
                    "type": "conversation.item.created",
                    "event_id": "result_created",
                    "previous_item_id": "tool_item",
                    "item": request["item"],
                }
            )
            await asyncio.wait_for(update, 1)
            assert session.chat_ctx.get_by_id("result_item") is not None
        finally:
            update.cancel()
            await asyncio.gather(update, return_exceptions=True)


async def test_beta_audio_events_produce_complete_24khz_audio_and_text_streams() -> None:
    async with connected_session(turn_detection=None) as (session, peer, ws):
        reply = session.generate_reply()
        await peer.receive("response.create")
        await ws.send_json(
            {
                "type": "response.created",
                "event_id": "created",
                "response": {"id": "speech", "status": "in_progress"},
            }
        )
        generation = await asyncio.wait_for(reply, 1)
        item = {"id": "speech_item", "type": "message", "role": "assistant", "content": []}
        pcm = b"\x01\x00" * 2400
        for event in [
            {"type": "response.output_item.added", "item": item, "output_index": 0},
            {"type": "response.audio_transcript.delta", "delta": "Hello"},
            {"type": "response.audio.delta", "delta": base64.b64encode(pcm).decode()},
            {"type": "response.output_item.done", "item": item, "output_index": 0},
            {
                "type": "response.done",
                "response": {"id": "speech", "status": "completed", "output": [item]},
            },
        ]:
            await ws.send_json(
                {
                    "event_id": "ev",
                    "response_id": "speech",
                    "item_id": "speech_item",
                    "content_index": 0,
                    **event,
                }
            )
        messages = [message async for message in generation.message_stream]
        assert len(messages) == 1
        assert "".join([text async for text in messages[0].text_stream]) == "Hello"
        audio = [frame async for frame in messages[0].audio_stream]
        assert len(audio) == 1
        assert audio[0].sample_rate == 24000
        assert audio[0].data.tobytes() == pcm


async def test_server_vad_supports_manual_and_automatic_replies() -> None:
    async with connected_session() as (session, peer, ws):

        @llm.function_tool
        def weather() -> str:
            """Return weather."""
            return "Sunny"

        await session.update_tools([weather])
        await peer.receive("session.update")
        reply = session.generate_reply()
        await peer.receive("response.create")
        await ws.send_json(
            {
                "type": "response.created",
                "event_id": "manual_created",
                "response": {"id": "manual", "status": "in_progress"},
            }
        )
        assert (await asyncio.wait_for(reply, 1)).user_initiated
        await ws.send_json(
            {
                "type": "response.done",
                "event_id": "manual_done",
                "response": {"id": "manual", "status": "completed", "output": []},
            }
        )
        automatic = asyncio.get_running_loop().create_future()
        session.on(
            "generation_created",
            lambda ev: automatic.set_result(ev) if not automatic.done() else None,
        )
        await ws.send_json(
            {
                "type": "response.created",
                "event_id": "auto_created",
                "response": {"id": "automatic", "status": "in_progress"},
            }
        )
        assert (await asyncio.wait_for(automatic, 1)).user_initiated is False
        await ws.send_json(
            {
                "type": "response.done",
                "event_id": "auto_done",
                "response": {"id": "automatic", "status": "completed", "output": []},
            }
        )


async def test_interrupt_waits_for_matching_terminal_event_before_new_reply() -> None:
    async with connected_session() as (session, peer, ws):
        reply = session.generate_reply()
        await peer.receive("response.create")
        await ws.send_json(
            {
                "type": "response.created",
                "event_id": "created",
                "response": {"id": "active", "status": "in_progress"},
            }
        )
        await asyncio.wait_for(reply, 1)
        with pytest.raises(llm.RealtimeError, match="active"):
            session.generate_reply()
        session.interrupt()
        await peer.receive("response.cancel")
        next_reply = session.generate_reply()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(peer.messages.get(), 0.05)
        done = asyncio.Event()
        session.on("input_speech_started", lambda _: done.set())
        await ws.send_json(
            {
                "type": "response.done",
                "event_id": "done",
                "response": {"id": "active", "status": "cancelled", "output": []},
            }
        )
        await ws.send_json(
            {
                "type": "input_audio_buffer.speech_started",
                "event_id": "barrier",
                "item_id": "user",
                "audio_start_ms": 0,
            }
        )
        await asyncio.wait_for(done.wait(), 1)
        await peer.receive("response.create")
        await ws.send_json(
            {
                "type": "response.created",
                "event_id": "created2",
                "response": {"id": "next", "status": "in_progress"},
            }
        )
        assert (await asyncio.wait_for(next_reply, 1)).response_id == "next"


async def test_stale_done_does_not_release_new_active_response() -> None:
    async with connected_session() as (session, peer, ws):
        reply = session.generate_reply()
        await peer.receive("response.create")
        await ws.send_json(
            {
                "type": "response.created",
                "event_id": "created",
                "response": {"id": "current", "status": "in_progress"},
            }
        )
        await asyncio.wait_for(reply, 1)
        # Use a following user-transcription event as an ordered receive barrier.
        barrier = asyncio.Event()
        session.on("input_speech_started", lambda _: barrier.set())
        await ws.send_json(
            {
                "type": "response.done",
                "event_id": "old_done",
                "response": {"id": "old", "status": "cancelled", "output": []},
            }
        )
        await ws.send_json(
            {
                "type": "input_audio_buffer.speech_started",
                "event_id": "speech",
                "item_id": "user",
                "audio_start_ms": 0,
            }
        )
        await asyncio.wait_for(barrier.wait(), 1)
        with pytest.raises(llm.RealtimeError, match="active"):
            session.generate_reply()


async def test_close_settles_pending_reply_and_closes_socket() -> None:
    async with connected_session() as (session, peer, ws):
        reply = session.generate_reply()
        await peer.receive("response.create")
        await asyncio.wait_for(session.aclose(), 2)
        with pytest.raises(llm.RealtimeError, match="closed"):
            await asyncio.wait_for(reply, 1)
        assert ws.closed


async def test_interrupt_closes_local_streams_before_server_acknowledgement() -> None:
    async with connected_session() as (session, peer, ws):
        reply = session.generate_reply()
        await peer.receive("response.create")
        await ws.send_json(
            {
                "type": "response.created",
                "event_id": "created",
                "response": {"id": "active", "status": "in_progress"},
            }
        )
        generation = await asyncio.wait_for(reply, 1)
        session.interrupt()

        async def drain() -> list[Any]:
            return [message async for message in generation.message_stream]

        assert await asyncio.wait_for(drain(), 0.5) == []
        assert session.has_active_generation


async def test_interrupt_before_created_discards_late_generation_until_done() -> None:
    async with connected_session() as (session, peer, ws):
        reply = session.generate_reply()
        await peer.receive("response.create")
        session.interrupt()
        await peer.receive("response.cancel")
        assert reply.cancelled()
        await ws.send_json(
            {
                "type": "response.created",
                "event_id": "late",
                "response": {"id": "late", "status": "in_progress"},
            }
        )
        await peer.receive("response.cancel")
        with pytest.raises(llm.RealtimeError, match="active"):
            session.generate_reply()


async def test_cancel_without_terminal_event_reconnects() -> None:
    async with connected_session() as (session, peer, ws):
        reply = session.generate_reply()
        await peer.receive("response.create")
        await ws.send_json(
            {
                "type": "response.created",
                "event_id": "created",
                "response": {"id": "active", "status": "in_progress"},
            }
        )
        await asyncio.wait_for(reply, 1)
        session.interrupt()
        await peer.receive("response.cancel")
        replacement = session.generate_reply()
        replacement.cancel()
        new_ws = await asyncio.wait_for(peer.connections.get(), 12)
        await peer.receive("session.update")
        assert new_ws is not ws


async def test_pending_or_cancelled_unacknowledged_reply_blocks_another_request() -> None:
    async with connected_session(turn_detection=None) as (session, peer, ws):
        reply = session.generate_reply()
        await peer.receive("response.create")
        with pytest.raises(llm.RealtimeError, match="already pending"):
            session.generate_reply()
        reply.cancel()
        await peer.receive("response.cancel")
        with pytest.raises(llm.RealtimeError, match="already pending"):
            session.generate_reply()
        await ws.send_json(
            {
                "type": "response.created",
                "event_id": "cancelled_created",
                "response": {"id": "cancelled", "status": "in_progress"},
            }
        )
        await peer.receive("response.cancel")


async def test_cancelled_request_queued_before_transmission_is_not_sent() -> None:
    async with connected_session(turn_detection=None) as (session, peer, ws):
        reply = session.generate_reply()
        reply.cancel()
        # A harmless update is an outgoing barrier: no create/cancel may precede it.
        session.update_options(voice="Cherry")
        await peer.receive("session.update")
        next_reply = session.generate_reply()
        await peer.receive("response.create")
        await ws.send_json(
            {
                "type": "response.created",
                "event_id": "new_created",
                "response": {"id": "new", "status": "in_progress"},
            }
        )
        assert (await asyncio.wait_for(next_reply, 1)).response_id == "new"


def test_default_model_and_config() -> None:
    assert DEFAULT_MODEL == "qwen-audio-3.1-realtime-plus"
    assert DEFAULT_VOICE == "longanqian"
    assert DEFAULT_REGION == "cn"

    model = RealtimeModel(api_key="test-api-key")
    assert model.model == "qwen-audio-3.1-realtime-plus"
    assert model._opts.voice == "longanqian"
    assert model._opts.base_url == "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="api_key client option must be set"):
        RealtimeModel()


def test_url_construction_for_regions_and_workspaces() -> None:
    # 1. Domestic China default
    url_cn = get_realtime_url(region="cn")
    assert url_cn == "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"

    # 2. International (Singapore)
    url_intl = get_realtime_url(region="intl")
    assert url_intl == "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"

    # 3. Domestic Workspace
    url_ws_cn = get_realtime_url(region="cn", workspace_id="ws-my-team")
    assert url_ws_cn == "wss://ws-my-team.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime"

    # 4. International Workspace
    url_ws_intl = get_realtime_url(region="intl", workspace_id="ws-my-intl")
    assert url_ws_intl == "wss://ws-my-intl.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime"

    # 5. Explicit base_url override
    url_custom = get_realtime_url(base_url="wss://custom.proxy/endpoint")
    assert url_custom == "wss://custom.proxy/endpoint"


async def test_session_initialization_formats_dashscope_payload() -> None:
    turn_detection = ServerVad(
        type="server_vad",
        threshold=0.6,
        prefix_padding_ms=250,
        silence_duration_ms=900,
        create_response=True,
    )
    async with connected_session(turn_detection=turn_detection) as (session, peer, _):
        payload = peer.session_updates[0]
        assert payload["modalities"] == ["text", "audio"]
        assert payload["input_audio_format"] == "pcm"
        assert payload["output_audio_format"] == "pcm"
        assert payload["voice"] == "longanqian"
        assert payload["input_audio_transcription"] == {"model": "gummy-realtime-v1"}
        assert payload["turn_detection"] == {
            "type": "server_vad",
            "threshold": 0.6,
            "prefix_padding_ms": 250,
            "silence_duration_ms": 900,
        }
        await session.update_instructions("Test prompt")
        assert (await peer.receive("session.update"))["session"] == {"instructions": "Test prompt"}


@pytest.mark.parametrize("options", [{"turn_detection_disabled": True}, {"turn_detection": None}])
async def test_session_initialization_and_reconnect_disable_turn_detection(
    options: dict[str, Any],
) -> None:
    async with connected_session(**options) as (session, peer, ws):
        assert peer.session_updates[0]["turn_detection"] is None
        assert session.capabilities.turn_detection is False
        await ws.close()
        await asyncio.wait_for(peer.connections.get(), 2)
        assert (await peer.receive("session.update"))["session"]["turn_detection"] is None


@pytest.mark.asyncio
async def test_dashscope_ws_adapter_event_normalization() -> None:
    # Mock WebSocket response
    mock_ws = MagicMock()

    test_messages = [
        # 1. response.audio.delta -> response.output_audio.delta
        aiohttp.WSMessage(
            type=aiohttp.WSMsgType.TEXT,
            data=json.dumps({"type": "response.audio.delta", "delta": "AQID"}),
            extra="",
        ),
        # 2. conversation.item.created -> conversation.item.added
        aiohttp.WSMessage(
            type=aiohttp.WSMsgType.TEXT,
            data=json.dumps({"type": "conversation.item.created", "item": {"id": "item_123"}}),
            extra="",
        ),
        # 3. response.audio_transcript.delta -> response.output_audio_transcript.delta
        aiohttp.WSMessage(
            type=aiohttp.WSMsgType.TEXT,
            data=json.dumps({"type": "response.audio_transcript.delta", "delta": "hello"}),
            extra="",
        ),
        # 4. Standard GA event untouched
        aiohttp.WSMessage(
            type=aiohttp.WSMsgType.TEXT,
            data=json.dumps({"type": "response.created", "response": {"id": "resp_1"}}),
            extra="",
        ),
    ]

    async def mock_receive(timeout: float | None = None) -> aiohttp.WSMessage:
        return test_messages.pop(0)

    mock_ws.receive = mock_receive

    adapter = _DashScopeWSAdapter(mock_ws, mock_ws.send_str)

    # Message 1 (test via receive, as called by OpenAI Realtime _recv_task)
    m1 = await adapter.receive()
    d1 = json.loads(m1.data)
    assert d1["type"] == "response.output_audio.delta"
    assert d1["delta"] == "AQID"

    # Message 2 (test via receive)
    m2 = await adapter.receive()
    d2 = json.loads(m2.data)
    assert d2["type"] == "conversation.item.added"
    assert d2["item"]["id"] == "item_123"

    # Message 3 (test via receive)
    m3 = await adapter.receive()
    d3 = json.loads(m3.data)
    assert d3["type"] == "response.output_audio_transcript.delta"
    assert d3["delta"] == "hello"

    # Message 4 (test via receive)
    m4 = await adapter.receive()
    d4 = json.loads(m4.data)
    assert d4["type"] == "response.created"


def test_response_created_associates_missing_metadata() -> None:
    sess = RealtimeSession.__new__(RealtimeSession)
    sess._pending_response_id = "client_resp_req_1"
    sess._pending_response_sent = True
    sess._response_idle = asyncio.Event()
    sess._response_timeout = None
    sess._correlation_lost = False
    fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.Future()
    sess._response_created_futures = {"client_resp_req_1": fut}
    sess._discarded_event_ids = set()

    # Create dummy response without metadata
    resp_ev = ResponseCreatedEvent(
        event_id="ev_resp_created",
        type="response.created",
        response=RealtimeResponse(
            id="server_resp_id_999",
            object="realtime.response",
            status="in_progress",
            metadata=None,
        ),
    )

    # Mock super handler behavior
    handled_events: list[Any] = []
    sess._close_current_generation = lambda reason=None: None  # type: ignore[method-assign]
    sess.emit = lambda name, ev: handled_events.append((name, ev))  # type: ignore[method-assign,misc]

    sess._handle_response_created(resp_ev)

    assert sess._pending_response_id is None
    assert resp_ev.response.metadata == {"client_event_id": "client_resp_req_1"}
    assert "client_resp_req_1" not in sess._response_created_futures
    assert fut.done()
    gen_ev = fut.result()
    assert gen_ev.user_initiated is True
    assert gen_ev.response_id == "server_resp_id_999"


def test_handle_error_ignores_benign_cancel_race() -> None:
    sess = RealtimeSession.__new__(RealtimeSession)
    emitted_errors: list[Any] = []
    sess._emit_error = lambda err, recoverable: emitted_errors.append((err, recoverable))  # type: ignore[method-assign,assignment]
    sess._realtime_model = SimpleNamespace(_provider_label="Alibaba Realtime API", _label="alibaba")  # type: ignore[assignment]
    sess._chat_ctx_event_futures = {}
    sess._response_created_futures = {}
    sess._opts = SimpleNamespace(turn_detection=None)  # type: ignore[assignment]

    # 1. "no ongoing response to cancel" -> ignored
    err_ev = RealtimeErrorEvent(
        event_id="err_1",
        type="error",
        error=RealtimeError(
            type="invalid_request_error",
            message="Conversation has no ongoing response to cancel",
        ),
    )
    sess._handle_error(err_ev)
    assert len(emitted_errors) == 0

    # 2. "Conversation has no active response" -> ignored
    err_ev_2 = RealtimeErrorEvent(
        event_id="err_2",
        type="error",
        error=RealtimeError(
            type="invalid_request_error",
            message="Conversation has no active response.",
        ),
    )
    sess._handle_error(err_ev_2)
    assert len(emitted_errors) == 0


def test_audio_resampler_outputs_16khz() -> None:
    sess = RealtimeSession.__new__(RealtimeSession)
    sess._input_resampler = None

    # Input: 48kHz mono frame with 100ms duration (4800 samples)
    samples_48k = 4800
    data_48k = b"\x00" * (samples_48k * 2)
    input_frame = rtc.AudioFrame(
        data=data_48k,
        sample_rate=48000,
        num_channels=1,
        samples_per_channel=samples_48k,
    )

    resampled_frames = list(sess._resample_audio(input_frame))
    assert len(resampled_frames) >= 1
    for f in resampled_frames:
        assert f.sample_rate == INPUT_SAMPLE_RATE
        assert f.num_channels == 1
