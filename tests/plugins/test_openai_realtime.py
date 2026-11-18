"""OpenAI Realtime: a VoiceSession against a local Realtime API peer speaking its literal JSON."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import Agent, VoiceSession, llm
from livekit.agents.voice.events import ErrorEvent
from livekit.plugins.openai.realtime import RealtimeModel, Voice
from tests.fakes import FakeAudioOutput, until


class _Peer:
    """The Realtime API's happy path: it keeps each `session.update`, acks every item the client
    creates and answers each `response.create` with one spoken message."""

    def __init__(self) -> None:
        self.sessions: list[dict[str, object]] = []
        self.authorizations: list[str | None] = []
        self._tail: str | None = None
        self._ids = 0
        self.reject_next_options = False

    def _id(self, prefix: str) -> str:
        self._ids += 1
        return f"{prefix}_{self._ids}"

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        self.authorizations.append(request.headers.get("Authorization"))
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        async for message in socket:
            event = json.loads(message.data)
            if event["type"] == "session.update":
                self.sessions.append(event["session"])
                if self.reject_next_options:
                    self.reject_next_options = False
                    await socket.send_json(
                        {
                            "type": "error",
                            "error": {
                                "type": "invalid_request_error",
                                "code": "cannot_update_voice",
                                "event_id": event["event_id"],
                                "message": "Cannot change voice after audio",
                            },
                        }
                    )
            elif event["type"] == "conversation.item.create":
                item = {**event["item"], "id": event["item"].get("id") or self._id("item")}
                await socket.send_json(
                    {
                        "type": "conversation.item.added",
                        "previous_item_id": self._tail,
                        "item": item,
                    }
                )
                self._tail = item["id"]
            elif event["type"] == "conversation.item.delete":
                await socket.send_json(
                    {"type": "conversation.item.deleted", "item_id": event["item_id"]}
                )
            elif event["type"] == "response.create":
                await self._speak(socket, event["response"]["metadata"])
        return socket

    async def _speak(self, socket: web.WebSocketResponse, metadata: object) -> None:
        response, item = self._id("resp"), self._id("item")
        audio = base64.b64encode(b"\x00\x00" * 4800).decode()
        transcript = "Hello from the Realtime API."
        message = {
            "id": item,
            "object": "realtime.item",
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        spoken = {"type": "output_audio", "transcript": transcript}
        done = {**message, "status": "completed", "content": [spoken]}
        for event in (
            {"type": "response.created", "response": {"id": response, "metadata": metadata}},
            {"type": "response.output_item.added", "item": message},
            {"type": "conversation.item.added", "previous_item_id": self._tail, "item": message},
            {
                "type": "response.content_part.added",
                "item_id": item,
                "part": {"type": "output_audio", "transcript": ""},
            },
            {
                "type": "response.output_audio_transcript.delta",
                "item_id": item,
                "delta": transcript,
            },
            {"type": "response.output_audio.delta", "item_id": item, "delta": audio},
            {"type": "response.output_item.done", "item": done},
            {"type": "response.done", "response": {"id": response, "status": "completed"}},
        ):
            await socket.send_json(event)
        self._tail = item


@asynccontextmanager
async def _voice(peer: _Peer, *, voice: Voice | None = None) -> AsyncIterator[VoiceSession]:
    app = web.Application()
    app.router.add_get("/v1/realtime", peer.handle)
    async with TestServer(app) as server:
        model = RealtimeModel(api_key="sk-local", base_url=str(server.make_url("/v1")), voice=voice)
        session = VoiceSession(llm=model, vad=None)
        session.output.audio = FakeAudioOutput()
        async with session:
            await session.start(Agent(instructions="Answer."))
            yield session


async def test_a_spoken_reply_is_recorded_with_its_transcript() -> None:
    """A reply the model speaks enters the history as the assistant's message, its text the
    audio transcript."""
    async with _voice(_Peer()) as session:
        await session.generate_reply(instructions="greet the user")

    said = [(i.role, i.text_content) for i in session.history.items if i.type == "message"]
    assert said == [("assistant", "Hello from the Realtime API.")]


async def test_the_configured_voice_is_sent_with_the_session() -> None:
    """The voice the model was built with reaches the API in the session configuration."""
    peer = _Peer()

    async with _voice(peer, voice="cedar") as session:
        await session.generate_reply(instructions="greet the user")

    audio = peer.sessions[0]["audio"]
    assert isinstance(audio, dict)
    assert audio["output"]["voice"] == "cedar"


async def test_the_api_key_given_to_the_model_authenticates_its_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model's own api_key, not OPENAI_API_KEY, authenticates the Realtime socket.

    https://github.com/livekit/agents/issues/1649
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-environment")
    peer = _Peer()

    async with _voice(peer) as session:
        await session.generate_reply(instructions="greet the user")

    assert peer.authorizations == ["Bearer sk-local"]


async def test_a_rejected_option_update_does_not_fail_the_reply_handle_or_the_next_turn() -> None:
    peer = _Peer()
    async with _voice(peer) as session:
        model = session.llm
        assert isinstance(model, RealtimeModel)
        errors: list[ErrorEvent] = []
        session.on("error", errors.append)
        first = session.generate_reply(instructions="first reply")
        await until(lambda: peer.sessions, "initial configuration sent")
        peer.reject_next_options = True
        model.update_options(voice="cedar")
        await until(lambda: errors, "the rejected update is reported")
        await first
        assert first.done() and first.exception() is None
        model.update_options(speed=1.1)
        second = session.generate_reply(instructions="next reply")
        await second
        assert second.done() and second.exception() is None
        assert first is not second
        assert len(first.chat_items) == len(second.chat_items) == 1
        assert first.chat_items[0].id != second.chat_items[0].id
        assert any(
            isinstance(e.error, llm.RealtimeModelError) and e.error.recoverable for e in errors
        )
