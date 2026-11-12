"""OpenAI Realtime: a VoiceSession against a local Realtime API peer speaking its literal JSON."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import Agent, VoiceSession
from livekit.plugins.openai.realtime import RealtimeModel, Voice
from tests_new.fakes import FakeAudioOutput


class _Peer:
    """The Realtime API's happy path: it keeps each `session.update`, acks every item the client
    creates and answers each `response.create` with one spoken message."""

    def __init__(self) -> None:
        self.sessions: list[dict[str, object]] = []
        self.authorizations: list[str | None] = []
        self._tail: str | None = None
        self._ids = 0

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
