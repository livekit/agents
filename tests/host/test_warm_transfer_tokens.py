"""The token that a warm transfer uses to join the human agent's room.

In key mode, the transfer signs it with the server's key and secret. In token mode, the server
creates it for the job.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import jwt
import pytest

from livekit import api
from livekit.agents import _core
from livekit.agents.beta import workflows

KEY, SECRET = "server-key", "server-secret-server-secret-server"


class _Job:
    def __init__(self, token_mode: bool) -> None:
        self.token_mode = token_mode
        self.asked: list[dict[str, Any]] = []
        self._info = SimpleNamespace(url="wss://project.livekit.cloud")

    def _token_mode(self) -> bool:
        return self.token_mode

    async def create_participant_token(self, **claims: Any) -> str:
        self.asked.append(claims)
        return "server-made-token"


class _Room:
    def __init__(self) -> None:
        self.joined: list[tuple[str, str]] = []

    async def connect(self, url: str, token: str) -> None:
        self.joined.append((url, token))


async def _connect_human(job: _Job, monkeypatch: pytest.MonkeyPatch) -> _Room:
    monkeypatch.setattr(_core, "_core_create_room_session", lambda room: room)
    host = workflows._TransferHost.__new__(workflows._TransferHost)
    room = _Room()
    host._job, host._room = job, room  # type: ignore[assignment]
    await host._core_connect_human("human-room", "human-agent")
    return room


# Connects the human agent in token mode and in key mode. Catches a token mode transfer that
# signs the token with no key or asks the server for the agent grant. Also catches a key mode
# transfer that asks the server for the token.
async def test_the_human_agents_room_token_comes_from_the_server_in_token_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _Job(token_mode=True)
    room = await _connect_human(job, monkeypatch)
    assert room.joined == [("wss://project.livekit.cloud", "server-made-token")]
    [claims] = job.asked
    assert (claims["identity"], claims["kind"]) == ("human-agent", "agent")
    grants: api.VideoGrants = claims["grants"]
    assert (grants.room_join, grants.room) == (True, "human-room")
    assert not grants.agent, "a job token never carries the agent grant"

    monkeypatch.setenv("LIVEKIT_API_KEY", KEY)
    monkeypatch.setenv("LIVEKIT_API_SECRET", SECRET)
    keyed = _Job(token_mode=False)
    room = await _connect_human(keyed, monkeypatch)
    [(_, token)] = room.joined
    signed = jwt.decode(token, SECRET, algorithms=["HS256"], issuer=KEY)
    assert (signed["sub"], signed["video"]["room"], signed["video"]["agent"]) == (
        "human-agent",
        "human-room",
        True,
    )
    assert keyed.asked == []
