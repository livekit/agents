"""Reusable helpers for end-to-end LiveKit room tests.

These are intentionally framework-light — they work on top of `livekit-rtc`
plus the `livekit_server` pytest fixture (see `tests/lk_server.py`) and are
meant to be composed by individual test files.

Examples
--------
A typical E2E test wires a "user" room and an "agent" room against the same
test server, then exercises whatever behavior is being verified::

    async with connect_room("user", room_name) as user, \
              connect_room("agent", room_name, agent=True) as agent:
        await wait_for_participant(user, identity="agent")
        ...
        await simulate_full_reconnect(agent)
        ...
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from livekit import api, rtc

from ..lk_server import LK_API_KEY, LK_API_SECRET, LK_URL

__all__ = [
    "ServerConfig",
    "lk_config",
    "make_token",
    "make_room_name",
    "connect_room",
    "wait_for_event",
    "simulate_resume",
    "simulate_full_reconnect",
]


@dataclass(frozen=True)
class ServerConfig:
    """LiveKit server credentials used by E2E test helpers."""

    url: str
    api_key: str
    api_secret: str


def lk_config() -> ServerConfig | None:
    """Resolve LiveKit server credentials from the environment.

    Looks at `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET`.
    Returns `None` when `LIVEKIT_URL` is unset — callers should treat this
    as "skip the test"; LiveKit Cloud / dev-server creds are environment-
    specific and we never want to assume them for unit-test runs.

    The `lk_server` pytest fixture (used by `tests/test_room.py`) is a
    separate path: it bootstraps a local server and uses fixed dev creds;
    callers that want to be compatible with both can fall through to
    those defaults if `LIVEKIT_URL` happens to be unset *and* the fixture
    is in use.
    """
    url = os.environ.get("LIVEKIT_URL")
    if not url:
        return None
    return ServerConfig(
        url=url,
        api_key=os.environ.get("LIVEKIT_API_KEY", LK_API_KEY),
        api_secret=os.environ.get("LIVEKIT_API_SECRET", LK_API_SECRET),
    )


def _resolve_config(config: ServerConfig | None) -> ServerConfig:
    if config is not None:
        return config
    cfg = lk_config()
    if cfg is not None:
        return cfg
    # Fall back to the lk_server fixture defaults — only meaningful when
    # the caller has the fixture active.
    return ServerConfig(url=LK_URL, api_key=LK_API_KEY, api_secret=LK_API_SECRET)


def make_token(
    identity: str,
    room: str,
    *,
    agent: bool = False,
    can_publish: bool = True,
    can_subscribe: bool = True,
    config: ServerConfig | None = None,
) -> str:
    """Mint a JWT against `config` (defaults to env vars / lk_server creds)."""
    cfg = _resolve_config(config)
    token = (
        api.AccessToken(cfg.api_key, cfg.api_secret)
        .with_identity(identity)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=can_publish,
                can_subscribe=can_subscribe,
                agent=agent,
            )
        )
    )
    if agent:
        token = token.with_kind("agent")
    return token.to_jwt()


def make_room_name(prefix: str = "test") -> str:
    """Generate a unique room name so concurrent test runs don't collide."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


@asynccontextmanager
async def connect_room(
    identity: str,
    room_name: str,
    *,
    agent: bool = False,
    options: rtc.RoomOptions | None = None,
    config: ServerConfig | None = None,
) -> AsyncIterator[rtc.Room]:
    """Connect a `rtc.Room` and disconnect on exit.

    `agent=True` mints a token with the `agent` grant + `kind=agent` claim,
    matching what the agents framework would send in production. Pass
    `config` to target a non-default server; otherwise environment vars
    (`LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET`) are used,
    falling back to the `lk_server` fixture defaults.
    """
    cfg = _resolve_config(config)
    room = rtc.Room()
    token = make_token(identity, room_name, agent=agent, config=cfg)
    await room.connect(cfg.url, token, options or rtc.RoomOptions())
    try:
        yield room
    finally:
        await room.disconnect()


async def wait_for_event(
    emitter: rtc.Room,
    event: str,
    *,
    timeout: float = 10.0,
    predicate: Callable[..., bool] | None = None,
) -> tuple:
    """Wait for an `EventEmitter` event to fire.

    Returns the event payload as a tuple (since `EventEmitter` callbacks may
    receive multiple positional args). Pass `predicate` to filter — only the
    first matching event resolves the future.
    """
    fut: asyncio.Future[tuple] = asyncio.get_event_loop().create_future()

    def _handler(*args) -> None:
        if predicate is not None and not predicate(*args):
            return
        if not fut.done():
            fut.set_result(args)

    emitter.on(event, _handler)
    try:
        return await asyncio.wait_for(fut, timeout=timeout)
    finally:
        emitter.off(event, _handler)


async def simulate_resume(room: rtc.Room) -> None:
    """Trigger a signal-only reconnect (Resume).

    The PeerConnection is preserved, existing publications stay valid, the
    SDK does NOT fire `reconnected`. Apps observe recovery via
    `connection_state_changed`.
    """
    await room.simulate_scenario(rtc.SimulateScenarioKind.SIMULATE_SIGNAL_RECONNECT)


async def simulate_full_reconnect(room: rtc.Room) -> None:
    """Trigger a full reconnect.

    The server issues `LeaveRequest{Reconnect}`; the SDK rebuilds the
    RtcSession, re-publishes existing local tracks (preserving the
    underlying `Track` handles but with new publication SIDs), and fires
    `reconnected`.
    """
    await room.simulate_scenario(rtc.SimulateScenarioKind.SIMULATE_FULL_RECONNECT)
