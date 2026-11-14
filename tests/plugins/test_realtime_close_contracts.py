"""Closing a realtime session and its model ends every wait the session holds: a pending
``generate_reply`` resolves, the provider socket closes and no task outlives the close (the
suite's leaked-task check fails a test that leaks one). Run over every realtime plugin against a
local peer of its provider; a plugin whose transport the peer cannot serve is skipped with what is
missing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from aiohttp import web

from livekit.agents import APIConnectOptions, _core, llm
from livekit.plugins import phonic, ultravox
from livekit.plugins.nvidia.experimental import realtime as personaplex
from tests.fakes import until

if TYPE_CHECKING:
    from tests.plugins.conftest import LivePeer, WsPeer


async def _ultravox_call(peer: WsPeer) -> web.Response:
    return web.json_response({"joinUrl": peer.ws_url + "/join"})


def _ultravox(peer: WsPeer) -> llm.RealtimeSessionModel:
    peer.http = lambda _: _ultravox_call(peer)
    return ultravox.realtime.RealtimeModel(api_key="key", base_url=peer.url + "/api")


MODELS: dict[str, Callable[[WsPeer], llm.RealtimeSessionModel]] = {
    "ultravox": _ultravox,
    "phonic": lambda peer: phonic.realtime.RealtimeModel(api_key="key", base_url=peer.ws_url),
    "nvidia": lambda peer: personaplex.RealtimeModel(base_url=peer.ws_url),
}


async def _open(model: llm.RealtimeSessionModel) -> llm.RealtimeSession:
    """A session as a VoiceSession opens one: its instructions and tools set first."""
    session = model.session()
    await session.update_instructions("Be brief.")
    await session.update_tools([])
    return session


async def _close_while_waiting(session: llm.RealtimeSession, model: llm.RealtimeModel) -> None:
    try:
        reply: asyncio.Future[llm.GenerationCreatedEvent] | None = session.generate_reply(
            instructions="Say hello."
        )
    except NotImplementedError:  # a speech-to-speech model that only answers audio
        reply = None
    await session.aclose()
    await model.aclose()
    if reply is not None:
        await until(reply.done, "the pending reply ends with the session")
        if not reply.cancelled():
            reply.exception()  # read: a closed session's reply fails


# Catches: a receive loop, keepalive or pending reply that outlives the session's close, leaking
# its task or leaving a caller awaiting generate_reply forever.
@pytest.mark.parametrize(
    "name",
    [
        *MODELS,
        *(
            pytest.param(name, marks=pytest.mark.skip(reason=f"{name}'s {what}"))
            for name, what in [
                ("xai", "sessions run in the core's OpenAI protocol client, not the plugin"),
                ("azure", "sessions run in the core's OpenAI protocol client, not the plugin"),
                ("openai", "GPT Live session needs a VoiceSession to open (GPTVoiceSession)"),
                ("aws", "Nova Sonic connects through the AWS SDK's bidirectional stream"),
            ]
        ),
    ],
)
async def test_closing_a_realtime_session_ends_every_pending_wait(
    name: str, ws_peer: WsPeer
) -> None:
    model = MODELS[name](ws_peer)
    session = await _open(model)
    connection = await ws_peer.connected()

    await _close_while_waiting(session, model)

    await until(lambda: connection.socket.closed, "the session closes its socket")


async def test_closing_a_gemini_live_session_ends_every_pending_wait(
    open_session: Callable[..., llm.RealtimeSession], gemini_live: LivePeer
) -> None:
    session = open_session()
    await until(lambda: gemini_live.sockets, "the session connects")

    await _close_while_waiting(session, session.realtime_model)

    await until(lambda: gemini_live.sockets[0].closed, "the session closes its socket")


NEVER_READY: dict[str, Callable[[WsPeer, APIConnectOptions], llm.RealtimeSessionModel]] = {
    "phonic": lambda peer, options: phonic.realtime.RealtimeModel(
        api_key="key", base_url=peer.ws_url, conn_options=options
    ),
    "nvidia": lambda peer, options: personaplex.RealtimeModel(
        base_url=peer.ws_url, conn_options=options
    ),
}


# Catches: a provider that accepts the socket but never says it is ready (Phonic's
# ready_to_start_conversation, PersonaPlex's handshake) leaving the session connected and silent
# forever, its replies and audio waiting on nothing.
@pytest.mark.parametrize("name", NEVER_READY)
async def test_a_provider_that_never_becomes_ready_fails_the_session_with_a_timeout(
    name: str, ws_peer: WsPeer
) -> None:
    options = APIConnectOptions(max_retry=0, connect_timeout=5)
    model = NEVER_READY[name](ws_peer, options)
    session = await _open(model)
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    try:
        await ws_peer.connected()
        _core.skip_host_time(6)  # past the connect limit
        await until(lambda: errors, "the session fails")
        [error] = errors
        assert not error.recoverable
        assert getattr(error.error, "kind", None) == "timeout"
    finally:
        await session.aclose()
        await model.aclose()
