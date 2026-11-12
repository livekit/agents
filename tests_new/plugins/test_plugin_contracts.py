"""Contracts shared across plugins: avatar session creation and TTS sample-rate sync.

Each avatar starts through ``AvatarSession.start`` against a local provider.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectionError, APIConnectOptions, http
from livekit.agents.voice.avatar import AvatarSession
from livekit.plugins import (
    anam,
    avatario,
    avatartalk,
    bey,
    bithuman,
    did,
    keyframe,
    lemonslice,
    liveavatar,
    runway,
)

if TYPE_CHECKING:
    from tests_new.plugins.conftest import AvatarHost

AVATARS: dict[str, Callable[[str, APIConnectOptions], AvatarSession]] = {
    "anam": lambda url, options: anam.AvatarSession(
        persona_config=anam.PersonaConfig(name="avatar", avatarId="avatar"),
        api_key="key",
        api_url=url,
        conn_options=options,
    ),
    "avatario": lambda url, options: avatario.AvatarSession(
        avatar_id="avatar", api_key="key", conn_options=options
    ),
    "avatartalk": lambda url, options: avatartalk.AvatarSession(api_secret="key", api_url=url),
    "bey": lambda url, options: bey.AvatarSession(api_key="key", api_url=url, conn_options=options),
    "bithuman": lambda url, options: bithuman.AvatarSession(
        api_secret="secret", avatar_id="avatar", api_url=url, conn_options=options
    ),
    "did": lambda url, options: did.AvatarSession(
        agent_id="avatar", api_key="key", api_url=url, conn_options=options
    ),
    "keyframe": lambda url, options: keyframe.AvatarSession(
        persona_id="avatar", api_key="key", api_url=url, conn_options=options
    ),
    "lemonslice": lambda url, options: lemonslice.AvatarSession(
        agent_id="avatar", api_key="key", api_url=url, conn_options=options
    ),
    "liveavatar": lambda url, options: liveavatar.AvatarSession(
        avatar_id="avatar", api_key="key", api_url=url, conn_options=options
    ),
    "runway": lambda url, options: runway.AvatarSession(
        avatar_id="avatar", api_key="key", api_url=url, conn_options=options
    ),
}


async def _start(
    provider: str,
    reply: object,
    options: APIConnectOptions,
    requests: list[object],
    avatar_host: AvatarHost,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Starts ``provider``'s avatar against a local provider that answers every POST with
    ``reply``, keeping each request body in ``requests``."""

    async def create(request: web.Request) -> web.Response:
        requests.append(await request.json())
        return web.json_response(reply)

    app = web.Application()
    app.router.add_post("/{path:.*}", create)
    async with TestServer(app) as server, http.open():
        url = str(server.make_url("")).rstrip("/")
        monkeypatch.setattr("livekit.plugins.avatario.api.DEFAULT_API_URL", url)
        avatar = AVATARS[provider](url, options)
        try:
            await avatar.start(avatar_host.agent, avatar_host.room, **avatar_host.credentials)
        finally:
            await avatar.aclose()


@pytest.mark.parametrize(
    "provider", ["anam", "avatario", "avatartalk", "did", "keyframe", "lemonslice", "liveavatar"]
)
async def test_invalid_success_response_does_not_repeat_creation(
    provider: str, avatar_host: AvatarHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful avatar POST with an invalid body fails non-retryably and is never repeated."""
    requests: list[object] = []
    options = APIConnectOptions(max_retry=2, retry_interval=0)

    with pytest.raises(APIConnectionError) as error:
        await _start(provider, [], options, requests, avatar_host, monkeypatch)

    assert error.value.retryable is False
    assert len(requests) == 1


@pytest.mark.parametrize("provider", ["avatario", "bey", "bithuman", "runway"])
async def test_zero_retries_still_creates_avatar(
    provider: str, avatar_host: AvatarHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With max_retry=0 the avatar is still created, once. https://github.com/livekit/agents/issues/7604"""
    reply = {"id": "session"}
    requests: list[object] = []

    await _start(
        provider, reply, APIConnectOptions(max_retry=0), requests, avatar_host, monkeypatch
    )

    assert len(requests) == 1


@pytest.mark.parametrize("plugin", ["inworld", "lmnt", "smallestai"])
def test_update_options_syncs_public_sample_rate(plugin: str) -> None:
    """``TTS.sample_rate``, which the fallback adapter resamples from, follows the option.

    https://github.com/livekit/agents/pull/7135
    """
    tts = __import__(f"livekit.plugins.{plugin}", fromlist=["TTS"]).TTS(api_key="key")
    assert tts.sample_rate != 16000

    tts.update_options(sample_rate=16000)

    assert tts.sample_rate == 16000
