"""bitHuman avatar plugin: cloud session requests through a local peer, and local runtimes."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import bithuman
import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from numpy.typing import NDArray

from livekit import api, rtc
from livekit.agents import (
    NOT_GIVEN,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    VoiceSession,
    http,
)
from livekit.agents.voice.avatar import QueueAudioOutput
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF
from livekit.plugins.bithuman import AvatarSession
from livekit.plugins.bithuman.avatar import RenderedFrame, RuntimeSettings
from tests.fakes import until

if TYPE_CHECKING:
    from tests.plugins.conftest import AvatarHost

_OPTS = APIConnectOptions(max_retry=2, retry_interval=0, connect_timeout=1, read_timeout=1)
HOSTED = "https://api.bithuman.ai/avatar"


class _Peer:
    def __init__(self, url: str) -> None:
        self.url = url
        self.statuses: list[int] = [200]
        self.requests: list[tuple[str, dict[str, str]]] = []  # content type, fields and headers


@pytest.fixture
async def peer(avatar_host: AvatarHost, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[_Peer]:
    """A local bitHuman API, also reached at the hosted URL through a local resolver."""

    async def create(request: web.Request) -> web.Response:
        content_type = request.content_type
        if content_type != "application/json":
            form = await request.post()
            fields = {k: v for k, v in form.items() if isinstance(v, str)}
        else:
            fields = {k: str(v) for k, v in (await request.json()).items() if v is not None}
        fields["authorization"] = request.headers.get("Authorization", "")
        fields["api-secret"] = request.headers.get("api-secret", "")
        state.requests.append((content_type, fields))
        status = state.statuses.pop(0) if len(state.statuses) > 1 else state.statuses[0]
        return web.json_response({"detail": f"provider says {status}"}, status=status)

    app = web.Application()
    app.router.add_post("/{path:.*}", create)
    async with TestServer(app) as server, http.open() as session:
        state = _Peer(str(server.make_url("/avatar")))
        post = session.post

        # Any: post()'s keyword options and request context, forwarded untouched
        def local_resolver(url: str, **kwargs: Any) -> Any:
            return post(url.replace(HOSTED, state.url, 1), **kwargs)

        monkeypatch.setattr(session, "post", local_resolver)
        yield state


async def _start(avatar: AvatarSession, avatar_host: AvatarHost) -> None:
    await avatar.start(avatar_host.agent, avatar_host.room, **avatar_host.credentials)


@pytest.mark.parametrize("model", ["expression-2", None])
async def test_cloud_session_names_its_agent_identity_and_model(
    peer: _Peer, avatar_host: AvatarHost, model: Literal["expression-2"] | None
) -> None:
    """The avatar publishes for the job's agent and renders the model asked for, if any.

    https://github.com/livekit/agents/pull/4029 https://github.com/livekit/agents/pull/7366
    https://github.com/livekit/agents/pull/3313
    """
    avatar = AvatarSession(
        api_url=HOSTED,
        api_secret="secret",
        avatar_id="A1",
        model=model or NOT_GIVEN,
        conn_options=_OPTS,
    )

    await _start(avatar, avatar_host)

    [(content_type, sent)] = peer.requests
    assert content_type == "application/json"
    assert sent["api-secret"] == "secret"
    assert sent["room_name"] == "room-1"
    assert sent["agent_id"] == "A1"
    assert sent.get("model") == model
    claims = api.TokenVerifier(
        avatar_host.credentials["livekit_api_key"], avatar_host.credentials["livekit_api_secret"]
    ).verify(sent["livekit_token"])
    assert claims.attributes is not None
    assert claims.attributes[ATTRIBUTE_PUBLISH_ON_BEHALF] == "agent-1"
    assert claims.attributes["agent_id"] == "A1"


async def test_expression_model_on_a_custom_endpoint_posts_a_form(
    peer: _Peer, avatar_host: AvatarHost
) -> None:
    """A self-hosted expression worker receives the session as form fields with a bearer token.

    https://github.com/livekit/agents/pull/4799
    """
    avatar = AvatarSession(
        api_url=peer.url + "?async=true",
        api_token="worker-token",
        api_secret="secret",
        avatar_id="A1",
        model="expression",
        conn_options=_OPTS,
    )

    await _start(avatar, avatar_host)

    [(content_type, sent)] = peer.requests
    assert content_type in ("multipart/form-data", "application/x-www-form-urlencoded")
    assert sent["authorization"] == "Bearer worker-token"
    assert sent["model"] == "expression"
    assert sent["avatar_id"] == "A1"
    assert sent["async_mode"] == "true"
    assert sent["room_name"] == "room-1"
    assert (
        api.TokenVerifier(
            avatar_host.credentials["livekit_api_key"],
            avatar_host.credentials["livekit_api_secret"],
        )
        .verify(sent["livekit_token"])
        .attributes
    )


@pytest.mark.parametrize(
    ("status", "attempts", "raised_type"),
    [(401, 1, APIStatusError), (503, _OPTS.max_retry + 1, APIStatusError)],
)
async def test_provider_status_survives_and_only_transient_errors_retry(
    peer: _Peer, avatar_host: AvatarHost, status: int, attempts: int, raised_type: type[APIError]
) -> None:
    """A refused session keeps the provider's status and body; only 5xx is retried, and
    exhausted retries raise from the last answer.

    https://github.com/livekit/agents/issues/7499
    """
    peer.statuses[:] = [status]
    avatar = AvatarSession(
        api_url=peer.url, api_secret="secret", avatar_id="A1", conn_options=_OPTS
    )
    with pytest.raises(raised_type) as error:
        await _start(avatar, avatar_host)

    refused = error.value if isinstance(error.value, APIStatusError) else error.value.__cause__
    assert isinstance(refused, APIStatusError)
    assert refused.status_code == status
    assert f"provider says {status}" in str(refused.body)
    assert len(peer.requests) == attempts


@dataclass
class _Settings:
    FPS: int = 25
    INPUT_SAMPLE_RATE: int = 16000


class _Runtime:
    """A local bitHuman SDK runtime: keeps the audio it is given, its stops and its cleanup."""

    def __init__(self) -> None:
        self.settings: RuntimeSettings = _Settings()
        self.audio: list[bytes] = []
        self.stops = 0
        self.cleaned_up = False
        self._frames: asyncio.Queue[RenderedFrame] = asyncio.Queue()

    def get_first_frame(self) -> NDArray[np.uint8]:
        return np.zeros((2, 2, 3), dtype=np.uint8)

    async def push_audio(self, data: bytes, sample_rate: int, *, last_chunk: bool) -> None:
        self.audio.append(data)

    async def flush(self) -> None:
        pass

    def interrupt(self) -> None:
        pass

    async def run(self) -> AsyncIterator[RenderedFrame]:
        while True:
            yield await self._frames.get()

    async def stop(self) -> None:
        self.stops += 1

    def cleanup(self) -> None:
        self.cleaned_up = True


@pytest.fixture
def created(monkeypatch: pytest.MonkeyPatch) -> list[_Runtime]:
    """The runtimes the bitHuman SDK creates for the plugin, in order."""
    runtimes: list[_Runtime] = []

    async def create(**_: object) -> _Runtime:
        runtimes.append(_Runtime())
        return runtimes[-1]

    monkeypatch.setattr(bithuman.AsyncBithuman, "create", create)
    return runtimes


async def _start_local(avatar: AvatarSession) -> QueueAudioOutput:
    """Starts a local avatar for a voice session and returns the session's audio sink."""
    voice = VoiceSession(vad=None)
    await avatar.start(voice, rtc.Room())
    audio = voice.output.audio
    assert isinstance(audio, QueueAudioOutput)
    return audio


async def test_borrowed_runtime_is_not_destroyed_or_reauthenticated(
    avatar_host: AvatarHost, created: list[_Runtime]
) -> None:
    """A caller's runtime is stopped after each session, never cleaned up or replaced."""
    runtime = _Runtime()

    for _ in range(2):
        avatar = AvatarSession(runtime=runtime)
        audio = await _start_local(avatar)
        await audio.capture_frame(rtc.AudioFrame(b"\0\0" * 160, 16000, 1, 160))
        await until(lambda: runtime.audio, "the runtime renders the speech")
        await avatar.aclose()
        runtime.audio.clear()

    assert runtime.stops == 2
    assert not runtime.cleaned_up
    assert created == []


async def test_owned_runtime_is_recreated_after_close(
    avatar_host: AvatarHost, created: list[_Runtime]
) -> None:
    """A runtime the plugin creates is stopped and cleaned up on close; a restart creates another."""
    avatar = AvatarSession(model_path="avatar.imx", api_secret="secret")
    running: list[object] = []

    for _ in range(2):
        await _start_local(avatar)
        running.append(avatar.runtime)
        await avatar.aclose()

    assert running == created
    assert created[0] is not created[1]
    assert [(r.stops, r.cleaned_up) for r in created] == [(1, True), (1, True)]
