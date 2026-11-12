"""Meta STT plugin: realtime handshake, retry boundary and secret redaction, against a local peer.

https://github.com/livekit/agents/pull/7097
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, http
from livekit.plugins import meta

ONE_TRY = APIConnectOptions(max_retry=0, retry_interval=0.0, connect_timeout=1.0, read_timeout=1.0)
AUDIO = rtc.AudioFrame(b"x" * 3840, sample_rate=24_000, num_channels=1, samples_per_channel=1920)


Step = dict[str, object] | str | tuple[str, int, str]
"""A scripted reply: JSON, raw text, ``("close", code, reason)`` or ``"drop"``."""


@dataclass
class Muse:
    """The local Muse endpoint: its script and what it received."""

    url: str = ""
    script: list[Step] = field(default_factory=list)
    refuse: str | None = None
    received: list[tuple[str, object]] = field(default_factory=list)
    requests: list[web.Request] = field(default_factory=list)


@pytest.fixture
async def muse(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Muse]:
    """A local Muse ASR endpoint behind a TLS terminator.

    It acknowledges the handshake with a session id, then plays ``script``: a dict or str is
    sent as text, ``("close", code, reason)`` closes, ``"drop"`` drops the connection after the
    first audio. ``refuse`` answers the upgrade with HTTP 500 and that body. ``received`` lists
    ``("text", payload)`` and ``("audio", acknowledged)`` in arrival order.
    """
    peer = Muse()

    async def realtime(request: web.Request) -> web.StreamResponse:
        peer.requests.append(request)
        if peer.refuse is not None:
            return web.Response(status=500, text=peer.refuse)
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        peer.received.append(("text", json.loads((await socket.receive()).data)))
        await socket.send_json({"sessionId": "session-1"})
        for step in peer.script:
            if isinstance(step, tuple):
                await socket.close(code=step[1], message=step[2].encode())
            elif step != "drop":
                await socket.send_str(step if isinstance(step, str) else json.dumps(step))
        async for message in socket:
            if message.type == web.WSMsgType.BINARY:
                peer.received.append(("audio", True))
                if "drop" in peer.script:
                    assert request.transport is not None
                    request.transport.close()
            elif json.loads(message.data) == {"type": "endStream"}:
                await socket.close()
        return socket

    app = web.Application()
    app.router.add_get("/v1/asr/realtime", realtime)
    async with TestServer(app) as server, http.open() as session:
        connect = session.ws_connect

        # Any: ws_connect's keyword options, forwarded untouched
        async def local_tls_terminator(url: str, **kwargs: Any) -> http.WebSocket:
            return await connect(url.replace("wss://", "ws://", 1), **kwargs)

        monkeypatch.setattr(session, "ws_connect", local_tls_terminator)
        peer.url = str(server.make_url("/v1/asr/realtime").with_scheme("wss"))
        yield peer


async def test_handshake_precedes_audio_and_matches_contract(muse: Muse) -> None:
    """The handshake carries the key and options; audio follows only its acknowledgement."""
    provider = meta.STT(
        api_key="explicit-secret",
        keywords=["Muse", "Muse"],
        language_bias=["English", "French"],
        url=muse.url,
    )

    async with provider.stream(language="fr-FR", conn_options=ONE_TRY) as stream:
        stream.push_frame(AUDIO)
        stream.end_input()
        events = [event async for event in stream]
    await provider.aclose()

    [handshake, *rest] = muse.received
    assert handshake == (
        "text",
        {
            "mode": "ENDPOINTING",
            "authorization": {"accessToken": "Bearer explicit-secret"},
            "audioEncoding": "PCM_24KHZ",
            "model": "muse-voice-transcribe-1.0",
            "partialMode": "CUMULATIVE",
            "emitAudioProgress": True,
            "keywords": ["Muse"],
            "languageBias": ["English", "French"],
        },
    )
    assert ("audio", True) in rest
    [request] = muse.requests
    assert "explicit-secret" not in str(request.url)
    assert "explicit-secret" not in repr(dict(request.headers))
    assert all(event.recognition_usage is None for event in events)


async def test_post_consumption_failure_is_not_retried(muse: Muse) -> None:
    """A connection lost after audio was sent fails the stream instead of reconnecting."""
    muse.script = ["drop"]
    provider = meta.STT(api_key="key", url=muse.url)
    options = APIConnectOptions(max_retry=1, retry_interval=0.0, read_timeout=1.0)

    with pytest.raises(APIError) as caught:
        async with provider.stream(conn_options=options) as stream:
            stream.push_frame(AUDIO)
            stream.end_input()
            async for _ in stream:
                pass
    await provider.aclose()

    assert caught.value.retryable is False
    assert len(muse.requests) == 1


@pytest.mark.parametrize(
    ("secret", "refuse", "script"),
    [
        ("connect-secret", "connect-secret", []),
        (
            "provider-secret",
            None,
            [{"type": "error", "errorCode": "invalid_request", "message": "provider-secret"}],
        ),
        ("close-secret", None, [("close", 1011, "close-secret")]),
        ("payload-secret", None, ["{payload-secret"]),
    ],
)
async def test_failures_redact_transport_and_provider_details(
    muse: Muse,
    secret: str,
    refuse: str | None,
    script: list[Step],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No failure carries the provider's text, the transport's details or the API key."""
    muse.refuse, muse.script = refuse, script
    provider = meta.STT(api_key="api-key-secret", url=muse.url)

    with pytest.raises(APIError) as caught:
        async with provider.stream(conn_options=ONE_TRY) as stream:
            async for _ in stream:
                pass
    await provider.aclose()

    rendered = f"{caught.value!s}\n{caught.value!r}\n{caught.value.__cause__!r}\n{caplog.text}"
    assert secret not in rendered
    assert "api-key-secret" not in rendered
    assert caught.value.body is None
