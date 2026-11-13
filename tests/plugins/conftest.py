"""Shared setup of the plugin wire tests."""

from __future__ import annotations

import asyncio
import base64
import datetime
import ipaddress
import json
import ssl
import sys
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import web
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from google.genai import types
from google.oauth2.credentials import Credentials

from livekit import rtc
from livekit.agents import VoiceSession, llm
from livekit.agents.voice.avatar import AvatarSession
from livekit.plugins.google.realtime import RealtimeModel
from livekit.plugins.google.realtime.realtime_api import RealtimeSession
from tests.fakes import until

# Any: decoded wire JSON, indexed by the provider's documented shape
Wire = dict[str, Any]


@dataclass
class AudioSink:
    """A DataStreamAudioOutput an avatar plugin built, with the options it was built with."""

    options: dict[str, object]
    aclose: AsyncMock = field(default_factory=AsyncMock)


@dataclass
class AvatarHost:
    """What an avatar plugin's ``start()`` is handed outside a job: see ``avatar_host``."""

    agent: VoiceSession
    room: rtc.Room
    audio_tail: Mock
    framework_start: AsyncMock
    credentials: dict[str, str]
    sinks: list[AudioSink]


class _Agent(rtc.EventEmitter[str]):
    """The VoiceSession an avatar attaches to: it emits events and takes the audio tail."""

    def __init__(self, audio_tail: Mock) -> None:
        super().__init__()
        self.output = SimpleNamespace(replace_audio_tail=audio_tail)


@pytest.fixture
def avatar_host(monkeypatch: pytest.MonkeyPatch) -> AvatarHost:
    """Lets an avatar plugin's ``start(host.agent, host.room, **host.credentials)`` run outside
    a job and room.

    ``host.agent`` emits like a VoiceSession and hands its audio tail to ``host.audio_tail``.
    The framework's half of ``start`` (``host.framework_start``) is stubbed, the job's agent
    identity is ``agent-1``, and every audio sink a plugin builds is kept in ``host.sinks``.
    """
    audio_tail = Mock()
    host = AvatarHost(
        # stand-ins for the session and room: an avatar only emits on and names them
        agent=cast(VoiceSession, _Agent(audio_tail)),
        room=cast(rtc.Room, SimpleNamespace(name="room-1", isconnected=lambda: False)),
        audio_tail=audio_tail,
        framework_start=AsyncMock(),
        credentials={
            "livekit_url": "wss://lk.test",
            "livekit_api_key": "lk-key",
            "livekit_api_secret": "lk-secret-with-enough-entropy-32b",
        },
        sinks=[],
    )
    job = SimpleNamespace(
        local_participant_identity="agent-1",
        token_claims=lambda: SimpleNamespace(identity="agent-1"),
        job=SimpleNamespace(id="AJ_1", room=SimpleNamespace(sid="RM_1")),
        add_shutdown_callback=Mock(),
    )

    def sink(**options: object) -> AudioSink:
        host.sinks.append(AudioSink(options))
        return host.sinks[-1]

    monkeypatch.setattr(AvatarSession, "start", host.framework_start)
    for name, module in list(sys.modules.items()):
        if name.startswith("livekit.plugins.") and hasattr(module, "get_job_context"):
            monkeypatch.setattr(module, "get_job_context", lambda *_, **__: job)
        if name.startswith("livekit.plugins.") and hasattr(module, "DataStreamAudioOutput"):
            monkeypatch.setattr(module, "DataStreamAudioOutput", sink)
    return host


class LivePeer:
    """Gemini Live on a local TLS socket: answers each setup, records what the plugin sends."""

    # 10 ms of 24 kHz mono PCM, base64 as Gemini sends it
    AUDIO_PART = {
        "inlineData": {"mimeType": "audio/pcm", "data": base64.b64encode(b"\0\1" * 240).decode()}
    }
    AUDIO = {"modelTurn": {"parts": [AUDIO_PART]}}

    def __init__(self, ssl_context: ssl.SSLContext) -> None:
        self.url = ""
        self.ssl_context = ssl_context
        self.setups: list[Wire] = []
        self.headers: list[dict[str, str]] = []
        self.received: list[list[Wire]] = []  # one list per connection
        self.sockets: list[web.WebSocketResponse] = []
        self.accepting = asyncio.Event()
        self.accepting.set()

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        self.headers.append(dict(request.headers))
        self.setups.append(json.loads((await socket.receive()).data)["setup"])
        received: list[Wire] = []
        self.received.append(received)
        await self.accepting.wait()
        await socket.send_json({"setupComplete": {}})
        self.sockets.append(socket)
        async for message in socket:
            received.append(json.loads(message.data))
        return socket

    @property
    def http_options(self) -> types.HttpOptions:
        return types.HttpOptions(base_url=self.url, async_client_args={"ssl": self.ssl_context})

    async def send(self, **message: object) -> None:
        await until(lambda: self.sockets, "the plugin connects")
        await self.sockets[-1].send_json(message)

    def sent(self, kind: str, connection: int = -1) -> list[Wire]:
        return [m[kind] for m in self.received[connection] if kind in m]

    async def wait_sent(self, kind: str, count: int = 1, connection: int = -1) -> list[Wire]:
        await until(lambda: self.received, "the plugin connects")
        await until(lambda: len(self.sent(kind, connection)) >= count, f"the plugin sends {kind}")
        return self.sent(kind, connection)

    async def ready(self, session: RealtimeSession, handle: str = "ready") -> None:
        """Hands out a resumption handle and waits until the session holds it: it is live."""
        await self.send(sessionResumptionUpdate={"newHandle": handle, "resumable": True})
        await until(lambda: session.session_resumption_handle == handle, "the session is live")

    async def drop(self) -> None:
        """Closes the current socket, as Gemini does when a connection ends."""
        connections = len(self.setups)
        await self.sockets[-1].close()
        await until(lambda: len(self.setups) > connections, "the plugin reconnects")

    @staticmethod
    async def drain(generation: llm.GenerationCreatedEvent) -> tuple[str, int, list[str]]:
        """A generation's spoken text, audio frame count and called tools."""
        text, frames = "", 0
        async for message in generation.message_stream:
            text += "".join(
                [c if isinstance(c, str) else c.text async for c in message.text_stream]
            )
            frames += len([frame async for frame in message.audio_stream])
        return text, frames, [call.name async for call in generation.function_stream]


@pytest.fixture(scope="module")
def gemini_tls(tmp_path_factory: pytest.TempPathFactory) -> tuple[ssl.SSLContext, ssl.SSLContext]:
    """A self-signed 127.0.0.1 certificate: the SDK only speaks wss to its base_url."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    folder = tmp_path_factory.mktemp("tls")
    (cert_file := folder / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (key_file := folder / "key.pem").write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    server = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    server.load_cert_chain(cert_file, key_file)
    return server, ssl.create_default_context(cafile=cert_file)


@pytest.fixture
async def gemini_live(gemini_tls: tuple[ssl.SSLContext, ssl.SSLContext]) -> AsyncIterator[LivePeer]:
    """A local Gemini Live peer (see LivePeer)."""
    server_ctx, client_ctx = gemini_tls
    live = LivePeer(client_ctx)
    app = web.Application()
    app.router.add_get("/{path:.*}", live.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0, ssl_context=server_ctx).start()
    live.url = f"https://127.0.0.1:{runner.addresses[0][1]}"
    try:
        yield live
    finally:
        live.accepting.set()
        await runner.cleanup()
        # the SDK's AsyncClient.__del__ schedules aclose() even on a closed client; settle it
        await asyncio.sleep(0)


@pytest.fixture
async def open_session(gemini_live: LivePeer) -> AsyncIterator[Callable[..., RealtimeSession]]:
    """Opens Gemini (or, with vertexai=True, Vertex AI) sessions on the peer; closed after."""
    sessions: list[RealtimeSession] = []

    def open_(**options: Any) -> RealtimeSession:  # Any: RealtimeModel's keyword options
        if options.get("vertexai"):
            # google-auth leaves Credentials.__init__ untyped
            credentials = Credentials(token="t")  # type: ignore[no-untyped-call]
            options = {"project": "test", "credentials": credentials, **options}
        model = RealtimeModel(api_key="test", http_options=gemini_live.http_options, **options)
        sessions.append(model.session())
        return sessions[-1]

    yield open_
    for session in sessions:
        await session.aclose()
