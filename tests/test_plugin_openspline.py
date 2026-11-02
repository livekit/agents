"""Provider I/O is mocked; lifecycle uses the real LiveKit session/base class."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openspline import Openspline
from openspline.integrations import livekit as native

from livekit import rtc
from livekit.agents import AgentSession, Plugin
from livekit.agents.voice import io
from livekit.plugins import openspline as plugin

pytestmark = pytest.mark.unit


class Output(io.AudioOutput):
    def __init__(self, room=None, **kwargs):
        super().__init__(
            label="test", sample_rate=24000, capabilities=io.AudioOutputCapabilities(pause=False)
        )
        self.options = kwargs

    async def capture_frame(self, frame):
        await super().capture_frame(frame)

    def flush(self):
        pass

    def clear_buffer(self):
        pass


@pytest.fixture
async def harness(monkeypatch):
    for name in [
        "OPENSPLINE_API_KEY",
        "OPENSPLINE_URL",
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
    ]:
        monkeypatch.delenv(name, raising=False)
    worker = SimpleNamespace(attach_livekit=AsyncMock(), close=AsyncMock())
    source = SimpleNamespace(__aenter__=AsyncMock(return_value=worker))
    client = Openspline(api_key="cloud-key")
    monkeypatch.setattr(client, "avatar", lambda *args, **kwargs: source)
    room = rtc.Room()
    room._info.name = "plugin-test"
    room._local_participant = SimpleNamespace(identity="agent")
    monkeypatch.setattr(room, "isconnected", lambda: True)
    monkeypatch.setattr(native, "DataStreamAudioOutput", Output)
    avatar = plugin.AvatarSession(
        avatar_id="saved",
        client=client,
        livekit_url="wss://room.example",
        livekit_api_key="room-key",
        livekit_api_secret="room-secret",
    )
    joined = AsyncMock()
    monkeypatch.setattr(avatar, "wait_for_join", joined)
    session = AgentSession()
    old = Output()
    session.output.audio = old
    return SimpleNamespace(
        avatar=avatar,
        worker=worker,
        source=source,
        room=room,
        session=session,
        old=old,
        joined=joined,
    )


def test_registration_and_sources(monkeypatch):
    monkeypatch.delenv("OPENSPLINE_API_KEY", raising=False)
    assert any(p.package == "livekit.plugins.openspline" for p in Plugin.registered_plugins)
    with pytest.raises(ValueError, match="exactly one"):
        plugin.AvatarSession()
    with pytest.raises(ValueError, match="exactly one"):
        plugin.AvatarSession(portrait=b"image", avatar_id="saved")
    with pytest.raises(ValueError, match="Cloud API key"):
        plugin.AvatarSession(avatar_id="saved")
    with pytest.raises(ValueError, match="not both"):
        plugin.AvatarSession(portrait=b"image", client=Openspline(), api_key="other")
    avatar = plugin.AvatarSession(
        avatar_id="saved", api_key="cloud", quality="high", background=b"bg"
    )
    assert avatar._source.avatar_id == "saved"
    assert avatar._source.quality == "high"
    assert avatar._source.background == b"bg"
    assert avatar.client.url == "https://dashboard.openspline.dev/api"
    local = plugin.AvatarSession(portrait=b"image", url="http://localhost:9999")
    assert local.client.url == "http://localhost:9999"
    assert local._source.quality == "low"


@pytest.mark.asyncio
async def test_start_routes_once_and_restores_output(harness):
    h = harness
    await h.avatar.start(h.session, h.room)
    assert h.session.output.audio is not h.old
    assert h.session.output.audio.options["wait_playback_start"] is True
    h.joined.assert_awaited_once_with(timeout=30.0)
    call = h.worker.attach_livekit.call_args.kwargs
    assert call["url"] == "wss://room.example" and call["sender_identity"] == "agent"
    import jwt

    claims = jwt.decode(
        call["token"], "room-secret", algorithms=["HS256"], options={"verify_aud": False}
    )
    assert claims["video"]["room"] == "plugin-test"
    assert claims["attributes"]["lk.publish_on_behalf"] == "agent"
    assert claims["sub"] == h.avatar.avatar_identity
    with pytest.raises(RuntimeError, match="single-use"):
        await h.avatar.start(h.session, h.room)
    await asyncio.gather(h.avatar.aclose(), h.avatar.aclose())
    h.worker.close.assert_awaited_once()
    assert h.session.output.audio is h.old
    assert h.avatar._wait_avatar_join_task is None


@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["allocate", "attach", "join"])
async def test_failure_cleans_without_replacing_original(harness, step):
    h = harness
    failure = RuntimeError("provider failure")
    target = {"allocate": h.source.__aenter__, "attach": h.worker.attach_livekit, "join": h.joined}[
        step
    ]
    target.side_effect = failure
    h.worker.close.side_effect = RuntimeError("cleanup failure")
    with pytest.raises(RuntimeError, match="provider failure"):
        await h.avatar.start(h.session, h.room)
    assert h.session.output.audio is h.old
    assert h.avatar._wait_avatar_join_task is None
    if step != "allocate":
        h.worker.close.assert_awaited_once()
    await h.avatar.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_start", [True, False])
async def test_cancel_or_close_during_attach(harness, cancel_start):
    h = harness
    entered = asyncio.Event()

    async def pending(**kwargs):
        entered.set()
        await asyncio.Future()

    h.worker.attach_livekit.side_effect = pending
    start = asyncio.create_task(h.avatar.start(h.session, h.room))
    await entered.wait()
    if cancel_start:
        start.cancel()
    else:
        await asyncio.wait_for(h.avatar.aclose(), 2)
    with pytest.raises(asyncio.CancelledError):
        await start
    await h.avatar.aclose()
    h.worker.close.assert_awaited_once()
    assert h.avatar._wait_avatar_join_task is None


@pytest.mark.asyncio
async def test_validate_before_allocation(harness):
    h = harness
    h.avatar.key = None
    with pytest.raises(ValueError, match="LIVEKIT_API_KEY"):
        await h.avatar.start(h.session, h.room)
    h.source.__aenter__.assert_not_awaited()
    await h.avatar.aclose()


@pytest.mark.asyncio
async def test_does_not_overwrite_replacement_output(harness):
    h = harness
    await h.avatar.start(h.session, h.room)
    replacement = Output()
    h.session.output.audio = replacement
    await h.avatar.aclose()
    assert h.session.output.audio is replacement
