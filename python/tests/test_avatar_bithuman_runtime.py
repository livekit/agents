"""A supplied BitHuman runtime remains owned and prepared by its caller."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from livekit import rtc
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession
from livekit.plugins.bithuman import AvatarSession

pytestmark = pytest.mark.unit


class Runtime:
    settings = SimpleNamespace(FPS=25, INPUT_SAMPLE_RATE=16000)

    def __init__(self):
        self.cleaned = False
        self.stop_calls = 0
        self.audio = []

    def get_first_frame(self):
        return np.zeros((2, 2, 3), dtype=np.uint8)

    async def push_audio(self, data, sample_rate, *, last_chunk):
        if self.cleaned:
            raise RuntimeError("caller runtime was destroyed")
        self.audio.append(data)

    async def flush(self):
        pass

    def interrupt(self):
        pass

    async def run(self):
        return
        yield

    async def stop(self):
        self.stop_calls += 1

    def cleanup(self):
        self.cleaned = True


class Runner:
    def __init__(self, room, *, video_gen, audio_recv, options):
        self.generator = video_gen
        self.receiver = audio_recv

    async def start(self):
        await self.generator.push_audio(rtc.AudioFrame(b"\0\0" * 160, 16000, 1, 160))

    async def aclose(self):
        await self.receiver.aclose()


@pytest.fixture(autouse=True)
def local_avatar(monkeypatch):
    monkeypatch.setattr(BaseAvatarSession, "start", AsyncMock())
    monkeypatch.setattr(BaseAvatarSession, "aclose", AsyncMock())
    monkeypatch.setattr("livekit.plugins.bithuman.avatar.AvatarRunner", Runner)


async def test_borrowed_runtime_is_not_destroyed_or_reauthenticated():
    runtime = Runtime()
    agent = SimpleNamespace(output=SimpleNamespace(replace_audio_tail=Mock()))

    for _ in range(2):
        avatar = AvatarSession(runtime=runtime)
        await avatar.start(agent, SimpleNamespace())
        assert avatar.runtime is runtime
        await avatar.aclose()
        assert runtime.cleaned is False

    assert runtime.stop_calls == 2
    assert len(runtime.audio) == 2


async def test_owned_runtime_is_recreated_after_close(monkeypatch):
    created = []

    async def create(**kwargs):
        runtime = Runtime()
        created.append(runtime)
        return runtime

    monkeypatch.setattr(
        "livekit.plugins.bithuman.avatar.import_module",
        lambda name: SimpleNamespace(AsyncBithuman=SimpleNamespace(create=create)),
    )
    avatar = AvatarSession(model_path="avatar.imx", api_secret="key")
    agent = SimpleNamespace(output=SimpleNamespace(replace_audio_tail=Mock()))
    for _ in range(2):
        await avatar.start(agent, SimpleNamespace())
        assert avatar.runtime is created[-1]
        await avatar.aclose()
        assert created[-1].stop_calls == 1
        assert created[-1].cleaned is True
    assert len(created) == 2
    assert created[0] is not created[1]
