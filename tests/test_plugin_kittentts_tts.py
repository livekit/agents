"""Unit tests for the KittenTTS TTS plugin."""

from __future__ import annotations

import asyncio
import sys
import threading
import types

import numpy as np
import pytest

from livekit.plugins.kittentts import tts as kittentts_tts


class FakeKittenTTS:
    instances: list[FakeKittenTTS] = []

    def __init__(self, model_name: str, cache_dir: str | None = None, backend: str | None = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.backend = backend
        self.calls = []
        FakeKittenTTS.instances.append(self)

    def generate_stream(self, text: str, *, voice: str, speed: float, clean_text: bool):
        self.calls.append(
            {
                "text": text,
                "voice": voice,
                "speed": speed,
                "clean_text": clean_text,
            }
        )
        yield np.array([0.0, 0.5, -0.5, 1.5, -1.5], dtype=np.float32)


@pytest.fixture(autouse=True)
def fake_kittentts(monkeypatch):
    FakeKittenTTS.instances = []
    module = types.ModuleType("kittentts")
    module.KittenTTS = FakeKittenTTS
    monkeypatch.setitem(sys.modules, "kittentts", module)


@pytest.mark.asyncio
async def test_synthesize_returns_pcm_audio() -> None:
    service = kittentts_tts.TTS(
        model="KittenML/kitten-tts-nano-0.8",
        voice="expr-voice-5-m",
        speed=1.2,
        clean_text=False,
        cache_dir="/tmp/kittentts-cache",
        backend="cpu",
    )

    frame = await service.synthesize("hello").collect()

    assert frame.sample_rate == 24000
    assert frame.num_channels == 1
    assert len(frame.data) > 0

    assert len(FakeKittenTTS.instances) == 1
    fake_model = FakeKittenTTS.instances[0]
    assert fake_model.model_name == "KittenML/kitten-tts-nano-0.8"
    assert fake_model.cache_dir == "/tmp/kittentts-cache"
    assert fake_model.backend == "cpu"
    assert fake_model.calls == [
        {
            "text": "hello",
            "voice": "expr-voice-5-m",
            "speed": 1.2,
            "clean_text": False,
        }
    ]


def test_audio_to_pcm16_clips_and_converts() -> None:
    pcm = kittentts_tts._audio_to_pcm16(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    samples = np.frombuffer(pcm, dtype="<i2")

    assert samples.tolist() == [-32767, -32767, 0, 32767, 32767]


def test_metadata_properties() -> None:
    service = kittentts_tts.TTS(model="KittenML/kitten-tts-mini-0.8")

    assert service.model == "KittenML/kitten-tts-mini-0.8"
    assert service.provider == "KittenML"
    assert service.sample_rate == 24000
    assert service.num_channels == 1
    assert service.capabilities.streaming is False


@pytest.mark.asyncio
async def test_update_options_discards_stale_concurrent_model_load() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingKittenTTS(FakeKittenTTS):
        def __init__(
            self, model_name: str, cache_dir: str | None = None, backend: str | None = None
        ):
            if model_name == "old-model":
                started.set()
                assert release.wait(timeout=5)
            super().__init__(model_name, cache_dir=cache_dir, backend=backend)

    sys.modules["kittentts"].KittenTTS = BlockingKittenTTS
    service = kittentts_tts.TTS(model="old-model")

    load_task = asyncio.create_task(service._ensure_model())
    assert await asyncio.to_thread(started.wait, 5)

    service.update_options(model="new-model")
    release.set()

    model = await load_task

    assert model.model_name == "new-model"
    assert [instance.model_name for instance in FakeKittenTTS.instances] == [
        "old-model",
        "new-model",
    ]
