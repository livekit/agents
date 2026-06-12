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

    def __init__(self, model_name: str, cache_dir: str | None = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.calls = []
        FakeKittenTTS.instances.append(self)

    def generate_stream(
        self, text: str, *, voice: str, speed: float, clean_text: bool, chunk_size: int
    ):
        self.calls.append(
            {
                "text": text,
                "voice": voice,
                "speed": speed,
                "clean_text": clean_text,
                "chunk_size": chunk_size,
            }
        )
        yield np.array([0.0, 0.5, -0.5, 1.5, -1.5], dtype=np.float32)


class FakeGenerateOnlyKittenTTS:
    instances: list[FakeGenerateOnlyKittenTTS] = []

    def __init__(self, model_name: str, cache_dir: str | None = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.calls = []
        FakeGenerateOnlyKittenTTS.instances.append(self)

    def generate(self, text: str, *, voice: str, speed: float, clean_text: bool):
        self.calls.append(
            {
                "text": text,
                "voice": voice,
                "speed": speed,
                "clean_text": clean_text,
            }
        )
        return np.array([0.0, 0.25, -0.25], dtype=np.float32)


@pytest.fixture(autouse=True)
def fake_kittentts(monkeypatch):
    FakeKittenTTS.instances = []
    FakeGenerateOnlyKittenTTS.instances = []
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
        chunk_size=120,
        cache_dir="/tmp/kittentts-cache",
    )

    frame = await service.synthesize("hello").collect()

    assert frame.sample_rate == 24000
    assert frame.num_channels == 1
    assert len(frame.data) > 0

    assert len(FakeKittenTTS.instances) == 1
    fake_model = FakeKittenTTS.instances[0]
    assert fake_model.model_name == "KittenML/kitten-tts-nano-0.8"
    assert fake_model.cache_dir == "/tmp/kittentts-cache"
    assert fake_model.calls == [
        {
            "text": "hello",
            "voice": "expr-voice-5-m",
            "speed": 1.2,
            "clean_text": False,
            "chunk_size": 120,
        }
    ]


@pytest.mark.asyncio
async def test_synthesize_supports_generate_only_api() -> None:
    sys.modules["kittentts"].KittenTTS = FakeGenerateOnlyKittenTTS
    FakeGenerateOnlyKittenTTS.instances = []

    service = kittentts_tts.TTS(voice="expr-voice-2-f", speed=0.9, clean_text=True)

    frame = await service.synthesize("local hello").collect()

    assert frame.sample_rate == 24000
    assert frame.num_channels == 1
    assert len(frame.data) > 0

    assert len(FakeGenerateOnlyKittenTTS.instances) == 1
    assert FakeGenerateOnlyKittenTTS.instances[0].calls == [
        {
            "text": "local hello",
            "voice": "expr-voice-2-f",
            "speed": 0.9,
            "clean_text": True,
        }
    ]


@pytest.mark.asyncio
async def test_stream_accepts_incremental_text() -> None:
    service = kittentts_tts.TTS()

    async with service.stream() as stream:
        stream.push_text("hello ")
        stream.push_text("stream")
        stream.end_input()
        events = [event async for event in stream]

    assert events
    assert all(event.frame.sample_rate == 24000 for event in events)
    assert all(event.frame.num_channels == 1 for event in events)

    assert len(FakeKittenTTS.instances) == 1
    assert FakeKittenTTS.instances[0].calls == [
        {
            "text": "hello stream",
            "voice": "expr-voice-5-m",
            "speed": 1.0,
            "clean_text": True,
            "chunk_size": 120,
        }
    ]


@pytest.mark.asyncio
async def test_stream_chunk_size_can_be_configured() -> None:
    service = kittentts_tts.TTS(chunk_size=80)

    async with service.stream() as stream:
        stream.push_text("hello chunk size")
        stream.end_input()
        events = [event async for event in stream]

    assert events
    assert FakeKittenTTS.instances[0].calls == [
        {
            "text": "hello chunk size",
            "voice": "expr-voice-5-m",
            "speed": 1.0,
            "clean_text": True,
            "chunk_size": 80,
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
    assert service.capabilities.streaming is True


@pytest.mark.asyncio
async def test_update_options_discards_stale_concurrent_model_load() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingKittenTTS(FakeKittenTTS):
        def __init__(self, model_name: str, cache_dir: str | None = None):
            if model_name == "old-model":
                started.set()
                assert release.wait(timeout=5)
            super().__init__(model_name, cache_dir=cache_dir)

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
