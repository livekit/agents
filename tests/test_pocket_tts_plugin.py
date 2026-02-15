from __future__ import annotations

import asyncio
import importlib
import sys
import threading
import time
import types
from collections.abc import Generator
from typing import Any

import numpy as np
import pytest

from livekit.agents import APIConnectionError


@pytest.fixture
def pocket_plugin(monkeypatch: pytest.MonkeyPatch) -> Any:
    calls: dict[str, Any] = {
        "num_chunks": 2,
        "chunk_samples": 9600,
        "per_chunk_sleep": 0.0,
        "pause_after_first_chunk": False,
        "allow_generation_finish": None,
        "raise_on_generate": None,
        "active_generations": 0,
        "max_active_generations": 0,
    }

    class _FakeModel:
        def get_state_for_audio_prompt(self, voice: str, truncate: bool = True) -> dict[str, str]:
            calls["voice"] = voice
            if voice == "missing":
                raise FileNotFoundError("voice prompt not found")
            if voice == "bad-voice":
                raise RuntimeError("bad voice")
            return {"voice": voice}

        def generate_audio_stream(
            self,
            state: dict[str, str],
            text: str,
            copy_state: bool = True,
        ) -> Generator[np.ndarray[Any, np.dtype[np.float32]], None, None]:
            calls["state"] = state
            calls["text"] = text
            calls["copy_state"] = copy_state
            calls["active_generations"] += 1
            calls["max_active_generations"] = max(
                calls["max_active_generations"], calls["active_generations"]
            )
            try:
                if calls["raise_on_generate"] is not None:
                    raise calls["raise_on_generate"]

                for i in range(calls["num_chunks"]):
                    if calls["per_chunk_sleep"] > 0:
                        time.sleep(calls["per_chunk_sleep"])

                    yield np.linspace(
                        -0.25 + i * 0.05,
                        0.25 - i * 0.05,
                        calls["chunk_samples"],
                        dtype=np.float32,
                    )

                    if i == 0 and calls["pause_after_first_chunk"]:
                        gate = calls["allow_generation_finish"]
                        if isinstance(gate, threading.Event):
                            gate.wait(timeout=2.0)
            finally:
                calls["active_generations"] -= 1

    class _FakeTTSModel:
        @staticmethod
        def load_model(*, temp: float, lsd_decode_steps: int) -> _FakeModel:
            calls["temperature"] = temp
            calls["lsd_decode_steps"] = lsd_decode_steps
            return _FakeModel()

    monkeypatch.setitem(sys.modules, "pocket_tts", types.SimpleNamespace(TTSModel=_FakeTTSModel))

    for name in list(sys.modules):
        if name == "livekit.plugins.pocket" or name.startswith("livekit.plugins.pocket."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    module = importlib.import_module("livekit.plugins.pocket")
    calls["module"] = module
    return calls


def test_imports_and_alias(pocket_plugin: Any) -> None:
    module = pocket_plugin["module"]
    assert module.TTS is module.PocketTTS
    assert module.DEFAULT_VOICE == "alba"


def test_fallback_voice_and_missing_voice_error(pocket_plugin: Any) -> None:
    module = pocket_plugin["module"]

    tts_fallback = module.TTS(voice="bad-voice")
    assert tts_fallback._voice == "alba"

    with pytest.raises(ValueError, match="Failed to load voice"):
        module.TTS(voice="missing")


def test_sample_rate_forced_to_native(pocket_plugin: Any, caplog: pytest.LogCaptureFixture) -> None:
    module = pocket_plugin["module"]
    caplog.set_level("WARNING")

    tts_native = module.TTS(sample_rate=48000)
    assert tts_native.sample_rate == 24000
    assert "Ignoring sample_rate=48000" in caplog.text


@pytest.mark.asyncio
async def test_stream_emits_audio(pocket_plugin: Any) -> None:
    module = pocket_plugin["module"]
    tts_v = module.TTS(voice="alba")

    async with tts_v.stream() as synth_stream:
        synth_stream.push_text("hola")
        synth_stream.end_input()
        streamed_events = await asyncio.wait_for(
            _collect_events(synth_stream),
            timeout=3.0,
        )

    assert streamed_events
    assert streamed_events[-1].is_final
    assert all(event.frame.sample_rate == 24000 for event in streamed_events)
    assert all(event.frame.num_channels == 1 for event in streamed_events)
    assert streamed_events[0].segment_id.startswith("SEG_")

    chunked = tts_v.synthesize("hola")
    chunked_events = await asyncio.wait_for(_collect_events(chunked), timeout=3.0)
    assert chunked_events
    assert chunked_events[-1].is_final


@pytest.mark.asyncio
async def test_stream_emits_before_generation_completes(pocket_plugin: Any) -> None:
    module = pocket_plugin["module"]
    pocket_plugin["pause_after_first_chunk"] = True
    pocket_plugin["allow_generation_finish"] = threading.Event()
    pocket_plugin["num_chunks"] = 2
    pocket_plugin["chunk_samples"] = 9600

    tts_v = module.TTS(voice="alba")
    async with tts_v.stream() as synth_stream:
        synth_stream.push_text("hola")
        synth_stream.end_input()

        first_event = await asyncio.wait_for(synth_stream.__anext__(), timeout=1.0)
        assert not first_event.is_final

        pocket_plugin["allow_generation_finish"].set()
        remaining_events = [event async for event in synth_stream]

    all_events = [first_event, *remaining_events]
    assert all_events[-1].is_final


@pytest.mark.asyncio
async def test_chunked_generation_does_not_block_event_loop(pocket_plugin: Any) -> None:
    module = pocket_plugin["module"]
    pocket_plugin["per_chunk_sleep"] = 0.05
    pocket_plugin["num_chunks"] = 6

    tts_v = module.TTS(voice="alba")
    heartbeat = 0
    done = asyncio.Event()

    async def ticker() -> None:
        nonlocal heartbeat
        while not done.is_set():
            heartbeat += 1
            await asyncio.sleep(0.01)

    ticker_task = asyncio.create_task(ticker())
    try:
        chunked_events = await asyncio.wait_for(
            _collect_events(tts_v.synthesize("hola")), timeout=5.0
        )
        assert chunked_events
    finally:
        done.set()
        await ticker_task

    assert heartbeat >= 10


@pytest.mark.asyncio
async def test_serializes_concurrent_generation(pocket_plugin: Any) -> None:
    module = pocket_plugin["module"]
    pocket_plugin["per_chunk_sleep"] = 0.03
    pocket_plugin["num_chunks"] = 3

    tts_v = module.TTS(voice="alba")
    await asyncio.gather(
        _collect_events(tts_v.synthesize("uno")),
        _collect_events(tts_v.synthesize("dos")),
    )
    assert pocket_plugin["max_active_generations"] == 1


@pytest.mark.asyncio
async def test_generation_errors_are_mapped_to_api_errors(pocket_plugin: Any) -> None:
    module = pocket_plugin["module"]
    pocket_plugin["raise_on_generate"] = RuntimeError("boom")

    tts_v = module.TTS(voice="alba")
    with pytest.raises(APIConnectionError):
        await _collect_events(tts_v.synthesize("hola"))


def test_tensor_to_pcm_bytes_handles_channel_first_and_last(pocket_plugin: Any) -> None:
    module = importlib.import_module("livekit.plugins.pocket.tts")

    channel_first = np.array(
        [[1.0, -1.0, 0.5, -0.5], [-1.0, 1.0, -0.5, 0.5]],
        dtype=np.float32,
    )
    channel_last = np.array(
        [[1.0, -1.0], [-1.0, 1.0], [0.5, -0.5], [-0.5, 0.5]],
        dtype=np.float32,
    )

    pcm_first = module._tensor_to_pcm_bytes(channel_first)
    pcm_last = module._tensor_to_pcm_bytes(channel_last)

    assert len(pcm_first) == 8
    assert len(pcm_last) == 8


def test_tensor_to_pcm_bytes_rejects_unsupported_shape(pocket_plugin: Any) -> None:
    module = importlib.import_module("livekit.plugins.pocket.tts")
    invalid = np.zeros((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="unsupported audio tensor shape"):
        module._tensor_to_pcm_bytes(invalid)


async def _collect_events(stream: Any) -> list[Any]:
    return [event async for event in stream]
