from __future__ import annotations

import asyncio
import importlib
import sys
import threading
import types
from collections.abc import Callable
from typing import Any

import pytest

from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectionError

pytestmark = pytest.mark.plugin("funasr")


def test_package_imports_with_declared_dependencies() -> None:
    import funasr as funasr_runtime
    import torch
    import torchaudio

    from livekit.plugins import funasr as livekit_funasr

    assert callable(funasr_runtime.AutoModel)
    assert torch.__version__
    assert torchaudio.__version__
    assert livekit_funasr.STT is livekit_funasr.FunASRSTT


class _FakeAutoModel:
    generate_impl: Callable[..., list[dict[str, str]]]
    init_calls: list[dict[str, Any]] = []
    init_threads: list[int] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.init_calls.append(kwargs)
        self.init_threads.append(threading.get_ident())

    def generate(self, **kwargs: Any) -> list[dict[str, str]]:
        return self.generate_impl(**kwargs)


def _load_funasr_stt_module(
    monkeypatch: pytest.MonkeyPatch,
    generate: Callable[..., list[dict[str, str]]],
) -> types.ModuleType:
    fake_funasr = types.ModuleType("funasr")
    fake_funasr.AutoModel = _FakeAutoModel

    fake_utils = types.ModuleType("funasr.utils")
    fake_utils.__path__ = []

    fake_postprocess = types.ModuleType("funasr.utils.postprocess_utils")
    fake_postprocess.rich_transcription_postprocess = lambda text: text

    monkeypatch.setitem(sys.modules, "funasr", fake_funasr)
    monkeypatch.setitem(sys.modules, "funasr.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "funasr.utils.postprocess_utils", fake_postprocess)

    _FakeAutoModel.generate_impl = staticmethod(generate)
    _FakeAutoModel.init_calls = []
    _FakeAutoModel.init_threads = []

    for name in tuple(sys.modules):
        if name == "livekit.plugins.funasr" or name.startswith("livekit.plugins.funasr."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    return importlib.import_module("livekit.plugins.funasr.stt")


def _make_audio_frame(*, sample_rate: int = 16000, num_channels: int = 1) -> rtc.AudioFrame:
    samples_per_channel = 80
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples_per_channel * num_channels,
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=samples_per_channel,
    )


async def test_model_loading_runs_off_event_loop_on_first_recognition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    funasr_stt = _load_funasr_stt_module(
        monkeypatch,
        lambda **kwargs: [{"text": "<|en|>hello"}],
    )

    event_loop_thread = threading.get_ident()
    stt = funasr_stt.FunASRSTT()

    assert _FakeAutoModel.init_calls == []

    await stt._recognize_impl(
        [_make_audio_frame()],
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
    )

    assert _FakeAutoModel.init_calls == [
        {"model": "iic/SenseVoiceSmall", "device": "cpu", "disable_update": True}
    ]
    assert _FakeAutoModel.init_threads[0] != event_loop_thread


async def test_cancelled_recognition_keeps_model_calls_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    calls_lock = threading.Lock()
    calls = 0

    def generate(**kwargs: Any) -> list[dict[str, str]]:
        nonlocal calls
        with calls_lock:
            calls += 1
            call_number = calls
        if call_number == 1:
            first_started.set()
            assert release_first.wait(timeout=5)
        else:
            second_started.set()
        return [{"text": "<|en|>hello"}]

    funasr_stt = _load_funasr_stt_module(monkeypatch, generate)
    stt = funasr_stt.FunASRSTT()
    first = asyncio.create_task(
        stt._recognize_impl(
            [_make_audio_frame()],
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
        )
    )
    assert await asyncio.to_thread(first_started.wait, 5)

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    second = asyncio.create_task(
        stt._recognize_impl(
            [_make_audio_frame()],
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
        )
    )
    try:
        await asyncio.sleep(0.1)
        assert not second_started.is_set()
    finally:
        release_first.set()
        await asyncio.wait_for(second, timeout=5)

    assert second_started.is_set()


def test_plugin_download_files_prefetches_default_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _load_funasr_stt_module(
        monkeypatch,
        lambda **kwargs: [{"text": "<|en|>hello"}],
    )
    funasr_plugin = sys.modules["livekit.plugins.funasr"]

    funasr_plugin.FunASRPlugin().download_files()

    assert _FakeAutoModel.init_calls == [
        {"model": "iic/SenseVoiceSmall", "device": "cpu", "disable_update": True}
    ]


async def test_recognize_uses_named_high_quality_resampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    funasr_stt = _load_funasr_stt_module(
        monkeypatch,
        lambda **kwargs: [{"text": "<|en|>hello"}],
    )

    captured: dict[str, Any] = {}

    class FakeAudioResampler:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["args"] = args
            captured["kwargs"] = kwargs

        def push(self, frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
            return [frame]

        def flush(self) -> list[rtc.AudioFrame]:
            return []

    monkeypatch.setattr(funasr_stt.rtc, "AudioResampler", FakeAudioResampler)

    stt = funasr_stt.FunASRSTT()

    await stt._recognize_impl(
        [_make_audio_frame(sample_rate=8000)],
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
    )

    assert captured["args"] == ()
    assert captured["kwargs"] == {
        "input_rate": 8000,
        "output_rate": 16000,
        "num_channels": 1,
        "quality": rtc.AudioResamplerQuality.HIGH,
    }


async def test_recognize_marks_local_inference_failures_non_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_bad_output(**kwargs: Any) -> list[dict[str, str]]:
        raise ValueError("bad FunASR output")

    funasr_stt = _load_funasr_stt_module(monkeypatch, raise_bad_output)
    stt = funasr_stt.FunASRSTT()

    with pytest.raises(APIConnectionError) as exc_info:
        await stt._recognize_impl(
            [_make_audio_frame()],
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
        )

    assert exc_info.value.retryable is False
    assert isinstance(exc_info.value.__cause__, ValueError)
