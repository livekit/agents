from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Callable
from typing import Any

import pytest

from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectionError

pytestmark = pytest.mark.plugin("funasr")


class _FakeAutoModel:
    generate_impl: Callable[..., list[dict[str, str]]]

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

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
