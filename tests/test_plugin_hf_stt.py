from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, stt
from livekit.plugins import hf
from livekit.plugins.hf import stt as hf_stt

pytestmark = pytest.mark.unit


class _FakePipeline:
    def __init__(self, text: str = " hello there ", error: Exception | None = None) -> None:
        self.model = SimpleNamespace(
            generation_config=SimpleNamespace(lang_to_id={"<|en|>": 50259})
        )
        self.text = text
        self.error = error
        self.calls: list[tuple[np.ndarray, int, dict[str, Any]]] = []

    def __call__(self, inputs: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        if self.error is not None:
            raise self.error
        self.calls.append((inputs["raw"], inputs["sampling_rate"], kwargs))
        return {"text": self.text}


def _install(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: _FakePipeline,
    *,
    is_whisper: bool = True,
    **stt_kwargs: Any,
) -> hf.STT:
    def _load(key: Any, *, local_files_only: bool) -> hf_stt._LoadedSTT:
        return hf_stt._LoadedSTT(pipeline=pipeline, sample_rate=16000, is_whisper=is_whisper)

    monkeypatch.setattr(hf_stt, "_load_stt", _load)
    # a unique model name per test keeps the process-wide cache from leaking between tests
    return hf.STT(model=f"fake/{id(pipeline)}", **stt_kwargs)


def _frame(seconds: float, sample_rate: int = 48000, channels: int = 1) -> rtc.AudioFrame:
    samples = int(seconds * sample_rate)
    t = np.arange(samples) / sample_rate
    tone = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
    data = np.repeat(tone, channels) if channels > 1 else tone
    return rtc.AudioFrame(
        data=data.tobytes(),
        sample_rate=sample_rate,
        num_channels=channels,
        samples_per_channel=samples,
    )


async def test_recognize_resamples_to_mono_model_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _FakePipeline()
    instance = _install(monkeypatch, pipeline)

    event = await instance.recognize([_frame(0.5, channels=2), _frame(0.5, channels=2)])

    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives[0].text == "hello there"
    audio, sample_rate, _ = pipeline.calls[0]
    assert sample_rate == 16000
    assert audio.dtype == np.float32
    assert abs(len(audio) - 16000) < 400  # one second, give or take resampler delay
    assert np.abs(audio).max() <= 1.0
    assert event.recognition_usage is not None
    assert event.recognition_usage.audio_duration == pytest.approx(len(audio) / 16000)


async def test_whisper_gets_language_and_task(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _FakePipeline()
    instance = _install(monkeypatch, pipeline, language="fr", generate_kwargs={"num_beams": 2})

    event = await instance.recognize(_frame(0.2), language="sw")

    _, _, kwargs = pipeline.calls[0]
    assert kwargs["generate_kwargs"] == {"num_beams": 2, "task": "transcribe", "language": "sw"}
    assert event.alternatives[0].language == "sw"


async def test_fine_tune_language_tokens_override_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # fine-tunes such as SunflowerASR select languages with their own token ids
    forced = [[1, 50332], [2, 50360], [3, 50364]]
    pipeline = _FakePipeline()
    instance = _install(
        monkeypatch, pipeline, language="lg", generate_kwargs={"forced_decoder_ids": forced}
    )

    await instance.recognize(_frame(0.2))

    _, _, kwargs = pipeline.calls[0]
    assert kwargs["generate_kwargs"]["forced_decoder_ids"] == forced
    assert "language" not in kwargs["generate_kwargs"]


async def test_fine_tune_language_token_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _FakePipeline()
    instance = _install(monkeypatch, pipeline, language_token_ids={"lug": 50332})

    await instance.recognize(_frame(0.2), language="lug")
    await instance.recognize(_frame(0.2), language="en")

    lug_kwargs = pipeline.calls[0][2]["generate_kwargs"]
    assert lug_kwargs["language"] == "<|lug|>"
    assert lug_kwargs["generation_config"].lang_to_id == {"<|en|>": 50259, "<|lug|>": 50332}
    # the model's own config is copied, never changed
    assert pipeline.model.generation_config.lang_to_id == {"<|en|>": 50259}
    assert pipeline.calls[1][2]["generate_kwargs"] == {"task": "transcribe", "language": "en"}


async def test_non_whisper_model_gets_no_language(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _FakePipeline()
    instance = _install(monkeypatch, pipeline, is_whisper=False, language="en")

    await instance.recognize(_frame(0.2))

    assert pipeline.calls[0][2] == {}


async def test_empty_buffer_skips_model(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _FakePipeline()
    instance = _install(monkeypatch, pipeline)

    event = await instance.recognize([])

    assert event.alternatives[0].text == ""
    assert pipeline.calls == []


async def test_model_error_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _FakePipeline(error=RuntimeError("CUDA out of memory"))
    instance = _install(monkeypatch, pipeline)

    with pytest.raises(APIConnectionError) as exc_info:
        await instance.recognize(_frame(0.2))
    assert exc_info.value.retryable is False


async def test_prewarm_loads_once_in_background(monkeypatch: pytest.MonkeyPatch) -> None:
    loads: list[Any] = []

    def _load(key: Any, *, local_files_only: bool) -> hf_stt._LoadedSTT:
        loads.append(key)
        return hf_stt._LoadedSTT(pipeline=_FakePipeline(), sample_rate=16000, is_whisper=True)

    monkeypatch.setattr(hf_stt, "_load_stt", _load)
    instance = hf.STT(model="fake/prewarm")

    instance.prewarm()
    instance.prewarm()
    assert instance._prewarm_task is not None
    await instance._prewarm_task
    await instance.recognize(_frame(0.1))

    assert len(loads) == 1
