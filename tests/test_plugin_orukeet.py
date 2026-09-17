"""Hermetic checks for local model integrity, audio conversion and lifecycle."""

from __future__ import annotations

import asyncio
import hashlib
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import stt

orukeet = pytest.importorskip("livekit.plugins.orukeet")
from livekit.plugins.orukeet import _model  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.fixture
def model_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Small pinned files exercise the downloader without fetching real weights."""
    folder = tmp_path / _model.SUBFOLDER
    folder.mkdir(parents=True)
    payload = b"verified fixture"
    hashes = {name: hashlib.sha256(payload).hexdigest() for name in _model.FILE_HASHES}
    for name in (*hashes, *_model.LICENSE_FILES):
        (folder / name).write_bytes(payload)
    monkeypatch.setattr(_model, "FILE_HASHES", hashes)
    return folder


def test_complete_cache_does_not_request_http(model_cache: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        _model,
        "try_to_load_from_cache",
        lambda repo, name, **kwargs: str(model_cache / Path(name).name),
    )
    download = Mock(side_effect=AssertionError("complete cache must not request HTTP"))
    monkeypatch.setattr(_model, "snapshot_download", download)
    assert _model.download_model(local_files_only=True) == model_cache
    download.assert_not_called()


def test_missing_cache_downloads_pinned_required_files(model_cache: Path, monkeypatch):
    monkeypatch.setattr(_model, "try_to_load_from_cache", lambda *args, **kwargs: None)
    download = Mock(return_value=str(model_cache.parents[1]))
    monkeypatch.setattr(_model, "snapshot_download", download)
    assert _model.download_model(cache_dir="chosen-cache") == model_cache
    assert download.call_args.kwargs == {
        "repo_id": "oruk/orukeet",
        "revision": _model.REVISION,
        "cache_dir": "chosen-cache",
        "local_files_only": False,
        "allow_patterns": [
            f"{_model.SUBFOLDER}/{name}" for name in (*_model.FILE_HASHES, *_model.LICENSE_FILES)
        ],
    }


@pytest.mark.parametrize("filename", tuple(_model.FILE_HASHES))
def test_corruption_is_rejected_without_deleting_other_files(model_cache, monkeypatch, filename):
    monkeypatch.setattr(
        _model,
        "try_to_load_from_cache",
        lambda repo, name, **kwargs: str(model_cache / Path(name).name),
    )
    (model_cache / filename).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _model.download_model(local_files_only=True)
    assert all((model_cache / name).exists() for name in _model.FILE_HASHES)
    (model_cache / filename).write_bytes(b"verified fixture")
    assert _model.download_model(local_files_only=True) == model_cache


def test_offline_miss_is_not_retried_online(monkeypatch):
    monkeypatch.setattr(_model, "try_to_load_from_cache", lambda *args, **kwargs: None)
    download = Mock(side_effect=FileNotFoundError("missing cached weights"))
    monkeypatch.setattr(_model, "snapshot_download", download)
    with pytest.raises(FileNotFoundError):
        _model.download_model(local_files_only=True)
    download.assert_called_once()
    assert download.call_args.kwargs["local_files_only"] is True


async def test_stereo_pcm_conversion_and_honest_capabilities():
    recognizer = orukeet.STT(local_files_only=True)
    recognize = Mock(return_value="bonjour")
    recognizer._runtime = SimpleNamespace(recognize=recognize, close=lambda: None)
    samples = np.array([[16384, 0], [-16384, 0]], dtype=np.int16)
    frame = rtc.AudioFrame(samples.tobytes(), 48000, 2, 2)
    event = await recognizer.recognize(frame)
    np.testing.assert_array_equal(recognize.call_args.args[0], np.array([0.25, -0.25]))
    assert recognize.call_args.args[1] == 48000
    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives[0].text == "bonjour"
    assert event.alternatives[0].language == ""
    assert event.alternatives[0].words is None
    assert recognizer.capabilities == stt.STTCapabilities(streaming=False, interim_results=False)
    await recognizer.aclose()


async def test_empty_input_does_not_load_model(monkeypatch):
    load = Mock(side_effect=AssertionError("empty audio must not load a model"))
    monkeypatch.setattr(_model.Model, "_load", load)
    recognizer = orukeet.STT(local_files_only=True)
    event = await recognizer.recognize([])
    assert event.alternatives[0].text == ""
    load.assert_not_called()
    await recognizer.aclose()


async def test_language_forcing_fails_before_model_work():
    recognizer = orukeet.STT(local_files_only=True)
    with pytest.raises(ValueError, match="automatically"):
        await recognizer.recognize([], language="en")
    await recognizer.aclose()


def test_prewarm_loads_once(monkeypatch, tmp_path):
    import onnx_asr

    monkeypatch.setattr(_model, "download_model", lambda **kwargs: tmp_path)
    load = Mock(return_value=SimpleNamespace(recognize=lambda *args, **kwargs: "hello"))
    monkeypatch.setattr(onnx_asr, "load_model", load)
    model = _model.Model(cache_dir=None, local_files_only=True)
    model.prewarm()
    model.prewarm()
    assert model.recognize(np.ones(16, dtype=np.float32), 16000) == "hello"
    load.assert_called_once_with(
        "nemo-conformer-tdt", path=tmp_path, quantization="int8", providers=["CPUExecutionProvider"]
    )
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.prewarm()


async def test_cancelled_native_work_finishes_before_close():
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def blocking_recognize(*args, **kwargs):
        started.set()
        assert release.wait(5)
        finished.set()
        return "stale transcript"

    recognizer = orukeet.STT(local_files_only=True)
    recognizer._runtime._recognizer = SimpleNamespace(recognize=blocking_recognize)
    frame = rtc.AudioFrame(np.ones(160, dtype=np.int16).tobytes(), 16000, 1, 160)
    task = asyncio.create_task(recognizer.recognize(frame))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        close = asyncio.create_task(recognizer.aclose())
        await asyncio.sleep(0)
        assert not close.done()
    finally:
        release.set()
    await asyncio.wait_for(close, 5)
    assert finished.is_set()
    with pytest.raises(RuntimeError, match="closed"):
        await recognizer.recognize(frame)


async def test_automatic_prewarm_keeps_loop_responsive_and_close_waits():
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def load():
        started.set()
        assert release.wait(5)
        finished.set()

    recognizer = orukeet.STT(local_files_only=True)
    recognizer._runtime._load = load
    try:
        recognizer.prewarm()
        recognizer.prewarm()
        assert await asyncio.to_thread(started.wait, 5)
        close = asyncio.create_task(recognizer.aclose())
        await asyncio.sleep(0)
        assert not close.done()
    finally:
        release.set()
    await asyncio.wait_for(close, 5)
    assert finished.is_set()
    assert recognizer._prewarm_task.done()


@pytest.mark.parametrize("rate", [12000, 96000])
async def test_uncommon_sample_rates_use_native_resampling(rate):
    recognizer = orukeet.STT(local_files_only=True)
    recognize = Mock(return_value="hello")
    recognizer._runtime = SimpleNamespace(recognize=recognize, close=lambda: None)
    samples = np.zeros(rate, dtype=np.int16)
    frame = rtc.AudioFrame(samples.tobytes(), rate, 1, rate)
    assert (await recognizer.recognize(frame)).alternatives[0].text == "hello"
    assert recognize.call_args.args[1] == 16000
    assert len(recognize.call_args.args[0]) == 16000
    await recognizer.aclose()
