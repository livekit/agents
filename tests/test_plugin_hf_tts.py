from __future__ import annotations

import json
import os
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from livekit.agents import APIConnectionError
from livekit.plugins import hf
from livekit.plugins.hf import download as hf_download, tts as hf_tts

pytestmark = pytest.mark.unit

_BASE = hf_tts._AUDIO_TOKEN_BASE
_SPF = hf_tts._SAMPLES_PER_FRAME


def _frame_tokens(frame_index: int) -> list[int]:
    """Orpheus token ids for one SNAC frame whose coarse code is the frame index."""
    codes = [frame_index, 1, 2, 3, 4, 5, 6]
    return [_BASE + position * 4096 + code for position, code in enumerate(codes)]


def _orpheus_tokens(frames: int) -> list[int]:
    tokens = [hf_tts._START_OF_SPEECH]
    for i in range(frames):
        tokens += _frame_tokens(i)
    return [*tokens, hf_tts._END_OF_SPEECH]


class _FakeCodec:
    """Decodes each frame to 2048 samples holding its coarse code, like SNAC's layout."""

    def __init__(self) -> None:
        import torch

        self._param = torch.zeros(1)
        self.windows: list[int] = []

    def parameters(self) -> Any:
        return iter([self._param])

    def decode(self, layers: list[Any]) -> Any:

        coarse = layers[0].reshape(-1)
        assert layers[1].numel() == 2 * coarse.numel()
        assert layers[2].numel() == 4 * coarse.numel()
        self.windows.append(int(coarse.numel()))
        audio = coarse.float().repeat_interleave(_SPF) / 1000.0
        return audio.reshape(1, 1, -1)


def _install_orpheus(
    monkeypatch: pytest.MonkeyPatch,
    tokens: list[int],
    *,
    error: Exception | None = None,
    **tts_kwargs: Any,
) -> tuple[hf.TTS, _FakeCodec, list[str]]:
    codec = _FakeCodec()
    prompts: list[str] = []

    def _load(key: Any, *, local_files_only: bool) -> hf_tts._LoadedTTS:
        return hf_tts._LoadedTTS(
            family="orpheus", model=None, tokenizer=None, sample_rate=24000, codec=codec
        )

    def _generate(
        loaded: Any, text: str, opts: Any, on_token: Any, stop_event: threading.Event
    ) -> None:
        prompts.append(f"{opts.voice}: {text}" if opts.voice else text)
        for token in tokens:
            if stop_event.is_set():
                return
            on_token(token)
        if error is not None:
            raise error

    monkeypatch.setattr(hf_tts, "_load_tts", _load)
    monkeypatch.setattr(hf_tts, "_generate_orpheus", _generate)
    instance = hf.TTS(model=f"fake/orpheus-{id(codec)}", **tts_kwargs)
    return instance, codec, prompts


async def _synthesize(instance: hf.TTS, text: str) -> tuple[np.ndarray, int]:
    audio = [a async for a in instance.synthesize(text)]
    # the framework closes a segment with a short silent marker frame; drop it
    if len(audio) > 1 and not np.frombuffer(audio[-1].frame.data, dtype=np.int16).any():
        audio = audio[:-1]
    pcm = np.concatenate([np.frombuffer(a.frame.data, dtype=np.int16) for a in audio])
    return pcm, audio[0].frame.sample_rate


async def test_orpheus_streams_every_frame_once(monkeypatch: pytest.MonkeyPatch) -> None:
    instance, codec, prompts = _install_orpheus(
        monkeypatch, _orpheus_tokens(10), voice="salt_lug_0001", chunk_frames=3
    )

    pcm, sample_rate = await _synthesize(instance, "Oli otya?")

    assert prompts == ["salt_lug_0001: Oli otya?"]
    assert sample_rate == 24000
    assert len(pcm) == 10 * _SPF
    # each frame's samples hold its index, so a dropped or repeated frame shows up here
    per_frame = pcm.reshape(10, _SPF)
    expected = (np.arange(10) / 1000.0 * 32767.0).astype(np.int16)
    assert (per_frame == expected[:, None]).all()
    # decoded in several windows, not once at the end
    assert len(codec.windows) > 1


async def test_orpheus_ignores_text_tokens_and_bad_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    misplaced = _BASE + 3 * 4096 + 7  # a position-3 code where position 0 is expected
    tokens = [
        hf_tts._START_OF_SPEECH,
        *_frame_tokens(0)[:1],
        42,  # a text token mid-stream
        *_frame_tokens(0)[1:],
        misplaced,
        *_frame_tokens(1),
        hf_tts._END_OF_SPEECH,
    ]
    instance, _, _ = _install_orpheus(monkeypatch, [42, 7, *tokens])

    pcm, _ = await _synthesize(instance, "hi")

    assert len(pcm) == 2 * _SPF


async def test_orpheus_without_speech_tokens_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    instance, _, _ = _install_orpheus(monkeypatch, [], error=RuntimeError("boom"))

    with pytest.raises(APIConnectionError):
        await _synthesize(instance, "hi")


def test_decoder_windows_keep_context_out_of_output() -> None:
    decoder = hf_tts._SnacStreamDecoder(chunk_frames=2, left_context=1, right_context=1)
    windows = []
    for token in _orpheus_tokens(5):
        if (window := decoder.push(token)) is not None:
            windows.append(window)
    if (window := decoder.flush()) is not None:
        windows.append(window)

    kept = [(w.end - w.start) // _SPF for w in windows]
    assert sum(kept) == 5
    # the second window reaches back one frame for context and keeps only new frames
    assert windows[1].start == _SPF
    assert len(windows[1].codes) // 7 == 4


async def test_waveform_model_is_resampled(monkeypatch: pytest.MonkeyPatch) -> None:
    def _load(key: Any, *, local_files_only: bool) -> hf_tts._LoadedTTS:
        return hf_tts._LoadedTTS(family="waveform", model=None, tokenizer=None, sample_rate=16000)

    monkeypatch.setattr(hf_tts, "_load_tts", _load)
    monkeypatch.setattr(
        hf_tts,
        "_synthesize_waveform",
        lambda loaded, text: np.sin(np.linspace(0, 100, 16000)).astype(np.float32) * 0.5,
    )
    instance = hf.TTS(model="fake/vits")

    pcm, sample_rate = await _synthesize(instance, "hello")

    assert sample_rate == 24000
    assert abs(len(pcm) - 24000) < 600


def test_detects_orpheus_from_config() -> None:
    assert hf_tts._is_orpheus(SimpleNamespace(model_type="llama", vocab_size=156939))
    assert not hf_tts._is_orpheus(SimpleNamespace(model_type="llama", vocab_size=128256))
    assert not hf_tts._is_orpheus(SimpleNamespace(model_type="vits", vocab_size=200000))


def test_download_fetches_snac_for_orpheus(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    import huggingface_hub

    repos = {
        "sunbird/orpheus": {
            "model.safetensors": "",
            "config.json": {"model_type": "llama", "vocab_size": 156939},
        },
        "facebook/mms": {
            "model.safetensors": "",
            "config.json": {"model_type": "vits", "vocab_size": 40},
        },
        hf_tts.SNAC_MODEL: {"pytorch_model.bin": "", "config.json": {}},
    }
    calls: list[tuple[str, list[str]]] = []

    def _snapshot(repo: str, *, revision: Any, allow_patterns: list[str]) -> str:
        calls.append((repo, allow_patterns))
        path = tmp_path / repo.replace("/", "--")
        path.mkdir(exist_ok=True)
        for name, content in repos[repo].items():
            (path / name).write_text(json.dumps(content))
        return os.fspath(path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _snapshot)

    hf_download.download_model("facebook/mms")
    assert [c[0] for c in calls] == ["facebook/mms"]

    calls.clear()
    hf_download.download_model("sunbird/orpheus")
    assert [c[0] for c in calls] == ["sunbird/orpheus", hf_tts.SNAC_MODEL, hf_tts.SNAC_MODEL]
    assert "*.bin" in calls[-1][1]
