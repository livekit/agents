# Copyright 2026 Oruk AI
# SPDX-License-Identifier: Apache-2.0

"""Pinned model acquisition and serialized CPU inference."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import snapshot_download, try_to_load_from_cache
from numpy.typing import NDArray

REPO_ID = "oruk/orukeet"
REVISION = "1751fce6ecde442f14543cf1804800c49b3e415c"
SUBFOLDER = "onnx/combined-v0.1.0-int8"
FILE_HASHES = {
    "encoder-model.int8.onnx": "7b55f2a504a20a8e462899f5befd45f4a1784948d76ed0127902d9cf39405487",
    "decoder_joint-model.int8.onnx": "95d3b1f53f9aadc5ef58e63664a3681a2184ee228b5010e1ef975a1c4ea8318a",
    "vocab.txt": "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d",
    "config.json": "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466",
}
LICENSE_FILES = (
    "LICENSE-WEIGHTS",
    "NOTICE.md",
    "LICENSE-CONVERTER.txt",
    "LICENSE-PREPROCESSOR.txt",
)


def download_model(*, cache_dir: str | None = None, local_files_only: bool = False) -> Path:
    """Acquire and verify the pinned model, reusing a complete cache without HTTP.

    The runtime consumes config.json; its normal Hub download participates in
    Hugging Face model download accounting. No extra counting request is made.
    """
    required = (*FILE_HASHES, *LICENSE_FILES)
    paths = [
        try_to_load_from_cache(
            REPO_ID, f"{SUBFOLDER}/{name}", revision=REVISION, cache_dir=cache_dir
        )
        for name in required
    ]
    if all(isinstance(path, str) for path in paths):
        directory = Path(str(paths[0])).parent
    else:
        snapshot = snapshot_download(
            repo_id=REPO_ID,
            revision=REVISION,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            allow_patterns=[f"{SUBFOLDER}/{name}" for name in required],
        )
        directory = Path(snapshot) / SUBFOLDER

    for name, expected in FILE_HASHES.items():
        digest = hashlib.sha256()
        with (directory / name).open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(
                f"Orukeet checksum mismatch for {name}. Remove the corrupted cached file "
                "and download it again before using this model."
            )
    return directory


class Model:
    """Keep loading, inference and closing under one native-runtime lock."""

    def __init__(self, *, cache_dir: str | None, local_files_only: bool) -> None:
        self._cache_dir = cache_dir
        self._local_files_only = local_files_only
        self._lock = threading.Lock()
        self._recognizer: Any = None
        self._closed = False

    def _load(self) -> Any:
        if self._closed:
            raise RuntimeError("Orukeet STT is closed")
        if self._recognizer is None:
            import onnx_asr

            directory = download_model(
                cache_dir=self._cache_dir, local_files_only=self._local_files_only
            )
            self._recognizer = onnx_asr.load_model(
                "nemo-conformer-tdt",
                path=directory,
                quantization="int8",
                providers=["CPUExecutionProvider"],
            )
        return self._recognizer

    def prewarm(self) -> None:
        """Download, verify and load the model once, outside the event loop."""
        with self._lock:
            self._load()

    def recognize(self, audio: NDArray[np.float32], sample_rate: int) -> str:
        """Recognize a mono utterance; native work continues until completion if canceled."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Orukeet STT is closed")
            if not audio.size:
                return ""
            text: str = self._load().recognize(audio, sample_rate=sample_rate)
            return text

    def close(self) -> None:
        """Wait for any native work to finish before releasing the recognizer."""
        with self._lock:
            self._closed = True
            self._recognizer = None
