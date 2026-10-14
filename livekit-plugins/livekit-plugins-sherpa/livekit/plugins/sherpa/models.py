# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

STTModels = Literal[
    "zipformer-en-2023-06-26",
    "zipformer-en-20M-2023-02-17",
    "moonshine-tiny-en-int8",
    "whisper-tiny.en-int8",
]


@dataclass(frozen=True)
class HubModel:
    """Model files on the Hugging Face Hub, pinned to a commit so they never change."""

    repo_id: str
    revision: str
    files: tuple[str, ...]
    """Glob patterns of the files the plugin needs from the repo."""


@dataclass(frozen=True)
class TransducerModel(HubModel):
    encoder: str = ""
    decoder: str = ""
    joiner: str = ""
    tokens: str = "tokens.txt"
    language: str = "en"


@dataclass(frozen=True)
class MoonshineModel(HubModel):
    preprocessor: str = "preprocess.onnx"
    encoder: str = "encode.int8.onnx"
    uncached_decoder: str = "uncached_decode.int8.onnx"
    cached_decoder: str = "cached_decode.int8.onnx"
    tokens: str = "tokens.txt"
    language: str = "en"


@dataclass(frozen=True)
class WhisperModel(HubModel):
    encoder: str = ""
    decoder: str = ""
    tokens: str = ""
    language: str = "en"


def _transducer(repo_id: str, revision: str, suffix: str) -> TransducerModel:
    encoder, decoder, joiner = (f"{part}-{suffix}" for part in ("encoder", "decoder", "joiner"))
    return TransducerModel(
        repo_id=repo_id,
        revision=revision,
        files=(encoder, decoder, joiner, "tokens.txt"),
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
    )


STT_MODELS: dict[STTModels, TransducerModel | MoonshineModel | WhisperModel] = {
    # Streaming (online) models.
    # Default. ~70M params, 80 MB int8.
    "zipformer-en-2023-06-26": _transducer(
        "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26",
        "672fbf1b30579d6585301139bb363f42a0ad4a24",
        "epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    ),
    # ~20M params, 42 MB int8. Drops the first words of an utterance when the audio was
    # resampled from 48 kHz (every LiveKit room), so it is not the default.
    "zipformer-en-20M-2023-02-17": _transducer(
        "csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
        "d42f2d9f7ca24806fb667456a18a9f1b60f70d16",
        "epoch-99-avg-1.int8.onnx",
    ),
    # Batch (offline) models: AgentSession runs them live through stt.StreamAdapter + VAD.
    # Default. Moonshine tiny, ~27M params, 120 MB int8.
    "moonshine-tiny-en-int8": MoonshineModel(
        repo_id="csukuangfj/sherpa-onnx-moonshine-tiny-en-int8",
        revision="bf2b762c076d8ea61e2af0b3851c9564fb77552e",
        files=(
            "preprocess.onnx",
            "encode.int8.onnx",
            "uncached_decode.int8.onnx",
            "cached_decode.int8.onnx",
            "tokens.txt",
        ),
    ),
    # Whisper tiny.en, ~39M params, 100 MB int8.
    "whisper-tiny.en-int8": WhisperModel(
        repo_id="csukuangfj/sherpa-onnx-whisper-tiny.en",
        revision="d026532c022fa99fd789d6b32446a1df7b6bfc43",
        files=("tiny.en-encoder.int8.onnx", "tiny.en-decoder.int8.onnx", "tiny.en-tokens.txt"),
        encoder="tiny.en-encoder.int8.onnx",
        decoder="tiny.en-decoder.int8.onnx",
        tokens="tiny.en-tokens.txt",
    ),
}


def download_model(model: HubModel) -> str:
    path: str = snapshot_download(
        model.repo_id, revision=model.revision, allow_patterns=list(model.files)
    )
    return path


def local_model_dir(model: HubModel) -> str:
    """Directory of an already downloaded model. Never touches the network."""
    try:
        path: str = snapshot_download(
            model.repo_id,
            revision=model.revision,
            allow_patterns=list(model.files),
            local_files_only=True,
        )
    except LocalEntryNotFoundError as e:
        raise RuntimeError(
            f"sherpa model {model.repo_id} is not downloaded. "
            "Run `python -m livekit.agents download-files` once while online."
        ) from e
    return path
