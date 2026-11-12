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
from typing import Literal, get_args

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

STTModels = Literal[
    "zipformer-en-2023-06-26",
    "zipformer-en-20M-2023-02-17",
    "moonshine-tiny-en-int8",
    "whisper-tiny.en-int8",
]
TTSModels = Literal["kokoro-int8-multi-lang-v1_0", "kitten-nano-en-v0_2", "piper-en_US-amy-low"]


# Speaker ids of the Kokoro v1.0 voice table, in order (see the model's `speaker2id` metadata).
KokoroVoice = Literal[
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole",
    "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
    "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma",
    "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis", "ef_dora",
    "em_alex", "ff_siwis", "hf_alpha", "hf_beta", "hm_omega", "hm_psi", "if_sara", "im_nicola",
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo", "pf_dora", "pm_alex",
    "pm_santa", "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi",
    "zm_yunxia", "zm_yunyang", "em_santa",
]  # fmt: skip


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


@dataclass(frozen=True)
class TTSModel(HubModel):
    """A sherpa-onnx TTS model and its speaker table (index = speaker id)."""

    model: str = ""
    tokens: str = "tokens.txt"
    data_dir: str = "espeak-ng-data"
    speakers: tuple[str, ...] = ()
    default_voice: str = ""
    language: str = "en"


@dataclass(frozen=True)
class KokoroModel(TTSModel):
    voices: str = "voices.bin"
    lexicon: str = "lexicon-us-en.txt"


@dataclass(frozen=True)
class KittenModel(TTSModel):
    voices: str = "voices.bin"


@dataclass(frozen=True)
class VitsModel(TTSModel):
    """Piper and other VITS voices."""


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


# Speaker tables come from each model's `speaker2id` metadata.
KITTEN_VOICES = (
    "expr-voice-2-m", "expr-voice-2-f", "expr-voice-3-m", "expr-voice-3-f",
    "expr-voice-4-m", "expr-voice-4-f", "expr-voice-5-m", "expr-voice-5-f",
)  # fmt: skip

TTS_MODELS: dict[TTSModels, TTSModel] = {
    # Default. Kokoro v1.0, 82M params, 114 MB int8, 54 voices.
    "kokoro-int8-multi-lang-v1_0": KokoroModel(
        repo_id="csukuangfj/kokoro-int8-multi-lang-v1_0",
        revision="2a360693d79b88b49b88e29aec2b53577f41f206",
        files=(
            "model.int8.onnx",
            "voices.bin",
            "tokens.txt",
            "lexicon-us-en.txt",
            "espeak-ng-data/**",
        ),
        model="model.int8.onnx",
        speakers=get_args(KokoroVoice),
        default_voice="af_heart",
    ),
    # Kitten nano v0.2, ~15M params, 24 MB fp16, 8 voices.
    "kitten-nano-en-v0_2": KittenModel(
        repo_id="csukuangfj/kitten-nano-en-v0_2-fp16",
        revision="7d14a97d38072576ddca7a673ab5bb49c43bb169",
        files=("model.fp16.onnx", "voices.bin", "tokens.txt", "espeak-ng-data/**"),
        model="model.fp16.onnx",
        speakers=KITTEN_VOICES,
        default_voice="expr-voice-2-f",
    ),
    # Piper en_US amy (low), 63 MB, one speaker, 16 kHz.
    "piper-en_US-amy-low": VitsModel(
        repo_id="csukuangfj/vits-piper-en_US-amy-low",
        revision="76da6ca287517a49f58b943c4c4fdb0c1e94d61f",
        files=("en_US-amy-low.onnx", "tokens.txt", "espeak-ng-data/**"),
        model="en_US-amy-low.onnx",
        speakers=("amy",),
        default_voice="amy",
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
