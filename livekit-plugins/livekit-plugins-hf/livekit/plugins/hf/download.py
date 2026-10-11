# Copyright 2023 LiveKit, Inc.
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

import os

from .log import logger

MODELS_ENV_VAR = "LIVEKIT_HF_MODELS"

# the files from_pretrained needs; skips duplicate .bin/.pth/.gguf weights some repos ship
_ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "*.py"]


def download_model(model: str, *, revision: str | None = None) -> str:
    """Download a model into the local Hugging Face cache and return its path.

    Uses the standard Hugging Face cache (``HF_HOME``) and token (``HF_TOKEN``), so a
    model fetched at build time is found by ``LLM(local_files_only=True)`` at runtime.
    """
    from huggingface_hub import snapshot_download

    logger.info("downloading model", extra={"model": model, "revision": revision})
    path: str = snapshot_download(model, revision=revision, allow_patterns=_ALLOW_PATTERNS)
    return path


def models_to_download(default_model: str) -> list[str]:
    """Models fetched by ``download-files``: ``LIVEKIT_HF_MODELS`` (comma separated)."""
    configured = os.environ.get(MODELS_ENV_VAR, "")
    models = [m.strip() for m in configured.split(",") if m.strip()]
    return models or [default_model]
