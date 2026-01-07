# Copyright 2025 LiveKit, Inc.
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

"""Vosk model management utilities."""

from pathlib import Path


class VoskModels:
    """
    Pre-defined Vosk model identifiers.

    Download models from: https://alphacephei.com/vosk/models
    """

    # English models
    EN_US_SMALL = "vosk-model-small-en-us-0.15"
    EN_US = "vosk-model-en-us-0.22"
    EN_US_LARGE = "vosk-model-en-us-0.22-lgraph"

    # Other languages
    CN = "vosk-model-cn-0.22"
    DE = "vosk-model-de-0.21"
    ES = "vosk-model-es-0.42"
    FR = "vosk-model-fr-0.22"
    IT = "vosk-model-it-0.22"
    JA = "vosk-model-ja-0.22"
    PT = "vosk-model-pt-0.3"
    RU = "vosk-model-ru-0.42"
    TR = "vosk-model-tr-0.3"
    VI = "vosk-model-vi-0.4"

    # Speaker identification model
    SPEAKER_MODEL = "vosk-model-spk-0.4"


DEFAULT_MODEL_DIR = Path.home() / ".cache" / "vosk" / "models"


def validate_model_path(model_path: str | Path) -> Path:
    """
    Validate that a model path exists and contains required files.

    Args:
        model_path: Path to Vosk model directory

    Returns:
        Validated Path object

    Raises:
        FileNotFoundError: If model path doesn't exist or is invalid
    """
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {path}\n"
            f"Download models from: https://alphacephei.com/vosk/models"
        )

    if not path.is_dir():
        raise FileNotFoundError(f"Model path must be a directory: {path}")

    # Check for required model files
    # Note: Structure varies between small and large models
    # Large models have graph/HCLG.fst, small models might have different graph files
    if not (path / "am/final.mdl").exists():
        raise FileNotFoundError(
            f"Model directory is missing 'am/final.mdl': {path}\n"
            f"This is required for all Vosk models."
        )

    if not (path / "conf/model.conf").exists():
        raise FileNotFoundError(f"Model directory is missing 'conf/model.conf': {path}")

    return path
