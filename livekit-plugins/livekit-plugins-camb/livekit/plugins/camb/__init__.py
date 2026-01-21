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

from camb.client import AsyncCambAI
from camb.core.api_error import ApiError
from livekit.agents import APIStatusError, Plugin

from .log import logger
from .tts import TTS
from .version import __version__

# Gender mapping from API integer to string
GENDER_MAP = {0: "Not Specified", 1: "Male", 2: "Female", 9: "Not Applicable"}


async def list_voices(
    *,
    api_key: str | None = None,
    base_url: str = "https://client.camb.ai/apis",
) -> list[dict]:
    """
    List available voices from Camb.ai.

    Args:
        api_key: Camb.ai API key (or use CAMB_API_KEY env var).
        base_url: API base URL.

    Returns:
        List of voice dicts with id, name, gender, age, language.

    Raises:
        ValueError: If no API key provided.
        APIStatusError: If API request fails.
    """
    api_key = api_key or os.environ.get("CAMB_API_KEY")
    if not api_key:
        raise ValueError("api_key required (or set CAMB_API_KEY environment variable)")

    async with AsyncCambAI(api_key=api_key, base_url=base_url) as client:
        try:
            voice_list = await client.voice_cloning.list_voices()
            voices = []

            for voice in voice_list:
                voice_id = voice.get("id")
                if voice_id is None:
                    continue

                gender_int = voice.get("gender")
                gender = GENDER_MAP.get(gender_int) if gender_int is not None else None

                voices.append(
                    {
                        "id": voice_id,
                        "name": voice.get("voice_name", ""),
                        "gender": gender,
                        "age": voice.get("age"),
                        "language": voice.get("language"),
                    }
                )

            return voices

        except ApiError as e:
            raise APIStatusError(
                f"Failed to list voices: {e.body}",
                status_code=e.status_code or 500,
            ) from e


class CambPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(CambPlugin())

__all__ = ["TTS", "list_voices", "__version__"]
