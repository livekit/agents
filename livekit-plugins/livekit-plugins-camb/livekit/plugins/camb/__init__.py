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

import aiohttp

from livekit.agents import APIStatusError, Plugin

from .log import logger
from .models import VoiceInfo
from .tts import TTS
from .version import __version__


async def list_voices(
    *,
    api_key: str | None = None,
    base_url: str = "https://client.camb.ai/apis",
) -> list[VoiceInfo]:
    """
    List available voices from Camb.ai.

    Returns voices categorized as:
    - Public voices (pre-built)
    - Shared voices
    - Custom voices (user-created)

    Args:
        api_key: Camb.ai API key (or use CAMB_API_KEY env var).
        base_url: API base URL.

    Returns:
        List of VoiceInfo objects with id, name, gender, language.

    Raises:
        ValueError: If no API key provided.
        APIStatusError: If API request fails.
    """
    api_key = api_key or os.environ.get("CAMB_API_KEY")
    if not api_key:
        raise ValueError("api_key required (or set CAMB_API_KEY environment variable)")

    headers = {
        "x-api-key": api_key,
        "Accept": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{base_url}/list-voices",
            headers=headers,
        ) as response:
            if response.status != 200:
                error_body = await response.text()
                raise APIStatusError(
                    f"Failed to list voices: {error_body}",
                    status_code=response.status,
                )

            data = await response.json()
            voices = []

            # Parse response - API returns a list directly
            voice_list = data if isinstance(data, list) else data.get("voices", [])

            for voice in voice_list:
                # Map gender integer to string (0=Not Specified, 1=Male, 2=Female, 9=Not Applicable)
                gender_map = {0: "Not Specified", 1: "Male", 2: "Female", 9: "Not Applicable"}
                gender = gender_map.get(voice.get("gender"), None)

                voices.append(
                    VoiceInfo(
                        id=voice["id"],
                        name=voice["voice_name"],  # API uses "voice_name" not "name"
                        gender=gender,
                        language=voice.get("language"),  # Returns integer language code
                    )
                )

            return voices


class CambPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(CambPlugin())

__all__ = ["TTS", "VoiceInfo", "list_voices", "__version__"]
