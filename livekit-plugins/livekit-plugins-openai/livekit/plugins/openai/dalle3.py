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

import os
import requests
import asyncio
import cv2
import openai
import numpy as np
from livekit import rtc
from typing import Literal, Optional
from .models import DalleModels


class Dalle3:
    def __init__(self, api_key: Optional[str] = None) -> None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set")

        self._client = openai.AsyncOpenAI(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"],
        model: DalleModels = "dall-e-3",
        quality: Literal["standard", "hd"] = "standard",
    ) -> rtc.ArgbFrame:
        response = await self._client.images.generate(
            model=model, prompt=prompt, size=size, quality=quality, n=1
        )
        image_url = response.data[0].url
        image = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_image, image_url
        )
        argb_array = bytearray(image.tobytes())

        # shape is (height, width, channels)
        argb_frame = rtc.ArgbFrame.create(
            rtc.VideoFormatType.FORMAT_ARGB, image.shape[1], image.shape[0]
        )
        argb_frame.data[:] = argb_array
        return argb_frame

    def _fetch_image(self, url):
        response = requests.get(url, timeout=10)
        arr = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        from_to = [0, 3, 1, 1, 2, 2, 3, 0]
        cv2.mixChannels([img], [img], from_to)
        return img
