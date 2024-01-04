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

import fal
import cv2
import requests
import numpy as np
from livekit import rtc
import asyncio
import os

class SDXL:
    def __init__(self, model: str ="110602490-fast-sdxl") -> None:
        if not os.getenv("FAL_KEY_ID") and os.getenv("FAL_KEY_SECRET"):
            raise ValueError(
                "The Fal plugin requires FAL_KEY_ID and FAL_KEY_SECRET environment variables to be set."
            )

        self.model = model

    async def generate(self, prompt: str) -> rtc.ArgbFrame:
        handler = fal.apps.submit(
            self.model,
            arguments={"prompt": prompt, "sync_mode": False},
        )
        result = handler.get()
        image_url = result["images"][0]["url"]
        image = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_image, image_url
        )
        argb_array = bytearray(image.tobytes())

        argb_frame = rtc.ArgbFrame.create(
            rtc.VideoFormatType.FORMAT_ARGB, image.shape[0], image.shape[1]
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
