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
import numpy as np
from openai import AsyncOpenAI
from livekit import rtc


class DALLE3Plugin:
    """DALL-E 3 Plugin
    """

    def __init__(self):
        self._client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    async def generate_image_from_prompt(self,
                                         prompt: str,
                                         size='1024x1024',
                                         model='dall-e-3',
                                         quality="standard",
                                        ) -> rtc.VideoFrame:
        """Generate an image from a prompt

        Args:
            prompt (str): Prompt to generate an image from
            size (str, optional): Size of image to generate (1024x1024, 1024x1792 or 1792x1024). Defaults to '1024x1024'.

        Returns:
            rtc.VideoFrame: Image generated from prompt
        """
        response = await self._client.images.generate(model=model,
                                                      prompt=prompt,
                                                      size=size,
                                                      quality=quality,
                                                      n=1)
        image_url = response.data[0].url
        image = await asyncio.get_event_loop().run_in_executor(None, self._fetch_image, image_url)
        argb_array = bytearray(image.tobytes())

        argb_frame = rtc.ArgbFrame.create(
            rtc.VideoFormatType.FORMAT_ARGB, image.shape[0], image.shape[1])
        argb_frame.data[:] = argb_array
        return rtc.VideoFrame(
            0,
            rtc.VideoRotation.VIDEO_ROTATION_0,
            argb_frame.to_i420())

    def _fetch_image(self, url):
        response = requests.get(url, timeout=10)
        arr = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        from_to = [0, 3, 1, 1, 2, 2, 3, 0]
        cv2.mixChannels([img], [img], from_to)
        return img
