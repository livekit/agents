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

import asyncio
import io
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import aiohttp
from ._utils import API_URL, generate_token
from livekit import rtc
from PIL import Image


class Detector:
    @dataclass
    class DetectorConfig:
        name: str
        examples_to_include: List[str]
        examples_to_exclude: List[str]
        detection_threshold: float

    @dataclass
    class DetectionResult:
        top_left: Tuple[float, float]
        bottom_right: Tuple[float, float]
        score: float
        name: str

    def __init__(
        self,
        *,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        detector_configs: List[DetectorConfig],
    ):
        self._client_id = client_id
        self._client_secret = client_secret
        if client_id is None or client_secret is None:
            try:
                self._client_id = os.environ["DIRECTAI_CLIENT_ID"]
                self._client_secret = os.environ["DIRECTAI_CLIENT_SECRET"]
            except KeyError:
                raise Exception(
                    "DIRECTAI_CLIENT_ID or DIRECTAI_CLIENT_SECRET not set. Set them as environment variables or pass them in as arguments."
                )
        self._token: Optional[str] = None
        self._deploy_id: Optional[str] = None
        self._detector_configs = detector_configs
        self._token_lock = asyncio.Lock()
        self._deploy_lock = asyncio.Lock()
        self._http_session = aiohttp.ClientSession(base_url=API_URL)
        asyncio.ensure_future(self._deploy())

    async def detect(self, frame: rtc.VideoFrame) -> List[DetectionResult]:
        deploy_id = await self._get_deploy_id()
        converted = frame.convert(rtc.VideoBufferType.RGBA)
        image = Image.frombytes(
            "RGBA", (converted.width, converted.height), converted.data
        ).convert("RGB")

        output_stream = io.BytesIO()
        image.save(output_stream, format="JPEG")
        output_stream.seek(0)

        token = await self._get_token()
        headers = {"Authorization": f"Bearer {token}"}

        data = aiohttp.FormData()
        data.add_field(
            "data", output_stream, filename="image.jpeg", content_type="image/jpeg"
        )

        url = f"/detect?deployed_id={deploy_id}"
        self._check_http_session()
        async with self._http_session.post(url, data=data, headers=headers) as response:
            result_json = await response.json()
            results = []
            for class_results in result_json:
                for r in class_results:
                    results.append(
                        Detector.DetectionResult(
                            top_left=(r["tlbr"][0], r["tlbr"][1]),
                            bottom_right=(r["tlbr"][2], r["tlbr"][3]),
                            score=r["score"],
                            name=r["class"],
                        )
                    )
            return results

    async def _get_token(self):
        async with self._token_lock:
            if self._token:
                return self._token
            self._check_http_session()
            self._token = await generate_token(
                http_session=self._http_session,
                client_id=self._client_id,
                client_secret=self._client_secret,
            )
            return self._token

    async def _get_deploy_id(self):
        async with self._deploy_lock:
            if self._deploy_id:
                return self._deploy_id

            self._deploy_id = await self._deploy()
            return self._deploy_id

    async def _deploy(self):
        token = await self._get_token()
        headers = {"Authorization": f"Bearer {token}"}
        detector_configs = []
        for c in self._detector_configs:
            detector_configs.append(
                {
                    "name": c.name,
                    "examples_to_include": c.examples_to_include,
                    "examples_to_exclude": c.examples_to_exclude,
                    "detection_threshold": c.detection_threshold,
                }
            )
        body = {"detector_configs": detector_configs}
        self._check_http_session()
        async with self._http_session.post(
            "/deploy_detector", headers=headers, json=body
        ) as deploy_response:
            if deploy_response.status != 200:
                error_message = (await deploy_response.json())["message"]
                raise ValueError(error_message)
            deployed_id = (await deploy_response.json())["deployed_id"]
            return deployed_id

    def _check_http_session(self):
        if self._http_session.closed:
            self._http_session = aiohttp.ClientSession(base_url=API_URL)
