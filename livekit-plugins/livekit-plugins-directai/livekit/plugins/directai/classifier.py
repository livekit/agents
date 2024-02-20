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
from typing import Optional, Dict, List

import aiohttp
from ._utils import API_URL, generate_token
from livekit import rtc
from PIL import Image


class Classifier:
    @dataclass
    class ClassifierConfig:
        name: str
        examples_to_include: List[str]
        examples_to_exclude: List[str]

    @dataclass
    class ClassificationResult:
        scores: Dict[str, float]
        raw_scores: Dict[str, float]
        pred: str

    def __init__(
        self,
        *,
        client_id: Optional[str],
        client_secret: Optional[str],
        classifier_configs: List[ClassifierConfig],
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
        self._classifier_configs = classifier_configs
        self._token_lock = asyncio.Lock()
        self._deploy_lock = asyncio.Lock()
        self._http_session = aiohttp.ClientSession(base_url=API_URL)
        asyncio.ensure_future(self._deploy())

    async def classify(self, frame: rtc.VideoFrame) -> ClassificationResult:
        deploy_id = await self._get_deploy_id()
        buffer = frame.convert(rtc.VideoBufferType.RGBA)
        image = Image.frombytes(
            "RGBA", (buffer.width, buffer.height), buffer.data
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
        url = f"/classify?deployed_id={deploy_id}"
        self._check_http_session()
        async with self._http_session.post(url, data=data, headers=headers) as response:
            resp_json = await response.json()
            return Classifier.ClassificationResult(
                scores=resp_json["scores"],
                raw_scores=resp_json["raw_scores"],
                pred=resp_json["pred"],
            )

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
        classifier_configs = []
        for c in self._classifier_configs:
            classifier_configs.append(
                {
                    "name": c.name,
                    "examples_to_include": c.examples_to_include,
                    "examples_to_exclude": c.examples_to_exclude,
                }
            )
        body = {"classifier_configs": classifier_configs}
        self._check_http_session()
        async with self._http_session.post(
            "/deploy_classifier", headers=headers, json=body
        ) as deploy_response:
            if deploy_response.status != 200:
                error_message = (await deploy_response.json())["message"]
                raise ValueError(error_message)
            deployed_classifier_id = (await deploy_response.json())["deployed_id"]
            return deployed_classifier_id

    def _check_http_session(self):
        if self._http_session.closed:
            self._http_session = aiohttp.ClientSession(base_url=API_URL)
