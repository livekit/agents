import asyncio
import aiohttp
from dataclasses import dataclass
from livekit import rtc
from typing import Optional
from PIL import Image
from common import generate_token, API_URL
import io


class Detector:
    @dataclass
    class DetectorConfig:
        name: str
        examples_to_include: [str]
        examples_to_exclude: [str]
        detection_threshold: float

    def __init__(self, detector_configs: [DetectorConfig], nms_threshold: float = 0.4):
        self._token: Optional[str] = None
        self._deploy_id: Optional[str] = None
        self._detector_configs = detector_configs
        self._nms_threshold = nms_threshold
        self._token_lock = asyncio.Lock()
        self._deploy_lock = asyncio.Lock()
        asyncio.ensure_future(self._deploy())

    async def _get_token(self):
        if self._token:
            return self._token

        with await self._token_lock:
            self._token = await generate_token()
            return self._token

    async def _get_deploy_id(self):
        if self._deploy_id:
            return self._deploy_id

        with await self._deploy_lock:
            return self._deploy_id

    async def _deploy(self):
        token = await self._get_token()
        headers = {"Authorization": f"Bearer {token}"}
        detector_configs = []
        for d in self._detector_configs:
            detector_configs.append(
                {
                    "name": d.name,
                    "examples_to_include": d.examples_to_include,
                    "examples_to_exclude": d.examples_to_exclude,
                    "detection_threshold": d.detection_threshold,
                }
            )
        body = {
            "detector_configs": detector_configs,
            "nms_threshold": self._nms_threshold,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{API_URL}/deploy_detector", headers=headers, json=body
            ) as deploy_response:
                if deploy_response.status != 200:
                    error_message = (await deploy_response.json())["message"]
                    raise ValueError(error_message)
                deployed_classifier_id = (await deploy_response.json())["deployed_id"]
                return deployed_classifier_id

    async def detect(self, frame: rtc.VideoFrame):
        deploy_id = await self._get_deploy_id()
        image = Image.frombytes(
            "RGBA", (frame.width, frame.height), frame.data
        ).convert("RGB")
        output_stream = io.BytesIO()
        image.save(output_stream, format="JPEG")
        data = aiohttp.FormData()

        token = await self._get_token()
        headers = {"Authorization": f"Bearer {token}"}

        data.add_field(
            "image",
            io.BytesIO(output_stream),
            filename="image.jpg",
            content_type="image/jpeg",
        )

        url = f"{API_URL}/detect?deploy_id={deploy_id}"
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(url, data=data) as response:
                return await response.json()
